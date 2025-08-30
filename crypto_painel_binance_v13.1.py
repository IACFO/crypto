# -*- coding: utf-8 -*-
"""
Painel – Binance v13.1 (UMFutures / binance-connector)
"""
from __future__ import annotations
import os, time, importlib.util, importlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ===== Streamlit (tem que vir antes de usar st.*) =====
st.set_page_config(page_title="📊 Painel – Binance v13.1 (UMFutures)", layout="wide")

# ===== Verificação do pacote binance-connector (diagnóstico robusto) =====
def _diagnose_binance_connector():
    # 1) O pacote 'binance-connector' expõe o módulo top-level 'binance'
    spec_pkg = importlib.util.find_spec("binance")
    pkg_path = getattr(spec_pkg, "origin", None) if spec_pkg else None

    # 2) Tentar localizar o submódulo 'binance.um_futures'
    spec_umf = importlib.util.find_spec("binance.um_futures")

    # 3) Versão (quando disponível)
    try:
        import binance  # type: ignore
        pkg_ver = getattr(binance, "__version__", "desconhecida")
    except Exception:
        pkg_ver = "não encontrado"

    return spec_pkg is not None, spec_umf is not None, pkg_ver, pkg_path

pkg_ok, umf_ok, pkg_ver, pkg_path = _diagnose_binance_connector()

if not pkg_ok or not umf_ok:
    st.error(
        "Falha ao importar `UMFutures` do **binance-connector**.\n\n"
        f"Detalhe: módulo 'binance' encontrado? {pkg_ok} | submódulo 'binance.um_futures' encontrado? {umf_ok}\n"
        f"versão reportada: {pkg_ver}\n"
        f"caminho carregado: {pkg_path}\n\n"
        "Como corrigir:\n"
        "1) Confirme no **requirements.txt**: `binance-connector==3.12.0`\n"
        "2) No Render: **Settings → Clear build cache → Deploy latest**\n"
        "3) Verifique que **não existe** `python-binance` no requirements\n"
        "4) Verifique que **não existe** arquivo/pasta chamada `binance` dentro do repositório (isso sombreia o pacote)\n"
        "5) Build Command: `pip install --upgrade pip wheel setuptools && pip install -r requirements.txt`\n"
        "6) Start Command: `streamlit run crypto_painel_binance_v13.1.py --server.port $PORT --server.address 0.0.0.0`\n"
    )
    st.stop()

# Importa definitivamente (sabemos que existe)
from binance.um_futures import UMFutures  # type: ignore

st.caption(f"🧩 Conector OK • binance-connector versão: {pkg_ver} • caminho: {pkg_path}")

BINANCE_REST = "https://api.binance.com"
RECV_WINDOW_MS = 60_000  # 60s

# ===== NTP (referência) =====
def show_ntp_reference() -> None:
    try:
        import ntplib
        from time import ctime
        ntp = ntplib.NTPClient()
        resp = ntp.request("pool.ntp.org", version=3, timeout=2)
        st.caption(f"🕒 NTP ref: {ctime(resp.tx_time)}")
    except Exception as e:
        st.caption(f"🕒 NTP ref indisponível: {e}")

show_ntp_reference()

# ===== Cliente UMFutures via ENV =====
@st.cache_resource
def get_client() -> UMFutures:
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        st.error("Credenciais ausentes. Defina BINANCE_API_KEY e BINANCE_API_SECRET como variáveis de ambiente.")
        st.stop()
    cl = UMFutures(key=api_key, secret=api_secret)
    try:
        cl.ping()
        srv = cl.time()  # {'serverTime': ...}
        st.caption(f"⏱️ Server time (ms): {srv.get('serverTime')}")
    except Exception as e:
        st.error(f"Falha ao conectar no UMFutures: {e}")
        st.stop()
    return cl

client = get_client()

# ============== Helpers REST públicos (klines spot) ==============
def fetch_klines_rest(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    r = requests.get(
        f"{BINANCE_REST}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": min(int(limit), 1500)},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    cols = ["openTime","open","high","low","close","volume","closeTime","quote","n","tb_base","tb_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)[["openTime","open","high","low","close","volume"]]
    df.columns = ["t","o","h","l","c","v"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for c in ["o","h","l","c","v"]:
        df[c] = df[c].astype(float)
    return df

# ============== Indicadores e features ==============
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        (df["h"] - df["l"]),
        (df["h"] - df["c"].shift()).abs(),
        (df["l"] - df["c"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = out["c"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["c"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["c"].ewm(span=200, adjust=False).mean()
    out["atr14"] = atr(out, 14)
    out["vol_ma20"] = out["v"].rolling(20).mean()
    out["vol_rel"] = out["v"] / (out["vol_ma20"] + 1e-9)
    out["spread_rel"] = (out["h"] - out["l"]) / (out["atr14"] + 1e-9)
    out["body_frac"] = (out["c"] - out["o"]).abs() / ((out["h"] - out["l"]).replace(0, 1e-9))
    out["strength_score"] = 0.45 * out["vol_rel"] + 0.35 * out["spread_rel"] + 0.20 * out["body_frac"]
    out["rsi14"] = rsi(out["c"], 14)
    return out

def trend_label(row: pd.Series) -> str:
    if row["ema20"] > row["ema50"] and row["c"] > row["ema200"] and row["rsi14"] > 55:
        return "ALTA"
    if row["ema20"] < row["ema50"] and row["c"] < row["ema200"] and row["rsi14"] < 45:
        return "BAIXA"
    return "NEUTRA"

def confidence_from_features(row: pd.Series, trend: str, align_count: int) -> int:
    sr = float(np.tanh(max(row.get("spread_rel", 0), 0)))
    vr = float(min(max(row.get("vol_rel", 0) / 2.0, 0), 1))
    stf = float(min(max(row.get("strength_score", 0) / 2.0, 0), 1))
    rsi14 = float(row.get("rsi14", 50))
    rsi_comp = (rsi14 - 50) / 50.0
    if trend == "ALTA":
        rsi_comp = max(0, rsi_comp)
    elif trend == "BAIXA":
        rsi_comp = max(0, -rsi_comp)
    else:
        rsi_comp = 0
    align_bonus = align_count / 3.0
    conf = (0.35 * stf + 0.25 * vr + 0.20 * sr + 0.10 * rsi_comp + 0.10 * align_bonus) * 100
    return int(round(min(max(conf, 0), 100)))

# ============== Multi-timeframe (cache) ==============
@st.cache_data(ttl=30)
def load_multitf(symbol: str) -> Dict[str, pd.DataFrame]:
    d1 = fetch_klines_rest(symbol, "1d", 400)
    h1 = fetch_klines_rest(symbol, "1h", 500)
    m5 = fetch_klines_rest(symbol, "5m", 500)
    return {"1D": add_core_features(d1),
            "1H": add_core_features(h1),
            "5M": add_core_features(m5)}

# ============== UI Controles principais ==============
with st.container():
    colA, colB, colC = st.columns([1.5, 1, 1])
    with colA:
        symbol = st.selectbox(
            "Par (USDT-M Futures)",
            ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"],
            index=1
        )
    with colB:
        margin_type = st.selectbox("Tipo de margem", ["ISOLATED","CROSSED"], index=0)
    with colC:
        leverage = st.number_input("Alavancagem", min_value=1, max_value=125, value=10, step=1)

mtf = load_multitf(symbol)
if not isinstance(mtf, dict) or any(df is None or df.empty for df in mtf.values()):
    st.error("Falha ao carregar dados multi-timeframe.")
    st.stop()

# Construir leitura
blocks: Dict[str, dict] = {}
for tf, df in mtf.items():
    row = df.iloc[-1]
    blocks[tf] = {
        "trend": trend_label(row),
        "rsi": float(row["rsi14"]),
        "price": float(row["c"]),
        "ema20": float(row["ema20"]),
        "ema50": float(row["ema50"]),
        "ema200": float(row["ema200"]),
        "vol_rel": float(row["vol_rel"]),
        "strength": float(row["strength_score"]),
        "atr": float(row["atr14"]),
        "spread_rel": float(row["spread_rel"]),
    }

align_map = {"ALTA": 1, "BAIXA": -1, "NEUTRA": 0}
vals = [align_map[blocks[tf]["trend"]] for tf in ["1D","1H","5M"]]
count_up = sum(1 for v in vals if v == 1)
count_dn = sum(1 for v in vals if v == -1)
align_count = max(count_up, count_dn)
consensus = "ALTA" if count_up > count_dn else ("BAIXA" if count_dn > count_up else "NEUTRA")
conf = confidence_from_features(mtf["5M"].iloc[-1], consensus, align_count)

c1, c2 = st.columns([2,1])
with c1:
    st.subheader("🔎 Leitura Multi-timeframe")
    a,b,c = st.columns(3)
    for tf, col in zip(["1D","1H","5M"], [a,b,c]):
        info = blocks[tf]
        badge = "🟢" if info["trend"] == "ALTA" else ("🔴" if info["trend"] == "BAIXA" else "⚪")
        col.metric(f"{tf} – {badge} {info['trend']}",
                   value=f"RSI {info['rsi']:.1f}",
                   help=f"EMA20/50/200: {info['ema20']:.4f}/{info['ema50']:.4f}/{info['ema200']:.4f}\n"
                        f"Vol.rel: {info['vol_rel']:.2f}× | Str: {info['strength']:.2f}")
    st.caption(f"Alinhamento: {align_count}/3 • Consenso: {consensus}")
with c2:
    st.subheader("🔐 Confiança do Sinal")
    st.metric("Nível", f"{conf}/100")

# ============== Parâmetros de risco / alvo ==============
st.subheader("🎯 Parâmetros de Trade")
col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
with col_r1:
    atr_mult_sl = st.number_input("SL = ATR ×", 0.2, 5.0, 1.0, step=0.1)
with col_r2:
    rr = st.number_input("R:R (TP = R × SL)", 0.5, 5.0, 1.5, step=0.1)
with col_r3:
    min_conf = st.number_input("Confiança mínima", 0, 100, 60, step=5)
with col_r4:
    stake = st.number_input("Stake (USDT)", 1.0, 1e9, 50.0, step=1.0)
with col_r5:
    banca = st.number_input("Banca (USDT)", 10.0, 1e9, 1000.0, step=10.0)

price_now = float(blocks["5M"]["price"])
atr5 = float(blocks["5M"]["atr"])

# Sugestão
suggestion = "AGUARDAR"; reason = "Condições insuficientes"
if align_count == 3 and conf >= min_conf:
    suggestion = "COMPRAR" if consensus == "ALTA" else ("VENDER" if consensus == "BAIXA" else "AGUARDAR")
    reason = f"Alinhamento 3/3 e confiança {conf}"
elif align_count == 2 and conf >= (min_conf + 10):
    suggestion = "COMPRAR" if count_up == 2 else ("VENDER" if count_dn == 2 else "AGUARDAR")
    reason = f"Alinhamento 2/3 com confiança {conf}"

# TP/SL sugeridos
sl_price = None; tp_price = None; stop_pct = 0.0; tp_pct = 0.0
if suggestion in ("COMPRAR","VENDER") and atr5 > 0:
    if suggestion == "COMPRAR":
        sl_price = price_now - atr_mult_sl * atr5
        tp_price = price_now + rr * atr_mult_sl * atr5
        stop_pct = (price_now - sl_price) / price_now
        tp_pct = (tp_price - price_now) / price_now
    else:
        sl_price = price_now + atr_mult_sl * atr5
        tp_price = price_now - rr * atr_mult_sl * atr5
        stop_pct = (sl_price - price_now) / price_now
        tp_pct = (price_now - tp_price) / price_now

# Heurística de alavancagem
profile_caps = {"Conservador": 5, "Moderado": 10, "Agressivo": 20}
perfil = st.selectbox("Perfil", list(profile_caps.keys()), index=1)
cap = profile_caps.get(perfil, 10)

base_from_conf = (
    2 if conf < 55 else
    4 if conf < 65 else
    6 if conf < 75 else
    9 if conf < 85 else
    12 if conf < 92 else
    15
)
if stop_pct > 0:
    if stop_pct < 0.003:
        base_from_conf = int(round(base_from_conf * 1.25))
    elif stop_pct > 0.006:
        base_from_conf = int(round(base_from_conf * 0.7))
lev_conf = max(1, min(base_from_conf, cap))

# Estimativas
taker_fee_pct = st.number_input("Taxa taker (%)", 0.0, 0.5, 0.04, step=0.01) / 100.0
fees_roundtrip = 2 * taker_fee_pct
gross_gain_pct = tp_pct
gross_loss_pct = stop_pct
net_gain_pct = max(gross_gain_pct - fees_roundtrip, 0) * lev_conf
net_loss_pct = (gross_loss_pct + fees_roundtrip) * lev_conf
est_gain_usdt = stake * net_gain_pct
est_loss_usdt = stake * net_loss_pct

colS1, colS2 = st.columns([1,2])
with colS1:
    color = "🟢" if suggestion == "COMPRAR" else ("🔴" if suggestion == "VENDER" else "⏸️")
    st.metric("Sugestão", f"{color} {suggestion}", help=reason)
with colS2:
    if sl_price and tp_price:
        st.info(f"Entrada: ~{price_now:.6f} | Stop: {sl_price:.6f} | Alvo: {tp_price:.6f}\n\n"
                f"Stake: {stake:.2f} USDT | Alavancagem sugerida: {lev_conf}×\n\n"
                f"Estimativa → TP: ~{est_gain_usdt:.2f} | SL: ~{est_loss_usdt:.2f}")
    else:
        st.info("Sem TP/SL sugeridos (aguarde melhor alinhamento/confiança).")

# ============== Funções Futures (UMFutures) ==============
def setup_futures_pair(symbol: str, leverage: int, margin_type: str) -> bool:
    try:
        client.change_margin_type(symbol=symbol, marginType=margin_type, recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        if "No need to change margin type" not in str(e):
            st.error(f"Erro ao definir margem: {e}")
            return False
    try:
        client.change_leverage(symbol=symbol, leverage=int(leverage), recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        st.error(f"Erro ao definir alavancagem: {e}")
        return False
    return True

def executar_ordem_mercado(symbol: str, side: str, quantity: float, sl_price: float, tp_price: float):
    try:
        # Entrada a mercado
        order_resp = client.new_order(
            symbol=symbol,
            side=side,  # "BUY" ou "SELL"
            type="MARKET",
            quantity=quantity,
            recvWindow=RECV_WINDOW_MS,
        )
        # SL/TP como closePosition
        if side == "BUY":
            client.new_order(
                symbol=symbol, side="SELL", type="STOP_MARKET",
                stopPrice=round(sl_price, 6), closePosition="true",
                timeInForce="GTC", recvWindow=RECV_WINDOW_MS
            )
            client.new_order(
                symbol=symbol, side="SELL", type="TAKE_PROFIT_MARKET",
                stopPrice=round(tp_price, 6), closePosition="true",
                timeInForce="GTC", recvWindow=RECV_WINDOW_MS
            )
        else:
            client.new_order(
                symbol=symbol, side="BUY", type="STOP_MARKET",
                stopPrice=round(sl_price, 6), closePosition="true",
                timeInForce="GTC", recvWindow=RECV_WINDOW_MS
            )
            client.new_order(
                symbol=symbol, side="BUY", type="TAKE_PROFIT_MARKET",
                stopPrice=round(tp_price, 6), closePosition="true",
                timeInForce="GTC", recvWindow=RECV_WINDOW_MS
            )
        return True, order_resp
    except Exception as e:
        return False, str(e)

def get_symbol_precision(symbol: str) -> Dict[str, int]:
    try:
        ex = client.exchange_info()
        for s in ex.get("symbols", []):
            if s.get("symbol") == symbol:
                qty_step = None; px_tick = None
                for f in s.get("filters", []):
                    if f.get("filterType") == "LOT_SIZE":
                        qty_step = f.get("stepSize")
                    if f.get("filterType") == "PRICE_FILTER":
                        px_tick = f.get("tickSize")
                def dec_places(x: Optional[str]) -> int:
                    if not x: return 3
                    s2 = x.rstrip("0")
                    return len(s2.split(".")[1]) if "." in s2 else 0
                return {"qty_precision": dec_places(qty_step), "px_precision": dec_places(px_tick)}
    except Exception:
        pass
    return {"qty_precision": 3, "px_precision": 2}

# ============== Execução ==============
st.subheader("⚡ Execução")
colE1, colE2, colE3, colE4 = st.columns([1,1,1,1])
with colE1:
    desired_side = st.selectbox("Direção", ["BUY","SELL"], index=0)

prec = get_symbol_precision(symbol)
qty_prec = int(prec["qty_precision"])
px_prec = int(prec["px_precision"])

qty_calc = (stake * max(leverage, 1)) / max(price_now, 1e-9)
qty = float(np.floor(qty_calc * (10**qty_prec)) / (10**qty_prec))

with colE2:
    qty = st.number_input(f"Quantidade ({symbol})", min_value=0.0, value=float(qty),
                          step=10**(-qty_prec), format=f"%.{qty_prec}f")
with colE3:
    sl_input = st.number_input("Stop (SL)", min_value=0.0,
                               value=float(round(sl_price or price_now, px_prec)),
                               step=10**(-px_prec), format=f"%.{px_prec}f")
with colE4:
    tp_input = st.number_input("Alvo (TP)", min_value=0.0,
                               value=float(round(tp_price or price_now, px_prec)),
                               step=10**(-px_prec), format=f"%.{px_prec}f")

colB1, colB2 = st.columns([1,1])
with colB1:
    if st.button("Aplicar margem + alavancagem", use_container_width=True):
        ok = setup_futures_pair(symbol, leverage, margin_type)
        st.success("Margem/alavancagem ajustadas!") if ok else st.error("Falha ao ajustar margem/alavancagem.")
with colB2:
    execute_now = st.button("🚀 Executar Ação", use_container_width=True)

# Saldo / conexão (opcional)
try:
    acc = client.account(recvWindow=RECV_WINDOW_MS)
    total_margin_balance = acc.get("totalMarginBalance")
    if total_margin_balance is not None:
        st.caption(f"✅ Conectado | Balance: {total_margin_balance} USDT")
except Exception as e:
    st.error(f"❌ Erro na conexão com Binance: {e}")

if execute_now:
    if setup_futures_pair(symbol, leverage, margin_type):
        ok, res = executar_ordem_mercado(
            symbol=symbol, side=desired_side, quantity=qty,
            sl_price=sl_input, tp_price=tp_input
        )
        if ok:
            st.success("✅ Ordem executada com sucesso!")
            st.json(res)
        else:
            st.error(f"❌ Erro na execução: {res}")

st.caption("⚠️ Uso educacional. Ajuste TP/SL/quantidade de acordo com sua gestão e as precisões do símbolo.")
