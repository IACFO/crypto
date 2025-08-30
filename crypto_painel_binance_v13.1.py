# -*- coding: utf-8 -*-
"""
Painel – Binance v13.1 (Scanner + Foco + Execução + PnL, robusto p/ Render)
-------------------------------------------------------------------------------
• Mantém leitura/scan mesmo se execução estiver indisponível por região.
• Modo de backend por ENV: BINANCE_MODE = off | testnet | live  (default: off)
• Testnet Futures suportada via python-binance (oficial).
• Trata o erro de região restrita e desabilita execução com aviso claro.
"""

from __future__ import annotations

import os, math, time
from typing import Optional, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# === Só usamos python-binance (como no seu ambiente local) ===
from binance.client import Client
from binance.enums import *

# ---------------- UI base ----------------
st.set_page_config(page_title="📊 Painel – Binance v13.1", layout="wide")
st.markdown("""
<style>
:root{
  --bg:#0b0e11;--card:#14181d;--card2:#161a1e;--text:#eaecef;--muted:#a7b1c2;
  --accent:#f0b90b;--success:#1f8b4c;--danger:#b02a37;--border:#232a31;
}
.stApp{background:var(--bg);color:var(--text);}
.block-container{padding-top:2.25rem !important;}
h1, h2, h3, .stMarkdown h1{color:var(--accent) !important;margin-top:0 !important; line-height:1.25;}
section[data-testid="stSidebar"]{background:var(--card2)}
div[role="alert"]{background:#0f1317 !important;border:1px solid var(--border) !important;color:var(--text) !important;}
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:10px;}
div[data-testid="stMetricLabel"]{color:var(--muted) !important;}
div[data-testid="stMetricValue"]{color:var(--text) !important;font-weight:700;}
div.stButton > button{background:var(--accent);color:#111;border:0;border-radius:10px;font-weight:700}
div.stButton > button:disabled{opacity:.45}
.sugestao-box{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px}
hr{border-color:var(--border);}
</style>
""", unsafe_allow_html=True)
st.title("📊 Painel – Binance v13.1")

# ---------------- NTP (visual) ----------------
def show_ntp_reference():
    try:
        import ntplib
        from time import ctime
        ntp = ntplib.NTPClient()
        resp = ntp.request("pool.ntp.org", version=3, timeout=2)
        st.caption(f"🕒 NTP ref: {ctime(resp.tx_time)}")
    except Exception as e:
        st.caption(f"🕒 NTP ref indisponível: {e}")
show_ntp_reference()

# ---------------- Constantes ----------------
BINANCE_REST = "https://api.binance.com"
RECV_WINDOW_MS = 90_000  # 90s

# ---------------- Helpers REST (klines públicos) ----------------
# Endpoints públicos com fallback (evita 451/403/região)
KLINES_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api3.binance.com",
    # Mirror público oficial para dados (geralmente não bloqueado)
    "https://data-api.binance.vision",
]

def _fetch_klines_from(base: str, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    # data-api.binance.vision usa a MESMA rota /api/v3/klines
    url = f"{base}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(int(limit), 1500)}
    r = requests.get(url, params=params, timeout=10)
    # Em caso de 451/403/5xx, deixamos levantar para tentar próximo base
    r.raise_for_status()
    data = r.json()

    cols = ["openTime","open","high","low","close","volume","closeTime","quote","n","tb_base","tb_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)[["openTime","open","high","low","close","volume"]]
    df.columns = ["t","o","h","l","c","v"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for c in ["o","h","l","c","v"]:
        df[c] = df[c].astype(float)
    return df

def fetch_klines_rest(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    last_error = None
    for base in KLINES_BASES:
        try:
            df = _fetch_klines_from(base, symbol, interval, limit)
            # Mostra uma vez qual mirror foi usado (para diagnóstico)
            st.session_state.setdefault("_klines_base_used", set())
            if base not in st.session_state["_klines_base_used"]:
                st.session_state["_klines_base_used"].add(base)
                st.caption(f"📡 Klines carregados via **{base}**")
            return df
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # 451/403/429/5xx → tenta próximo mirror
            if status in (451, 403, 429) or (status and 500 <= status < 600):
                last_error = e
                continue
            # Outros erros HTTP: propaga
            raise
        except Exception as e:
            # Timeout / rede etc → tenta próximo
            last_error = e
            continue
    # Se chegou aqui, todos falharam
    raise last_error if last_error else RuntimeError("Falha ao obter klines em todos os endpoints")

# ---------------- Indicadores ----------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        (df["h"] - df["l"]),
        (df["h"] - df["c"].shift()).abs(),
        (df["l"] - df["c"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"]  = out["c"].ewm(span=20,  adjust=False).mean()
    out["ema50"]  = out["c"].ewm(span=50,  adjust=False).mean()
    out["ema200"] = out["c"].ewm(span=200, adjust=False).mean()
    out["atr14"]  = atr(out, 14)
    out["vol_ma20"] = out["v"].rolling(20).mean()
    out["vol_rel"]  = out["v"] / (out["vol_ma20"] + 1e-9)
    out["spread_rel"] = (out["h"] - out["l"]) / (out["atr14"] + 1e-9)
    out["body_frac"]  = (out["c"] - out["o"]).abs() / ((out["h"] - out["l"]).replace(0, 1e-9))
    out["strength_score"] = 0.45*out["vol_rel"] + 0.35*out["spread_rel"] + 0.20*out["body_frac"]
    out["rsi14"] = rsi(out["c"], 14)
    return out

def trend_label(row: pd.Series) -> str:
    if row["ema20"] > row["ema50"] and row["c"] > row["ema200"] and row["rsi14"] > 55:
        return "ALTA"
    if row["ema20"] < row["ema50"] and row["c"] < row["ema200"] and row["rsi14"] < 45:
        return "BAIXA"
    return "NEUTRA"

def confidence_from_features(row: pd.Series, trend: str, align_count: int) -> int:
    sr  = float(np.tanh(max(row.get("spread_rel", 0), 0)))
    vr  = float(min(max(row.get("vol_rel", 0)/2.0, 0), 1))
    stf = float(min(max(row.get("strength_score", 0)/2.0, 0), 1))
    rsi14 = float(row.get("rsi14", 50))
    rsi_comp = (rsi14 - 50) / 50.0
    if trend == "ALTA":   rsi_comp = max(0,  rsi_comp)
    elif trend == "BAIXA":rsi_comp = max(0, -rsi_comp)
    else:                 rsi_comp = 0
    align_bonus = align_count / 3.0
    conf = (0.35*stf + 0.25*vr + 0.20*sr + 0.10*rsi_comp + 0.10*align_bonus) * 100
    return int(round(min(max(conf, 0), 100)))

# ---------------- Multi-timeframe ----------------
@st.cache_data(ttl=30)
def load_multitf(symbol: str) -> Dict[str, pd.DataFrame]:
    d1 = add_core_features(fetch_klines_rest(symbol, "1d", 400))
    h1 = add_core_features(fetch_klines_rest(symbol, "1h", 500))
    m5 = add_core_features(fetch_klines_rest(symbol, "5m", 500))
    return {"1D": d1, "1H": h1, "5M": m5}

# ---------------- Backend mode (ENV) ----------------
BINANCE_MODE = os.environ.get("BINANCE_MODE", "off").strip().lower()  # 'off' | 'testnet' | 'live'
EXECUTION_ENABLED = BINANCE_MODE in ("testnet", "live")
execution_blocked_reason = None

# ---------------- Client helpers ----------------
def sync_client_time(c: Client) -> None:
    try:
        srv = c.get_server_time()
        drift = int(srv["serverTime"]) - int(time.time() * 1000)
        setattr(c, "timestamp_offset", drift)
        setattr(c, "TIME_OFFSET", drift)
        st.caption(f"⏱️ Offset aplicado (server - local): {drift} ms")
    except Exception as e:
        st.warning(f"⚠️ Falha ao sincronizar tempo: {e}")

def make_client() -> Optional[Client]:
    """Cria client conforme BINANCE_MODE. Se houver bloqueio regional, retorna None e seta motivo."""
    global execution_blocked_reason
    if not EXECUTION_ENABLED:
        return None

    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        execution_blocked_reason = "Credenciais ausentes (BINANCE_API_KEY / BINANCE_API_SECRET)"
        return None

    try:
        if BINANCE_MODE == "testnet":
            c = Client(api_key, api_secret, testnet=True)
            # URLs de futures testnet (python-binance)
            c.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            c.FUTURES_DATA_URL = "https://testnet.binancefuture.com/futures/data"
        else:
            c = Client(api_key, api_secret)

        # Primeiro ping privado (pode disparar restrição de região)
        c.ping()
        sync_client_time(c)
        return c

    except Exception as e:
        s = str(e)
        # Mensagem típica de região restrita
        if "restricted location" in s.lower():
            execution_blocked_reason = "Execução desabilitada: localização do servidor não é elegível p/ Binance.com."
            return None
        execution_blocked_reason = f"Falha ao inicializar client: {s}"
        return None

# Cria client sob demanda (não trava o app)
client: Optional[Client] = make_client()

def _call_with_resync(fn, *args, **kwargs):
    if client is None:
        raise RuntimeError("Client indisponível para chamadas privadas.")
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        s = str(e)
        if "-1021" in s or "outside of the recvWindow" in s:
            sync_client_time(client)
            return fn(*args, **kwargs)
        raise

# ---------------- Filtros / arredondamentos ----------------
@st.cache_data(ttl=600)
def get_symbol_filters(symbol: str) -> Dict[str, float]:
    if client is None:
        # fallback genérico só pra calcular qty localmente
        return {"stepSize": 0.001, "minQty": 0.001, "tickSize": 0.01}
    info = _call_with_resync(client.futures_exchange_info)
    sym = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
    if not sym:
        return {"stepSize": 0.001, "minQty": 0.001, "tickSize": 0.01}
    lot = next((f for f in sym["filters"] if f["filterType"] == "LOT_SIZE"), None)
    prc = next((f for f in sym["filters"] if f["filterType"] == "PRICE_FILTER"), None)
    step = float(lot["stepSize"]) if lot else 0.001
    minq = float(lot["minQty"])   if lot else 0.001
    tick = float(prc["tickSize"]) if prc else 0.01
    return {"stepSize": step, "minQty": minq, "tickSize": tick}

def round_step(value: float, step: float, mode: str = "floor") -> float:
    if step <= 0: return value
    if mode == "floor":
        return math.floor(value/step) * step
    elif mode == "ceil":
        return math.ceil(value/step) * step
    return round(value/step) * step

# ---------------- Setup / execução ----------------
def setup_futures_pair(symbol: str, leverage: int = 10, margin_type: str = "ISOLATED") -> bool:
    try:
        _call_with_resync(client.futures_change_margin_type, symbol=symbol, marginType=margin_type, recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        if "No need to change margin type" not in str(e):
            st.error(f"Erro ao definir margem: {e}")
            return False
    try:
        _call_with_resync(client.futures_change_leverage, symbol=symbol, leverage=int(leverage), recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        st.error(f"Erro ao definir alavancagem: {e}")
        return False
    return True

def get_position_info(symbol: str) -> dict:
    infos = _call_with_resync(client.futures_position_information, symbol=symbol, recvWindow=RECV_WINDOW_MS)
    return infos[0] if infos else {}

def get_position_amt(symbol: str) -> float:
    info = get_position_info(symbol)
    try:
        return float(info.get("positionAmt", 0) or 0)
    except Exception:
        return 0.0

def close_position_market(symbol: str) -> Optional[dict]:
    amt = get_position_amt(symbol)
    if abs(amt) < 1e-12:
        return None
    side = SIDE_SELL if amt > 0 else SIDE_BUY
    qty = abs(amt)
    return _call_with_resync(
        client.futures_create_order,
        symbol=symbol, side=side, type=ORDER_TYPE_MARKET, quantity=qty,
        reduceOnly=True, recvWindow=RECV_WINDOW_MS
    )

def executar_ordem_mercado(symbol: str, side: str, quantity: float, sl_price: float, tp_price: float):
    try:
        ordem = _call_with_resync(
            client.futures_create_order,
            symbol=symbol,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
            recvWindow=RECV_WINDOW_MS,
        )

        if side == "BUY":
            _call_with_resync(client.futures_create_order, symbol=symbol, side=SIDE_SELL,
                              type=ORDER_TYPE_STOP_MARKET, stopPrice=sl_price,
                              closePosition=True, timeInForce="GTC", recvWindow=RECV_WINDOW_MS)
            _call_with_resync(client.futures_create_order, symbol=symbol, side=SIDE_SELL,
                              type=ORDER_TYPE_TAKE_PROFIT_MARKET, stopPrice=tp_price,
                              closePosition=True, timeInForce="GTC", recvWindow=RECV_WINDOW_MS)
        else:
            _call_with_resync(client.futures_create_order, symbol=symbol, side=SIDE_BUY,
                              type=ORDER_TYPE_STOP_MARKET, stopPrice=sl_price,
                              closePosition=True, timeInForce="GTC", recvWindow=RECV_WINDOW_MS)
            _call_with_resync(client.futures_create_order, symbol=symbol, side=SIDE_BUY,
                              type=ORDER_TYPE_TAKE_PROFIT_MARKET, stopPrice=tp_price,
                              closePosition=True, timeInForce="GTC", recvWindow=RECV_WINDOW_MS)
        return True, ordem
    except Exception as e:
        return False, str(e)

def executar_trade_automatico(suggestion_local, symbol_local, price_now_local,
                              stake_local, lev_conf_local, sl_price_local, tp_price_local):
    if suggestion_local not in ("COMPRAR", "VENDER"):
        st.warning("Sem sugestão de trade válida.")
        return

    f = get_symbol_filters(symbol_local)
    step, minQty, tick = f["stepSize"], f["minQty"], f["tickSize"]

    sl_price_local = round_step(sl_price_local, tick, "floor")
    tp_price_local = round_step(tp_price_local, tick, "floor")

    if not setup_futures_pair(symbol_local, lev_conf_local):
        return

    raw_qty = (stake_local * lev_conf_local) / price_now_local
    qty = max(minQty, round_step(raw_qty, step, "floor"))
    if qty < minQty:
        st.error(f"Quantidade calculada ({qty}) menor que minQty ({minQty}). Aumente o stake.")
        return

    pos = get_position_amt(symbol_local)
    new_side = "BUY" if suggestion_local == "COMPRAR" else "SELL"
    if pos > 0 and new_side == "SELL":
        close_position_market(symbol_local)
    elif pos < 0 and new_side == "BUY":
        close_position_market(symbol_local)

    success, result = executar_ordem_mercado(symbol_local, new_side, qty, sl_price_local, tp_price_local)
    if success:
        st.success(f"✅ Ordem executada ({suggestion_local}) | Qty: {qty}")
        info = get_position_info(symbol_local)
        entry_price = float(info.get("entryPrice", 0) or 0)
        st.session_state.active_trade = {
            "symbol": symbol_local,
            "side": new_side,
            "qty": qty,
            "entry_time_ms": int(time.time()*1000),
            "entry_price": entry_price if entry_price>0 else price_now_local,
            "sl": sl_price_local,
            "tp": tp_price_local,
            "stake": stake_local,
            "lev": lev_conf_local
        }
    else:
        st.error(f"❌ Erro na execução: {result}")

# ---------------- Controles principais ----------------
st.subheader("📡 Painel")
colA, colB, colC = st.columns([1.5, 1, 1])
with colA:
    focus_symbol_list = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"]
    symbol = st.selectbox("Par (USDT-M Futures)", focus_symbol_list, index=1)
with colB:
    margin_type = st.selectbox("Tipo de margem", ["ISOLATED","CROSSED"], index=0)
with colC:
    leverage = st.number_input("Alavancagem", 1, 125, 10, step=1)

mtf = load_multitf(symbol)
if any(df.empty for df in mtf.values()):
    st.error("Falha ao carregar dados multi-timeframe.")
    st.stop()

# --------- Leitura ----------
blocks: Dict[str, dict] = {}
for tf, df in mtf.items():
    r = df.iloc[-1]
    blocks[tf] = {
        "trend": trend_label(r), "rsi": float(r["rsi14"]), "ema20": float(r["ema20"]),
        "ema50": float(r["ema50"]), "ema200": float(r["ema200"]), "vol_rel": float(r["vol_rel"]),
        "strength": float(r["strength_score"]), "atr": float(r["atr14"]), "close": float(r["c"]),
        "spread_rel": float(r["spread_rel"])
    }

align_map = {"ALTA":1,"BAIXA":-1,"NEUTRA":0}
vals = [align_map[blocks[tf]["trend"]] for tf in ["1D","1H","5M"]]
count_up = sum(1 for v in vals if v==1)
count_dn = sum(1 for v in vals if v==-1)
align_count = max(count_up, count_dn)
consensus = "ALTA" if count_up>count_dn else ("BAIXA" if count_dn>count_up else "NEUTRA")
conf = confidence_from_features(mtf["5M"].iloc[-1], consensus, align_count)

cL, cR = st.columns([2,1])
with cL:
    st.subheader("🔎 Leitura Multi-timeframe")
    a,b,c = st.columns(3)
    for tf, col in zip(["1D","1H","5M"], [a,b,c]):
        info = blocks[tf]
        badge = "🟢" if info["trend"]=="ALTA" else ("🔴" if info["trend"]=="BAIXA" else "⚪")
        col.metric(f"{tf} – {badge} {info['trend']}",
                   value=f"RSI {info['rsi']:.1f}",
                   help=f"EMA20/50/200: {info['ema20']:.4f} / {info['ema50']:.4f} / {info['ema200']:.4f}\n"
                        f"Vol.rel: {info['vol_rel']:.2f}× | Str: {info['strength']:.2f}")
    st.caption(f"Alinhamento: {align_count}/3 • Consenso: {consensus}")
with cR:
    st.subheader("🔐 Confiança do Sinal")
    st.metric("Nível", f"{conf}/100")

# --------- Parâmetros ----------
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

price_now = blocks["5M"]["close"]
atr5 = blocks["5M"]["atr"] or 0.0

# Sugestão
suggestion, reason = "AGUARDAR", "Condições insuficientes"
if align_count==3 and conf>=min_conf:
    suggestion = "COMPRAR" if consensus=="ALTA" else ("VENDER" if consensus=="BAIXA" else "AGUARDAR")
    reason = f"Alinhamento 3/3 e confiança {conf}"
elif align_count==2 and conf>=(min_conf+10):
    suggestion = "COMPRAR" if count_up==2 else ("VENDER" if count_dn==2 else "AGUARDAR")
    reason = f"Alinhamento 2/3 com confiança {conf}"

# TP/SL
sl_price = tp_price = None
stop_pct = tp_pct = 0.0
if suggestion in ("COMPRAR","VENDER") and atr5>0:
    if suggestion=="COMPRAR":
        sl_price = price_now - atr_mult_sl*atr5
        tp_price = price_now + rr*atr_mult_sl*atr5
        stop_pct = (price_now - sl_price)/price_now
        tp_pct   = (tp_price - price_now)/price_now
    else:
        sl_price = price_now + atr_mult_sl*atr5
        tp_price = price_now - rr*atr_mult_sl*atr5
        stop_pct = (sl_price - price_now)/price_now
        tp_pct   = (price_now - tp_price)/price_now

# Alavancagem sugerida
profile_caps = {"Conservador":5,"Moderado":10,"Agressivo":20}
cap = profile_caps.get("Moderado", 10)  # simples
base_from_conf = 2 if conf<55 else 4 if conf<65 else 6 if conf<75 else 9 if conf<85 else 12 if conf<92 else 15
if stop_pct>0:
    if stop_pct < 0.003: base_from_conf = int(round(base_from_conf*1.25))
    elif stop_pct > 0.006: base_from_conf = int(round(base_from_conf*0.7))
lev_conf = max(1, min(base_from_conf, cap))

# Estimativas
taker_fee_pct = st.number_input("Taxa taker (%)", 0.0, 0.5, 0.04, step=0.01) / 100.0
fees_roundtrip = 2*taker_fee_pct
net_gain_pct = max((tp_pct - fees_roundtrip), 0) * lev_conf if tp_price else 0.0
net_loss_pct = (stop_pct + fees_roundtrip) * lev_conf if sl_price else 0.0
est_gain_usdt = stake * net_gain_pct if tp_price else 0.0
est_loss_usdt = stake * net_loss_pct if sl_price else 0.0

colS1, colS2 = st.columns([1,2])
with colS1:
    color = "🟢" if suggestion=="COMPRAR" else ("🔴" if suggestion=="VENDER" else "⏸️")
    st.metric("Sugestão", f"{color} {suggestion}", help=reason)
with colS2:
    if sl_price and tp_price:
        st.markdown('<div class="sugestao-box">', unsafe_allow_html=True)
        st.write(f"**Entrada**: ~{price_now:.6f} | **Stop**: {sl_price:.6f} | **Alvo**: {tp_price:.6f}")
        st.write(f"**Stake**: {stake:.2f} USDT | **Lev sugerida**: {lev_conf}×")
        st.write(f"**Estimativa** → TP: ~{est_gain_usdt:.2f} | SL: ~{est_loss_usdt:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Sem TP/SL sugeridos (aguarde melhor alinhamento/confiança).")

# --------- Execução ----------
st.subheader("⚡ Execução")
if not EXECUTION_ENABLED:
    st.info("Execução está **desativada** (BINANCE_MODE=off). Defina BINANCE_MODE=testnet ou live para habilitar.")
elif client is None:
    # Execução pretendida, mas indisponível (ex.: região bloqueada)
    st.error(f"Execução indisponível. {execution_blocked_reason or ''}".strip())
else:
    can_execute = suggestion in ("COMPRAR","VENDER") and sl_price and tp_price and lev_conf>=1
    btn_label = f"🚀 Executar {suggestion} {symbol}"
    clicked = st.button(btn_label, type="primary", use_container_width=True, disabled=not can_execute)

    # Status de conta (opcional)
    try:
        acc = _call_with_resync(client.futures_account, recvWindow=RECV_WINDOW_MS)
        bal = acc.get("totalWalletBalance") or acc.get("availableBalance")
        st.caption(f"✅ Conectado | Balance: {bal} USDT | Modo: {BINANCE_MODE.upper()}")
    except Exception as e:
        st.warning(f"Conexão privada indisponível: {e}")

    if clicked and can_execute:
        executar_trade_automatico(
            suggestion_local=suggestion,
            symbol_local=symbol,
            price_now_local=price_now,
            stake_local=stake,
            lev_conf_local=lev_conf,
            sl_price_local=sl_price,
            tp_price_local=tp_price
        )

# --------- Trade Ativo / PnL ----------
st.subheader("📘 Trade Ativo (tempo real)")
st_autorefresh(interval=3000, key="live_trade_refresh_v13_1")

if "active_trade" not in st.session_state:
    st.session_state.active_trade = None
active = st.session_state.active_trade

def get_income_pnl_since(symbol: str, start_ms: int) -> float:
    try:
        hist = _call_with_resync(
            client.futures_income_history,
            symbol=symbol, incomeType="REALIZED_PNL",
            startTime=start_ms, recvWindow=RECV_WINDOW_MS
        )
        return float(sum(float(x.get("income", 0) or 0) for x in hist))
    except Exception:
        return 0.0

if EXECUTION_ENABLED and client is not None and active and active.get("symbol"):
    sym = active["symbol"]
    info = get_position_info(sym)
    amt = float(info.get("positionAmt", 0) or 0)
    entry_price = float(info.get("entryPrice", 0) or active.get("entry_price", 0))
    mark_price = float(info.get("markPrice", 0) or 0)
    unrl = float(info.get("unRealizedProfit", 0) or 0)

    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1: st.metric("Símbolo", sym)
    with c2: st.metric("Lado", "Long" if amt>0 else ("Short" if amt<0 else "—"))
    with c3: st.metric("Qtd", f"{abs(amt):.6f}")
    with c4: st.metric("Entrada", f"{entry_price:.6f}")

    c5,c6,c7 = st.columns([1,1,1])
    margin_approx = active.get("stake", 0.0)
    roe = (unrl / margin_approx * 100) if margin_approx>0 else 0.0
    with c5: st.metric("Preço (mark)", f"{mark_price:.6f}")
    with c6: st.metric("PnL não realizado", f"{unrl:.4f} USDT")
    with c7: st.metric("ROE (aprox.)", f"{roe:.2f}%")

    st.caption(f"SL: {active.get('sl'):.6f} | TP: {active.get('tp'):.6f} | Lev: {active.get('lev')}×")

    col_close, _ = st.columns([1,3])
    with col_close:
        if st.button("✖️ Fechar posição agora", use_container_width=True, type="primary"):
            close_position_market(sym)

    if abs(amt) < 1e-12:
        realized = get_income_pnl_since(sym, active.get("entry_time_ms", int(time.time()*1000))-1_000)
        if realized >= 0:
            st.success(f"✅ Posição encerrada | PnL realizado: +{realized:.4f} USDT")
        else:
            st.error(f"❌ Posição encerrada | PnL realizado: {realized:.4f} USDT")
        st.session_state.active_trade = None
else:
    st.info("Nenhum trade ativo ou execução desabilitada / indisponível no momento.")

st.caption("⚠️ Observação legal: respeite os Termos de Uso da exchange e a elegibilidade por região. Testnet disponível para testes.")
