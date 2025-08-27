# -*- coding: utf-8 -*-
"""
Painel – Binance v13 (Scanner + Foco + Execução + PnL)
----------------------------------------------------------------------------
• Multi-timeframe (1D, 1H, 5M) com EMA20/50/200 + RSI
• Sinal: COMPRAR / VENDER / AGUARDAR
• TP/SL a partir do ATR(5m): SL = k*ATR, TP = R*SL
• Alavancagem sugerida (heurística) com teto por perfil e por risco de banca
• Execução no painel de Foco: ajusta qty por stepSize/minQty, arredonda preço por tickSize,
  define margem/alavancagem, fecha posição oposta se necessário e abre nova com SL/TP.
• Tratamento de erro -1021 (tempo): sincroniza e repete a chamada.
• Acompanhamento do Trade Ativo: PnL ao vivo e PnL realizado ao encerrar.

Rodar:
    streamlit run crypto_painel_binance_v13.py
Requisitos:
    pip install --upgrade streamlit pandas numpy requests python-binance plotly streamlit-autorefresh ntplib
"""
from __future__ import annotations

# ===================== Imports =====================
import math, time, json
from typing import Optional, Dict
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from binance.client import Client
from binance.enums import *

# ===================== Config visual =====================
st.set_page_config(page_title="Painel – Binance v13", layout="wide")
st.markdown("""
<style>
:root{
  --bg:#0b0e11;--card:#14181d;--card2:#161a1e;--text:#eaecef;--muted:#a7b1c2;
  --accent:#f0b90b;--success:#1f8b4c;--danger:#b02a37;--border:#232a31;
}
.stApp{background:var(--bg);color:var(--text);}
.block-container{padding-top:2.25rem !important;}
h1{color:var(--accent) !important;margin-top:0 !important; line-height:1.25;}
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
st.title("📊 Painel – Binance v13")

# ===================== NTP (referência visual) =====================
def show_ntp_reference():
    try:
        import ntplib
        from time import ctime
        ntp = ntplib.NTPClient()
        resp = ntp.request('pool.ntp.org', version=3, timeout=2)
        st.caption(f"✅ Relógio de referência NTP: {ctime(resp.tx_time)}")
    except Exception as e:
        st.caption(f"⚠️ NTP indisponível: {e}")
show_ntp_reference()

# ===================== Constantes =====================
BINANCE_REST = "https://api.binance.com"
RECV_WINDOW_MS = 90_000  # 90s

# ===================== Helpers REST (klines) =====================
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

# ===================== Indicadores =====================
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

# ===================== Multi-timeframe Loader =====================
@st.cache_data(ttl=30)
def load_multitf(symbol: str) -> Dict[str, pd.DataFrame]:
    d1 = add_core_features(fetch_klines_rest(symbol, "1d", 400))
    h1 = add_core_features(fetch_klines_rest(symbol, "1h", 500))
    m5 = add_core_features(fetch_klines_rest(symbol, "5m", 500))
    return {"1D": d1, "1H": h1, "5M": m5}

# ===================== Binance Client + Offset/Retry =====================
def sync_client_time(client: Client) -> None:
    try:
        srv = client.get_server_time()  # {"serverTime": <ms>}
        drift = int(srv["serverTime"]) - int(time.time() * 1000)
        setattr(client, "timestamp_offset", drift)  # atributo atual
        setattr(client, "TIME_OFFSET", drift)       # compatibilidade
        st.caption(f"⏱️ Offset aplicado (server - local): {drift} ms")
    except Exception as e:
        st.warning(f"⚠️ Falha ao sincronizar tempo pelo serverTime: {e}")

@st.cache_resource
def get_binance_client():
    api_key = st.secrets["binance"]["api_key"]
    api_secret = st.secrets["binance"]["api_secret"]
    c = Client(api_key=api_key, api_secret=api_secret)
    sync_client_time(c)
    return c

client = get_binance_client()

def _call_with_resync(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        s = str(e)
        if "-1021" in s or "outside of the recvWindow" in s:
            sync_client_time(client)
            return fn(*args, **kwargs)
        raise

# ===================== Exchange Info (filtros) =====================
@st.cache_data(ttl=600)
def get_symbol_filters(symbol: str) -> Dict[str, float]:
    info = _call_with_resync(client.futures_exchange_info)
    sym = next((s for s in info["symbols"] if s["symbol"] == symbol), None)
    if not sym:  # fallback
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

# ===================== Setup de Par =====================
def setup_futures_pair(client, symbol: str, leverage: int = 10, margin_type: str = "ISOLATED") -> bool:
    try:
        _call_with_resync(
            client.futures_change_margin_type,
            symbol=symbol, marginType=margin_type, recvWindow=RECV_WINDOW_MS
        )
    except Exception as e:
        if "No need to change margin type" not in str(e):
            st.error(f"Erro ao definir margem: {e}")
            return False
    try:
        _call_with_resync(
            client.futures_change_leverage,
            symbol=symbol, leverage=int(leverage), recvWindow=RECV_WINDOW_MS
        )
    except Exception as e:
        st.error(f"Erro ao definir alavancagem: {e}")
        return False
    return True

# ===================== Posição atual / Fechar =====================
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

# ===================== Execução de Ordem =====================
def executar_ordem_mercado(client, symbol: str, side: str, quantity: float, sl_price: float, tp_price: float):
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
            _call_with_resync(
                client.futures_create_order,
                symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_price, closePosition=True, timeInForce="GTC",
                recvWindow=RECV_WINDOW_MS,
            )
            _call_with_resync(
                client.futures_create_order,
                symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_price, closePosition=True, timeInForce="GTC",
                recvWindow=RECV_WINDOW_MS,
            )
        else:
            _call_with_resync(
                client.futures_create_order,
                symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_price, closePosition=True, timeInForce="GTC",
                recvWindow=RECV_WINDOW_MS,
            )
            _call_with_resync(
                client.futures_create_order,
                symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=tp_price, closePosition=True, timeInForce="GTC",
                recvWindow=RECV_WINDOW_MS,
            )

        return True, ordem
    except Exception as e:
        return False, str(e)

def executar_trade_automatico(suggestion_local, symbol_local, price_now_local,
                              stake_local, lev_conf_local, sl_price_local, tp_price_local):
    if suggestion_local not in ("COMPRAR", "VENDER"):
        st.warning("Sem sugestão de trade válida.")
        return

    # Ajuste de filtros do símbolo
    f = get_symbol_filters(symbol_local)
    step, minQty, tick = f["stepSize"], f["minQty"], f["tickSize"]

    # Ajuste de preços para o tickSize
    sl_price_local = round_step(sl_price_local, tick, "floor")
    tp_price_local = round_step(tp_price_local, tick, "floor")

    # Define margem e alavancagem
    if not setup_futures_pair(client, symbol_local, lev_conf_local):
        return

    # Quantidade sugerida em moeda base
    raw_qty = (stake_local * lev_conf_local) / price_now_local
    qty = max(minQty, round_step(raw_qty, step, "floor"))
    if qty < minQty:
        st.error(f"Quantidade calculada ({qty}) menor que minQty ({minQty}). Aumente o stake.")
        return

    # Fechar/inverter se necessário
    pos = get_position_amt(symbol_local)
    new_side = "BUY" if suggestion_local == "COMPRAR" else "SELL"
    if pos > 0 and new_side == "SELL":
        close_position_market(symbol_local)
    elif pos < 0 and new_side == "BUY":
        close_position_market(symbol_local)

    success, result = executar_ordem_mercado(client, symbol_local, new_side, qty, sl_price_local, tp_price_local)
    if success:
        st.success(f"✅ Ordem executada ({suggestion_local}) | Qty: {qty}")

        # Salva estado do trade ativo
        info = get_position_info(symbol_local)
        entry_price = float(info.get("entryPrice", 0) or 0)
        st.session_state.active_trade = {
            "symbol": symbol_local,
            "side": new_side,           # BUY/SELL
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

# ===================== Teste de conexão =====================
try:
    acc = _call_with_resync(client.futures_account, recvWindow=RECV_WINDOW_MS)
    bal = acc.get("totalWalletBalance") or acc.get("availableBalance")
    st.success(f"✅ Conectado à Binance Futures | Saldo: {bal} USDT")
except Exception as e:
    st.error(f"❌ Erro na conexão com Binance: {e}")

# ===================== Scanner =====================
st.subheader("🧭 Scanner de pares – alinhamento + confiança")
pairs = st.multiselect(
    "Selecione os pares para escanear",
    ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"],
    default=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"]
)
colsc1, colsc2, colsc3 = st.columns([1,1,1])
with colsc1:
    scan_interval = st.number_input("⏱️ Intervalo auto-scan (s)", 10, 300, 60, step=5)
with colsc2:
    alert_thresh = st.number_input("🔔 Confiança para alertar", 60, 95, 80, step=1)
with colsc3:
    auto_scan = st.toggle("Auto-refresh", True)
if auto_scan:
    st_autorefresh(interval=scan_interval*1000, key="auto_scan_v13")

@st.cache_data(ttl=25)
def scan_pair(sym: str) -> Dict:
    d1 = add_core_features(fetch_klines_rest(sym, "1d", 400))
    h1 = add_core_features(fetch_klines_rest(sym, "1h", 500))
    m5 = add_core_features(fetch_klines_rest(sym, "5m", 500))

    b1, b2, b3 = d1.iloc[-1], h1.iloc[-1], m5.iloc[-1]
    trends = [trend_label(x) for x in [b1, b2, b3]]
    count_up = trends.count("ALTA"); count_dn = trends.count("BAIXA")
    align = max(count_up, count_dn)
    consensus = "ALTA" if count_up > count_dn else ("BAIXA" if count_dn > count_up else "NEUTRA")
    conf = confidence_from_features(b3, consensus, align)
    price = float(b3["c"]); atr5 = float(b3["atr14"]) or 0.0

    sug = "AGUARDAR"
    if align == 3 and conf >= 60:
        sug = "COMPRAR" if consensus == "ALTA" else ("VENDER" if consensus == "BAIXA" else "AGUARDAR")
    elif align == 2 and conf >= 70:
        sug = "COMPRAR" if consensus == "ALTA" else ("VENDER" if consensus == "BAIXA" else "AGUARDAR")

    sl = tp = None; stop_pct = tp_pct = 0.0
    if sug != "AGUARDAR" and atr5 > 0:
        if sug == "COMPRAR":
            sl = price - 1.0*atr5; tp = price + 1.5*atr5
            stop_pct = (price - sl)/price; tp_pct = (tp - price)/price
        else:
            sl = price + 1.0*atr5; tp = price - 1.5*atr5
            stop_pct = (sl - price)/price; tp_pct = (price - tp)/price

    base_from_conf = 2 if conf < 55 else 4 if conf < 65 else 6 if conf < 75 else 9 if conf < 85 else 12 if conf < 92 else 15
    if stop_pct > 0:
        if stop_pct < 0.003: base_from_conf = int(round(base_from_conf*1.25))
        elif stop_pct > 0.006: base_from_conf = int(round(base_from_conf*0.7))
    lev = max(1, min(base_from_conf, 20))

    score = align * conf
    return {
        "Par": sym, "Consenso": consensus, "Alinhamento": align, "Confiança": conf,
        "Sugestão": sug, "Preço": round(price, 6),
        "Stop": (round(sl, 6) if sl else None), "Alvo": (round(tp, 6) if tp else None),
        "Stop%": round(stop_pct*100, 2) if stop_pct>0 else None,
        "Alvo%": round(tp_pct*100, 2) if tp_pct>0 else None,
        "Lev sug": lev, "Score": score
    }

rows = [scan_pair(p) for p in pairs] if pairs else []
df_scan = pd.DataFrame(rows)
if not df_scan.empty:
    sug_order = {"COMPRAR":0,"VENDER":1,"AGUARDAR":2}
    df_scan["SugOrd"] = df_scan["Sugestão"].map(sug_order)
    df_scan = df_scan.sort_values(["Confiança","SugOrd"], ascending=[False, True]).drop(columns="SugOrd")
    st.dataframe(df_scan, use_container_width=True, height=360)
else:
    st.info("Selecione pares para escanear.")

best_symbol = df_scan.iloc[0]["Par"] if not df_scan.empty else "ETHUSDT"

# ===================== Painel de Foco (Leitura + Execução) =====================
st.subheader("🎯 Foco do Painel (Leitura + Execução)")
colf1, colf2 = st.columns([2,1])
with colf1:
    focus_symbol_default_list = pairs or ["ETHUSDT"]
    default_index = focus_symbol_default_list.index(best_symbol) if best_symbol in focus_symbol_default_list else 0
    focus_symbol = st.selectbox("Par em foco", focus_symbol_default_list, index=default_index)
with colf2:
    st.caption("Dica: por padrão usamos o melhor do scanner, mas você pode escolher outro aqui.")

mtf = load_multitf(focus_symbol)
if any(df is None or df.empty for df in mtf.values()):
    st.error("Falha ao carregar dados multi-timeframe para o foco.")
    st.stop()

blocks = {}
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

# Parâmetros do sinal
st.subheader("🧮 Parâmetros do Sinal")
cA,cB,cC,cD,cE = st.columns([1,1,1,1,1])
with cA:
    atr_mult_sl = st.number_input("SL = ATR ×", 0.2, 5.0, 1.0)
with cB:
    rr = st.number_input("R:R (TP = R × SL)", 0.5, 5.0, 1.5)
with cC:
    min_conf = st.number_input("Confiança mínima", 0, 100, 60)
with cD:
    perfil = st.selectbox("Perfil", ["Conservador","Moderado","Agressivo"], index=1)
with cE:
    banca = st.number_input("Banca (USDT)", 10.0, 1e9, 1000.0, step=10.0)

cF,cG,cH = st.columns([1,1,1])
with cF:
    stake = st.number_input("Stake (USDT)", 1.0, 1e9, 50.0, step=1.0)
with cG:
    risco_pct = st.number_input("Risco máx por trade (%)", 0.1, 10.0, 1.0)/100.0
with cH:
    taker_fee_pct = st.number_input("Taxa taker (%)", 0.0, 0.5, 0.04)/100.0

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
cap = profile_caps.get(perfil, 10)
base_from_conf = 2 if conf<55 else 4 if conf<65 else 6 if conf<75 else 9 if conf<85 else 12 if conf<92 else 15
if stop_pct>0:
    if stop_pct < 0.003: base_from_conf = int(round(base_from_conf*1.25))
    elif stop_pct > 0.006: base_from_conf = int(round(base_from_conf*0.7))
lev_conf = max(1, min(base_from_conf, cap))
if stop_pct>0:
    lev_cap_risk = (banca*risco_pct) / (stake*stop_pct)
    lev_conf = int(max(1, min(lev_conf, cap, lev_cap_risk)))

# Estimativas
fees_roundtrip = 2*taker_fee_pct
net_gain_pct = max(tp_pct - fees_roundtrip, 0) * lev_conf if tp_price else 0.0
net_loss_pct = (stop_pct + fees_roundtrip) * lev_conf if sl_price else 0.0
est_gain_usdt = stake * net_gain_pct if tp_price else 0.0
est_loss_usdt = stake * net_loss_pct if sl_price else 0.0

# Métricas
cL, cR = st.columns([2,1])
with cL:
    st.subheader("🔎 Leitura Multi-timeframe")
    ca, cb, cc = st.columns(3)
    for tf, col in zip(["1D","1H","5M"], [ca,cb,cc]):
        info = blocks[tf]
        badge = "🟢" if info["trend"]=="ALTA" else ("🔴" if info["trend"]=="BAIXA" else "⚪")
        col.metric(f"{tf} – {badge} {info['trend']}", value=f"RSI {info['rsi']:.1f}",
                   help=f"EMA20/50/200: {info['ema20']:.4f} / {info['ema50']:.4f} / {info['ema200']:.4f}\n"
                        f"Vol.rel: {info['vol_rel']:.2f}× | Str: {info['strength']:.2f}")
    st.caption(f"Alinhamento: {align_count}/3 • Consenso: {consensus}")
with cR:
    st.subheader("🔐 Confiança do Sinal")
    st.metric("Nível de confiança", f"{conf}/100")

st.subheader("🎯 Sugestão de Ação + TP/SL + Alavancagem")
cSug1, cSug2 = st.columns([1,2])
with cSug1:
    color = "🟢" if suggestion=="COMPRAR" else ("🔴" if suggestion=="VENDER" else "⏸️")
    st.metric("Sugestão", f"{color} {suggestion}", help=reason)
with cSug2:
    if sl_price and tp_price:
        st.markdown('<div class="sugestao-box">', unsafe_allow_html=True)
        st.write(f"**Entrada**: ~{price_now:.6f} | **Stop**: {sl_price:.6f} | **Alvo**: {tp_price:.6f}")
        st.write(f"**Stake**: {stake:.2f} USDT | **Lev sugerida**: {lev_conf}×")
        st.write(f"**Estimativa** → TP: ~{est_gain_usdt:.2f} | SL: ~{est_loss_usdt:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Sem TP/SL e alavancagem sugeridos (aguarde melhor alinhamento/confiança).")

# ===================== Execução =====================
st.subheader("⚙️ Execução")
can_execute = suggestion in ("COMPRAR","VENDER") and sl_price and tp_price and lev_conf>=1
btn_label = f"🚀 Executar {suggestion} {focus_symbol}"
clicked = st.button(btn_label, type="primary", use_container_width=True, disabled=not can_execute)

if clicked and can_execute:
    executar_trade_automatico(
        suggestion_local=suggestion,
        symbol_local=focus_symbol,
        price_now_local=price_now,
        stake_local=stake,
        lev_conf_local=lev_conf,
        sl_price_local=sl_price,
        tp_price_local=tp_price
    )

# ===================== Trade Ativo – acompanhamento e PnL =====================
st.subheader("📘 Trade Ativo (tempo real)")
# auto refresh curto nessa seção
st_autorefresh(interval=3000, key="live_trade_refresh_v13")

if "active_trade" not in st.session_state:
    st.session_state.active_trade = None

active = st.session_state.active_trade

def get_income_pnl_since(symbol: str, start_ms: int) -> float:
    """Soma PnL realizado desde start_ms para o símbolo."""
    try:
        hist = _call_with_resync(
            client.futures_income_history,
            symbol=symbol, incomeType="REALIZED_PNL",
            startTime=start_ms, recvWindow=RECV_WINDOW_MS
        )
        total = sum(float(x.get("income", 0) or 0) for x in hist)
        return float(total)
    except Exception:
        return 0.0

if active and active.get("symbol"):
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
    with c4: st.metric("Preço de entrada", f"{entry_price:.6f}")

    c5,c6,c7 = st.columns([1,1,1])
    # Margem aproximada ≈ stake informado na abertura
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

    # Se a posição zerou, mostrar PnL realizado e limpar
    if abs(amt) < 1e-12:
        realized = get_income_pnl_since(sym, active.get("entry_time_ms", int(time.time()*1000))-1_000)
        if realized >= 0:
            st.success(f"✅ Posição encerrada | PnL realizado: +{realized:.4f} USDT")
        else:
            st.error(f"❌ Posição encerrada | PnL realizado: {realized:.4f} USDT")
        st.session_state.active_trade = None
else:
    st.info("Nenhum trade ativo no momento. Execute um sinal no painel de foco para começar.")

st.caption("⚠️ Educacional. A alavancagem sugerida é heurística (confiança+stop+perfil+risco). Ajuste conforme sua conta/gestão.")
