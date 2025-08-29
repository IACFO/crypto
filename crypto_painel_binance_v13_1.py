# -*- coding: utf-8 -*-
"""
Crypto Painel Binance v13.1 – Foco + Scanner + Execução
-------------------------------------------------------
• Multi-timeframe (1D / 1H / 5M) com EMA20/50/200, RSI, ATR 
• Scanner de pares ordenado por confiança
• Painel de foco com cálculo de SL/TP, alavancagem sugerida e execução real
• Botões: COMPRAR/VENDER, FECHAR posição e INVERTER
• PnL realizado exibido após fechamento
• Compatível com Render (env vars BINANCE_API_KEY / BINANCE_API_SECRET)

Requisitos:
  streamlit, python-binance, pandas, numpy, requests, plotly (opcional), websocket-client, ntplib

Execução local:
  streamlit run crypto_painel_binance_v13_1.py
"""

from __future__ import annotations
import os, time, json, threading
from collections import deque
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Binance
from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_STOP_MARKET,
    ORDER_TYPE_TAKE_PROFIT_MARKET,
)

# =========================
# Config e Aparência
# =========================
st.set_page_config(page_title="Crypto Painel – Binance v13.1", layout="wide")
st.markdown("""
<style>
:root{
  --bg:#0b0e11; --card:#14181d; --card-2:#161a1e; --text:#eaecef; --muted:#a7b1c2;
  --accent:#f0b90b; --success:#1f8b4c; --danger:#b02a37; --border:#232a31;
}
.stApp{ background:var(--bg); color:var(--text); }
.block-container{ padding-top:1.2rem !important; }
h1,h2,h3{ color:var(--accent) !important; }
div[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 14px; }
div[role="alert"]{ background:#0f1317 !important; border:1px solid var(--border) !important; color:var(--text) !important; }
div.stButton > button{ background:var(--accent); color:#111; font-weight:700; border:0; border-radius:10px; }
div.stButton > button:hover{ background:#ffce32; }
[data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"]{
  background:var(--card-2) !important; color:var(--text) !important; border:1px solid var(--border) !important; border-radius:8px;
}
.stDataFrame thead tr th{ background:var(--card) !important; color:var(--text) !important; border-bottom:1px solid var(--border) !important; }
.stDataFrame tbody tr td{ background:var(--card-2) !important; color:var(--text) !important; border-bottom:1px solid var(--border) !important; }
.sugestao-box{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px 12px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Crypto Painel – Binance v13.1")

# =========================
# Util – NTP referência
# =========================
def show_ntp_reference():
    try:
        import ntplib
        from time import ctime
        ntp = ntplib.NTPClient()
        resp = ntp.request('pool.ntp.org', version=3, timeout=2)
        st.caption(f"🕒 Referência NTP: {ctime(resp.tx_time)}")
    except Exception as e:
        st.caption(f"🕒 NTP: indisponível ({e})")

show_ntp_reference()

# =========================
# Segredos/Cliente Binance
# =========================
RECV_WINDOW_MS = 60000  # 60s
BINANCE_REST = "https://api.binance.com"

def _get_secret(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    if v: return v
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    try:
        if "binance" in st.secrets:
            if key == "BINANCE_API_KEY":
                return st.secrets["binance"].get("api_key", default)
            if key == "BINANCE_API_SECRET":
                return st.secrets["binance"].get("api_secret", default)
    except Exception:
        pass
    return default

def has_binance_keys() -> bool:
    return bool(_get_secret("BINANCE_API_KEY") and _get_secret("BINANCE_API_SECRET"))

@st.cache_resource
def get_binance_client() -> Client | None:
    if not has_binance_keys():
        return None
    api_key = _get_secret("BINANCE_API_KEY")
    api_secret = _get_secret("BINANCE_API_SECRET")
    try:
        client = Client(api_key=api_key, api_secret=api_secret)
        try:
            srv = client.get_server_time()
            client.TIME_OFFSET = int(srv["serverTime"]) - int(time.time() * 1000)
            st.caption(f"⏱️ Offset aplicado (server - local): {client.TIME_OFFSET} ms")
        except Exception as e:
            st.caption(f"⚠️ Não foi possível obter server time: {e}")
        return client
    except Exception as e:
        st.error(f"Falha ao inicializar Binance Client: {e}")
        return None

client = get_binance_client()
if client is None:
    st.warning("🔒 Sem chaves da Binance. Painel em **modo leitura** (scanner/análise). Configure BINANCE_API_KEY/SECRET no ambiente para habilitar execução.")
else:
    with st.sidebar.expander("🔐 Conexão Binance", expanded=False):
        if st.button("Validar conexão / saldo", use_container_width=True):
            try:
                info = client.futures_account(recvWindow=RECV_WINDOW_MS)
                bal = info.get("totalWalletBalance") or info.get("availableBalance")
                st.success(f"✅ Conectado à Binance Futures | Saldo: {bal} USDT")
            except Exception as e:
                st.error(f"❌ Erro na conexão com Binance: {e}")

# =========================
# Dados – REST helpers
# =========================
def fetch_klines_rest(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    r = requests.get(f"{BINANCE_REST}/api/v3/klines",
                     params={"symbol":symbol, "interval":interval, "limit":min(int(limit),1500)}, timeout=10)
    r.raise_for_status()
    data = r.json()
    cols = ["openTime","open","high","low","close","volume","closeTime","quote","n","tb_base","tb_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)[["openTime","open","high","low","close","volume"]]
    df.columns = ["t","o","h","l","c","v"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for c in ["o","h","l","c","v"]: df[c] = df[c].astype(float)
    return df

def get_current_price(symbol: str) -> float:
    try:
        r = requests.get(f"{BINANCE_REST}/api/v3/ticker/price", params={"symbol":symbol}, timeout=5)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return np.nan

# =========================
# Indicadores
# =========================
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
    out["ema20"] = out["c"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["c"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["c"].ewm(span=200, adjust=False).mean()
    out["atr14"] = atr(out, 14)
    out["vol_ma20"] = out["v"].rolling(20).mean()
    out["vol_rel"] = out["v"] / (out["vol_ma20"] + 1e-9)
    out["spread_rel"] = (out["h"] - out["l"]) / (out["atr14"] + 1e-9)
    out["body_frac"] = (out["c"] - out["o"]).abs() / ((out["h"] - out["l"]).replace(0,1e-9))
    out["strength_score"] = 0.45*out["vol_rel"] + 0.35*out["spread_rel"] + 0.20*out["body_frac"]
    out["rsi14"] = rsi(out["c"], 14)
    return out

def trend_label(row: pd.Series) -> str:
    if row["ema20"]>row["ema50"] and row["c"]>row["ema200"] and row["rsi14"]>55:
        return "ALTA"
    if row["ema20"]<row["ema50"] and row["c"]<row["ema200"] and row["rsi14"]<45:
        return "BAIXA"
    return "NEUTRA"

def confidence_from_features(row: pd.Series, trend: str, align_count: int) -> int:
    sr = float(np.tanh(max(row.get("spread_rel",0),0)))
    vr = float(min(max(row.get("vol_rel",0)/2.0, 0), 1))
    stf = float(min(max(row.get("strength_score",0)/2.0, 0), 1))
    rsi14 = float(row.get("rsi14",50))
    rsi_comp = (rsi14-50)/50.0
    if trend == "ALTA": rsi_comp = max(0, rsi_comp)
    elif trend == "BAIXA": rsi_comp = max(0, -rsi_comp)
    else: rsi_comp = 0
    align_bonus = align_count/3.0
    conf = (0.35*stf + 0.25*vr + 0.20*sr + 0.10*rsi_comp + 0.10*align_bonus) * 100
    return int(round(min(max(conf,0),100)))

# =========================
# Multi-TF
# =========================
@st.cache_data(ttl=30)
def load_multitf(symbol: str) -> Dict[str, pd.DataFrame]:
    d1 = add_core_features(fetch_klines_rest(symbol, "1d", 400))
    h1 = add_core_features(fetch_klines_rest(symbol, "1h", 500))
    m5 = add_core_features(fetch_klines_rest(symbol, "5m", 500))
    return {"1D": d1, "1H": h1, "5M": m5}

# =========================
# Execução – Futures
# =========================
def setup_futures_pair(client: Client, symbol: str, leverage: int = 10, margin_type: str = "ISOLATED") -> bool:
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type, recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        if "No need to change margin type" not in str(e):
            st.error(f"Erro ao definir margem: {e}")
            return False
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage, recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        st.error(f"Erro ao definir alavancagem: {e}")
        return False
    return True

def place_entry_and_brackets(client: Client, symbol: str, side: str, qty: float, sl_price: float, tp_price: float):
    try:
        entry = client.futures_create_order(
            symbol=symbol, side=SIDE_BUY if side=="BUY" else SIDE_SELL,
            type=ORDER_TYPE_MARKET, quantity=qty, recvWindow=RECV_WINDOW_MS
        )
        # Brackets como closePosition
        opp = SIDE_SELL if side=="BUY" else SIDE_BUY
        client.futures_create_order(
            symbol=symbol, side=opp, type=ORDER_TYPE_STOP_MARKET,
            stopPrice=round(sl_price, 6), closePosition=True, timeInForce="GTC", recvWindow=RECV_WINDOW_MS
        )
        client.futures_create_order(
            symbol=symbol, side=opp, type=ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=round(tp_price, 6), closePosition=True, timeInForce="GTC", recvWindow=RECV_WINDOW_MS
        )
        return True, entry
    except Exception as e:
        return False, str(e)

def market_close_position(client: Client, symbol: str) -> Tuple[bool, str | dict]:
    try:
        pos = client.futures_position_information(symbol=symbol, recvWindow=RECV_WINDOW_MS)
        if not pos or "positionAmt" not in pos[0]:
            return False, "Posição não encontrada"
        amt = float(pos[0]["positionAmt"])
        if abs(amt) < 1e-8:
            return False, "Sem posição aberta"
        side = SIDE_SELL if amt > 0 else SIDE_BUY
        res = client.futures_create_order(
            symbol=symbol, side=side, type=ORDER_TYPE_MARKET, quantity=abs(amt), reduceOnly=True, recvWindow=RECV_WINDOW_MS
        )
        return True, res
    except Exception as e:
        return False, str(e)

def invert_position(client: Client, symbol: str) -> Tuple[bool, str | dict]:
    try:
        pos = client.futures_position_information(symbol=symbol, recvWindow=RECV_WINDOW_MS)
        if not pos or "positionAmt" not in pos[0]:
            return False, "Posição não encontrada"
        amt = float(pos[0]["positionAmt"])
        if abs(amt) < 1e-8:
            return False, "Sem posição aberta"
        # Fecha a atual
        side_close = SIDE_SELL if amt > 0 else SIDE_BUY
        client.futures_create_order(
            symbol=symbol, side=side_close, type=ORDER_TYPE_MARKET, quantity=abs(amt), reduceOnly=True, recvWindow=RECV_WINDOW_MS
        )
        # Abre invertida com mesma quantidade
        side_new = SIDE_BUY if amt < 0 else SIDE_SELL
        res = client.futures_create_order(
            symbol=symbol, side=side_new, type=ORDER_TYPE_MARKET, quantity=abs(amt), recvWindow=RECV_WINDOW_MS
        )
        return True, res
    except Exception as e:
        return False, str(e)

def get_position_snapshot(client: Client, symbol: str) -> dict:
    snap = {"amt":0.0,"entry":np.nan,"uPnL":0.0,"margin":np.nan}
    try:
        pos = client.futures_position_information(symbol=symbol, recvWindow=RECV_WINDOW_MS)
        if pos and isinstance(pos, list):
            p = pos[0]
            snap["amt"] = float(p.get("positionAmt", 0))
            snap["entry"] = float(p.get("entryPrice", np.nan))
            snap["uPnL"] = float(p.get("unRealizedProfit", 0))
            snap["margin"] = float(p.get("isolatedMargin", np.nan))
    except Exception:
        pass
    return snap

# =========================
# Estado
# =========================
if "last_trade" not in st.session_state:
    st.session_state.last_trade = None  # dict com info da última execução
if "realized_pnl" not in st.session_state:
    st.session_state.realized_pnl = 0.0

# =========================
# Scanner + Foco
# =========================
st.subheader("📡 Monitoramento e Sinais")

colA, colB, colC = st.columns([2,1,1])
with colA:
    symbol = st.selectbox("Par (símbolo foco)", [
        "ETHUSDT","BTCUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"
    ], index=0)
with colB:
    scan_interval = st.number_input("⏱️ Auto-scan (s)", 15, 300, 45, step=5)
with colC:
    auto = st.toggle("Auto-refresh", True)

if auto:
    st_autorefresh(interval=scan_interval*1000, key="auto_refresh_v13_1")

# Multi-TF foco
mtf = load_multitf(symbol)
if not isinstance(mtf, dict) or any(df is None or df.empty for df in mtf.values()):
    st.error("Falha ao carregar dados multi-timeframe.")
    st.stop()

# Construir leitura foco
blocks = {}
for tf, df in mtf.items():
    row = df.iloc[-1]
    blocks[tf] = {
        "trend": trend_label(row),
        "rsi": float(row["rsi14"]),
        "ema20": float(row["ema20"]),
        "ema50": float(row["ema50"]),
        "ema200": float(row["ema200"]),
        "vol_rel": float(row["vol_rel"]),
        "strength": float(row["strength_score"]),
        "atr": float(row["atr14"]),
        "close": float(row["c"]),
        "spread_rel": float(row["spread_rel"]),
    }

align_map = {"ALTA":1,"BAIXA":-1,"NEUTRA":0}
values = [align_map[blocks[tf]["trend"]] for tf in ["1D","1H","5M"]]
count_up = sum(1 for v in values if v==1)
count_dn = sum(1 for v in values if v==-1)
align_count = max(count_up, count_dn)
consensus = "ALTA" if count_up>count_dn else ("BAIXA" if count_dn>count_up else "NEUTRA")
conf = confidence_from_features(mtf["5M"].iloc[-1], consensus, align_count)

c1,c2 = st.columns([2,1])
with c1:
    st.markdown("### 🔎 Leitura Multi-timeframe")
    a,b,c = st.columns(3)
    for tf, col in zip(["1D","1H","5M"], [a,b,c]):
        info = blocks[tf]
        badge = "🟢" if info["trend"]=="ALTA" else ("🔴" if info["trend"]=="BAIXA" else "⚪")
        col.metric(f"{tf} – {badge} {info['trend']}",
                   value=f"RSI {info['rsi']:.1f}",
                   help=f"EMA20/50/200: {info['ema20']:.4f} / {info['ema50']:.4f} / {info['ema200']:.4f}\nVol.rel: {info['vol_rel']:.2f}× | Str: {info['strength']:.2f}")
    st.caption(f"Alinhamento: {align_count}/3 • Consenso: {consensus}")
with c2:
    st.markdown("### 🔐 Confiança")
    st.metric("Nível de confiança", f"{conf}/100")

# =========================
# Parâmetros de Risco/Execução
# =========================
st.markdown("### 🎯 Setup do Trade (Painel de Foco)")
colR1, colR2, colR3, colR4, colR5 = st.columns(5)
with colR1:
    atr_mult_sl = st.number_input("SL = ATR ×", 0.2, 5.0, 1.0)
with colR2:
    rr = st.number_input("R:R (TP = R × SL)", 0.5, 5.0, 1.5)
with colR3:
    min_conf = st.number_input("Confiança mínima", 0, 100, 60)
with colR4:
    perfil = st.selectbox("Perfil", ["Conservador","Moderado","Agressivo"], index=1)
with colR5:
    banca = st.number_input("Banca (USDT)", 10.0, 1e9, 1000.0, step=10.0)

colE1, colE2, colE3 = st.columns(3)
with colE1:
    stake = st.number_input("Stake (USDT)", 1.0, 1e9, 50.0, step=1.0)
with colE2:
    risco_pct = st.number_input("Risco máx por trade (%)", 0.1, 10.0, 1.0)/100.0
with colE3:
    taker_fee_pct = st.number_input("Taxa taker (%)", 0.0, 0.5, 0.04)/100.0

price_now = float(blocks["5M"]["close"])
atr5 = float(blocks["5M"]["atr"])

# Direção sugerida
suggestion = "AGUARDAR"; reason = "Condições insuficientes"
if align_count==3 and conf>=min_conf:
    suggestion = "COMPRAR" if consensus=="ALTA" else ("VENDER" if consensus=="BAIXA" else "AGUARDAR")
    reason = f"Alinhamento 3/3 e confiança {conf}"
elif align_count==2 and conf>=(min_conf+10):
    suggestion = "COMPRAR" if count_up==2 else ("VENDER" if count_dn==2 else "AGUARDAR")
    reason = f"Alinhamento 2/3 com confiança {conf}"

# SL/TP
sl_price = None; tp_price = None; stop_pct=0.0; tp_pct=0.0
if suggestion in ("COMPRAR","VENDER") and atr5>0:
    if suggestion=="COMPRAR":
        sl_price = price_now - atr_mult_sl*atr5
        tp_price = price_now + rr*atr_mult_sl*atr5
        stop_pct = (price_now - sl_price)/price_now
        tp_pct = (tp_price - price_now)/price_now
    else:
        sl_price = price_now + atr_mult_sl*atr5
        tp_price = price_now - rr*atr_mult_sl*atr5
        stop_pct = (sl_price - price_now)/price_now
        tp_pct = (price_now - tp_price)/price_now

# Alavancagem sugerida
profile_caps = {"Conservador":5, "Moderado":10, "Agressivo":20}
cap = profile_caps.get(perfil, 10)
base_from_conf = (2 if conf<55 else 4 if conf<65 else 6 if conf<75 else 9 if conf<85 else 12 if conf<92 else 15)
if stop_pct>0:
    if stop_pct<0.003: base_from_conf = int(round(base_from_conf*1.25))
    elif stop_pct>0.006: base_from_conf = int(round(base_from_conf*0.7))
lev_conf = max(1, min(base_from_conf, cap))
# Cap por risco: stake*lev*stop_pct <= banca*risco_pct
if stop_pct>0:
    lev_cap_risk = (banca*risco_pct)/(stake*stop_pct)
    lev_conf = int(max(1, min(lev_conf, cap, lev_cap_risk)))

# Estimativas
fees_roundtrip = 2*taker_fee_pct
net_gain_pct = max(tp_pct - fees_roundtrip, 0) * lev_conf
net_loss_pct = (stop_pct + fees_roundtrip) * lev_conf
est_gain_usdt = stake * net_gain_pct
est_loss_usdt = stake * net_loss_pct

cS1,cS2 = st.columns([1,2])
with cS1:
    color="🟢" if suggestion=="COMPRAR" else ("🔴" if suggestion=="VENDER" else "⏸️")
    st.metric("Sugestão", f"{color} {suggestion}", help=reason)
with cS2:
    if sl_price and tp_price:
        st.markdown('<div class="sugestao-box">', unsafe_allow_html=True)
        st.write(f"**Entrada**: ~{price_now:.6f} | **Stop**: {sl_price:.6f} | **Alvo**: {tp_price:.6f}")
        st.write(f"**Stake**: {stake:.2f} USDT | **Lev sugerida**: {lev_conf}×")
        st.write(f"**Estimativa** → TP: ~{est_gain_usdt:.2f} | SL: ~{est_loss_usdt:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Sem TP/SL no momento.")

st.divider()

# =========================
# Execução – Painel de Foco
# =========================
st.markdown("### ⚡ Execução (Foco)")
disabled_exec = (client is None)
colX1, colX2, colX3 = st.columns([1,1,1])

def calc_order_qty(stake_usdt: float, leverage: int, price: float, step: float = 0.001) -> float:
    if price <= 0: return 0.0
    raw = (stake_usdt * leverage) / price
    # arredonda para múltiplo do step (ex: 0.001 para BTC, 0.01 em outros símbolos – ajuste se quiser por símbolo)
    q = int(raw/step) * step
    return max(round(q, 6), 0.0)

# Exibe posição atual
pos_snap = get_position_snapshot(client, symbol) if client else {"amt":0.0,"entry":np.nan,"uPnL":0.0,"margin":np.nan}
pos_side = "LONG" if pos_snap["amt"]>0 else ("SHORT" if pos_snap["amt"]<0 else "FLAT")
st.caption(f"📌 Posição atual {symbol}: {pos_side} | qty={pos_snap['amt']} | entry={pos_snap['entry']} | uPnL={pos_snap['uPnL']:.2f} USDT")

with colX1:
    if st.button("🟢 Comprar (Mercado)", use_container_width=True, disabled=disabled_exec or suggestion=="AGUARDAR"):
        if sl_price and tp_price and lev_conf>0:
            if setup_futures_pair(client, symbol, lev_conf):
                qty = calc_order_qty(stake, lev_conf, price_now, step=0.001)
                ok, res = place_entry_and_brackets(client, symbol, "BUY", qty, sl_price, tp_price)
                if ok:
                    st.session_state.last_trade = {"symbol":symbol, "side":"BUY", "qty":qty, "entry_price":price_now, "time":time.time()}
                    st.success(f"✅ BUY executado. Qty={qty}")
                else:
                    st.error(f"❌ Falha BUY: {res}")
        else:
            st.warning("Sem parâmetros válidos de SL/TP/Alavancagem.")

with colX2:
    if st.button("🔴 Vender (Mercado)", use_container_width=True, disabled=disabled_exec or suggestion=="AGUARDAR"):
        if sl_price and tp_price and lev_conf>0:
            if setup_futures_pair(client, symbol, lev_conf):
                qty = calc_order_qty(stake, lev_conf, price_now, step=0.001)
                ok, res = place_entry_and_brackets(client, symbol, "SELL", qty, sl_price, tp_price)
                if ok:
                    st.session_state.last_trade = {"symbol":symbol, "side":"SELL", "qty":qty, "entry_price":price_now, "time":time.time()}
                    st.success(f"✅ SELL executado. Qty={qty}")
                else:
                    st.error(f"❌ Falha SELL: {res}")
        else:
            st.warning("Sem parâmetros válidos de SL/TP/Alavancagem.")

with colX3:
    if st.button("✖️ Fechar posição", use_container_width=True, disabled=disabled_exec):
        ok, res = market_close_position(client, symbol)
        if ok:
            # Atualiza PnL realizado com base no snapshot antes do fechamento
            snap_before = pos_snap if pos_snap else {"uPnL":0.0}
            pnl_add = float(snap_before.get("uPnL", 0.0))
            st.session_state.realized_pnl += pnl_add
            st.success(f"✅ Posição fechada. PnL realizado +{pnl_add:.2f} USDT | Total dia: {st.session_state.realized_pnl:.2f} USDT")
        else:
            st.error(f"❌ Falha ao fechar: {res}")

# Botão inverter
if st.button("🔁 Inverter posição", use_container_width=True, disabled=disabled_exec):
    ok, res = invert_position(client, symbol)
    if ok:
        st.success("✅ Posição invertida.")
    else:
        st.error(f"❌ Falha ao inverter: {res}")

# PnL total do dia
st.metric("💰 PnL Realizado (sessão)", f"{st.session_state.realized_pnl:.2f} USDT")

st.divider()

# =========================
# Scanner de Pares
# =========================
st.markdown("### 🧭 Scanner de Pares")

colS1, colS2 = st.columns([3,1])
with colS1:
    pairs = st.multiselect("Selecione os pares", [
        "BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"
    ], default=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"])
with colS2:
    look = st.number_input("Lookback 5m (candles)", 100, 500, 200, step=50)

do_scan = st.button("🔎 Scan agora", use_container_width=True)

@st.cache_data(ttl=25)
def scan_pair(sym: str) -> Dict:
    d1 = add_core_features(fetch_klines_rest(sym, "1d", 400))
    h1 = add_core_features(fetch_klines_rest(sym, "1h", 500))
    m5 = add_core_features(fetch_klines_rest(sym, "5m", 500))
    b1, b2, b3 = d1.iloc[-1], h1.iloc[-1], m5.iloc[-1]
    trends = [trend_label(x) for x in [b1,b2,b3]]
    count_up = trends.count("ALTA")
    count_dn = trends.count("BAIXA")
    align = max(count_up, count_dn)
    consensus = "ALTA" if count_up>count_dn else ("BAIXA" if count_dn>count_up else "NEUTRA")
    conf = confidence_from_features(b3, consensus, align)
    price = float(b3["c"]); atr5 = float(b3["atr14"]) or 0.0

    sug = "AGUARDAR"; sl = None; tp = None; stop_pct = 0.0; tp_pct = 0.0
    if align==3 and conf>=60:
        sug = "COMPRAR" if consensus=="ALTA" else ("VENDER" if consensus=="BAIXA" else "AGUARDAR")
    elif align==2 and conf>=70:
        sug = "COMPRAR" if count_up==2 else ("VENDER" if count_dn==2 else "AGUARDAR")
    if sug!="AGUARDAR" and atr5>0:
        if sug=="COMPRAR":
            sl = price - 1.0*atr5; tp = price + 1.5*atr5
            stop_pct = (price-sl)/price; tp_pct = (tp-price)/price
        else:
            sl = price + 1.0*atr5; tp = price - 1.5*atr5
            stop_pct = (sl-price)/price; tp_pct = (price-tp)/price

    base_from_conf = 2 if conf<55 else 4 if conf<65 else 6 if conf<75 else 9 if conf<85 else 12 if conf<92 else 15
    if stop_pct>0:
        if stop_pct<0.003: base_from_conf = int(round(base_from_conf*1.25))
        elif stop_pct>0.006: base_from_conf = int(round(base_from_conf*0.7))
    lev = max(1, min(base_from_conf, 20))
    score = align*conf
    return {
        "Par": sym, "Consenso": consensus, "Alinhamento": align, "Confiança": conf,
        "Sugestão": sug, "Preço": round(price, 6),
        "Stop": (round(sl, 6) if sl else None), "Alvo": (round(tp, 6) if tp else None),
        "Stop%": round(stop_pct*100, 2) if stop_pct>0 else None,
        "Alvo%": round(tp_pct*100, 2) if tp_pct>0 else None,
        "Lev sug": lev, "Score": score
    }

if do_scan:
    with st.spinner("Escaneando..."):
        if not pairs:
            df_scan = pd.DataFrame(columns=["Par","Consenso","Alinhamento","Confiança","Sugestão","Preço","Stop","Alvo","Stop%","Alvo%","Lev sug","Score"])
        else:
            rows = [scan_pair(p) for p in pairs]
            df_scan = pd.DataFrame(rows)
            sug_order = {"COMPRAR":0,"VENDER":1,"AGUARDAR":2}
            df_scan["Sugestão_ord"] = df_scan["Sugestão"].map(sug_order)
            df_scan = df_scan.sort_values(["Confiança","Sugestão_ord"], ascending=[False, True]).drop(columns="Sugestão_ord")
        st.dataframe(df_scan, use_container_width=True)
else:
    st.info("Clique em **Scan agora** para atualizar a lista.")

st.caption("⚠️ Conteúdo educacional. Ajuste alavancagem e risco conforme seu perfil e conta.")
