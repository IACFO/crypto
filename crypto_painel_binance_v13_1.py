# -*- coding: utf-8 -*-
"""
Streamlit – Crypto Painel (Render safe boot)
- Boot leve: sem NTP, sem chamadas à Binance no import.
- Scanner usa REST público da Binance (sem chave, sem assinatura).
- Conexão Futures (ordens) só quando o usuário clicar no botão.
"""

from __future__ import annotations
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# Config / Constantes
# =========================
BINANCE_REST = "https://api.binance.com"
RECV_WINDOW_MS = 60000  # 60s para chamadas assinadas
PUBLIC_TIMEOUT = 8       # timeout das chamadas públicas

st.set_page_config(page_title="Crypto Painel – Binance", page_icon="📊", layout="wide")
st.title("📊 Crypto Painel – Binance")
st.caption("Boot leve para Render: conexão à Binance só quando solicitado.")

# =========================
# Utils REST público
# =========================
def fetch_klines_rest(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Klines via API pública (sem chave)."""
    r = requests.get(
        f"{BINANCE_REST}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": min(int(limit), 1500)},
        timeout=PUBLIC_TIMEOUT,
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
    if trend == "ALTA":
        rsi_comp = max(0, rsi_comp)
    elif trend == "BAIXA":
        rsi_comp = max(0, -rsi_comp)
    else:
        rsi_comp = 0
    align_bonus = align_count/3.0
    conf = (0.35*stf + 0.25*vr + 0.20*sr + 0.10*rsi_comp + 0.10*align_bonus) * 100
    return int(round(min(max(conf,0),100)))

# =========================
# Multi-timeframe
# =========================
@st.cache_data(ttl=30)
def load_multitf(symbol: str) -> Dict[str, pd.DataFrame]:
    d1 = fetch_klines_rest(symbol, "1d", 400)
    h1 = fetch_klines_rest(symbol, "1h", 500)
    m5 = fetch_klines_rest(symbol, "5m", 500)
    return {"1D": add_core_features(d1), "1H": add_core_features(h1), "5M": add_core_features(m5)}

# =========================
# UI – Seleção + Leitura do Par
# =========================
st.subheader("📡 Painel de Monitoramento e Sinais")
symbol = st.selectbox("Par (símbolo)", [
    "ETHUSDT","BTCUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT"
], index=0)

try:
    mtf = load_multitf(symbol)
except Exception as e:
    st.error(f"Falha ao carregar dados públicos: {e}")
    st.stop()

if any(df is None or df.empty for df in mtf.values()):
    st.error("Sem dados para o par/intervalo no momento. Tente novamente.")
    st.stop()

# Construir leitura do símbolo focado
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
count_up = values.count(1)
count_dn = values.count(-1)
align_count = max(count_up, count_dn)
consensus = "ALTA" if count_up>count_dn else ("BAIXA" if count_dn>count_up else "NEUTRA")
conf = confidence_from_features(mtf["5M"].iloc[-1], consensus, align_count)

cL, cR = st.columns([2,1])
with cL:
    st.markdown("### 🔎 Leitura Multi-timeframe")
    colA, colB, colC = st.columns(3)
    for tf, col in zip(["1D","1H","5M"], [colA,colB,colC]):
        info = blocks[tf]
        badge = "🟢" if info["trend"]=="ALTA" else ("🔴" if info["trend"]=="BAIXA" else "⚪")
        col.metric(f"{tf} – {badge} {info['trend']}", value=f"RSI {info['rsi']:.1f}",
                   help=f"EMA20/50/200: {info['ema20']:.4f} / {info['ema50']:.4f} / {info['ema200']:.4f}\n"
                        f"Vol.rel: {info['vol_rel']:.2f}× | Str: {info['strength']:.2f}")
    st.caption(f"Alinhamento: {align_count}/3 • Consenso: {consensus}")
with cR:
    st.markdown("### 🔐 Confiança do Sinal")
    st.metric("Nível de confiança", f"{conf}/100")

st.divider()

# ===================================================
# Seção EXECUÇÃO (lazy import da Binance no clique)
# ===================================================
st.markdown("## 🚀 Execução – Binance Futures")
st.caption("Esta seção carrega a SDK da Binance apenas quando necessário (evita travar o boot no Render).")

colE1, colE2, colE3, colE4 = st.columns(4)
with colE1:
    stake = st.number_input("Stake (USDT)", 1.0, 1e9, 50.0, step=1.0)
with colE2:
    perfil = st.selectbox("Perfil", ["Conservador","Moderado","Agressivo"], index=1)
with colE3:
    atr_mult_sl = st.number_input("SL = ATR ×", 0.2, 5.0, 1.0)
with colE4:
    rr = st.number_input("R:R (TP = R × SL)", 0.5, 5.0, 1.5)

price_now = float(blocks["5M"]["close"])
atr5 = float(blocks["5M"]["atr"]) or 0.0

# Direção sugerida simples
suggestion = "AGUARDAR"
if align_count == 3 and conf >= 60:
    suggestion = "COMPRAR" if consensus == "ALTA" else ("VENDER" if consensus == "BAIXA" else "AGUARDAR")
elif align_count == 2 and conf >= 70:
    suggestion = "COMPRAR" if count_up == 2 else ("VENDER" if count_dn == 2 else "AGUARDAR")

sl_price = tp_price = None
if suggestion in ("COMPRAR","VENDER") and atr5>0:
    if suggestion == "COMPRAR":
        sl_price = price_now - atr_mult_sl*atr5
        tp_price = price_now + rr*atr_mult_sl*atr5
    else:
        sl_price = price_now + atr_mult_sl*atr5
        tp_price = price_now - rr*atr_mult_sl*atr5

profile_caps = {"Conservador": 5, "Moderado": 10, "Agressivo": 20}
lev_conf = profile_caps.get(perfil, 10)

st.info(f"🎯 Sinal: **{suggestion}** | Preço ~ {price_now:.6f} | SL: {sl_price and f'{sl_price:.6f}'} | TP: {tp_price and f'{tp_price:.6f}'} | Alav: {lev_conf}×")

# Botões de ação (conexão e execução só aqui)
c1, c2, c3 = st.columns(3)
do_connect = c1.button("🔌 Conectar Binance", use_container_width=True)
do_execute = c2.button("🚀 Executar Ação", use_container_width=True, disabled=(suggestion=="AGUARDAR" or not sl_price or not tp_price))
do_close_reverse = c3.button("🔁 Fechar e Inverter", use_container_width=True)

# Guardamos objetos em session_state para reuso
if "bn_client" not in st.session_state:
    st.session_state.bn_client = None

def _get_client() -> Optional["Client"]:
    return st.session_state.bn_client

if do_connect:
    try:
        # Import só aqui (lazy)
        from binance.client import Client
        api_key = st.secrets["binance"]["api_key"]
        api_secret = st.secrets["binance"]["api_secret"]
        client = Client(api_key=api_key, api_secret=api_secret)

        # Ajuste de offset pelo server time
        srv = client.get_server_time()  # {"serverTime": ms}
        client.TIME_OFFSET = int(srv["serverTime"]) - int(time.time() * 1000)

        # Teste Futures
        info = client.futures_account(recvWindow=RECV_WINDOW_MS)
        bal = info.get("totalWalletBalance") or info.get("availableBalance")
        st.session_state.bn_client = client
        st.success(f"✅ Conectado! Saldo Futures: {bal} USDT")
    except Exception as e:
        st.error(f"❌ Falha na conexão: {e}")

def _setup_pair(client, symbol: str, leverage: int = 10, margin_type: str = "ISOLATED") -> bool:
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type, recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        if "No need to change margin type" not in str(e):
            st.error(f"Erro margem: {e}")
            return False
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage, recvWindow=RECV_WINDOW_MS)
    except Exception as e:
        st.error(f"Erro alavancagem: {e}")
        return False
    return True

def _market_order_with_brackets(client, symbol: str, side: str, qty: float, sl: float, tp: float):
    from binance.enums import *
    try:
        # entrada
        ordem = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side=="BUY" else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=qty,
            recvWindow=RECV_WINDOW_MS
        )
        # proteções
        if side=="BUY":
            client.futures_create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_STOP_MARKET,
                                        stopPrice=round(sl,6), closePosition=True, timeInForce="GTC",
                                        recvWindow=RECV_WINDOW_MS)
            client.futures_create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                                        stopPrice=round(tp,6), closePosition=True, timeInForce="GTC",
                                        recvWindow=RECV_WINDOW_MS)
        else:
            client.futures_create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_STOP_MARKET,
                                        stopPrice=round(sl,6), closePosition=True, timeInForce="GTC",
                                        recvWindow=RECV_WINDOW_MS)
            client.futures_create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                                        stopPrice=round(tp,6), closePosition=True, timeInForce="GTC",
                                        recvWindow=RECV_WINDOW_MS)
        return True, ordem
    except Exception as e:
        return False, str(e)

def _position_info(client, symbol: str):
    pos = client.futures_position_information(symbol=symbol, recvWindow=RECV_WINDOW_MS)
    if not pos:
        return None
    p = pos[0]
    return {
        "positionAmt": float(p["positionAmt"]),
        "entryPrice": float(p["entryPrice"]),
        "unRealizedProfit": float(p["unRealizedProfit"]),
        "marginType": p.get("marginType"),
        "leverage": int(p.get("leverage", 0)),
    }

def _close_and_reverse(client, symbol: str):
    from binance.enums import *
    try:
        # fecha posição atual (se houver)
        pos = _position_info(client, symbol)
        if pos and abs(pos["positionAmt"])>0:
            if pos["positionAmt"]>0:
                client.futures_create_order(symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET,
                                            quantity=abs(pos["positionAmt"]), reduceOnly=True,
                                            recvWindow=RECV_WINDOW_MS)
                new_side = "SELL"
            else:
                client.futures_create_order(symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET,
                                            quantity=abs(pos["positionAmt"]), reduceOnly=True,
                                            recvWindow=RECV_WINDOW_MS)
                new_side = "BUY"
        else:
            # se não tinha posição, apenas inverte em relação ao sinal
            new_side = "SELL" if suggestion=="COMPRAR" else "BUY"

        # calcula qty para nova entrada (mesmo stake * lev)
        price = float(blocks["5M"]["close"])
        lev = int(profile_caps.get(perfil, 10))
        qty = round((stake * lev) / price, 3)
        return _market_order_with_brackets(client, symbol, new_side, qty, sl_price, tp_price)
    except Exception as e:
        return False, str(e)

# Execução dos botões
client_obj = _get_client()

if do_execute:
    if client_obj is None:
        st.error("Conecte à Binance primeiro.")
    elif suggestion not in ("COMPRAR","VENDER") or not sl_price or not tp_price:
        st.warning("Sem sinal/TP/SL válidos no momento.")
    else:
        if _setup_pair(client_obj, symbol, lev_conf):
            side = "BUY" if suggestion=="COMPRAR" else "SELL"
            qty = round((stake * lev_conf) / price_now, 3)
            ok, res = _market_order_with_brackets(client_obj, symbol, side, qty, sl_price, tp_price)
            if ok:
                st.success("✅ Ordem enviada com SL/TP.")
            else:
                st.error(f"❌ Erro: {res}")

if do_close_reverse:
    if client_obj is None:
        st.error("Conecte à Binance primeiro.")
    else:
        ok, res = _close_and_reverse(client_obj, symbol)
        if ok:
            st.success("🔁 Posição fechada e invertida com sucesso.")
        else:
            st.error(f"❌ Erro ao fechar/inverter: {res}")

# Info de posição (se conectado)
if client_obj is not None:
    try:
        p = _position_info(client_obj, symbol)
        if p and abs(p["positionAmt"])>0:
            st.info(f"📌 Posição: {p['positionAmt']} @ {p['entryPrice']:.4f} | UPNL: {p['unRealizedProfit']:.2f} USDT | {p['leverage']}× {p['marginType']}")
        else:
            st.caption("Sem posição aberta no momento.")
    except Exception as e:
        st.caption(f"(pos) {e}")
