# -*- coding: utf-8 -*-
"""
Painel – Binance v13.1 (scanner + foco) – robusto em rede
---------------------------------------------------------
• Multi-timeframe (1D, 1H, 5M) com EMA20/50/200 + RSI
• Sinal: COMPRAR / VENDER / AGUARDAR
• TP/SL via ATR(5m) para referência
• Scanner de pares + Painel de Foco
• Carregamento resiliente (timeout curto, retries, fallback de host)
• Botão '🔄 Recarregar dados' para limpar cache
• Execução está DESABILITADA por padrão em cloud (Bloqueios regionais Binance).
"""

from __future__ import annotations
import os
import time
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ===== Config visual/geral =====
st.set_page_config(page_title="📊 Painel – Binance v13.1", layout="wide")
st.title("📊 Painel – Binance v13.1")
st.caption("Carregamento resiliente de dados de mercado (data-api.binance.vision → api.binance.com).")

# Execução (ordens) desativada na nuvem para evitar erros de elegibilidade/451.
ENABLE_EXECUTION = False

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

# ===== Pares padrão (scanner/foco) =====
PAIRS_DEFAULT: List[str] = [
    "BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT",
    "SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT",
]

# ===== Rede / Timeouts =====
REQUEST_TIMEOUT = 6      # segundos
MAX_RETRIES     = 3
BINANCE_SPOT_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",  # preferido
    "https://api.binance.com/api/v3/klines",          # fallback
]
HEADERS_HTTP = {"User-Agent": "streamlit-binance-panel/13.1"}

def http_get_json(url: str, params: dict) -> list:
    """GET com timeout curto e até MAX_RETRIES, retornando JSON; levanta última exceção em caso de falha."""
    last_err = None
    for i in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS_HTTP, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.5 * i)  # backoff simples
    raise last_err

def fetch_klines_rest(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": min(int(limit), 1500)}
    last_err = None
    for base in BINANCE_SPOT_ENDPOINTS:
        try:
            data = http_get_json(base, params)
            cols = ["openTime","open","high","low","close","volume","closeTime","quote","n","tb_base","tb_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)[["openTime","open","high","low","close","volume"]]
            df.columns = ["t","o","h","l","c","v"]
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            for c in ["o","h","l","c","v"]:
                df[c] = df[c].astype(float)
            return df
        except Exception as e:
            last_err = e
            continue
    st.error(f"Falha ao buscar klines ({symbol}, {interval}): {last_err}")
    return pd.DataFrame(columns=["t","o","h","l","c","v"])

# ===== Indicadores / Features =====
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
        (df["l"] - df["c"].shift()).abs(),
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

# ===== Multi-timeframe Loader (com proteção) =====
@st.cache_data(ttl=30)
def load_multitf(symbol: str) -> Dict[str, pd.DataFrame]:
    d1 = fetch_klines_rest(symbol, "1d", 400)
    h1 = fetch_klines_rest(symbol, "1h", 500)
    m5 = fetch_klines_rest(symbol, "5m", 500)
    if any(df.empty for df in (d1, h1, m5)):
        return {"1D": pd.DataFrame(), "1H": pd.DataFrame(), "5M": pd.DataFrame()}
    return {"1D": add_core_features(d1),
            "1H": add_core_features(h1),
            "5M": add_core_features(m5)}

# ===== Scanner =====
@st.cache_data(ttl=25)
def scan_pair(sym: str) -> Dict:
    d1 = fetch_klines_rest(sym, "1d", 400)
    h1 = fetch_klines_rest(sym, "1h", 500)
    m5 = fetch_klines_rest(sym, "5m", 500)
    if any(df.empty for df in (d1, h1, m5)):
        return {"Par": sym, "Confiança": 0, "Sugestão": "—"}

    b1, b2, b3 = add_core_features(d1).iloc[-1], add_core_features(h1).iloc[-1], add_core_features(m5).iloc[-1]
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

    score = align * conf
    return {
        "Par": sym, "Consenso": consensus, "Alinhamento": align, "Confiança": conf,
        "Sugestão": sug, "Preço": round(price, 6),
        "Stop": (round(sl, 6) if sl else None), "Alvo": (round(tp, 6) if tp else None),
        "Stop%": round(stop_pct*100, 2) if stop_pct>0 else None,
        "Alvo%": round(tp_pct*100, 2) if tp_pct>0 else None,
        "Lev sug": max(1, min(12, 2 + conf//10)), "Score": score
    }

# ===== Controles principais =====
with st.container():
    colA, colB, colC = st.columns([1.5, 1, 1])
    with colA:
        symbol = st.selectbox("Par (USDT-M Futures)", PAIRS_DEFAULT, index=1)
    with colB:
        margin_type = st.selectbox("Tipo de margem", ["ISOLATED","CROSSED"], index=0)
    with colC:
        leverage = st.number_input("Alavancagem", min_value=1, max_value=125, value=10, step=1)

# Botão de recarga/limpeza de cache antes de carregar
c_cache_left, c_cache_right = st.columns([1,4])
with c_cache_left:
    if st.button("🔄 Recarregar dados", help="Limpa cache e busca klines novamente"):
        load_multitf.clear()
        scan_pair.clear()
        st.cache_data.clear()

# ===== Painel de Foco (Leitura + Parâmetros) =====
st.subheader("🔎 Leitura Multi-timeframe")

with st.spinner("Carregando multi-timeframe…"):
    mtf = load_multitf(symbol)

if any(mtf[x].empty for x in ("1D","1H","5M")):
    st.warning("Não foi possível carregar todos os timeframes. Tente novamente (ou verifique a conectividade).")
    st.stop()

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

# ===== Parâmetros de Trade (referência) =====
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

# Sugestão + TP/SL de referência
suggestion, reason = "AGUARDAR", "Condições insuficientes"
if align_count==3 and conf>=min_conf:
    suggestion = "COMPRAR" if consensus=="ALTA" else ("VENDER" if consensus=="BAIXA" else "AGUARDAR")
    reason = f"Alinhamento 3/3 e confiança {conf}"
elif align_count==2 and conf>=(min_conf+10):
    suggestion = "COMPRAR" if count_up==2 else ("VENDER" if count_dn==2 else "AGUARDAR")
    reason = f"Alinhamento 2/3 com confiança {conf}"

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

# Alavancagem sugerida (heurística simples)
profile_caps = {"Conservador":5,"Moderado":10,"Agressivo":20}
perfil = st.selectbox("Perfil", list(profile_caps.keys()), index=1)
cap = profile_caps.get(perfil, 10)
base_from_conf = 2 if conf<55 else 4 if conf<65 else 6 if conf<75 else 9 if conf<85 else 12 if conf<92 else 15
if stop_pct>0:
    if stop_pct < 0.003: base_from_conf = int(round(base_from_conf*1.25))
    elif stop_pct > 0.006: base_from_conf = int(round(base_from_conf*0.7))
lev_conf = max(1, min(base_from_conf, cap))

# Estimativas simples
taker_fee_pct = st.number_input("Taxa taker (%)", 0.0, 0.5, 0.04, step=0.01) / 100.0
fees_roundtrip = 2*taker_fee_pct
net_gain_pct = max(tp_pct - fees_roundtrip, 0) * lev_conf if tp_price else 0.0
net_loss_pct = (stop_pct + fees_roundtrip) * lev_conf if sl_price else 0.0
est_gain_usdt = stake * net_gain_pct if tp_price else 0.0
est_loss_usdt = stake * net_loss_pct if sl_price else 0.0

cSug1, cSug2 = st.columns([1,2])
with cSug1:
    color = "🟢" if suggestion=="COMPRAR" else ("🔴" if suggestion=="VENDER" else "⏸️")
    st.metric("Sugestão", f"{color} {suggestion}", help=reason)
with cSug2:
    if sl_price and tp_price:
        st.info(f"Entrada: ~{price_now:.6f} | Stop: {sl_price:.6f} | Alvo: {tp_price:.6f}\n\n"
                f"Stake: {stake:.2f} USDT | Lev sugerida: {lev_conf}×\n\n"
                f"Estimativa → TP: ~{est_gain_usdt:.2f} | SL: ~{est_loss_usdt:.2f}")
    else:
        st.info("Sem TP/SL sugeridos (aguarde melhor alinhamento/confiança).")

# ===== Execução (desligada em cloud) =====
st.subheader("⚡ Execução")
if not ENABLE_EXECUTION:
    st.info("Execução está desativada (cloud). Ative localmente se desejar enviar ordens reais/testnet.")
else:
    st.warning("Coloque aqui sua camada de execução (python-binance / binance-connector), se for usar localmente.")

# ===== Scanner de pares =====
st.subheader("🧭 Scanner de pares – alinhamento + confiança")
pairs = st.multiselect(
    "Selecione os pares para escanear",
    PAIRS_DEFAULT,
    default=PAIRS_DEFAULT
)
colsc1, colsc2 = st.columns([1,1])
with colsc1:
    if st.button("🔎 Rodar scan agora"):
        scan_pair.clear()

rows = [scan_pair(p) for p in pairs] if pairs else []
df_scan = pd.DataFrame(rows)
if not df_scan.empty and "Confiança" in df_scan.columns:
    sug_order = {"COMPRAR":0,"VENDER":1,"AGUARDAR":2,"—":3}
    df_scan["SugOrd"] = df_scan["Sugestão"].map(sug_order).fillna(9)
    df_scan = df_scan.sort_values(["Confiança","SugOrd"], ascending=[False, True]).drop(columns="SugOrd")
    st.dataframe(df_scan, use_container_width=True, height=360)
else:
    st.info("Selecione pares para escanear.")

st.caption("⚠️ Observação: em ambientes cloud, o acesso a endpoints da Binance pode estar sujeito a restrições regionais. Em caso de falha, use o botão 'Recarregar dados' ou rode localmente.")
