import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import os
import sys

# --- 0. MOD VE TASARIM AYARLARI ---
is_headless = "--headless" in sys.argv
if not is_headless:
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh
    # Web sitesi 15 dakikada bir yenilenir
    st_autorefresh(interval=900 * 1000, key="sentinel_silent_v22")
    st.set_page_config(page_title="Alpha Sentinel Silent", layout="wide")
    st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

# SECRETS
if is_headless:
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHAT_ID = os.environ.get("CHAT_ID")
    FRED_API_KEY = os.environ.get("FRED_API_KEY")
else:
    TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", None)
    CHAT_ID = st.secrets.get("CHAT_ID", None)
    FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

STATE_FILE = "last_scores.csv"

def send_telegram(text):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try: requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
        except: pass

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=5).mean()) / (s.rolling(win, min_periods=5).std() + 1e-9)

def fetch_data():
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    df_y = yf.download(list(syms.values()), period="5y", interval="1d", progress=False)['Close'].ffill().bfill()
    df_y = df_y.rename(columns={v: k for k, v in syms.items()})
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="2d", interval="15m", progress=False)
    df_h_close = df_h['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    df_h_vol = df_h['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    
    df_f = pd.DataFrame(index=df_y.index)
    if FRED_API_KEY:
        try:
            r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id=WALCL&api_key={FRED_API_KEY}&file_type=json", timeout=5).json()
            if 'observations' in r:
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f['WALCL'] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
        except: pass
    
    if 'WALCL' not in df_f.columns: df_f['WALCL'] = 7150000
    df_f['T10YIE'] = 2.1
    df_f['SPREAD'] = (100 - df_y['HYG']).rolling(20).mean()
    return df_y, df_h_close, df_h_vol, df_f.ffill().bfill()

# --- ANA MOTOR ---
try:
    df_y, df_h, df_v, df_f = fetch_data()
    master = pd.concat([df_y, df_f], axis=1).ffill()
    
    z_rf = z_score(master['TNX'] - master['T10YIE'])
    z_dxy = z_score(master['DXY'])
    z_liq = z_score(master['WALCL'])
    z_spr = z_score(master['SPREAD'])

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    current_results = {}

    # Önceki skorları yükle
    if os.path.exists(STATE_FILE):
        try:
            old_df = pd.read_csv(STATE_FILE)
            old_scores = old_df.set_index(old_df.columns[0])['score'].to_dict()
        except: old_scores = {}
    else: old_scores = {}

    for i, (name, signs) in enumerate(assets.items()):
        # 1. Makro Temel
        m_env = (0.5 * z_liq.iloc[-1] + 0.3 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.2 * z_spr.iloc[-1] * signs[2])
        # 2. Hacim Onaylı Momentum (Pure Quant)
        roc_1h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        v_series = df_v[name].tail(96)
        v_z = (df_v[name].iloc[-1] - v_series.mean()) / (v_series.std() + 1e-9)
        
        # Momentum çarpanı (Hacim varsa 4.5x, yoksa 0.5x)
        mom_impact = roc_1h * 4.5 if v_z > 0.5 else roc_1h * 0.5
        total_score = round((m_env * 0.4) + (mom_impact * 0.6), 2)
        
        signal = "AL" if total_score > 0.30 else ("SAT" if total_score < -0.30 else "NOTR")
        current_results[name] = total_score

        # ALARM DENETİMİ (SADECE DEĞİŞİM VARSA)
        if name in old_scores:
            old_val = old_scores[name]
            diff = total_score - old_val
            
            # Sinyal Dönüşü (Hacimle destekleniyorsa)
            old_sig = "AL" if old_val > 0.30 else ("SAT" if old_val < -0.30 else "NOTR")
            if old_sig != signal and signal != "NOTR" and v_z > 0.6:
                emoji = "🚀" if signal == "AL" else "📉"
                send_telegram(f"{emoji} *SİNYAL DÖNÜŞÜ: {name}*\n`{old_sig}` -> *{signal}*\nSkor: `{total_score}` (Hacim: {v_z:.1f}z)")
            
            # Güçlü Skor Sıçraması (Balina/Haber Kontrolü)
            if abs(diff) > 1.0 and v_z > 0.5:
                send_telegram(f"🚨 *GÜÇLÜ SKOR DEĞİŞİMİ: {name}*\nEski: `{old_val}` -> Yeni: *{total_score}*\nFark: `{diff:+.2f}`")

    # WEB ARAYÜZÜ (Dashboard her zaman taze kalır)
    if not is_headless:
        st.title("🏛️ ALPHA SENTINEL V22 - SILENT")
        cols = st.columns(4)
        for i, (name, sc) in enumerate(current_results.items()):
            with cols[i]: st.metric(name, sc, "AL" if sc > 0.30 else "SAT" if sc < -0.30 else "NOTR")

    # Yeni skorları kaydet (Bir sonraki 15 dk için hafıza)
    pd.DataFrame.from_dict(current_results, orient='index', columns=['score']).to_csv(STATE_FILE)

except Exception as e:
    if is_headless: print(f"Hata: {e}")
    else: st.error(f"Hata: {e}")
