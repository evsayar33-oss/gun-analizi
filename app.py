import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import os
import sys

# --- MOD KONTROLÜ ---
is_headless = "--headless" in sys.argv
if not is_headless:
    import streamlit as st
    st.set_page_config(page_title="Sentinel V20 Pro", layout="wide")
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
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="2d", interval="15m", progress=False)['Close'].ffill().bfill()
    df_h = df_h.rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})
    
    df_f = pd.DataFrame(index=df_y.index)
    if FRED_API_KEY:
        fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
        for name, s_id in fred_ids.items():
            try:
                r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={FRED_API_KEY}&file_type=json", timeout=5).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
            except: pass
    
    # Fallback for FRED
    if 'SPREAD' not in df_f.columns: df_f['SPREAD'] = (100 - df_y['HYG']).rolling(20).mean()
    if 'T10YIE' not in df_f.columns: df_f['T10YIE'] = 2.1
    if 'WALCL' not in df_f.columns: df_f['WALCL'] = 7150000

    return df_y, df_h, df_f.ffill().bfill()

# --- ANA MOTOR ---
try:
    df_y, df_h, df_f = fetch_data()
    master = pd.concat([df_y, df_f], axis=1).ffill()
    
    z_rf = z_score(master['TNX'] - master['T10YIE'])
    z_dxy = z_score(master['DXY'])
    z_spr = z_score(master['SPREAD'])
    z_liq = z_score(master['WALCL'])

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    current_results = {}

    # Hafıza Dosyasını Oku
    if os.path.exists(STATE_FILE):
        try:
            old_df = pd.read_csv(STATE_FILE, index_col=0)
            old_scores = old_df.to_dict()['score']
        except: old_scores = {}
    else: old_scores = {}

    if not is_headless:
        st.title("🏛️ ALPHA SENTINEL V20 - ALERTS")
        cols = st.columns(4)

    for i, (name, signs) in enumerate(assets.items()):
        # Hesaplama
        m_env = (0.5 * z_liq.iloc[-1] + 0.3 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.2 * z_spr.iloc[-1] * signs[2])
        roc = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        total_score = round((m_env * 0.4) + (roc * 4.5), 2)
        
        signal = "AL" if total_score > 0.15 else ("SAT" if total_score < -0.15 else "NOTR")
        current_results[name] = total_score

        # ALARM KONTROLÜ
        if name in old_scores:
            diff = total_score - old_scores[name]
            if abs(diff) > 0.60:
                send_telegram(f"🚨 *GÜÇLÜ SKOR DEĞİŞİMİ: {name}*\nFark: `{diff:+.2f}` | Yeni Skor: *{total_score}*")
            
            old_sig = "AL" if old_scores[name] > 0.15 else ("SAT" if old_scores[name] < -0.15 else "NOTR")
            if old_sig != signal and signal != "NOTR":
                send_telegram(f"🚀 *SİNYAL DÖNÜŞÜ: {name}*\nYön: `{old_sig}` -> *{signal}*")

        if not is_headless:
            with cols[i]:
                st.metric(name, total_score, signal)

    # Skorları kalıcı olarak kaydet
    pd.DataFrame.from_dict(current_results, orient='index', columns=['score']).to_csv(STATE_FILE)

except Exception as e:
    if is_headless: print(f"Hata: {e}")
    else: st.error(f"Sistem Hatası: {e}")
