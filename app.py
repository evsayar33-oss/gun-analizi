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
    st_autorefresh(interval=900 * 1000, key="sentinel_pure_v22")
    st.set_page_config(page_title="Sentinel V22 Pure Quant", layout="wide")
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

# --- 1. VERİ MOTORU ---
def fetch_pure_data():
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    # Günlük Makro + Hacim
    raw_y = yf.download(list(syms.values()), period="5y", interval="1d", progress=False)
    df_y = raw_y['Close'].ffill().bfill().rename(columns={v: k for k, v in syms.items()})
    
    # 15 Dakikalık Fiyat ve Hacim (Hacim Onayı İçin Şart)
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="5d", interval="15m", progress=False)
    df_h_close = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    df_h_vol = h_raw['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    
    df_f = pd.DataFrame(index=df_y.index)
    if FRED_API_KEY:
        try:
            fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
            for name, s_id in fred_ids.items():
                r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={FRED_API_KEY}&file_type=json", timeout=5).json()
                if 'observations' in r:
                    obs = pd.DataFrame(r['observations'])[['date', 'value']]
                    obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                    obs['date'] = pd.to_datetime(obs['date'])
                    df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
        except: pass
    
    if 'SPREAD' not in df_f.columns: df_f['SPREAD'] = (100 - df_y['HYG']).rolling(20).mean()
    if 'T10YIE' not in df_f.columns: df_f['T10YIE'] = 2.1
    if 'WALCL' not in df_f.columns: df_f['WALCL'] = 7150000
    return df_y, df_h_close, df_h_vol, df_f.ffill().bfill()

# --- 2. ANALİZ MOTORU ---
try:
    df_y, df_h, df_v, df_f = fetch_pure_data()
    master = pd.concat([df_y, df_f], axis=1).ffill()
    
    # Makro Temel (Z-Score)
    z_rf = z_score(master['TNX'] - master['T10YIE'])
    z_dxy = z_score(master['DXY'])
    z_liq = z_score(master['WALCL'])
    z_spr = z_score(master['SPREAD'])

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    current_results = {}

    if os.path.exists(STATE_FILE):
        try:
            old_df = pd.read_csv(STATE_FILE, index_col=0)
            old_scores = old_df.to_dict()['score']
        except: old_scores = {}
    else: old_scores = {}

    for i, (name, signs) in enumerate(assets.items()):
        # 1. MAKRO SKOR (Ağırlık %40)
        m_env = (0.5 * z_liq.iloc[-1] + 0.3 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.2 * z_spr.iloc[-1] * signs[2])
        
        # 2. HACİM ONAYLI MOMENTUM (Ağırlık %60 - EMASIZ)
        # 1 saatlik (4 bar) Fiyat ve Hacim analizi
        roc_1h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        
        # Hacim Z-Skoru (Son 24 saatlik 15dk'lık hacimlere göre)
        v_series = df_v[name].tail(96) # 24 saat = 96 adet 15dk
        v_z = (df_v[name].iloc[-1] - v_series.mean()) / (v_series.std() + 1e-9)
        
        # --- KRİTİK: HACİM KİLİDİ ---
        # Eğer hacim Z-skoru 1.0'dan küçükse (hacim zayıfsa), momentumun etkisini %80 buda!
        # Bu, NDX'teki gibi hacimsiz "testere" hareketlerini engeller.
        if v_z < 1.0:
            momentum_impact = roc_1h * 0.5 
        else:
            momentum_impact = roc_1h * 5.0 # Hacim varsa ivmeyi ciddiye al
            
        total_score = round((m_env * 0.4) + (momentum_impact * 0.6), 2)
        
        signal = "AL" if total_score > 0.30 else ("SAT" if total_score < -0.30 else "NOTR")
        current_results[name] = total_score

        # ALARM KONTROLÜ
        if name in old_scores:
            old_val = old_scores[name]
            diff = total_score - old_val
            
            # SADECE HACİM ONAYLI VE GÜÇLÜ DEĞİŞİMLERDE MESAJ AT
            if abs(diff) > 0.90 and v_z > 0.5:
                direction = "YÜKSELİŞ" if diff > 0 else "DÜŞÜŞ"
                send_telegram(f"🚨 *HACİM ONAYLI ŞOK: {name}*\nFiyat bu hareketi `{v_z:.1f}z` hacimle yaptı.\nEski: `{old_val}` -> Yeni: *{total_score}*")
            
            old_sig = "AL" if old_val > 0.30 else ("SAT" if old_val < -0.30 else "NOTR")
            if old_sig != signal and signal != "NOTR" and v_z > 0.8:
                emoji = "🚀" if signal == "AL" else "📉"
                send_telegram(f"{emoji} *SİNYAL DÖNÜŞÜ (HACİMLİ): {name}*\nYön: `{old_sig}` -> *{signal}*\nSkor: `{total_score}` (Hacim: {v_z:.1f}z)")

    # WEB ARAYÜZÜ (Görselleştirme)
    if not is_headless:
        st.title("🏛️ ALPHA SENTINEL V22 - PURE QUANT")
        cols = st.columns(4)
        for i, (name, score) in enumerate(current_results.items()):
            sig = "AL" if score > 0.30 else ("SAT" if score < -0.30 else "NOTR")
            with cols[i]:
                st.metric(name, score, sig)

    pd.DataFrame.from_dict(current_results, orient='index', columns=['score']).to_csv(STATE_FILE)

except Exception as e:
    if is_headless: print(f"Hata: {e}")
    else: st.error(f"Sistem Hatası: {e}")
