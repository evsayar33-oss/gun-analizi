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
    st_autorefresh(interval=900 * 1000, key="sentinel_institutional_v23")
    st.set_page_config(page_title="Alpha Sentinel V23 - Guard", layout="wide")
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
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="5d", interval="15m", progress=False)
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
    vix_now = df_y['VIX'].iloc[-1]

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    current_results = {}
    alert_triggered = False

    if os.path.exists(STATE_FILE):
        try:
            old_df = pd.read_csv(STATE_FILE)
            old_scores = old_df.set_index(old_df.columns[0])['score'].to_dict()
        except: old_scores = {}
    else: old_scores = {}

    for i, (name, signs) in enumerate(assets.items()):
        # 1. MAKRO ÇAPA (%50 Ağırlık - Güçlendirildi)
        m_env = (0.5 * z_liq.iloc[-1] + 0.3 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.2 * z_spr.iloc[-1] * signs[2])
        
        # 2. KURUMSAL MOMENTUM (%50 Ağırlık)
        # 1 Saatlik İvme
        roc_1h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        
        # Hacim Z-Skoru (Hassasiyet 168 saat / 7 güne çıkarıldı)
        v_series = df_v[name].tail(168 * 4) # 7 günlük 15m verileri
        v_z = (df_v[name].iloc[-1] - v_series.mean()) / (v_series.std() + 1e-9)
        
        # --- INSTITUTIONAL FILTER: VOLATİLİTEYE ADAPTİF ÇARPAN ---
        # VIX yüksekse çarpanı kıs (gürültüden korun), VIX düşükse çarpanı aç (trend takibi)
        dynamic_mult = np.clip(60 / (vix_now + 1e-9), 1.5, 4.0)
        
        # Hacimli mi? (Hacim Z > 1.2 sigma şartı - Sadece kurumsal hacim)
        if v_z > 1.2:
            momentum_impact = roc_1h * dynamic_mult
        else:
            # Hacim yoksa fiyat hareketini gürültü say, etkiyi %90 azalt
            momentum_impact = roc_1h * 0.1
            
        total_score = round((m_env * 0.5) + (momentum_impact * 0.5), 2)
        
        # Eşikler Genişletildi (Kararsız bölgeyi %50 artırdık)
        signal = "AL" if total_score > 0.45 else ("SAT" if total_score < -0.45 else "NOTR")
        current_results[name] = total_score

        if name in old_scores:
            old_val = old_scores[name]
            diff = total_score - old_val
            
            # Teyitli Sinyal Dönüşü
            old_sig = "AL" if old_val > 0.45 else ("SAT" if old_val < -0.45 else "NOTR")
            if old_sig != signal and signal != "NOTR":
                # Sadece hacim ve yön uyumluysa alarm at
                if v_z > 1.5:
                    emoji = "🛡️🚀" if signal == "AL" else "🛡️📉"
                    send_telegram(f"{emoji} *KURUMSAL TEYİTLİ DÖNÜŞ: {name}*\nYön: `{old_sig}` -> *{signal}*\nSkor: `{total_score}` (Hacim Z: {v_z:.1f})")
                    alert_triggered = True

    if not is_headless:
        st.title("🏛️ ALPHA SENTINEL V23 - INSTITUTIONAL")
        cols = st.columns(4)
        for i, (name, sc) in enumerate(current_results.items()):
            sig = "AL" if sc > 0.45 else ("SAT" if sc < -0.45 else "NOTR")
            with cols[i]: st.metric(name, sc, sig)

    pd.DataFrame.from_dict(current_results, orient='index', columns=['score']).to_csv(STATE_FILE)

except Exception as e:
    if is_headless: print(f"Hata: {e}")
    else: st.error(f"Hata: {e}")
