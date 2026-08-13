import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME ---
st_autorefresh(interval=1800 * 1000, key="macro_final_fix_v17")
st.set_page_config(page_title="Macro Alpha Sentinel V17", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 22px; border-radius: 15px; margin-bottom: 20px; border-left: 12px solid; background: #0f121a; }
    .status-AL { border-left-color: #00ff00; background: linear-gradient(90deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-SAT { border-left-color: #ff4b4b; background: linear-gradient(90deg, rgba(255,75,75,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-MOMENTUM_BOĞA { border-left-color: #76ff03; background: linear-gradient(90deg, rgba(118,255,3,0.15) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 15px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU ---
@st.cache_data(ttl=300)
def fetch_v17_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    df_raw = yf.download(list(syms.values()), period="5y", interval="1d")
    df_y = df_raw['Close'].ffill().rename(columns={v: k for k, v in syms.items()})
    df_v = df_raw['Volume'].ffill().rename(columns={v: k for k, v in syms.items()})
    
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

    df_f = pd.DataFrame(index=df_y.index)
    fred_ids = {'WALCL': 'WALCL', 'TGA': 'WTREGEN', 'RRP': 'RRPONTSYD', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
    for name, s_id in fred_ids.items():
        if api_key:
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json"
                r = requests.get(url, timeout=5).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
            except: pass
        if name not in df_f.columns or df_f[name].isna().all():
            if name == 'SPREAD': df_f['SPREAD'] = (100 - df_y['HYG']).rolling(20).mean()
            elif name == 'T10YIE': df_f['T10YIE'] = 2.1
            elif name == 'WALCL': df_f['WALCL'] = 7150000
    
    return df_y, df_v, df_h, df_f.ffill().bfill()

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=10).mean()) / (s.rolling(win, min_periods=10).std() + 1e-9)

# --- 2. HESAPLAMA ---
try:
    df_y, df_v, df_h, df_f = fetch_v17_data(FRED_API_KEY)
    
    # Likidite İvmesi
    net_liq = df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)
    z_liq_accel = z_score(net_liq.rolling(20).mean(), 126)
    
    # REEL FAİZ TANIMLAMASI (Hata burada giderildi)
    reel_faiz = df_y['TNX'] - df_f['T10YIE']
    z_rf = z_score(reel_faiz)
    
    z_dxy = z_score(df_y['DXY'])
    z_spr = z_score(df_f['SPREAD'])

    # OOS HIT-RATE
    sig_series = (0.5 * z_liq_accel) + (0.25 * -(z_rf + z_dxy)/2) + (0.25 * -z_spr)
    prediction = sig_series.shift(2)
    actual_move = df_y['SPX'].pct_change(2)
    mask = ~prediction.isna() & ~actual_move.isna()
    hr = float((np.sign(prediction[mask]) == np.sign(actual_move[mask])).tail(60).mean() * 100)

    # --- UI ---
    st.title("🏛️ ALPHA SENTINEL V17 - FINAL")
    
    c_time = datetime.now().strftime('%H:%M:%S')
    h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};"><small>MODEL GÜVENİ (OOS %)</small><br><b style="font-size:32px;">%{hr:.1f}</b><br><span>🕒 SON GÜNCELLEME: {c_time}</span></div>', unsafe_allow_html=True)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        m_env = (0.50 * z_liq_accel.iloc[-1] + 0.30 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.20 * z_spr.iloc[-1] * signs[2])
        roc_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        mom_impact = (roc_4h * 1.5) if roc_4h > 0 else (roc_4h)
        total_score = (m_env * 0.4) + (mom_impact * 0.6)
        
        if total_score > 0.4: signal, desc = "AL", "GÜÇLÜ TREND"
        elif total_score > 0: signal, desc = "MOMENTUM_BOĞA", "İVME TAKİBİ"
        else: signal, desc = "SAT", "MAKRO BASKI"
            
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{signal}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:28px;">{name}</b>
                    <b style="font-size:22px;">{signal.replace("_", " ")}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:24px;">Skor: <b>{total_score:.2f}</b></p>
                <div style="font-size:14px; color:#aaa; line-height:1.6;">
                    💧 Likidite İvmesi: {z_liq_accel.iloc[-1]:.2f}z<br>
                    🚀 4s Fiyat Hızı: %{roc_4h:.2f} | 📊 Durum: {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # SIDEBAR
    st.sidebar.markdown("### 📊 MAKRO ÇAPALAR")
    st.sidebar.write(f"Net Likidite: ${net_liq.iloc[-1]/1e6:.2f}T")
    st.sidebar.write(f"Likidite Trendi: {z_liq_accel.iloc[-1]:.2f}z")
    st.sidebar.write(f"Reel Faiz: %{reel_faiz.iloc[-1]:.2f}")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
