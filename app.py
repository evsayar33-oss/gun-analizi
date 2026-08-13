import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME ---
st_autorefresh(interval=3600 * 1000, key="macro_final_precision_v6")

st.set_page_config(page_title="Macro Precision Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card { padding: 18px; border-radius: 12px; margin-bottom: 15px; border-left: 8px solid; background: #161b22; }
    .trend { border-left-color: #00ff00; background-color: rgba(0, 255, 0, 0.05); }
    .diverjanz { border-left-color: #ffcc00; background-color: rgba(255, 204, 0, 0.05); }
    .zayif { border-left-color: #ff4b4b; background-color: rgba(255, 75, 75, 0.05); }
    .hit-rate-badge { border: 2px solid; padding: 15px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU (SAATLİK HACİM Z-SKORU İÇİN) ---
@st.cache_data(ttl=300)
def get_precision_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    df_y = yf.download(list(syms.values()), period="3y", interval="1d")['Close'].ffill().rename(columns={v: k for k, v in syms.items()})
    
    # Saatlik Hacim ve Fiyat (Son 24 saat için 7 günlük çekiyoruz)
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")
    df_h_close = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    df_h_vol = h_raw['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

    # FRED
    df_f = pd.DataFrame(index=df_y.index)
    if api_key:
        fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
        for name, s_id in fred_ids.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=5"
                r = requests.get(url).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
            except: pass
    return df_y, df_h_close, df_h_vol, df_f.ffill()

def z_roll(s, win=126):
    return (s - s.rolling(win, min_periods=1).mean()) / (s.rolling(win, min_periods=1).std() + 1e-9)

# --- 2. HESAPLAMA ---
try:
    df_y, df_h, df_v, df_f = get_precision_data(FRED_API_KEY)
    
    breakeven = df_f['T10YIE'] if 'T10YIE' in df_f.columns else 2.1
    reel_faiz = df_y['TNX'] - breakeven
    
    z_rf, z_dxy = z_roll(reel_faiz), z_roll(df_y['DXY'])
    z_liq = z_roll(df_f['WALCL']) if 'WALCL' in df_f.columns else -z_roll(df_y['TNX'])
    z_spr = z_roll(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_roll(100 - df_y['HYG'])

    # --- REJİM ADAPTİF HIT-RATE ---
    # Likidite ağırlığını Hit-Rate %50 altındayken artırıyoruz
    sig_series = (0.4 * z_liq) + (0.3 * -(z_rf + z_dxy)/2) + (0.3 * -z_spr)
    prediction = sig_series.shift(2) # 2 günlük gecikme (Makro etki süresi)
    actual_move = df_y['SPX'].pct_change(2)
    hr = float((np.sign(prediction) == np.sign(actual_move)).tail(60).mean() * 100)

    # --- UI ---
    st.title("🏛️ MACRO FORCE PRECISION TERMINAL")
    h_c = "#00ff00" if hr >= 50 else "#ffff00" if hr >= 40 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};"><small>OOS HIT-RATE (2D LAG)</small><br><b style="font-size:32px;">%{hr:.1f}</b></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reel Faiz", f"%{reel_faiz.iloc[-1]:.2f}", f"{z_rf.iloc[-1]:.2f}z")
    c2.metric("DXY", f"{df_y['DXY'].iloc[-1]:.2f}", f"{z_dxy.iloc[-1]:.2f}z")
    l_v = float(df_f['WALCL'].iloc[-1]/1000000) if 'WALCL' in df_f.columns else 0
    c3.metric("Likidite", f"{l_v:.2f}T", f"{z_liq.iloc[-1]:.2f}z")
    c4.metric("Spread Z", f"{z_spr.iloc[-1]:.2f}z")

    st.divider()

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # Makro Skor
        m_skor = float((0.35 * -(z_rf + z_dxy).iloc[-1]/2) + (0.35 * z_liq.iloc[-1] * signs[1]) + (0.30 * z_spr.iloc[-1] * signs[2]))
        
        # --- PRECISISON VOLUME Z-SCORE ---
        # Son 24 saatlik SAATLİK hacim verileri
        last_24h_vol = df_v[name].tail(24)
        v_now = last_24h_vol.iloc[-1]
        vol_z = (v_now - last_24h_vol.mean()) / (last_24h_vol.std() + 1e-9)
        
        # Fiyat ROC (Son 4 Saat)
        roc = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        
        # Karar: Hacim Z-Skoru > 0.5 ise "Hacimli" kabul et
        if abs(roc) > 0.02:
            if vol_z > 0.5: mom_dir, mom_lbl = np.sign(roc), "HACİMLİ"
            else: mom_dir, mom_lbl = 0, "HACİMSİZ"
        else: mom_dir, mom_lbl = 0, "YATAY"

        is_div = (np.sign(m_skor) != mom_dir) and (mom_dir != 0)
        
        if is_div: cls, lbl, clr = "diverjanz", "⚠️ DIVERJANZ", "#ffcc00"
        elif (np.sign(m_skor) == mom_dir) and (mom_dir != 0): cls, lbl, clr = "trend", "✅ TREND", "#00ff00"
        else: cls, lbl, clr = "zayif", "❌ ZAYIF", "#ff4b4b"
            
        sig = "AL" if m_skor > 0.05 else ("SAT" if m_skor < -0.05 else "NOTR")
        p_clr = "#00ff00" if roc > 0 else "#ff4b4b"
        
        with cols[i%2]:
            st.markdown(f"""
            <div class="card {cls}">
                <div style="display:flex; justify-content:space-between;">
                    <b style="font-size:22px;">{name}</b>
                    <b style="color:{p_clr}; font-size:20px;">%{roc:.2f}</b>
                </div>
                <p style="margin:5px 0; font-size:14px; color:#ccc;">
                    Makro: {m_skor:.2f} | Hacim Z-Skor: {vol_z:.2f} ({mom_lbl})
                </p>
                <div style="background:rgba(0,0,0,0.3); padding:10px; border-radius:8px; margin-top:8px;">
                    <b style="color:{clr}; font-size:18px;">{sig} - {lbl}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
