import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME ---
st_autorefresh(interval=3600 * 1000, key="macro_core_v14")
st.set_page_config(page_title="Macro Core Terminal V14", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 22px; border-radius: 15px; margin-bottom: 20px; border-left: 10px solid; background: #0f121a; }
    .status-AL { border-left-color: #00ff00; background: linear-gradient(90deg, rgba(0,255,0,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-SAT { border-left-color: #ff4b4b; background: linear-gradient(90deg, rgba(255,75,75,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-NOTR { border-left-color: #8b949e; }
    .hit-rate-badge { border: 2px solid; padding: 12px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU ---
@st.cache_data(ttl=300)
def fetch_core_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    df_y_raw = yf.download(list(syms.values()), period="5y", interval="1d")
    df_y = df_y_raw['Close'].ffill().rename(columns={v: k for k, v in syms.items()})
    df_v = df_y_raw['Volume'].ffill().rename(columns={v: k for k, v in syms.items()})
    
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
            elif name == 'WALCL': df_f['WALCL'] = 7100000
    
    return df_y, df_v, df_h, df_f.ffill().bfill()

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=20).mean()) / (s.rolling(win, min_periods=20).std() + 1e-9)

# --- 2. HESAPLAMA ÇEKİRDEĞİ ---
try:
    df_y, df_v, df_h, df_f = fetch_core_data(FRED_API_KEY)
    master = pd.concat([df_y, df_f], axis=1).ffill()
    
    # Makro Faktörler (Z-Skor Bazlı Omurga)
    z_rf = z_score(master['TNX'] - master['T10YIE'])
    z_dxy = z_score(master['DXY'])
    z_liq = z_score(master['WALCL'] - master.get('TGA', 0) - master.get('RRP', 0))
    z_spr = z_score(master['SPREAD'])

    # OOS Hit-Rate (% success)
    sig_vec = ((0.5 * z_liq) + (0.3 * -(z_rf + z_dxy)/2) + (0.2 * -z_spr)).values
    ret_vec = master['SPX'].pct_change(1).shift(-1).values
    mask = ~np.isnan(sig_vec) & ~np.isnan(ret_vec)
    hr = float((np.sign(sig_vec[mask]) == np.sign(ret_vec[mask]))[-126:].mean() * 100)

    st.title("🏛️ MACRO CORE DETERMINISTIC ENGINE")
    h_c = "#00ff00" if hr >= 52 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};">MAKRO GÜVEN (OOS): %{hr:.1f}</div>', unsafe_allow_html=True)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # 1. MAKRO SKOR (Ağırlık %80 - Asla Değişmez Temel)
        # Likidite(%40) + FaizDolar(%40) + Spread(%20)
        m_skor = (0.40 * z_liq.iloc[-1] + 
                  0.40 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 
                  0.20 * z_spr.iloc[-1] * signs[2])
        
        # 2. MOMENTUM KONTROL (Ağırlık %20 - Sadece Onaylayıcı)
        roc_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        v_now = df_v[name].iloc[-1]
        v_z = (v_now - df_v[name].tail(20).mean()) / (df_v[name].tail(20).std() + 1e-9)
        
        # Hacimli mi? (Hacim Z > -0.5 ise fiyat hareketini ciddiye al)
        mom_impact = (roc_4h / 5) if v_z > -0.5 else 0

        # NİHAİ COMPOSITE SKOR
        total_score = (m_skor * 0.8) + (mom_impact * 0.2)
        
        # SİNYAL KARAR (Net ve Kararlı)
        signal = "AL" if total_score > 0.15 else ("SAT" if total_score < -0.15 else "NOTR")
        
        # Diverjanz Tespiti (Makro ve Fiyat zıt yönlüyse)
        divergence = "⚠️ UYUŞMAZLIK" if np.sign(m_skor) != np.sign(roc_4h) and abs(roc_4h) > 0.1 else "✅ UYUMLU"
            
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{signal}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:32px;">{name}</b>
                    <b style="font-size:26px;">{signal}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:24px;">Skor: <b>{total_score:.2f}</b></p>
                <div style="font-size:13px; color:#aaa; line-height:1.6;">
                    🌍 Makro Temel: {m_skor:.2f} | 🚀 Momentum Etkisi: {mom_impact:.2f}<br>
                    📊 Durum: {divergence} | 4s İvme: %{roc_4h:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # MAKRO PANO
    st.sidebar.markdown("### 📊 MAKRO ÇAPALAR")
    st.sidebar.write(f"Net Likidite Z: {z_liq.iloc[-1]:.2f}z")
    st.sidebar.write(f"Reel Faiz Z: {z_rf.iloc[-1]:.2f}z")
    st.sidebar.write(f"Kredi Spread Z: {z_spr.iloc[-1]:.2f}z")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
