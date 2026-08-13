import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. AGRESİF YENİLEME (HER 2 DAKİKADA BİR) ---
st_autorefresh(interval=120 * 1000, key="macro_flash_v18")
st.set_page_config(page_title="Macro Flash Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 12px solid; background: #0f121a; }
    .status-AL { border-left-color: #00ff00; background: linear-gradient(90deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-SAT { border-left-color: #ff4b4b; background: linear-gradient(90deg, rgba(255,75,75,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-FLASH_BOĞA { border-left-color: #76ff03; background: linear-gradient(90deg, rgba(118,255,3,0.2) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 12px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. HIZLI VERİ MOTORU ---
@st.cache_data(ttl=60) # Önbellek sadece 60 saniye tutulur
def fetch_flash_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    # Günlük makro veriler
    df_raw = yf.download(list(syms.values()), period="5y", interval="1d")
    df_y = df_raw['Close'].ffill().rename(columns={v: k for k, v in syms.items()})
    
    # ANLIK İVME: 15 Dakikalık Veri (Son 2 gün)
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="2d", interval="15m")
    df_h = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

    # FRED Verileri
    df_f = pd.DataFrame(index=df_y.index)
    fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
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
    
    return df_y, df_h, df_f.ffill().bfill()

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=5).mean()) / (s.rolling(win, min_periods=5).std() + 1e-9)

# --- 2. ANALİZ MOTORU ---
try:
    df_y, df_h, df_f = fetch_flash_data(FRED_API_KEY)
    
    # Likidite İvmesi
    net_liq = df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)
    z_liq_accel = z_score(net_liq.rolling(20).mean(), 126)
    reel_faiz = df_y['TNX'] - df_f['T10YIE']
    z_rf, z_dxy, z_spr = z_score(reel_faiz), z_score(df_y['DXY']), z_score(df_f['SPREAD'])

    # OOS Hit-Rate
    sig_series = (0.5 * z_liq_accel) + (0.25 * -(z_rf + z_dxy)/2) + (0.25 * -z_spr)
    prediction = sig_series.shift(2)
    actual_move = df_y['SPX'].pct_change(2)
    hr = float((np.sign(prediction.dropna()) == np.sign(actual_move.dropna())).tail(60).mean() * 100)

    # --- UI ---
    st.title("🏛️ ALPHA SENTINEL V18 - FLASH")
    c_time = datetime.now().strftime('%H:%M:%S')
    h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};"><small>MODEL GÜVENİ (OOS %)</small><br><b style="font-size:32px;">%{hr:.1f}</b><br><span>⚡ CANLI TAKİP: {c_time}</span></div>', unsafe_allow_html=True)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # Makro Katmanı
        m_env = (0.50 * z_liq_accel.iloc[-1] + 0.30 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.20 * z_spr.iloc[-1] * signs[2])
        
        # ANLIK MOMENTUM (Son 15m bar vs 1 saat önceki 15m bar)
        # 1 saatte 4 adet 15m bar vardır.
        roc_flash = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        
        # Hızlandırma: Fiyat anlık fırlarsa skoru agresifçe yukarı çek
        total_score = (m_env * 0.3) + (roc_flash * 4.0) # Fiyat hızı çarpanı 4x yapıldı
        
        if total_score > 0.3: signal, desc = "AL", "GÜÇLÜ TREND"
        elif total_score > 0: signal, desc = "FLASH_BOĞA", "ANLIK İVME"
        else: signal, desc = "SAT", "BASKI VAR"
            
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
                    💧 Likidite: {z_liq_accel.iloc[-1]:.2f}z<br>
                    ⚡ **1s Fiyat Hızı (Flash): %{roc_flash:.2f}** | {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.sidebar.markdown("### 📡 HIZLI VERİ AYARI")
    st.sidebar.write("Veri Periyodu: 15 Dakika")
    st.sidebar.write("Yenileme: 2 Dakika")
    st.sidebar.info("Flash mode aktif. Fiyat hareketleri makro veriye göre 4 kat daha hızlı tepki verir.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
