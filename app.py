import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME (HER 1 SAAT) ---
st_autorefresh(interval=3600 * 1000, key="macro_force_timer")

st.set_page_config(page_title="Macro Force Terminal", layout="wide")

# CSS: Gelişmiş Kart Tasarımı
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card {
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
        border-left: 8px solid;
    }
    .trend { border-left-color: #00ff00; background-color: rgba(0, 255, 0, 0.05); }
    .diverjanz { border-left-color: #ffcc00; background-color: rgba(255, 204, 0, 0.05); }
    .zayif { border-left-color: #ff4b4b; background-color: rgba(255, 75, 75, 0.05); }
    </style>
    """, unsafe_allow_html=True)

# SECRETS
fred_api_key = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU ---
@st.cache_data(ttl=300)
def get_engine_data(api_key):
    # Yahoo Verileri (3 Yıl Günlük + 7 Gün Saatlik)
    y_syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX'}
    df_y = yf.download(list(y_syms.values()), period="3y", interval="1d")['Close'].ffill()
    df_y = df_y.rename(columns={v: k for k, v in y_syms.items()})
    
    # Saatlik veride Hacim (Volume) çekimi kritik
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")
    # Multiindex sütunları düzeltme
    df_h_close = df_h['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})
    df_h_vol = df_h['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})

    # FRED Verileri
    df_f = pd.DataFrame(index=df_y.index)
    if api_key:
        fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
        for name, s_id in fred_ids.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=5"
                r = requests.get(url, timeout=5).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
            except: pass
    return df_y, df_h_close.ffill(), df_h_vol.ffill(), df_f.ffill()

def z_roll(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()

# --- 2. ANA AKIŞ ---
try:
    df_y, df_h, df_v, df_f = get_engine_data(fred_api_key)
    reel_faiz = df_y['TNX'] - (df_f['T10YIE'] if 'T10YIE' in df_f.columns else 2.1)
    
    st.title("🏛️ MACRO FORCE TERMINAL")
    st.caption(f"🕒 Son Güncelleme: {datetime.now().strftime('%H:%M:%S')} | Force Momentum (Price x Volume) Aktif")

    # --- Z-SCORES ---
    z_rf, z_dxy = z_roll(reel_faiz), z_roll(df_y['DXY'])
    z_liq = z_roll(df_f['WALCL']) if 'WALCL' in df_f.columns else -z_roll(df_y['TNX'])
    z_spr = z_roll(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_roll(df_y['VIX'])
    
    def get_m_skor(signs):
        return float((0.4 * -(z_rf + z_dxy).iloc[-1]/2) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2]))

    st.divider()

    # VARLIK KARTLARI
    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        m_skor = get_m_skor(signs)
        
        # --- MECHANICAL MOMENTUM (NO INDICATORS) ---
        p_now = df_h[name].iloc[-1]
        p_prev = df_h[name].iloc[-5] # 4 saat önce
        v_now = df_v[name].iloc[-1]
        v_avg = df_v[name].rolling(24).mean().iloc[-1] # 24 saatlik hacim ortalaması
        
        # 1. Price Velocity (%)
        roc = ((p_now / p_prev) - 1) * 100
        
        # 2. Force Index (Price Change x Volume)
        force_momentum = (p_now - p_prev) * v_now
        
        # 3. Volume Confirmation
        vol_confirmed = v_now > (v_avg * 0.9) # Hacim ortalamanın %90'ından büyükse onaylı
        
        # Momentum Yönü Belirleme (Tamamen Mekanik)
        # Fiyat hızı > %0.10 ve Hacim Onaylıysa BOĞA
        if roc > 0.10 and vol_confirmed:
            mom_direction = 1
            mom_status = "GÜÇLÜ BOĞA"
        elif roc < -0.10 and vol_confirmed:
            mom_direction = -1
            mom_status = "GÜÇLÜ AYI"
        else:
            mom_direction = 0
            mom_status = "HACİMSİZ/ZAYIF"

        # Trend/Diverjanz Mantığı
        is_divergent = (np.sign(m_skor) != mom_direction) and (mom_direction != 0)
        
        if is_divergent:
            card_class = "diverjanz"
            status_label = "⚠️ DIVERJANZ"
            s_clr = "#ffcc00"
        elif (np.sign(m_skor) == mom_direction) and (mom_direction != 0):
            card_class = "trend"
            status_label = "✅ TREND"
            s_clr = "#00ff00"
        else:
            card_class = "zayif"
            status_label = "❌ ZAYIF"
            s_clr = "#ff4b4b"
            
        sig = "AL" if m_skor > 0.15 else ("SAT" if m_skor < -0.15 else "NOTR")
        p_clr = "#00ff00" if roc > 0 else "#ff4b4b"
        
        with cols[i%2]:
            st.markdown(f"""
            <div class="card {card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:20px;">{name}</b>
                    <b style="color:{p_clr}; font-size:18px;">%{roc:.2f}</b>
                </div>
                <p style="margin:5px 0; font-size:14px; color:#ccc;">
                    Makro: {m_skor:.2f} ({'BOĞA' if m_skor>0 else 'AYI'}) | İvme: {mom_status}
                </p>
                <div style="background:rgba(0,0,0,0.3); padding:8px; border-radius:5px; margin-top:5px;">
                    <b style="color:{s_clr}; font-size:16px;">{sig} - {status_label}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
