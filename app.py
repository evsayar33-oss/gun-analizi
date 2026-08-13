import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME (HER 1 SAAT) ---
st_autorefresh(interval=3600 * 1000, key="macro_oos_zvol_timer")

st.set_page_config(page_title="Macro Force Pro Terminal", layout="wide")

# CSS: Gelişmiş Kart Tasarımı
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card { padding: 18px; border-radius: 12px; margin-bottom: 15px; border-left: 8px solid; }
    .trend { border-left-color: #00ff00; background-color: rgba(0, 255, 0, 0.05); }
    .diverjanz { border-left-color: #ffcc00; background-color: rgba(255, 204, 0, 0.05); }
    .zayif { border-left-color: #ff4b4b; background-color: rgba(255, 75, 75, 0.05); }
    .hit-rate-badge { border: 2px solid; padding: 10px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# SECRETS
fred_api_key = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU ---
@st.cache_data(ttl=300)
def get_pro_engine_data(api_key):
    # Yahoo Verileri
    y_syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX'}
    df_y = yf.download(list(y_syms.values()), period="3y", interval="1d")['Close'].ffill()
    df_y = df_y.rename(columns={v: k for k, v in y_syms.items()})
    
    # Saatlik veride Close ve Volume çekimi
    df_h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")
    df_h = df_h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    df_v = df_h_raw['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

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
    return df_y, df_h, df_v, df_f.ffill()

def z_roll(s, win=126): return (s - s.rolling(win).mean()) / s.rolling(win).std()

# --- 2. ANA AKIŞ ---
try:
    df_y, df_h, df_v, df_f = get_pro_engine_data(fred_api_key)
    reel_faiz = df_y['TNX'] - (df_f['T10YIE'] if 'T10YIE' in df_f.columns else 2.1)
    
    st.title("🏛️ MACRO FORCE TERMINAL PRO")
    st.caption(f"🕒 Son Güncelleme: {datetime.now().strftime('%H:%M:%S')} | Hacim Z-Skoru Kalibrasyonu Aktif")

    # --- OOS HIT-RATE HESAPLAMA ---
    z_rf, z_dxy = z_roll(reel_faiz), z_roll(df_y['DXY'])
    z_liq = z_roll(df_f['WALCL']) if 'WALCL' in df_f.columns else -z_roll(df_y['TNX'])
    z_spr = z_roll(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_roll(df_y['VIX'])
    
    # 5 Günlük Tahmin Başarısı (S&P 500 bazlı)
    total_sig_series = (0.4 * -(z_rf + z_dxy)/2) + (0.3 * z_liq) + (0.3 * -z_spr)
    prediction = total_sig_series.shift(5)
    outcome = df_y['SPX'].pct_change(5)
    hr = float((np.sign(prediction) == np.sign(outcome)).tail(126).mean() * 100)

    # OOS ROZETİ
    h_c = "#00ff00" if hr >= 60 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f"""
        <div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};">
            <small>OOS HIT-RATE GÜVEN ROZETİ</small><br>
            <b style="font-size:24px;">%{hr:.1f}</b>
        </div>
    """, unsafe_allow_html=True)

    # --- MAKRO METRIKLER ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reel Faiz", f"%{reel_faiz.iloc[-1]:.2f}", f"{z_rf.iloc[-1]:.2f}z")
    m2.metric("DXY", f"{df_y['DXY'].iloc[-1]:.2f}", f"{z_dxy.iloc[-1]:.2f}z")
    l_v = float(df_f['WALCL'].iloc[-1]/1000000) if 'WALCL' in df_f.columns else 0
    m3.metric("Likidite", f"{l_v:.2f}T", f"{z_liq.iloc[-1]:.2f}z")
    s_v = float(df_f['SPREAD'].iloc[-1]) if 'SPREAD' in df_f.columns else 0
    m4.metric("Kredi Spread", f"%{s_v:.2f}", f"{z_spr.iloc[-1]:.2f}z")

    st.divider()

    # VARLIK KARTLARI
    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # Makro Skor
        m_skor = float((0.4 * -(z_rf + z_dxy).iloc[-1]/2) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2]))
        
        # --- VOL Z-SCORE & FORCE MOMENTUM ---
        p_now, p_prev = df_h[name].iloc[-1], df_h[name].iloc[-5]
        v_now = df_v[name].iloc[-1]
        
        # Hacim Z-Skoru (Son 24 saatlik pencerede)
        vol_z = (v_now - df_v[name].rolling(24).mean().iloc[-1]) / df_v[name].rolling(24).std().iloc[-1]
        
        roc = ((p_now / p_prev) - 1) * 100
        
        # Karar Mekanizması (Hacim Z-Skoru > 0 ise Onaylı)
        if roc > 0.05 and vol_z > 0:
            mom_direction, mom_status = 1, "GÜÇLÜ BOĞA"
        elif roc < -0.05 and vol_z > 0:
            mom_direction, mom_status = -1, "GÜÇLÜ AYI"
        else:
            mom_direction, mom_status = 0, "DÜŞÜK HACİM/BELİRSİZ"

        # Diverjanz / Trend Mantığı
        is_divergent = (np.sign(m_skor) != mom_direction) and (mom_direction != 0)
        
        if is_divergent:
            card_class, status_label, s_clr = "diverjanz", "⚠️ DIVERJANZ", "#ffcc00"
        elif (np.sign(m_skor) == mom_direction) and (mom_direction != 0):
            card_class, status_label, s_clr = "trend", "✅ TREND", "#00ff00"
        else:
            card_class, status_label, s_clr = "zayif", "❌ ZAYIF", "#ff4b4b"
            
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
                    Makro: {m_skor:.2f} ({'BOĞA' if m_skor>0 else 'AYI'}) | Hacim Z: {vol_z:.2f}
                </p>
                <div style="background:rgba(0,0,0,0.3); padding:8px; border-radius:5px; margin-top:5px;">
                    <b style="color:{s_clr}; font-size:16px;">{sig} - {status_label}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
