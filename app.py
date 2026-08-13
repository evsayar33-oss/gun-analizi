import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME VE AYARLAR ---
st_autorefresh(interval=3600 * 1000, key="macro_final_production_v10")
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 25px; border-radius: 15px; margin-bottom: 20px; border-top: 10px solid; background: #0f121a; box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
    .status-GÜÇLÜ_AL { border-top-color: #00ff00; background: linear-gradient(180deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-DİKKATLİ_AL { border-top-color: #76ff03; background: linear-gradient(180deg, rgba(118,255,3,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-DİKKATLİ_SAT { border-top-color: #ff9100; background: linear-gradient(180deg, rgba(255,145,0,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-GÜÇLÜ_SAT { border-top-color: #ff1744; background: linear-gradient(180deg, rgba(255,23,68,0.1) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 12px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU ---
@st.cache_data(ttl=300)
def get_v10_data(api_key):
    # Semboller
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'VXN':'^VXN', 'HYG':'HYG'}
    df_y = yf.download(list(syms.values()), period="4y", interval="1d")['Close'].ffill().rename(columns={v: k for k, v in syms.items()})
    
    # Hacim ve Saatlik İvme
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")
    df_h = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    df_v = h_raw['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

    # FRED (Likidite, Enflasyon, Spread)
    df_f = pd.DataFrame(index=df_y.index)
    if api_key:
        fred_ids = {'WALCL': 'WALCL', 'TGA': 'WTREGEN', 'RRP': 'RRPONTSYD', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
        for name, s_id in fred_ids.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json"
                r = requests.get(url, timeout=5).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
            except: pass
    return df_y, df_h, df_v, df_f.ffill()

def z_roll(s, win=126):
    return (s - s.rolling(win, min_periods=1).mean()) / (s.rolling(win, min_periods=1).std() + 1e-9)

# --- 2. ANALİZ MOTORU ---
try:
    df_y, df_h, df_v, df_f = get_v10_data(FRED_API_KEY)
    
    # Makro Faktörler (Hata Korumalı)
    breakeven = df_f['T10YIE'].iloc[-1] if ('T10YIE' in df_f.columns and not pd.isna(df_f['T10YIE'].iloc[-1])) else 2.1
    reel_faiz = df_y['TNX'] - breakeven
    
    z_rf, z_dxy = z_roll(reel_faiz), z_roll(df_y['DXY'])
    z_liq = z_roll(df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)) if 'WALCL' in df_f.columns else -z_roll(df_y['TNX'])
    z_spr = z_roll(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_roll(100 - df_y['HYG'])

    # --- OOS HIT-RATE (2 GÜNLÜK GECİKMELİ TEST) ---
    # Sinyal: Likidite(+), Faiz/Dolar(-), Spread(-)
    macro_agg = (0.5 * z_liq) + (0.25 * -(z_rf + z_dxy)/2) + (0.25 * -z_spr)
    prediction = macro_agg.shift(2)
    actual = df_y['SPX'].pct_change(2)
    hr = float((np.sign(prediction.dropna()) == np.sign(actual.dropna())).tail(126).mean() * 100)

    # --- UI BAŞLIĞI VE ÜST PANOLAR ---
    st.title("🏛️ MACRO QUANT DIFFERENTIAL")
    
    c_time = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
        <div style="background:#161b22; padding:10px; border-radius:10px; border:1px solid #333; display:flex; justify-content:space-between; margin-bottom:20px; font-size:14px;">
            <span>🕒 SON GÜNCELLEME: <b>{c_time}</b></span>
            <span>📡 VERİ DURUMU: <b>Canlı / Online</b></span>
        </div>
    """, unsafe_allow_html=True)

    h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f"""
        <div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};">
            <small>MODEL GÜVEN ROZETİ (OOS HIT-RATE)</small><br>
            <b style="font-size:32px;">%{hr:.1f}</b>
        </div>
    """, unsafe_allow_html=True)

    # VARLIKLAR
    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # 1. MAKRO ÇEVRE (nan korumalı)
        fwd_ret = df_y[name].pct_change(5).shift(-5)
        m_factors = [(z_rf + z_dxy)/2, z_liq, z_spr]
        weights = []
        for j, f_z in enumerate(m_factors):
            ic = f_z.rolling(126).corr(fwd_ret).iloc[-1]
            if np.isnan(ic): ic = 0.0
            weights.append(0.33 * np.clip(1 + (ic * signs[j]), 0.6, 1.4))
        w = np.array(weights) / sum(weights)
        
        m_env = (w[0] * (z_rf.iloc[-1] + z_dxy.iloc[-1])/2 * -1 + 
                 w[1] * z_liq.iloc[-1] * 1 + 
                 w[2] * z_spr.iloc[-1] * signs[2])
        m_env = np.nan_to_num(m_env)

        # 2. ÖZGÜN DEĞERLEME
        z_self = np.nan_to_num(z_roll(df_y[name], win=200).iloc[-1] * -1)

        # 3. MOMENTUM
        last_v = df_v[name].tail(24)
        v_z = (df_v[name].iloc[-1] - last_v.mean()) / (last_v.std() + 1e-9)
        roc_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        mom_power = (roc_4h / 5) if v_z > -0.3 else 0

        # NİHAİ SKOR
        total_score = (m_env * 0.7) + (z_self * 0.3) + mom_power
        
        if total_score > 0.8: status = "GÜÇLÜ_AL"
        elif total_score > 0: status = "DİKKATLİ_AL"
        elif total_score > -0.8: status = "DİKKATLİ_SAT"
        else: status = "GÜÇLÜ_SAT"
            
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{status}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:32px;">{name}</b>
                    <b style="font-size:22px;">{status.replace("_", " ")}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:18px;">Total Score: <b>{total_score:.2f}</b></p>
                <div style="font-size:13px; color:#aaa; line-height:1.6;">
                    🌍 Makro Çevre: {m_env:.2f} | 💎 Özgün Değerleme: {z_self:.2f}<br>
                    📊 Hacim Gücü: {v_z:.2f}z | 4s İvme: %{roc_4h:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
