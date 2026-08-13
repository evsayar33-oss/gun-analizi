import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. SİSTEM AYARLARI ---
st_autorefresh(interval=3600 * 1000, key="macro_momentum_fix_v12")
st.set_page_config(page_title="Macro Force Alpha V12", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 25px; border-radius: 15px; margin-bottom: 20px; border-top: 10px solid; background: #0f121a; }
    .status-GÜÇLÜ_AL { border-top-color: #00ff00; background: linear-gradient(180deg, rgba(0,255,0,0.15) 0%, rgba(15,18,26,1) 100%); }
    .status-BOĞA_MOMENTUM { border-top-color: #76ff03; background: linear-gradient(180deg, rgba(118,255,3,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-DİKKATLİ_İZLE { border-top-color: #ff9100; background: linear-gradient(180deg, rgba(255,145,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-GÜÇLÜ_SAT { border-top-color: #ff1744; background: linear-gradient(180deg, rgba(255,23,68,0.15) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 15px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU ---
@st.cache_data(ttl=300)
def fetch_alpha_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    raw_y = yf.download(list(syms.values()), period="5y", interval="1d")['Close'].ffill().bfill()
    df_y = raw_y.rename(columns={v: k for k, v in syms.items()})
    
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")
    df_h = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    df_v = h_raw['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

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
    
    return df_y, df_h, df_v, df_f.ffill().bfill()

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=1).mean()) / (s.rolling(win, min_periods=1).std() + 1e-9)

# --- 2. ANALİZ MOTORU ---
try:
    df_y, df_h, df_v, df_f = fetch_alpha_data(FRED_API_KEY)
    master = pd.concat([df_y, df_f], axis=1).ffill()
    
    # Makro Faktörler
    reel_faiz = master['TNX'] - master['T10YIE']
    z_rf, z_dxy, z_spr = z_score(reel_faiz), z_score(master['DXY']), z_score(master['SPREAD'])
    z_liq = z_score(master['WALCL'] - master.get('TGA', 0) - master.get('RRP', 0))

    # OOS Hit-Rate (Gelişmiş)
    sig_vec = ((0.5 * z_liq) + (0.25 * -(z_rf + z_dxy)/2) + (0.25 * -z_spr)).values
    ret_vec = master['SPX'].pct_change(1).shift(-1).values # 1 günlük ileri getiri
    mask = ~np.isnan(sig_vec) & ~np.isnan(ret_vec)
    hr = float((np.sign(sig_vec[mask]) == np.sign(ret_vec[mask]))[-126:].mean() * 100)

    # --- UI ---
    st.title("🏛️ MACRO FORCE ALPHA TERMINAL")
    
    c_time = datetime.now().strftime('%H:%M:%S')
    h_c = "#00ff00" if hr >= 52 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};">📡 SON VERİ: {c_time} | MODEL GÜVENİ (OOS): %{hr:.1f}</div>', unsafe_allow_html=True)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # 1. MAKRO ÇEVRE (Ağırlık %50)
        m_env = (0.35 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.45 * z_liq.iloc[-1] + 0.20 * z_spr.iloc[-1] * signs[2])
        
        # 2. VALÜASYON (Ağırlık %20) - Aşırı ısınma koruması
        z_self = z_score(master[name], win=252).iloc[-1] * -1 
        
        # 3. AGRESİF MOMENTUM (Ağırlık %30) - İVME ÖNCELİKLİ
        v_now = df_v[name].iloc[-1]
        v_std = df_v[name].tail(24).std() + 1e-9
        v_z = (v_now - df_v[name].tail(24).mean()) / v_std
        
        roc_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        # Momentum Bonus: Hacim eşiğini -1.0z'e düşürdüm (Hemen her yükselişi kabul etmesi için)
        mom_power = (roc_4h / 2.5) if v_z > -1.0 else 0 

        # NİHAİ SKOR HESAPLAMA
        total_score = (m_env * 0.5) + (z_self * 0.2) + (mom_power * 0.3)
        
        # SİNYAL KARAR (Nötr Yasak)
        if total_score > 0.4: status = "GÜÇLÜ_AL"
        elif total_score > 0: status = "BOĞA_MOMENTUM"
        elif total_score > -0.4: status = "DİKKATLİ_İZLE"
        else: status = "GÜÇLÜ_SAT"
            
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{status}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:32px;">{name}</b>
                    <b style="font-size:20px;">{status.replace("_", " ")}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:24px;">Skor: <b>{total_score:.2f}</b></p>
                <div style="font-size:13px; color:#aaa; line-height:1.6;">
                    🌍 Makro: {m_env:.2f} | 💎 Değerleme Drag: {z_self:.2f}<br>
                    📊 Hacim Gücü: {v_z:.2f}z | 🚀 4s İvme: %{roc_4h:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Kritik Sistem Hatası: {e}")
