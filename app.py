import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME ---
st_autorefresh(interval=1800 * 1000, key="macro_meltup_v15")
st.set_page_config(page_title="Macro Alpha Sentinel V15", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 22px; border-radius: 15px; margin-bottom: 20px; border-left: 12px solid; background: #0f121a; }
    .status-AL { border-left-color: #00ff00; background: linear-gradient(90deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-SAT { border-left-color: #ff4b4b; background: linear-gradient(90deg, rgba(255,75,75,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-MOMENTUM_BOĞA { border-left-color: #76ff03; background: linear-gradient(90deg, rgba(118,255,3,0.15) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 12px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

@st.cache_data(ttl=300)
def fetch_meltup_data(api_key):
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
            elif name == 'WALCL': df_f['WALCL'] = 7150000
    
    return df_y, df_v, df_h, df_f.ffill().bfill()

def z_score(s, win=126): return (s - s.rolling(win, min_periods=10).mean()) / (s.rolling(win, min_periods=10).std() + 1e-9)

# --- ANALİZ ÇEKİRDEĞİ ---
try:
    df_y, df_v, df_h, df_f = fetch_meltup_data(FRED_API_KEY)
    
    # Likidite Momenti (Hızlandırılmış)
    net_liq = df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)
    z_liq_level = z_score(net_liq, 126)
    z_liq_accel = z_score(net_liq.pct_change(5), 20) # 5 günlük hızlanma
    
    z_rf = z_score(df_y['TNX'] - df_f['T10YIE'])
    z_dxy = z_score(df_y['DXY'])
    z_spr = z_score(df_f['SPREAD'])

    st.title("🏛️ ALPHA SENTINEL - RALLY EDITION")
    st.caption(f"🕒 Son Güncelleme: {datetime.now().strftime('%H:%M:%S')} | Mod: Momentum & Liquidity Dominance")

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # 1. MAKRO SKOR (%40) - Likidite İvmesi Eklendi
        m_env = (0.50 * z_liq_accel.iloc[-1] + 
                 0.30 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 
                 0.20 * z_spr.iloc[-1] * signs[2])
        
        # 2. AGRESİF MOMENTUM (%60) - Fiyat Uçuyorsa Her Şeyi Override Et
        roc_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        v_now = df_v[name].iloc[-1]
        v_z = (v_now - df_v[name].tail(20).mean()) / (df_v[name].tail(20).std() + 1e-9)
        
        # Melt-up Koşulu: Eğer fiyat %0.10 üzerindeyse momentumu skora 2x vur
        mom_impact = (roc_4h * 2) if roc_4h > 0.05 else (roc_4h)

        # NİHAİ SKOR
        total_score = (m_env * 0.4) + (mom_impact * 0.6)
        
        # SİNYAL KARAR (SADECE AL/SAT)
        if total_score > 0:
            signal = "AL" if total_score > 0.5 else "MOMENTUM_BOĞA"
            status_desc = "GÜÇLÜ TREND" if signal == "AL" else "İVME TAKİBİ"
        else:
            signal = "SAT"
            status_desc = "MAKRO BASKI"
            
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{signal}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:32px;">{name}</b>
                    <b style="font-size:24px;">{signal.replace("_", " ")}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:26px;">Skor: <b>{total_score:.2f}</b></p>
                <div style="font-size:14px; color:#aaa; line-height:1.6;">
                    💧 Likidite İvmesi: {z_liq_accel.iloc[-1]:.2f}z<br>
                    🚀 4s Fiyat Hızı: %{roc_4h:.2f} | 📊 Durum: {status_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # MAKRO PANO (Altın Bilgi)
    st.sidebar.markdown("### 🧬 RALLİNİN YAKITI")
    st.sidebar.info(f"Piyasa şu an yüksek faize rağmen **Likidite İvmesi ({z_liq_accel.iloc[-1]:.2f}z)** ve **Short Squeeze** ile yükseliyor. Sistem momentum ağırlığını %60'a çıkardı.")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
