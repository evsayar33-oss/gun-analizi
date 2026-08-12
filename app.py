import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# 1. TEMEL SAYFA AYARLARI
st.set_page_config(page_title="Macro Matrix Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card { border: 1px solid #30363d; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ MACRO YÖN MATRİSİ")

# 2. VERİ ÇEKME
@st.cache_data(ttl=600)
def get_data():
    tickers = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F',
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'VXN': '^VXN', 'HYG': 'HYG'
    }
    df = yf.download(list(tickers.values()), period="1y", interval="1d")['Close']
    df = df.rename(columns={v: k for k, v in tickers.items()})
    df = df.ffill().dropna()
    df['REEL_FAIZ'] = df['TNX'] - 2.1
    df['SPREAD'] = 100 - df['HYG']
    return df

# 3. ANA HESAPLAMA MOTORU
def run_macro_engine(df):
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    
    z_rf = z(df['REEL_FAIZ'])
    z_dxy = z(df['DXY'])
    z_df_comp = (z_rf + z_dxy) / 2
    z_liq = -z(df['TNX'])
    z_spr = z(df['SPREAD'])

    assets = {
        'SPX': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'NDX': {'vol': 'VXN', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'XAU': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 1]}, 
        'XAG': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]}
    }
    
    results = {}
    for name, cfg in assets.items():
        # Rolling IC ve Ağırlıklandırma
        fwd = df[name].pct_change(5).shift(-5)
        factors = [z_df_comp, z_liq, z_spr]
        weights = []
        for i, f in enumerate(factors):
            ic = f.rolling(126).corr(fwd).iloc[-1]
            if np.isnan(ic): ic = 0.05
            ic_m = np.clip(1 + (ic * cfg['signs'][i]), 0.5, 1.5)
            weights.append(cfg['base'][i] * ic_m)
        
        w = np.array(weights) / sum(weights)
        w = np.clip(w, 0.15, 0.60)
        w = w / sum(w)
        
        # Skor
        m_skor = (w[0] * z_df_comp.iloc[-1] * cfg['signs'][0] +
                  w[1] * z_liq.iloc[-1] * cfg['signs'][1] +
                  w[2] * z_spr.iloc[-1] * cfg['signs'][2])
        
        # Momentum Gatekeeper
        ret = df[name].pct_change().iloc[-1]
        v_z = z(df[cfg['vol']]).iloc[-1]
        m_adj = 0.0
        gk = "NEUTRAL"
        if ret > 0.005 and v_z < 0.2:
            gk, m_adj = "CONFIRMED_BULLISH", 0.75
        elif ret < -0.004 or v_z > 1.2:
            gk
