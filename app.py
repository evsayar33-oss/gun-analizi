import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# --- 1. AYARLAR VE STİL ---
st.set_page_config(page_title="Macro Matrix Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; padding: 10px; border-radius: 10px; border: 1px solid #30363d; }
    [data-testid="stMetricValue"] { font-size: 22px !important; }
    .signal-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 15px;
        background-color: #0d1117;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERİ MOTORU (GELİŞMİŞ) ---
@st.cache_data(ttl=600)
def get_full_data():
    symbols = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F',
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'VXN': '^VXN', 'HYG': 'HYG'
    }
    # Veri çekme ve temizleme
    df = yf.download(list(symbols.values()), period="1y", interval="1d")['Close']
    df = df.rename(columns={v: k for k, v in symbols.items()})
    df = df.ffill().dropna()
    
    # Makro Değişkenler
    df['REEL_FAIZ'] = df['TNX'] - 2.0  # Proxy Reel Faiz
    df['SPREAD'] = 100 - df['HYG']    # Kredi Riski Proxy
    return df

def z_score(series, window=126):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

# --- 3. ANA HESAPLAMA MOTORU ---
def run_full_engine(df):
    # Çapalar: Dolar/Faiz, Likidite, Spread
    z_rf = z_score(df['REEL_FAIZ'])
    z_dxy = z_score(df['DXY'])
    z_df_comp = (z_rf + z_dxy) / 2
    z_liq = -z_score(df['TNX'])
    z_spr = z_score(df['SPREAD'])

    assets = {
        'SPX': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'NDX': {'vol': 'VXN', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'XAU': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 1]}, # XAU Spread Sign +1
        'XAG': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]}
    }
    
    results = {}
    for name, cfg in assets.items():
        # Rolling IC (126 Gün Korelasyonu)
        ret_5d = df[name].pct_change(5).shift(-5)
        factors = [z_df_comp, z_liq, z_spr]
        
        # Dinamik Ağırlık Hesaplama
        d_weights = []
        for i, f in enumerate(factors):
            ic = f.rolling(126).corr(ret_5d).iloc[-1]
            if np.isnan(ic): ic = 0.05
            # Sönümlenmiş IC Çarpanı
            ic_m = np.clip(1 + (ic * cfg['signs'][i]), 0.50, 1.50)
            d_weights.append(cfg['base'][i] * ic_m)
            
        # Normalize Et ve Sınırla (%15 - %60)
        d_weights = np.array(d_weights) / sum(d_weights)
        d_weights = np.clip(d_weights, 0.15, 0.60)
        d_weights = d_weights / sum(d_weights)

        # Ham Makro Skor
        m_skor = (d_weights[0] * z_df_comp.iloc[-1] * cfg['signs'][0] +
                  d_weights[1] * z_liq.iloc[-1] * cfg['signs'][1] +
                  d_weights[2] * z_spr.iloc[-1] * cfg['signs'][2])
        
        # Momentum Gatekeeper (Turbo Etki)
        daily_ret = df[name].pct_change().iloc[-1]
        if np.isnan(daily_ret): daily_ret = 0.0
        
        vol_z = z_score(df[cfg['vol']]).iloc[-1]
        
        m_adj = 0.0
        gate = "NEUTRAL"
        
        # Eşikler ve Turbo Düzeltme
        if daily_ret > 0.005 and vol_z < 0.5:
            gate, m_adj = "CONFIRMED_BULL
