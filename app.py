import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# --- 1. SİSTEM AYARLARI VE DARK MODE TERMİNAL TASARIMI ---
st.set_page_config(page_title="Macro Matrix Pro Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; font-family: 'Courier New', Courier, monospace; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    [data-testid="stMetricValue"] { font-size: 24px !important; color: #58a6ff; }
    .signal-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
        background-color: #0d1117;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERİ MİMARİSİ (DETERMİNİSTİK KATMAN) ---
@st.cache_data(ttl=600)
def fetch_financial_data():
    # SPX, NDX, Altın, Gümüş, DXY, 10Y Faiz, VIX, VXN, HY Bond (Kredi Spread için)
    tickers = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F',
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'VXN': '^VXN', 'HYG': 'HYG'
    }
    # Verileri çek ve sembolleri eşle
    raw = yf.download(list(tickers.values()), period="1y", interval="1d")['Close']
    df = raw.rename(columns={v: k for k, v in tickers.items()})
    
    # Eksik verileri doldur (Veri bütünlüğü için kritik)
    df = df.ffill().dropna()
    
    # Makro Çapa Hesaplamaları
    df['REEL_FAIZ'] = df['TNX'] - 2.1  # 10Y Nominal - Enflasyon Beklentisi (Proxy)
    df['SPREAD'] = 100 - df['HYG']     # Kredi Riski / Spread Proxy
    return df

def get_z_score(series, window=126):
    """Deterministik Z-Skoru Hesaplama"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std

# --- 3. SÖNÜMLENMİŞ DİNAMİK IC VE REJİM MOTORU ---
def calculate_engine(df):
    # Çapaların Z-Skorlarını Hazırla
    z_rf = get_z_score(df['REEL_FAIZ'])
    z_dxy = get_z_score(df['DXY'])
    z_dolar_faiz = (z_rf + z_dxy) / 2
    z_liq = -get_z_score(df['TNX'])  # Faiz arttıkça likidite azalır (-)
    z_spread = get_z_score(df['SPREAD'])

    assets = {
        'SPX': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'NDX': {'vol': 'VXN', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'XAU': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 1]}, 
        'XAG': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]}
    }
    
    results = {}
    for name, config in assets.items():
        # Rolling IC (126 Günlük Pearson Korelasyonu)
        # Faktörlerin 5 gün sonraki getiri ile ilişkisi
        fwd_ret = df[name].pct_change(5).shift(-5)
        factors = [z_dolar_faiz, z_liq, z_spread]
        
        raw_weights = []
        for i, f in enumerate(factors):
            ic = f.rolling(126).corr(fwd_ret).iloc[-1]
            if np.isnan(ic): ic = 0.0
            
            # Sönümlenmiş IC Çarpanı (Damped IC)
            ic_multiplier = np.clip(1 + (ic * config['signs'][i]), 0.40, 1.60)
            
            # Rejim Adaptörü (VIX veya Spread fırlarsa ağırlığı artır)
            vix_perc = df[config['vol']].rolling(126).rank(pct=True).iloc[-1]
            regime_mult = 1.6 if (z_spread.iloc[-1] > 1.4 or vix_perc > 0.85) else 1.0
            
            raw_weights.append(config['base'][i] * ic_multiplier * regime_mult)
            
        # Ağırlıkları Normalize Et ve Sınırla (%15 - %60)
        w = np.array(raw_weights) / sum(raw_weights)
        w = np.clip(w, 0.15, 0.60)
        w = w / sum(w) # Re-normalize
        
        # Ham Makro Skor (Σ Weight * Z_Score * Sign)
        ham_skor = (w[0] * z_dolar_faiz.iloc[-1] * config['signs'][0] +
                    w[1] * z_liq.iloc[-1] * config['signs'][1] +
                    w[2] * z_spread.iloc[-1] * config['signs'][2])
        
        # --- 4. GÜN İÇİ GATEKEEPER / MOMENTUM OVERRIDE ---
        daily_ret = df[name].pct_change().iloc[-1]
        vol_z = get_z_score(df[config['vol']]).iloc[-1]
