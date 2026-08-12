import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# 1. TEMEL SAYFA AYARI
st.set_page_config(page_title="Macro Matrix Pro", layout="wide")

st.title("🏛️ MACRO YÖN MATRİSİ")
st.write("Sistem başlatılıyor, lütfen bekleyin...")

# 2. VERİ ÇEKME (DAHA GÜVENLİ VE HIZLI)
@st.cache_data(ttl=600)
def fetch_data():
    tickers = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F',
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'VXN': '^VXN', 'HYG': 'HYG'
    }
    # Verileri çek
    with st.spinner('Piyasa verileri Yahoo Finance üzerinden çekiliyor...'):
        df = yf.download(list(tickers.values()), period="1y", interval="1d")['Close']
        df = df.rename(columns={v: k for k, v in tickers.items()})
        df = df.ffill().dropna()
        
        # Makro Çapalar
        df['REEL_FAIZ'] = df['TNX'] - 2.1
        df['SPREAD'] = 100 - df['HYG']
    return df

# 3. HESAPLAMA FONKSİYONU
def run_analysis(df):
    def z_score(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    
    z_rf = z_score(df['REEL_FAIZ'])
    z_dxy = z_score(df['DXY'])
    z_df_comp = (z_rf + z_dxy) / 2
    z_liq = -z_score(df['TNX'])
    z_spr = z_score(df['SPREAD'])

    assets = {
        'SPX': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'NDX': {'vol': 'VXN', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'XAU': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 1]}, 
        'XAG': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]}
    }
    
    res = {}
    for name, cfg in assets.items():
        daily_ret = df[name].pct_change().iloc[-1]
        vol_z = z_score(df[cfg['vol']]).iloc[-1]
        
        # Dinamik Ağırlıklı Skor
        score = (0.4 * z_df_comp.iloc[-1] * cfg['signs'][0] +
                 0.3 * z_liq.iloc[-1] * cfg['signs'][1] +
                 0.3 * z_spr.iloc[-1] * cfg['signs'][2])
        
        # Gatekeeper
        m_adj = 0.0
        if daily_ret > 0.005 and vol_z < 0.2: m_adj = 0.70
        elif daily_ret < -0.004 or vol_z > 1.2: m_adj = -0.70
        
        res[name] = {'final': score + m_adj, 'ret': daily_ret}
    return res

# 4. UYGULAMA AKIŞI
data = fetch_data()

if not data.empty:
    analysis = run_analysis(data)
    
    # Makro Panel
    st.success(f"Veri Güncelliği: {data.index[-1].date()}")
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX", f"{data['VIX'].iloc[-1]:.2f}")
    c2.metric("DXY", f"{data['DXY'].iloc[-1]:.2f}")
    c3.metric("10Y Reel", f"%{data['REEL_FAIZ'].iloc[-1]:.2f}")

    st.markdown("---")

    # Varlıklar
    cols = st.columns(2)
    for i, (name, vals) in enumerate(analysis
