import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# 1. SAYFA YAPILANDIRMASI
st.set_page_config(page_title="Macro Pulse Terminal", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .stMetric { border: 1px solid #333; padding: 10px; border-radius: 5px; }
    .macro-box { border-left: 5px solid #58a6ff; padding-left: 15px; margin-bottom: 20px; }
    .pulse-box { border-left: 5px solid #ffaa00; padding-left: 15px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. VERİ MOTORU (GÜNLÜK VE SAATLİK)
@st.cache_data(ttl=300) # 5 dakikada bir yenilenir
def get_all_data():
    symbols = {'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F', 'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'HYG': 'HYG'}
    
    # Makro Veri (Günlük - 1 Yıl)
    df_daily = yf.download(list(symbols.values()), period="1y", interval="1d")['Close'].ffill()
    df_daily = df_daily.rename(columns={v: k for k, v in symbols.items()})
    
    # Canlı İvme (Saatlik - Son 7 Gün)
    df_hourly = yf.download(list(symbols.values()), period="7d", interval="1h")['Close'].ffill()
    df_hourly = df_hourly.rename(columns={v: k for k, v in symbols.items()})
    
    return df_daily, df_hourly

# 3. HESAPLAMA MOTORU
def process_signals(df_d, df_h):
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    
    # Makro Çapalar
    z_rf = (df_d['TNX'] - 2.1).pipe(z)
    z_dxy = z(df_d['DXY'])
    z_liq = -z(df_d['TNX'])
    z_spr = z(100 - df_d['HYG'])
    z_comp = (z_rf + z_dxy) / 2

    assets = {
        'SPX': {'signs': [-1, 1, -1]},
        'NDX': {'signs': [-1, 1, -1]},
        'XAU': {'signs': [-1, 1, 1]},
        'XAG': {'signs': [-1, 1, -1]}
    }
    
    results = {}
    for name, cfg in assets.items():
        # A) MAKRO SKOR (GÜNLÜK)
        m_skor = (0.4 * z_comp.iloc[-1] * cfg['signs'][0] + 
                  0.3 * z_liq.iloc[-1] * cfg['signs'][1] + 
                  0.3 * z_spr.iloc[-1] * cfg['signs'][2])
        
        # B) İNTRADAY IVME (SAATLİK)
        # Son 4 saatlik değişim ortalaması vs Son 24 saat
        short_mom = df_h[name].pct_change(4).iloc[-1] # Son 4 saatlik hız
        velocity = (df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1
        
        # C) PROJEKSİYON (GÜNÜN KALANI)
        # Eğer hız pozitif ve makro skor destekliyorsa = AGGRESSIVE BUY
        # Eğer hız pozitif ama makro skor negatifse = BULL TRAP (TUZAK)
        if velocity > 0.002: # %0.20 üstü ivme
            pulse = "GÜÇLÜ YUKARI"
            if m_skor > 0: signal = "TREND DEVAM"
            else: signal = "AYI PİYASASI TEPKİSİ (DİKKAT)"
        elif velocity < -0.002:
            pulse = "GÜÇLÜ AŞAĞI"
            if m_skor < 0: signal = "SATIŞ BASKISI"
            else: signal = "BOĞA PİYASASI DÜZELTMESİ"
        else:
            pulse = "YATAY/ZAYIF"
            signal = "BEKLE"

        results[name] = {
            'm_skor': m_skor,
            'velocity': velocity * 100,
            'pulse': pulse,
            'signal': signal,
            'last_price': df_h[name].iloc[-1]
        }
    return results

# 4. DASHBOARD UI
st.title("🏛️ DUAL-LAYER DIRECTIONAL MATRIX")
st.write(f"Canlı Analiz: {datetime.now().strftime('%H:%M:%S')}")

try:
    df_d, df_h = get_all_data()
    res = process_signals(df_d, df_h)

    # Üst Göstergeler
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("10Y Reel Faiz", f"%{df_d['TNX'].iloc[-1]-2.1:.2f}")
    c2.metric("Dolar Endeksi", f"{df_d['DXY'].iloc[-1]:.2f}")
    c3.metric("VIX (Oynaklık)", f"{df_d['VIX'].iloc[-1]:.2f}")
    c4.metric("Kredi Riski", f"{100-df_d['HYG'].iloc[-1]:.2f}")

    st.markdown("---")

    for name, v in res.items():
        with st.container():
            col_main, col_pulse = st.columns([2, 1])
            
            with col_main:
                st.markdown(f"### {name} | {v['last_price']:.2f}")
                m_color = "green" if v['m_skor'] > 0 else "red"
                st.write(f"**MAKRO ANA YÖN:** :{m_color}[{ 'BOĞA' if v['m_skor'] > 0 else 'AYI' }] (Skor: {v['m_skor']:.2f})")
                st.info(f"📍 **BUGÜN NE OLUR?** {v['signal']}"
