import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# 1. AYARLAR VE TASARIM
st.set_page_config(page_title="Macro Matrix Pro Live", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card { border: 1px solid #333; padding: 15px; border-radius: 12px; background: #161b22; margin-bottom: 15px; }
    .metric-val { font-size: 28px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. 24 SAAT CANLI VERİ (FUTURES SEMBOLLERİ)
@st.cache_data(ttl=120) # 2 dakikada bir tazele
def get_futures_data():
    # Endeksler için vadeli semboller kullanıldı: ES=F (S&P), NQ=F (Nasdaq)
    syms = {
        'SPX': 'ES=F', 'NDX': 'NQ=F', 'XAU': 'GC=F', 'XAG': 'SI=F', 
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'HYG': 'HYG'
    }
    # Makro (Günlük)
    df_d = yf.download(list(syms.values()), period="1y", interval="1d")['Close'].ffill()
    df_d = df_d.rename(columns={v: k for k, v in syms.items()})
    # Canlı İvme (1 Saatlik - 24 saatlik kesintisiz akış için)
    df_h = yf.download(list(syms.values()), period="5d", interval="1h")['Close'].ffill()
    df_h = df_h.rename(columns={v: k for k, v in syms.items()})
    return df_d, df_h

# 3. ANALİZ MOTORU
def run_live_engine():
    df_d, df_h = get_futures_data()
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    
    # Makro (Top)
    z_comp = (z(df_d['TNX'] - 2.1) + z(df_d['DXY'])) / 2
    z_liq = -z(df_d['TNX'])
    z_spr = z(100 - df_d['HYG'])

    assets = {
        'SPX': [-1, 1, -1], 'NDX': [-1, 1, -1],
        'XAU': [-1, 1, 1], 'XAG': [-1, 1, -1]
    }
    
    res = {}
    for name, signs in assets.items():
        # Makro Yön
        m_skor = (0.4 * z_comp.iloc[-1] * signs[0]) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2])
        
        # Son 1 Saat ve Son 4 Saat İvme
        v_1h = (df_h[name].iloc[-1] / df_h[name].iloc[-2]) - 1
        v_4h = (df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1
        
        # Durum Analizi (Sentiment + Macro)
        if v_4h > 0.001: # Güçlü Yükseliş
            pulse = "YUKARI"
            if m_skor > 0: signal = "GÜÇLÜ TREND"
            else: signal = "MAKROYA DİRENEN YÜKSELİŞ (DİKKAT)"
        elif v_4h < -0.001: # Güçlü Düşüş
            pulse = "AŞAĞI"
            if m_skor < 0: signal = "GÜÇLÜ SATIŞ"
            else: signal = "BOĞA PİYASASI DÜZELTMESİ"
        else:
            pulse = "YATAY"
            signal = "KARARSIZ BÖLGE"

        res[name] = {
            'm_skor': m_skor, 'v_1h': v_1h * 100, 'v_4h': v_4h * 100,
            'pulse': pulse, 'signal': signal, 'price': df_h[name].iloc[-1]
        }
    return res, df_d

# 4. DASHBOARD UI
try:
    results, raw_d = run_live_engine()
    st.title("🏛️ 24/7 DUAL-LAYER MATRIX")
    st.caption(f"Son Canlı Veri: {datetime.now().strftime('%H:%M:%S')} (Vadeli Veriler Dahil)")

    # Makro Özet
    m1, m2, m3 = st.columns(3)
    m1.metric("10Y Reel Faiz", f"%{raw_d['TNX'].iloc[-1]-2.1:.2f}")
    m2.metric("Dolar Endeksi", f"{raw_d['DXY'].iloc[-1]:.2f}")
    m3.metric("Kredi Spread", f"{100-raw_d['HYG'].iloc[-1]:.2f}")

    st.divider()

    # Varlıklar
    for name, v in results.items():
        p_clr = "#00ff00" if v['v_4h'] > 0 else "#ff4b4b"
        m_clr = "green" if v['m_skor'] > 0 else "red"
        
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="margin:0;">{name}</h2>
                <div style="text-align:right;">
                    <div class="metric-val" style="color:{p_clr};">%{v['v_4h']:.2f}</div>
                    <small style="color:#888;">SON 4 SAATLİK HIZ</small>
                </div>
            </div>
            <p style="margin:10px 0;">Fiyat: <b>{v['price']:.2f}</b> | Makro Yön: <b style="color:{m_clr};">{'BOĞA' if v['m_skor']>0 else 'AYI'}</b></p>
            <div style="background:#0d1117; padding:12px; border-radius:8px; border-left:5px solid {p_clr};">
                <span style="font-size:13px; color:#888;">GÜNCEL DURUM VE TAVSİYE</span><br>
                <b style="font-size:16px;">{v['signal']}</b> ({v['pulse']})
            </div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Veri Bağlantı Hatası: {e}")
