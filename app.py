import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# 1. AYARLAR VE TASARIM
st.set_page_config(page_title="Dual-Layer Matrix", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card { border: 1px solid #333; padding: 15px; border-radius: 10px; background: #161b22; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# 2. VERİ ÇEKME (GÜNLÜK VE SAATLİK)
@st.cache_data(ttl=300)
def get_dual_data():
    syms = {'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F', 'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'HYG': 'HYG'}
    # Makro Veri
    df_d = yf.download(list(syms.values()), period="1y", interval="1d")['Close'].ffill()
    df_d = df_d.rename(columns={v: k for k, v in syms.items()})
    # Canlı İvme (7 Günlük Saatlik Veri)
    df_h = yf.download(list(syms.values()), period="7d", interval="1h")['Close'].ffill()
    df_h = df_h.rename(columns={v: k for k, v in syms.items()})
    return df_d, df_h

# 3. ANALİZ MOTORU
def analyze():
    df_d, df_h = get_dual_data()
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    
    # Makro Faktörler
    z_comp = (z(df_d['TNX'] - 2.1) + z(df_d['DXY'])) / 2
    z_liq = -z(df_d['TNX'])
    z_spr = z(100 - df_d['HYG'])

    assets = {
        'SPX': [-1, 1, -1], 'NDX': [-1, 1, -1],
        'XAU': [-1, 1, 1], 'XAG': [-1, 1, -1]
    }
    
    res = {}
    for name, signs in assets.items():
        # Makro Hesap (Günlük)
        m_skor = (0.4 * z_comp.iloc[-1] * signs[0]) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2])
        
        # Canlı İvme (Son 4 Saatlik Hız)
        velocity = (df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1
        
        # Karar Mekanizması
        if velocity > 0.0015: # Pozitif İvme
            status = "GÜÇLÜ YUKARI"
            msg = "TREND DEVAM" if m_skor > 0 else "AYI PİYASASI TEPKİSİ"
        elif velocity < -0.0015: # Negatif İvme
            status = "GÜÇLÜ AŞAĞI"
            msg = "SATIŞ BASKISI" if m_skor < 0 else "BOĞA DÜZELTMESİ"
        else:
            status = "ZAYIF/YATAY"
            msg = "BEKLE"

        res[name] = {
            'm_skor': m_skor, 'vel': velocity * 100, 
            'status': status, 'msg': msg, 'price': df_h[name].iloc[-1]
        }
    return res, df_d

# 4. ARAYÜZ
try:
    results, raw_d = analyze()
    st.title("🏛️ DUAL-LAYER MATRIX")
    st.caption(f"Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}")

    # Üst Bar
    c1, c2, c3 = st.columns(3)
    c1.metric("DXY", f"{raw_d['DXY'].iloc[-1]:.2f}")
    c2.metric("10Y Reel", f"%{raw_d['TNX'].iloc[-1]-2.1:.2f}")
    c3.metric("VIX", f"{raw_d['VIX'].iloc[-1]:.2f}")

    st.divider()

    # Varlık Kartları
    for name, v in results.items():
        m_clr = "green" if v['m_skor'] > 0 else "red"
        p_clr = "#00ff00" if v['vel'] > 0 else "#ff4b4b"
        
        st.markdown(f"""
        <div class="card">
            <table style="width:100%">
                <tr>
                    <td><h2 style="margin:0;">{name}</h2></td>
                    <td style="text-align:right;"><h2 style="margin:0; color:{p_clr};">%{v['vel']:.2f}</h2></td>
                </tr>
            </table>
            <p style="margin:5px 0;">Fiyat: <b>{v['price']:.2f}</b> | Makro Yön: <b style="color:{m_clr};">{'BOĞA' if v['m_skor']>0 else 'AYI'}</b></p>
            <div style="background:#0d1117; padding:8px; border-radius:5px; border-left:4px solid {p_clr};">
                <b>📍 DURUM:</b> {v['msg']} ({v['status']})
            </div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Hata: {e}")

# KOD SONU (BU SATIRIN OLDUĞUNDAN EMİN OLUN)
