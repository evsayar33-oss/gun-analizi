import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# 1. AYARLAR
st.set_page_config(page_title="Macro Matrix Pro", layout="wide")
st.markdown("<style>.main { background-color: #0d1117; color: white; }</style>", unsafe_allow_html=True)

# 2. VERİ MOTORU
@st.cache_data(ttl=120)
def get_data():
    # Vadeli semboller: ES=F (S&P), NQ=F (Nasdaq), GC=F (Altın), SI=F (Gümüş)
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    df_d = yf.download(list(syms.values()), period="1y", interval="1d")['Close'].ffill()
    df_d = df_d.rename(columns={v: k for k, v in syms.items()})
    df_h = yf.download(list(syms.values()), period="5d", interval="1h")['Close'].ffill()
    df_h = df_h.rename(columns={v: k for k, v in syms.items()})
    return df_d, df_h

# 3. OOS HIT-RATE HESAPLAMA
def calc_hit_rate(df):
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    # 5 gün önceki makro sinyal vs 5 gün sonraki getiri
    z_c = (z(df['TNX'] - 2.1) + z(df['DXY'])) / 2
    past_sig = -z_c.shift(5)
    fwd_ret = df['SPX'].pct_change(5)
    hits = (np.sign(past_sig) == np.sign(fwd_ret)).tail(60)
    return hits.mean() * 100

# 4. MOTORU ÇALIŞTIR
try:
    df_d, df_h = get_data()
    hr = calc_hit_rate(df_d)
    
    st.title("🏛️ MACRO MATRIX TERMINAL")

    # --- OOS HIT-RATE ROZETİ ---
    if hr >= 60:
        h_c, h_t = "#00ff00", "YEŞİL (ABD/Dolar Mantığı Kusursuz)"
    elif hr >= 45:
        h_c, h_t = "#ffff00", "SARI (Verim Düşüyor, Rejim Değişiyor)"
    else:
        h_c, h_t = "#ff4b4b", "KIRMIZI (PARADİGMA KAYMASI! Güncelle)"

    st.markdown(f"""
        <div style="border: 2px solid {h_c}; padding:10px; border-radius:10px; text-align:center; background:rgba(0,0,0,0.2);">
            <small style="color:{h_c}">OOS HIT-RATE GÜVEN ROZETİ</small><br>
            <b style="font-size:24px; color:{h_c}">%{hr:.1f}</b><br>
            <span style="color:{h_c}; font-size:12px;">{h_t}</span>
        </div>
    """, unsafe_allow_html=True)

    # Üst Göstergeler
    st.write("---")
    def z_val(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    z_comp = (z_val(df_d['TNX'] - 2.1) + z_val(df_d['DXY'])) / 2
    z_liq = -z_val(df_d['TNX'])
    z_spr = z_val(100 - df_d['HYG'])

    c1, c2, c3 = st.columns(3)
    c1.metric("10Y Reel", f"%{df_d['TNX'].iloc[-1]-2.1:.2f}")
    c2.metric("DXY", f"{df_d['DXY'].iloc[-1]:.2f}")
    c3.metric("VIX", f"{df_d['VIX'].iloc[-1]:.2f}")

    # Varlıklar
    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    
    for name, signs in assets.items():
        # Dinamik Altın Rejimi
        spr_sign = signs[2]
        if name == 'XAU' and z_val(df_d['VIX']).iloc[-1] > 2.0:
            spr_sign = -1
            
        m_skor = (0.4 * z_comp.iloc[-1] * signs[0]) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * spr_sign)
        v_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        
        p_clr = "#00ff00" if v_4h > 0 else "#ff4b4b"
        m_clr = "green" if m_skor > 0 else "red"
        
        # Durum Mesajı
        if v_4h > 0.15:
            msg = "GÜÇLÜ TREND" if m_skor > 0 else "MAKROYA DİRENEN YÜKSELİŞ"
        elif v_4h < -0.15:
            msg = "GÜÇLÜ SATIŞ" if m_skor < 0 else "BOĞA DÜZELTMESİ"
        else:
            msg = "KARARSIZ / BEKLE"

        st.markdown(f"""
            <div style="border:1px solid #333; padding:15px; border-radius:10px; margin-bottom:10px; background:#161b22;">
                <div style="display:flex; justify-content:space-between;">
                    <b>{name}</b> <b style="color:{p_clr}">%{v_4h:.2f}</b>
                </div>
                <div style="font-size:12px; margin:5px 0;">
                    Fiyat: {df_h[name].iloc[-1]:.2f} | Makro: <span style="color:{m_clr}">{'BOĞA' if m_skor>0 else 'AYI'}</span>
                </div>
                <div style="background:#0d1117; padding:8px; border-radius:5px; border-left:4px solid {p_clr}; font-size:14px;">
                    {msg}
                </div>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Hata: {e}")
