import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# 1. AYARLAR VE TASARIM
st.set_page_config(page_title="Macro Matrix Pro + OOS", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card { border: 1px solid #333; padding: 15px; border-radius: 12px; background: #161b22; margin-bottom: 15px; }
    .hit-rate-badge {
        padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px;
        border: 2px solid;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. VERİ ÇEKME
@st.cache_data(ttl=120)
def get_pro_data():
    syms = {
        'SPX': 'ES=F', 'NDX': 'NQ=F', 'XAU': 'GC=F', 'XAG': 'SI=F', 
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'HYG': 'HYG'
    }
    df_d = yf.download(list(syms.values()), period="1y", interval="1d")['Close'].ffill()
    df_d = df_d.rename(columns={v: k for k, v in syms.items()})
    df_h = yf.download(list(syms.values()), period="5d", interval="1h")['Close'].ffill()
    df_h = df_h.rename(columns={v: k for k, v in syms.items()})
    return df_d, df_h

# 3. OOS HIT-RATE HESAPLAMA (DETERMİNİSTİK)
def calculate_oos_hit_rate(df):
    # Makro sinyali geçmişte ne kadar haklıydı? (Son 60 gün test edilir)
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    
    z_comp = (z(df['TNX'] - 2.1) + z(df['DXY'])) / 2
    # Basitleştirilmiş makro sinyal geçmişi
    # Sign: -1 * z_comp (Dolar/Faiz arttıkça borsa düşer mantığı)
    past_signals = -z_comp.shift(5) # 5 gün önceki sinyal
    fwd_returns = df['SPX'].pct_change(5) # 5 gün sonraki gerçek getiri
    
    # Sinyal ile getirinin yönü aynı mı?
    hits = (np.sign(past_signals) == np.sign(fwd_returns)).tail(60)
    hit_rate = hits.mean() * 100
    return hit_rate

# 4. ANALİZ MOTORU
def run_engine():
    df_d, df_h = get_pro_data()
    hit_rate = calculate_oos_hit_rate(df_d)
    
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    z_comp = (z(df_d['TNX'] - 2.1) + z(df_d['DXY'])) / 2
    z_liq = -z(df_d['TNX'])
    z_spr = z(100 - df_d['HYG'])

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    res = {}
    
    for name, signs in assets.items():
        m_skor = (0.4 * z_comp.iloc[-1] * signs[0]) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2])
        v_4h = (df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1
        
        if v_4h > 0.001:
            pulse, sig = "YUKARI", ("GÜÇLÜ TREND" if m_skor > 0 else "MAKROYA DİRENEN YÜKSELİŞ")
        elif v_4h < -0.001:
            pulse, sig = "AŞAĞI", ("GÜÇLÜ SATIŞ" if m_skor < 0 else "BOĞA DÜZELTMESİ")
        else:
            pulse, sig = "YATAY", "KARARSIZ BÖLGE"

        res[name] = {'m_skor': m_skor, 'v_4h': v_4h*100, 'pulse': pulse, 'sig': sig, 'price': df_h[name].iloc[-1]}
        
    return res, df_d, hit_rate

# 5. UI
try:
    results, df_d, hr = run_engine()
    st.title("🏛️ MACRO MATRIX PRO")

    # --- OOS HIT-RATE ROZETİ ---
    if hr >= 60:
        hr_color, hr_bg, hr_txt = "#00ff00", "rgba(0,255,0,0.1)", "YEŞİL (ABD/Dolar Mantığı Kusursuz Çalışıyor)"
    elif hr >= 45:
        hr_color, hr_bg, hr_txt = "#ffff00", "rgba(255,255,0,0.1)", "SARI (Verim Düşüyor, Piyasa Rejim Değiştiriyor)"
    else:
        hr_color, hr_bg, hr_txt = "#ff4b4b", "rgba(255,75,75,0.1)", "KIRMIZI (PARADİGMA KAYMASI! Çapaları Güncelle)"

    st.markdown(f"""
        <div class="hit-rate-badge" style="background-color:{hr_bg}; border-color:{hr_color}; color:{hr_color};">
            <small>OOS HIT-RATE GÜVEN ROZETİ</small><br>
            <span style="font-size:24px;">%{hr:.1f}</span><br>
            <span>{hr_txt}</span>
        </div>
    """, unsafe_allow_html=True)

    # Üst Bilgi
    m1, m2, m3 = st.columns(3)
    m1.metric("10Y Reel", f"%{df_d['TNX'].iloc[-1]-2.1:.2f}")
    m2.metric("DXY", f"{df_d['DXY'].iloc[-1]:.2f}")
    m3.metric("Kredi Spread", f"{100-df_d['HYG'].iloc[-1]:.2f}")

    st.divider()

    for name, v in results.items():
        p_clr = "#00ff00" if v['v_4h'] > 0 else "#ff4b4b"
        m_clr = "green" if v['m_skor'] > 0 else "red"
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between;">
                <h2 style="margin:0;">{name}</h2>
                <h2 style="margin:0; color:{p_clr};">%{v['v_4h']:.2f}</h2>
