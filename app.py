import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests

# 1. AYARLAR
st.set_page_config(page_title="Macro Matrix V4 Final", layout="wide")
st.markdown("<style>.main { background-color: #0d1117; color: white; }</style>", unsafe_allow_html=True)

# SECRETS
fred_api_key = st.secrets.get("FRED_API_KEY", None)

# 2. VERİ ÇEKME (STABİLİTE İÇİN 3 YILLIK VERİ)
@st.cache_data(ttl=300)
def get_stabilized_data(api_key):
    y_syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX'}
    # Daha uzun veri seti (3 yıl) Z-Skorlarını stabilize eder
    df_y = yf.download(list(y_syms.values()), period="3y", interval="1d")['Close'].ffill()
    df_y = df_y.rename(columns={v: k for k, v in y_syms.items()})
    
    # Canlı İvme
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="5d", interval="1h")['Close'].ffill()
    df_h = df_h.rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})

    df_f = pd.DataFrame(index=df_y.index)
    if api_key:
        fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
        for name, s_id in fred_ids.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json"
                r = requests.get(url).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                obs = obs.set_index('date')
                df_f[name] = obs['value'].reindex(df_y.index, method='ffill')
            except: pass
    return df_y, df_h, df_f.ffill()

def z_rolling(series):
    # 126 günlük tam pencere
    return (series - series.rolling(window=126).mean()) / series.rolling(window=126).std()

# 3. HESAPLAMA MOTORU
try:
    df_y, df_h, df_f = get_stabilized_data(fred_api_key)
    
    # MAKRO ÇAPALAR (ROLLING)
    breakeven = df_f['T10YIE'] if 'T10YIE' in df_f.columns else 2.1
    reel_faiz = df_y['TNX'] - breakeven
    
    z_rf = z_rolling(reel_faiz)
    z_dxy = z_rolling(df_y['DXY'])
    z_dolar_faiz = (z_rf + z_dxy) / 2
    
    z_liq = z_rolling(df_f['WALCL']) if 'WALCL' in df_f.columns else -z_rolling(df_y['TNX'])
    z_spr = z_rolling(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_rolling(df_y['VIX'])

    # --- NİHAİ MAKRO SKOR (İŞARET KONTROLÜ YAPILDI) ---
    # Kurallar: DolarFaiz(-), Likidite(+), Spread(-)
    def calculate_macro_final(name, is_xau=False):
        s_faiz_dolar = 0.40 * (z_dolar_faiz.iloc[-1] * -1)
        s_liq = 0.30 * (z_liq.iloc[-1] * 1)
        
        # Altın özel spread rejimi
        spr_sign = 1 if (is_xau and z_rolling(df_y['VIX']).iloc[-1] < 2.0) else -1
        s_spr = 0.30 * (z_spr.iloc[-1] * spr_sign)
        
        return float(s_faiz_dolar + s_liq + s_spr)

    # --- OOS HIT-RATE (HATA DÜZELTME) ---
    # Geçmişteki makro skorun başarısını ölç
    total_macro_series = (0.4 * (z_dolar_faiz * -1)) + (0.3 * (z_liq * 1)) + (0.3 * (z_spr * -1))
    prediction = total_macro_series.shift(5) # 5 gün önceki skor
    outcome = df_y['SPX'].pct_change(5)      # 5 gün sonraki getiri
    
    # Hit-Rate hesapla (Son 126 günlük performans)
    compare = (np.sign(prediction) == np.sign(outcome)).tail(126)
    hr = float(compare.mean() * 100)

    # 4. ARAYÜZ
    st.title("🏛️ TERMINAL V4 - FINAL CALIBRATION")
    
    # Hit-Rate Rozeti (Dinamik Renk)
    h_c = "#00ff00" if hr >= 60 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f"""
        <div style="border:2px solid {h_c}; padding:15px; border-radius:12px; text-align:center; background:rgba(0,0,0,0.1); margin-bottom:20px;">
            <small style="color:{h_c}">OOS HIT-RATE (GÜVEN ENDEKSİ)</small><br>
            <b style="font-size:32px; color:{h_c}">%{hr:.1f}</b><br>
            <span style="color:{h_c}">{ 'ABD/DOLAR REJİMİ UYUMLU' if hr >= 60 else 'PİYASA REJİM DEĞİŞTİRİYOR' if hr >= 45 else 'PARADİGMA KAYMASI VAR!' }</span>
        </div>
    """, unsafe_allow_html=True)

    # Çapalar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reel Faiz (z)", f"%{reel_faiz.iloc[-1]:.2f}", f"{z_rf.iloc[-1]:.2f}z")
    m2.metric("DXY (z)", f"{df_y['DXY'].iloc[-1]:.2f}", f"{z_dxy.iloc[-1]:.2f}z")
    l_val = float(df_f['WALCL'].iloc[-1]/1000000) if 'WALCL' in df_f.columns else 0
    m3.metric("Likidite (WALCL)", f"{l_val:.2f}T", f"{z_liq.iloc[-1]:.2f}z")
    s_val = float(df_f['SPREAD'].iloc[-1]) if 'SPREAD' in df_f.columns else 0
    m4.metric("Kredi Spread", f"%{s_val:.2f}", f"{z_spr.iloc[-1]:.2f}z")

    st.divider()

    # Varlıklar
    assets = {'SPX':False, 'NDX':False, 'XAU':True, 'XAG':False}
    cols = st.columns(2)
    for i, (name, is_xau) in enumerate(assets.items()):
        m_skor = calculate_macro_final(name, is_xau)
        v_4h = float(((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100)
        
        sig = "AL" if m_skor > 0.15 else "SAT" if m_skor < -0.15 else "NOTR"
        p_clr = "#00ff00" if v_4h > 0 else "#ff4b4b"
        s_clr = "#00ff00" if sig == "AL" else "#ff4b4b" if sig == "SAT" else "white"

        with cols[i%2]:
            st.markdown(f"""
            <div style="border:1px solid #333; padding:15px; border-radius:10px; margin-bottom:10px; background:#161b22;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:20px;">{name}</b>
                    <b style="color:{p_clr}; font-size:18px;">%{v_4h:.2f}</b>
                </div>
                <p style="margin:5px 0; font-size:14px; color:#aaa;">Makro Skor: {m_skor:.2f} ({'BOĞA' if m_skor>0 else 'AYI'})</p>
                <div style="background:#0d1117; padding:10px; border-radius:5px; border-left:5px solid {s_clr};">
                    <b style="color:{s_clr}; font-size:18px;">{sig}</b> - 
                    <span style="font-size:14px;">{'TREND UYUMLU' if np.sign(m_skor) == np.sign(v_4h) else 'DIVERJANZ'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
