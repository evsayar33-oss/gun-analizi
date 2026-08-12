import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests

# 1. AYARLAR VE TASARIM
st.set_page_config(page_title="Macro Matrix V3 PRO", layout="wide")
st.markdown("<style>.main { background-color: #0d1117; color: white; }</style>", unsafe_allow_html=True)

# SECRETS KONTROLÜ
fred_api_key = st.secrets.get("FRED_API_KEY", None)

# 2. VERİ MOTORU (GELİŞMİŞ EŞLEŞTİRME)
@st.cache_data(ttl=3600)
def get_macro_data(api_key):
    # Yahoo Verileri
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX'}
    df_y = yf.download(list(syms.values()), period="1y", interval="1d")['Close'].ffill()
    df_y = df_y.rename(columns={v: k for k, v in syms.items()})
    
    # FRED Verileri
    fred_series = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
    df_f = pd.DataFrame(index=df_y.index)
    
    if api_key:
        try:
            for name, s_id in fred_series.items():
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json"
                r = requests.get(url).json()
                if 'observations' in r:
                    obs = pd.DataFrame(r['observations'])[['date', 'value']]
                    obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                    obs['date'] = pd.to_datetime(obs['date'])
                    obs = obs.set_index('date')
                    # Veriyi Yahoo indexine göre hizala ve boşlukları doldur
                    df_f[name] = obs['value'].reindex(df_y.index, method='ffill')
        except:
            pass
            
    return df_y, df_f.ffill()

def z_rolling(series):
    # 126 günlük akan pencere (Rolling)
    return (series - series.rolling(126).mean()) / series.rolling(126).std()

# 3. HESAPLAMA VE UI
try:
    df_y, df_f = get_macro_data(fred_api_key)
    
    # ÇAPA HESAPLAMALARI
    breakeven = df_f['T10YIE'].iloc[-1] if 'T10YIE' in df_f.columns else 2.1
    reel_faiz_raw = df_y['TNX'] - breakeven
    
    # Rolling Z-Scores
    z_rf = z_rolling(reel_faiz_raw)
    z_dxy = z_rolling(df_y['DXY'])
    z_comp = (z_rf + z_dxy) / 2
    
    # Likidite ve Spread (FRED yoksa fallback)
    z_liq = z_rolling(df_f['WALCL']) if 'WALCL' in df_f.columns and not df_f['WALCL'].empty else -z_rolling(df_y['TNX'])
    z_spr = z_rolling(df_f['SPREAD']) if 'SPREAD' in df_f.columns and not df_f['SPREAD'].empty else z_rolling(df_y['VIX'])

    # OOS HIT-RATE (60 Günlük Pencere)
    past_sig = -z_comp.shift(5)
    fwd_ret = df_y['SPX'].pct_change(5)
    hits = (np.sign(past_sig) == np.sign(fwd_ret)).tail(60)
    hr = hits.mean() * 100

    # UI ROZETİ
    st.title("🏛️ MACRO MATRIX TERMINAL V3")
    h_c = "#00ff00" if hr >= 60 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div style="border:2px solid {h_c}; padding:10px; border-radius:10px; text-align:center; background:rgba(0,0,0,0.2); margin-bottom:20px;"><b style="color:{h_c}; font-size:24px;">%{hr:.1f} OOS HIT-RATE</b></div>', unsafe_allow_html=True)

    # ÜST GÖSTERGELER (Hata Giderilmiş Formatlama)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("10Y Reel", f"%{reel_faiz_raw.iloc[-1]:.2f}", f"{z_rf.iloc[-1]:.2f}z")
    m2.metric("DXY", f"{df_y['DXY'].iloc[-1]:.2f}", f"{z_dxy.iloc[-1]:.2f}z")
    
    # Likidite Değeri (Trilyon Dolar Formatı)
    liq_val = df_f['WALCL'].iloc[-1] / 1000000 if 'WALCL' in df_f.columns else 0
    m3.metric("Likidite (WALCL)", f"{liq_val:.2f}T" if liq_val > 0 else "N/A", f"{z_liq.iloc[-1]:.2f}z")
    
    spr_val = df_f['SPREAD'].iloc[-1] if 'SPREAD' in df_f.columns else 0
    m4.metric("Kredi Spread", f"%{spr_val:.2f}" if spr_val > 0 else "N/A", f"{z_spr.iloc[-1]:.2f}z")

    st.divider()

    # VARLIK KARTLARI
    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    for i, (name, signs) in enumerate(assets.items()):
        # Nihai Makro Skor
        m_skor = (0.4 * z_comp.iloc[-1] * signs[0]) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2])
        
        # Canlı İvme (Futures Verisiyle)
        df_h = yf.download('ES=F' if name=='SPX' else 'NQ=F' if name=='NDX' else ('GC=F' if name=='XAU' else 'SI=F'), period="2d", interval="1h")['Close']
        v_4h = ((df_h.iloc[-1] / df_h.iloc[-5]) - 1) * 100
        
        p_clr = "#00ff00" if v_4h > 0 else "#ff4b4b"
        sig = "AL" if m_skor > 0.4 else "SAT" if m_skor < -0.4 else "NOTR"
        
        with cols[i%2]:
            st.markdown(f"""
            <div style="border:1px solid #333; padding:15px; border-radius:10px; margin-bottom:10px; background:#161b22;">
                <div style="display:flex; justify-content:space-between;">
                    <b>{name}</b> <b style="color:{p_clr}">%{v_4h:.2f}</b>
                </div>
                <div style="font-size:12px; margin:5px 0;">
                    Makro Skor: {m_skor:.2f} | Yön: {'BOĞA' if m_skor>0 else 'AYI'}
                </div>
                <div style="background:#0d1117; padding:8px; border-radius:5px; border-left:4px solid {p_clr};">
                    <b>{sig}</b> - {'TREND' if np.sign(m_skor)==np.sign(v_4h) else 'DIVERJANZ'}
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
