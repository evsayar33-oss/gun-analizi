import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests

# 1. AYARLAR VE TASARIM
st.set_page_config(page_title="Macro Matrix V3 - FRED Edition", layout="wide")
st.markdown("<style>.main { background-color: #0d1117; color: white; }</style>", unsafe_allow_html=True)

# SIDEBAR - FRED API KEY GİRİŞİ
st.sidebar.header("⚙️ Veri Ayarları")
fred_api_key = st.sidebar.text_input("FRED API Key Girin:", type="password")
st.sidebar.info("Likidite verisi (WALCL) için FRED Key gereklidir.")

# 2. VERİ MOTORU (YAHOO + FRED)
@st.cache_data(ttl=3600)
def get_macro_data(api_key):
    # Yahoo Sembolleri
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX'}
    df_y = yf.download(list(syms.values()), period="1y", interval="1d")['Close'].ffill()
    df_y = df_y.rename(columns={v: k for k, v in syms.items()})
    
    # FRED Verileri (WALCL, T10YIE, Kredi Spread)
    # API key yoksa boş döner
    fred_data = pd.DataFrame(index=df_y.index)
    if api_key:
        try:
            # WALCL (Likidite), T10YIE (Enflasyon Beklentisi), BAMLH0A0HYM2 (Spread)
            for series in ['WALCL', 'T10YIE', 'BAMLH0A0HYM2']:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={api_key}&file_type=json"
                r = requests.get(url).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                obs = obs.set_index('date')
                fred_data[series] = obs['value']
            fred_data = fred_data.ffill().reindex(df_y.index).ffill()
        except:
            st.sidebar.error("FRED API Hatası! Key'i kontrol edin.")
            
    return df_y, fred_data

# 3. ROLLING Z-SCORE (SON 126 GÜN)
def z_rolling(series):
    return (series - series.rolling(126).mean()) / series.rolling(126).std()

# 4. ANALİZ MOTORU
try:
    df_y, df_f = get_macro_data(fred_api_key)
    
    # REEL FAİZ: Rolling Z-Score (126 Gün)
    # (10Y Nominal - Breakeven) kullanılıyor, yoksa proxy.
    if 'T10YIE' in df_f.columns:
        reel_faiz_raw = df_y['TNX'] - df_f['T10YIE']
    else:
        reel_faiz_raw = df_y['TNX'] - 2.1 # Fallback
    
    z_rf = z_rolling(reel_faiz_raw)
    z_dxy = z_rolling(df_y['DXY'])
    z_comp = (z_rf + z_dxy) / 2
    
    # LİKİDİTE (WALCL) Z-SCORE
    if 'WALCL' in df_f.columns:
        z_liq = z_rolling(df_f['WALCL'])
    else:
        z_liq = -z_rolling(df_y['TNX']) # Fallback (Ters korelasyon)

    # KREDİ SPREAD (BAML) Z-SCORE
    if 'BAMLH0A0HYM2' in df_f.columns:
        z_spr = z_rolling(df_f['BAMLH0A0HYM2'])
    else:
        z_spr = z_rolling(100 - yf.download('HYG', period="1y")['Close'])

    # --- OOS HIT-RATE HESAPLAMA ---
    past_sig = -z_comp.shift(5)
    fwd_ret = df_y['SPX'].pct_change(5)
    hits = (np.sign(past_sig) == np.sign(fwd_ret)).tail(60)
    hr = hits.mean() * 100

    # UI BAŞLIĞI VE ROZET
    st.title("🏛️ MACRO MATRIX TERMINAL V3")
    
    h_c = "#00ff00" if hr >= 60 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f"""
        <div style="border:2px solid {h_c}; padding:10px; border-radius:10px; text-align:center; background:rgba(0,0,0,0.2); margin-bottom:20px;">
            <b style="color:{h_c}; font-size:24px;">%{hr:.1f} OOS HIT-RATE</b><br>
            <span style="color:{h_c};">Dinamik 126g Rolling Z-Skor Aktif</span>
        </div>
    """, unsafe_allow_html=True)

    # GÖSTERGE PANELLERİ
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("10Y Reel (Rolling)", f"%{reel_faiz_raw.iloc[-1]:.2f}", f"{z_rf.iloc[-1]:.2f}z")
    m2.metric("DXY", f"{df_y['DXY'].iloc[-1]:.2f}", f"{z_dxy.iloc[-1]:.2f}z")
    m3.metric("Likidite (WALCL)", f"{df_f['WALCL'].iloc[-1]/1000 if 'WALCL' in df_f else 0:.1f}T", f"{z_liq.iloc[-1]:.2f}z")
    m4.metric("Kredi Spread", f"%{df_f['BAMLH0A0HYM2'].iloc[-1] if 'BAMLH0A0HYM2' in df_f else 0:.2f}", f"{z_spr.iloc[-1]:.2f}z")

    st.divider()

    # VARLIK KARTLARI
    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # Skor: 40% Faiz/Dolar, 30% Likidite, 30% Spread
        m_skor = (0.4 * z_comp.iloc[-1] * signs[0]) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2])
        
        # Canlı İvme (Son 4 Saat)
        df_h = yf.download(yf.Ticker(name).ticker if name not in ['SPX','NDX'] else ('ES=F' if name=='SPX' else 'NQ=F'), period="2d", interval="1h")['Close']
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
                        {sig} - {'TREND DEVAM' if np.sign(m_skor)==np.sign(v_4h) else 'DIVERJANZ / TUZAK'}
                    </div>
                </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
