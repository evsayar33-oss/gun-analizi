import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# 1. OTOMATİK YENİLEME (HER 1 SAAT)
st_autorefresh(interval=3600 * 1000, key="macro_sentinel_timer")

st.set_page_config(page_title="Macro Matrix Pro Terminal", layout="wide")

# CSS: Gelişmiş Kart Tasarımı ve Renkler
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .stMetric { background-color: #161b22; padding: 10px; border-radius: 5px; }
    .card {
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        border-left: 6px solid;
    }
    .trend { border-left-color: #00ff00; background-color: rgba(0, 255, 0, 0.05); }
    .diverjanz { border-left-color: #ffcc00; background-color: rgba(255, 204, 0, 0.05); }
    .contrend { border-left-color: #ff4b4b; background-color: rgba(255, 75, 75, 0.05); }
    </style>
    """, unsafe_allow_html=True)

# SECRETS
fred_api_key = st.secrets.get("FRED_API_KEY", None)

# 2. VERİ MOTORU
@st.cache_data(ttl=300)
def get_full_engine_data(api_key):
    # Yahoo Verileri
    y_syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'DBC':'DBC'}
    df_y = yf.download(list(y_syms.values()), period="3y", interval="1d")['Close'].ffill()
    df_y = df_y.rename(columns={v: k for k, v in y_syms.items()})
    
    # Saatlik Veri (İvme için)
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="5d", interval="1h")['Close'].ffill()
    df_h = df_h.rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})

    # FRED Verileri
    df_f = pd.DataFrame(index=df_y.index)
    if api_key:
        fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2', 'ECB': 'ECBASSETSW'}
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

def z_roll(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()

# 3. ANA AKIŞ
try:
    df_y, df_h, df_f = get_full_engine_data(fred_api_key)
    reel_faiz = df_y['TNX'] - (df_f['T10YIE'] if 'T10YIE' in df_f.columns else 2.1)
    
    st.title("🏛️ MACRO MATRIX TERMINAL")
    st.caption(f"🕒 Son Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Her saat otomatik yenilenir)")

    # --- DIŞ MAKRO PARADİGMA BEKÇİSİ ---
    st.markdown("### 🛡️ DIŞ MAKRO PARADİGMA BEKÇİSİ")
    anomalies = []
    
    corr_xau_rr = df_y['XAU'].pct_change().rolling(60).corr(reel_faiz.pct_change()).iloc[-1]
    if corr_xau_rr > 0.20:
        anomalies.append(f"🚨 **PARADİGMA ALARMI:** Altın - Reel Faiz İlişkisi Pozitif Koptu ({corr_xau_rr:.2f})!")

    if 'DBC' in df_y.columns:
        corr_dxy_comm = df_y['DXY'].pct_change().rolling(30).corr(df_y['DBC'].pct_change()).iloc[-1]
        if corr_dxy_comm > 0.30:
            anomalies.append(f"⚠️ **MOMENTUM ALARMI:** DXY ve Emtialar Eşzamanlı Yükseliyor ({corr_dxy_comm:.2f})!")

    if 'WALCL' in df_f.columns and 'ECB' in df_f.columns:
        fed_ratio = (df_f['WALCL'].iloc[-1] / (df_f['WALCL'].iloc[-1] + df_f['ECB'].iloc[-1])) * 100
        if fed_ratio < 35:
            anomalies.append(f"⚠️ **KÜRESEL LİKİDİTE ALARMI:** Fed Dominansı Zayıfladı (%{fed_ratio:.1f})!")

    if not anomalies:
        st.success("🟢 **KÜRESEL PARADİGMA DENGEDE:** Makro omurga ilişkileri tarihsel normlar dahilinde.")
    else:
        for alert in anomalies:
            if "🚨" in alert: st.error(alert)
            else: st.warning(alert)

    # --- HIT RATE ROZETİ ---
    z_rf, z_dxy = z_roll(reel_faiz), z_roll(df_y['DXY'])
    z_liq = z_roll(df_f['WALCL']) if 'WALCL' in df_f.columns else -z_roll(df_y['TNX'])
    z_spr = z_roll(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_roll(df_y['VIX'])
    
    total_sig = (0.4 * -(z_rf + z_dxy)/2) + (0.3 * z_liq) + (0.3 * -z_spr)
    hr = float((np.sign(total_sig.shift(5)) == np.sign(df_y['SPX'].pct_change(5))).tail(126).mean() * 100)
    
    h_c = "#00ff00" if hr >= 60 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div style="border:2px solid {h_c}; padding:10px; border-radius:10px; text-align:center; background:rgba(0,0,0,0.1); margin: 15px 0;"><b style="color:{h_c}; font-size:24px;">%{hr:.1f} OOS HIT-RATE</b></div>', unsafe_allow_html=True)

    # GÖSTERGELER
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("10Y Reel", f"%{reel_faiz.iloc[-1]:.2f}", f"{z_rf.iloc[-1]:.2f}z")
    m2.metric("DXY", f"{df_y['DXY'].iloc[-1]:.2f}", f"{z_dxy.iloc[-1]:.2f}z")
    l_v = float(df_f['WALCL'].iloc[-1]/1000000) if 'WALCL' in df_f.columns else 0
    m3.metric("Likidite", f"{l_v:.2f}T", f"{z_liq.iloc[-1]:.2f}z")
    s_v = float(df_f['SPREAD'].iloc[-1]) if 'SPREAD' in df_f.columns else 0
    m4.metric("Spread", f"%{s_v:.2f}", f"{z_spr.iloc[-1]:.2f}z")

    st.divider()

    # VARLIK KARTLARI
    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    for i, (name, signs) in enumerate(assets.items()):
        # Makro Skor Hesaplama
        m_skor = float((0.4 * -(z_rf + z_dxy).iloc[-1]/2 * 1) + (0.3 * z_liq.iloc[-1] * signs[1]) + (0.3 * z_spr.iloc[-1] * signs[2]))
        
        # Son 4 Saatlik İvme
        v_4h = float(((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100)
        
        # Trend/Diverjanz Analizi
        is_divergent = np.sign(m_skor) != np.sign(v_4h) and abs(v_4h) > 0.05
        
        if is_divergent:
            card_class = "diverjanz"
            status_label = "⚠️ DIVERJANZ"
            s_clr = "#ffcc00"
        elif np.sign(m_skor) == np.sign(v_4h):
            card_class = "trend"
            status_label = "✅ TREND"
            s_clr = "#00ff00"
        else:
            card_class = "contrend"
            status_label = "❌ ZAYIF"
            s_clr = "#ff4b4b"
            
        sig = "AL" if m_skor > 0.15 else "SAT" if m_skor < -0.15 else "NOTR"
        p_clr = "#00ff00" if v_4h > 0 else "#ff4b4b"
        
        with cols[i%2]:
            st.markdown(f"""
            <div class="card {card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:20px;">{name}</b>
                    <b style="color:{p_clr}; font-size:18px;">%{v_4h:.2f}</b>
                </div>
                <p style="margin:5px 0; font-size:14px; color:#ccc;">Makro: {m_skor:.2f} | Yön: {'BOĞA' if m_skor>0 else 'AYI'}</p>
                <div style="background:rgba(0,0,0,0.3); padding:8px; border-radius:5px; margin-top:5px;">
                    <b style="color:{s_clr}; font-size:16px;">{sig} - {status_label}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
