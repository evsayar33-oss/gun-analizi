import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# --- TEMA VE AYARLAR ---
st.set_page_config(page_title="Macro Matrix Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff00; }
    .stProgress > div > div > div > div { background-color: #00ff00; }
    </style>
    """, unsafe_allow_html=True)

# --- VERİ MOTORU ---
@st.cache_data(ttl=3600)
def get_data():
    # Semboller: SPX, NDX, Gold, Silver, DXY, 10Y Yield, VIX, VXN, HY Corporate Bond (Spread için)
    symbols = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F',
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'VXN': '^VXN', 'HYG': 'HYG'
    }
    data = yf.download(list(symbols.values()), period="1y", interval="1d")['Close'].ffill()
    data.columns = [list(symbols.keys())[list(symbols.values()).index(col)] for col in data.columns]
    
    # Reel Faiz Hesaplama (Basitleştirilmiş: 10Y - %2 Enflasyon Beklentisi)
    data['REEL_FAIZ'] = data['TNX'] - 2.0 
    # Spread Hesaplama (HYG ters korelasyonu üzerinden kredi riski proxy)
    data['SPREAD'] = 100 - data['HYG']
    
    return data

def z_score(series, window=126):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

# --- HESAPLAMA ÇEKİRDEĞİ ---
def process_engine(data):
    # Z-Skorları
    z_rf = z_score(data['REEL_FAIZ'])
    z_dxy = z_score(data['DXY'])
    z_df_comp = (z_rf + z_dxy) / 2
    z_spread = z_score(data['SPREAD'])
    
    # Likidite Proxy (Basitleştirilmiş: DXY ve Tahvil fiyatı üzerinden ters likidite)
    z_liq = -z_score(data['TNX']) 

    assets = {
        'SPX': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'NDX': {'vol': 'VXN', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'XAU': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 0]},
        'XAG': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]}
    }
    
    output = {}
    for name, cfg in assets.items():
        # Rolling IC (Basit Pearson)
        ret_5d = data[name].pct_change(5).shift(-5)
        factors = [z_df_comp, z_liq, z_spread]
        
        final_w = []
        for i, f in enumerate(factors):
            ic = f.rolling(126).corr(ret_5d).iloc[-1]
            if np.isnan(ic): ic = 0.1
            ic_c = np.clip(1 + (ic * cfg['signs'][i]), 0.30, 1.70)
            
            # Rejim Adaptörü
            vix_p = data[cfg['vol']].rolling(126).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1]
            rej_c = 1.8 if (z_spread.iloc[-1] > 1.5 or vix_p > 0.85) else 1.0
            
            final_w.append(cfg['base'][i] * ic_c * rej_c)
            
        # Normalize & Clamp
        final_w = np.array(final_w) / sum(final_w)
        final_w = np.clip(final_w, 0.15, 0.60)
        final_w = final_w / sum(final_w)
        
        # XAU Özel Sign
        spr_sign = cfg['signs'][2]
        if name == 'XAU':
            vix_z = z_score(data['VIX']).iloc[-1]
            spr_sign = -1 if (vix_p > 0.90 and vix_z > 2.0) else 1
            
        # Skor
        ham_skor = (final_w[0] * z_df_comp.iloc[-1] * cfg['signs'][0] +
                    final_w[1] * z_liq.iloc[-1] * cfg['signs'][1] +
                    final_w[2] * z_spread.iloc[-1] * spr_sign)
        
        # Gatekeeper
        daily_ret = (data[name].iloc[-1] / data[name].iloc[-2]) - 1
        vol_z = z_score(data[cfg['vol']]).iloc[-1]
        
        m_adj = 0.0
        gk = "NEUTRAL"
        if vol_z > 1.0:
            if daily_ret < -0.004:
                gk, m_adj = "DIVERGING_BEARISH", -0.60
            else:
                gk, m_adj = "WATCH_BEARISH", -0.25
        elif vol_z < -0.8 and daily_ret > 0.005:
            gk, m_adj = "CONFIRMED_BULLISH", 0.60
            
        output[name] = {'score': ham_skor, 'adj': m_adj, 'gk': gk, 'ret': daily_ret, 'w': final_w}
        
    return output

# --- ARAYÜZ ---
data = get_data()
results = process_engine(data)

st.title("🏛️ MACRO DIRECTIONAL MATRIX")
st.write(f"Son Güncelleme: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

cols = st.columns(4)
for i, (name, res) in enumerate(results.items()):
    with cols[i]:
        f_skor = np.clip(res['score'] + res['adj'], -3.0, 3.0)
        signal = "AL" if f_skor > 0.5 else "SAT" if f_skor < -0.5 else "NOTR"
        color = "#00ff00" if signal == "AL" else "#ff4b4b" if signal == "SAT" else "#ffffff"
        
        st.markdown(f"### {name}")
        st.markdown(f"<h1 style='color:{color};'>{signal}</h1>", unsafe_allow_html=True)
        st.metric("Skor", f"{f_skor:.2f}")
        st.write(f"İvme: `{res['gk']}`")
        st.write(f"Günlük: %{res['ret']*100:.2f}")
        
        # Ağırlıklar
        st.caption(f"DF: %{res['w'][0]*100:.0f} | LIQ: %{res['w'][1]*100:.0f} | SPR: %{res['w'][2]*100:.0f}")
        st.divider()

st.sidebar.header("Makro Durum")
st.sidebar.write(f"VIX: {data['VIX'].iloc[-1]:.2f}")
st.sidebar.write(f"DXY: {data['DXY'].iloc[-1]:.2f}")
st.sidebar.write(f"10Y Reel: %{data['REEL_FAIZ'].iloc[-1]:.2f}")
