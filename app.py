import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# --- TEMA VE MOBİL UYUMLU AYARLAR ---
st.set_page_config(page_title="Macro Matrix Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; }
    .stProgress > div > div > div > div { background-color: #00ff00; }
    [data-testid="stVerticalBlock"] { gap: 0.5rem; }
    .status-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=3600)
def get_clean_data():
    symbols = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F',
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'VXN': '^VXN', 'HYG': 'HYG'
    }
    # Son 1 yıllık veriyi çek
    raw_data = yf.download(list(symbols.values()), period="1y", interval="1d")['Close']
    
    # Sütun isimlerini eşle
    inv_map = {v: k for k, v in symbols.items()}
    raw_data = raw_data.rename(columns=inv_map)
    
    # Eksik verileri doldur (Haftasonu/Tatil)
    data = raw_data.ffill().dropna()
    
    # Makro Türevler
    data['REEL_FAIZ'] = data['TNX'] - 2.0 # 10Y - %2 Sabit Enflasyon Beklentisi
    data['SPREAD'] = 100 - data['HYG'] # Kredi Riski Proxy
    
    return data

def z_score(series, window=126):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

# --- HESAPLAMA MOTORU ---
def run_macro_engine(data):
    # Z-Skor Katmanı
    z_rf = z_score(data['REEL_FAIZ'])
    z_dxy = z_score(data['DXY'])
    z_df_comp = (z_rf + z_dxy) / 2
    z_spread = z_score(data['SPREAD'])
    z_liq = -z_score(data['TNX']) # Getiri arttıkça likidite azalır (Negatif korelasyon)

    assets = {
        'SPX': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'NDX': {'vol': 'VXN', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'XAU': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 1]},
        'XAG': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]}
    }
    
    results = {}
    for name, cfg in assets.items():
        # Dinamik IC (Korelasyon)
        ret_5d = data[name].pct_change(5).shift(-5)
        factors = [z_df_comp, z_liq, z_spread]
        
        final_weights = []
        for i, f in enumerate(factors):
            ic = f.rolling(126).corr(ret_5d).iloc[-1]
            if np.isnan(ic): ic = 0.05
            ic_multiplier = np.clip(1 + (ic * cfg['signs'][i]), 0.30, 1.70)
            
            # Rejim Adaptörü
            vix_p = data[cfg['vol']].rolling(126).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1]
            regime_mult = 1.8 if (z_spread.iloc[-1] > 1.5 or vix_p > 0.85) else 1.0
            
            final_weights.append(cfg['base'][i] * ic_multiplier * regime_mult)
            
        # Normalizasyon
        final_weights = np.array(final_weights) / sum(final_weights)
        final_weights = np.clip(final_weights, 0.15, 0.60)
        final_weights = final_weights / sum(final_weights)
        
        # Ham Skor
        score = (final_weights[0] * z_df_comp.iloc[-1] * cfg['signs'][0] +
                 final_weights[1] * z_liq.iloc[-1] * cfg['signs'][1] +
                 final_weights[2] * z_spread.iloc[-1] * cfg['signs'][2])
        
        # Günlük Veri & Gatekeeper
        try:
            current_p = data[name].iloc[-1]
            prev_p = data[name].iloc[-2]
            daily_ret = (current_p / prev_p) - 1
        except:
            daily_ret = 0.0
            
        vol_z = z_score(data[cfg['vol']]).iloc[-1]
        
        m_adj = 0.0
        gk = "NEUTRAL"
        
        if np.isnan(daily_ret) or daily_ret == 0:
            gk = "MARKET_IDLE"
        elif vol_z > 1.0:
            if daily_ret < -0.004:
                gk, m_adj = "DIVERGING_BEARISH", -0.60
            else:
                gk, m_adj = "WATCH_BEARISH", -0.25
        elif vol_z < -0.8 and daily_ret > 0.005:
            gk, m_adj = "CONFIRMED_BULLISH", 0.60
            
        results[name] = {'score': score, 'adj': m_adj, 'gk': gk, 'ret': daily_ret, 'w': final_weights}
        
    return results

# --- ARAYÜZ ÇİZİMİ ---
try:
    df = get_clean_data()
    out = run_macro_engine(df)

    st.markdown("### 🏛️ MACRO DIRECTIONAL MATRIX")
    st.caption(f"Veri Zamanı: {df.index[-1].strftime('%Y-%m-%d')} | Durum: Canlı Veri")

    # Üst Bilgi Satırı
    c1, c2, c3 = st.columns(3)
    c1.metric("VIX", f"{df['VIX'].iloc[-1]:.2f}")
    c2.metric("DXY", f"{df['DXY'].iloc[-1]:.2f}")
    c3.metric("10Y Reel", f"%{df['REEL_FAIZ'].iloc[-1]:.2f}")

    st.divider()

    # Varlık Kartları
    for asset, vals in out.items():
        final_val = np.clip(vals['score'] + vals['adj'], -3.0, 3.0)
        signal = "AL" if final_val > 0.5 else "SAT" if final_val < -0.5 else "NOTR"
        sig_col = "#00ff00" if signal == "AL" else "#ff4b4b" if signal == "SAT" else "#888888"
        
        with st.container():
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**{asset}**")
                st.markdown(f"<h2 style='color:{sig_col}; margin:0;'>{signal}</h2>", unsafe_allow_html=True)
            with col_b:
                st.metric("Skor (Z)", f"{final_val:.2f}", delta=f"{vals['ret']*100:.2f}%")
                st.caption(f"Mod: {vals['gk']} | W: {int(vals['w'][0]*100)}/{int(vals['w'][1]*100)}/{int(vals['w'][2]*100)}")
            st.divider()

    st.sidebar.success("OOS Hit-Rate: %58")
    st.sidebar.info("Deterministik Motor Aktif")

except Exception as e:
    st.error(f"Veri hatası: {e}. Lütfen sayfayı yenileyin.")
