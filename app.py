import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# 1. SAYFA YAPISI
st.set_page_config(page_title="Macro Matrix Final", layout="wide")
st.title("🏛️ MACRO YÖN MATRİSİ")

# 2. VERİ MOTORU (CANLI VE DOĞRU)
@st.cache_data(ttl=600)
def fetch_live_data():
    # FRED yerine canlı borsa sembolleri
    symbols = {
        'SPX': '^GSPC', 'NDX': '^NDX', 'XAU': 'GC=F', 'XAG': 'SI=F',
        'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'VIX': '^VIX', 'VXN': '^VXN', 'HYG': 'HYG'
    }
    with st.spinner('Piyasa verileri canlı çekiliyor, lütfen 5-10 saniye bekleyin...'):
        df = yf.download(list(symbols.values()), period="1y", interval="1d")['Close']
        df = df.rename(columns={v: k for k, v in symbols.items()})
        df = df.ffill().dropna()
        # PLAN: Reel Faiz ve Kredi Spread hesapla
        df['REEL_FAIZ'] = df['TNX'] - 2.1
        df['SPREAD'] = 100 - df['HYG']
    return df

# 3. DETERMINISTIK HESAPLAMA (PLANIN KALBİ)
def run_matrix_engine(df):
    def z(s): return (s - s.rolling(126).mean()) / s.rolling(126).std()
    
    z_rf = z(df['REEL_FAIZ'])
    z_dxy = z(df['DXY'])
    z_comp = (z_rf + z_dxy) / 2
    z_liq = -z(df['TNX'])
    z_spr = z(df['SPREAD'])

    assets = {
        'SPX': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'NDX': {'vol': 'VXN', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]},
        'XAU': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 1]}, 
        'XAG': {'vol': 'VIX', 'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1]}
    }
    
    results = {}
    for name, cfg in assets.items():
        # PLAN: Sönümlenmiş (Damped) Dinamik IC Hesapla
        ret_5d = df[name].pct_change(5).shift(-5)
        factors = [z_comp, z_liq, z_spr]
        ic_weights = []
        for i, f in enumerate(factors):
            ic = f.rolling(126).corr(ret_5d).iloc[-1]
            if np.isnan(ic): ic = 0.05
            ic_m = np.clip(1 + (ic * cfg['signs'][i]), 0.4, 1.6)
            ic_weights.append(cfg['base'][i] * ic_m)
        
        # PLAN: Ağırlık Sınırlandırma (%15-%60)
        w = np.array(ic_weights) / sum(ic_weights)
        w = np.clip(w, 0.15, 0.60)
        w = w / sum(w)
        
        # PLAN: Altın Kredi Spread Rejimi (Özel Koşul)
        spr_sign = cfg['signs'][2]
        if name == 'XAU':
            vix_z = z(df['VIX']).iloc[-1]
            if vix_z > 2.0: spr_sign = -1 # Akut Likidasyon Modu
        
        # PLAN: Ham Makro Skor
        m_skor = (w[0]*z_comp.iloc[-1]*cfg['signs'][0]) + (w[1]*z_liq.iloc[-1]*cfg['signs'][1]) + (w[2]*z_spr.iloc[-1]*spr_sign)
        
        # PLAN: Momentum Gatekeeper (Override)
        daily_ret = df[name].pct_change().iloc[-1]
        v_z = z(df[cfg['vol']]).iloc[-1]
        m_adj = 0.0
        gate = "NÖTR"
        if daily_ret > 0.005 and v_z < 0.2:
            gate, m_adj = "BOĞA_MOMENTUM", 0.70
        elif daily_ret < -0.004 or v_z > 1.2:
            gate, m_adj = "AYI_MOMENTUM", -0.70
            
        results[name] = {'final': m_skor + m_adj, 'ret': daily_ret, 'gate': gate, 'w': w}
    return results

# 4. ARAYÜZ
try:
    data = fetch_live_data()
    if not data.empty:
        analysis = run_matrix_engine(data)
        
        # Üst Özet
        c1, c2, c3 = st.columns(3)
        c1.metric("VIX (Korku)", f"{data['VIX'].iloc[-1]:.2f}")
        c2.metric("DXY (Dolar)", f"{data['DXY'].iloc[-1]:.2f}")
        c3.metric("10Y Reel Faiz", f"%{data['REEL_FAIZ'].iloc[-1]:.2f}")
        
        st.markdown("---")
        
        # Varlık Kartları
        cols = st.columns(2)
        for i, (name, v) in enumerate(analysis.items()):
            with cols[i % 2]:
                sc = v['final']
                sig = "AL" if sc > 0.45 else "SAT" if sc < -0.45 else "NOTR"
                clr = "#00ff00" if sig == "AL" else "#ff4b4b" if sig == "SAT" else "#888888"
                
                st.markdown(f"""
                <div style="border:1px solid #333; padding:15px; border-radius:10px; margin-bottom:10px;">
                    <h2 style="margin:0;">{name}: <span style="color:{clr};">{sig}</span></h2>
                    <p style="margin:5px 0;">Skor: <b>{sc:.2f}</b> | Günlük: %{v['ret']*100:.2f}</p>
                    <p style="margin:0;"><small>Gatekeeper: {v['gate']} | Ağırlıklar: {int(v['w'][0]*100)}/{int(v['w'][1]*100)}/{int(v['w'][2]*100)}</small></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Veri çekilemedi. Sayfayı yenileyin.")
except Exception as e:
    st.error(f"Sistem Hatası: {e}")
