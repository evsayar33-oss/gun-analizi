import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. AGRESİF YENİLEME (HER 2 DAKİKADA BİR) ---
st_autorefresh(interval=120 * 1000, key="macro_flash_alignment_v19")
st.set_page_config(page_title="Macro Flash Terminal V19", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 12px solid; background: #0f121a; }
    .status-AL { border-left-color: #00ff00; background: linear-gradient(90deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-SAT { border-left-color: #ff4b4b; background: linear-gradient(90deg, rgba(255,75,75,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-FLASH_BOĞA { border-left-color: #76ff03; background: linear-gradient(90deg, rgba(118,255,3,0.2) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 12px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU (TAM HİZALAMA) ---
@st.cache_data(ttl=60)
def fetch_synchronized_data(api_key):
    # Semboller
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    
    # 1. Günlük Makro Veri (5 Yıllık)
    df_raw = yf.download(list(syms.values()), period="5y", interval="1d")['Close'].ffill().bfill()
    df_y = df_raw.rename(columns={v: k for k, v in syms.items()})
    
    # 2. Canlı İvme: 15 Dakikalık (Son 2 Gün)
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="2d", interval="15m")['Close'].ffill().bfill()
    df_h = h_raw.rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})

    # 3. FRED Verileri (Dizine Hizalanmış)
    df_f = pd.DataFrame(index=df_y.index)
    fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
    for name, s_id in fred_ids.items():
        success = False
        if api_key:
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=5"
                r = requests.get(url, timeout=5).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
                success = True
            except: pass
        if not success:
            if name == 'SPREAD': df_f['SPREAD'] = (100 - df_y['HYG']).rolling(20).mean()
            elif name == 'T10YIE': df_f['T10YIE'] = 2.1
            elif name == 'WALCL': df_f['WALCL'] = 7150000
            
    return df_y, df_h, df_f.ffill().bfill()

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=5).mean()) / (s.rolling(win, min_periods=5).std() + 1e-9)

# --- 2. ANALİZ VE HİT-RATE ---
try:
    df_y, df_h, df_f = fetch_synchronized_data(FRED_API_KEY)
    
    # Faktörler (Günlük Dizin Üzerinden)
    net_liq = df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)
    z_liq_accel = z_score(net_liq.rolling(20).mean(), 126)
    reel_faiz = df_y['TNX'] - df_f['T10YIE']
    z_rf, z_dxy, z_spr = z_score(reel_faiz), z_score(df_y['DXY']), z_score(df_f['SPREAD'])

    # --- HIT-RATE HESAPLAMA (NUMPY VECTORING - Hata Çözümü) ---
    # Pandas Series objelerini numpy array'e çeviriyoruz ki label hatası oluşmasın
    sig_raw = ((0.5 * z_liq_accel) + (0.25 * -(z_rf + z_dxy)/2) + (0.25 * -z_spr)).values
    ret_raw = df_y['SPX'].pct_change(2).shift(-2).values # 2 günlük ileri getiri
    
    # NaN temizliği ve karşılaştırma
    mask = ~np.isnan(sig_raw) & ~np.isnan(ret_raw)
    hits = (np.sign(sig_raw[mask]) == np.sign(ret_raw[mask]))
    hr = float(hits[-60:].mean() * 100) if len(hits) > 0 else 0.0

    # --- UI ---
    st.title("🏛️ ALPHA SENTINEL V19 - SYNCHRONIZED")
    c_time = datetime.now().strftime('%H:%M:%S')
    h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};"><small>MODEL GÜVENİ (HIZALANMIŞ OOS)</small><br><b style="font-size:32px;">%{hr:.1f}</b><br><span>⚡ CANLI TAKİP: {c_time}</span></div>', unsafe_allow_html=True)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # Makro Puanı (Günlük Son Veri)
        m_env = (0.50 * z_liq_accel.iloc[-1] + 0.30 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.20 * z_spr.iloc[-1] * signs[2])
        
        # Flash Momentum (15 Dakikalık Son Veri)
        # 1 saatlik değişim: Şimdiki bar / 4 bar önceki bar
        roc_flash = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        
        # Skor Birleştirme (Ağırlıklı)
        total_score = (m_env * 0.3) + (roc_flash * 4.5) # İvme etkisi 4.5x
        
        if total_score > 0.3: signal, desc = "AL", "GÜÇLÜ TREND"
        elif total_score > 0: signal, desc = "FLASH_BOĞA", "ANLIK İVME"
        else: signal, desc = "SAT", "BASKI VAR"
            
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{signal}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:28px;">{name}</b>
                    <b style="font-size:22px;">{signal.replace("_", " ")}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:24px;">Skor: <b>{total_score:.2f}</b></p>
                <div style="font-size:13px; color:#aaa; line-height:1.6;">
                    💧 Likidite: {z_liq_accel.iloc[-1]:.2f}z | 💎 Makro: {m_env:.2f}<br>
                    ⚡ **1s Fiyat Hızı (15m-Base): %{roc_flash:.2f}** | {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası (Hizalama): {e}")
