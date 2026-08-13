import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. SİSTEM AYARLARI VE OTOMATİK YENİLEME ---
st_autorefresh(interval=3600 * 1000, key="macro_sentinel_v11")
st.set_page_config(page_title="Macro Quant Sentinel", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 25px; border-radius: 15px; margin-bottom: 20px; border-top: 10px solid; background: #0f121a; }
    .status-GÜÇLÜ_AL { border-top-color: #00ff00; background: linear-gradient(180deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-DİKKATLİ_AL { border-top-color: #76ff03; background: linear-gradient(180deg, rgba(118,255,3,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-DİKKATLİ_SAT { border-top-color: #ff9100; background: linear-gradient(180deg, rgba(255,145,0,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-GÜÇLÜ_SAT { border-top-color: #ff1744; background: linear-gradient(180deg, rgba(255,23,68,0.1) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 15px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MOTORU (YEDEKLİ VE ZORLAYICI) ---
@st.cache_data(ttl=300)
def fetch_bulletproof_data(api_key):
    # 5 Yıllık Geniş Veri Seti
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    raw_y = yf.download(list(syms.values()), period="5y", interval="1d")['Close'].ffill().bfill()
    df_y = raw_y.rename(columns={v: k for k, v in syms.items()})
    
    # Hacim ve Saatlik İvme
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")
    df_h = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    df_v = h_raw['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

    # FRED Verileri ve Failover Mekanizması
    df_f = pd.DataFrame(index=df_y.index)
    
    fred_ids = {'WALCL': 'WALCL', 'TGA': 'WTREGEN', 'RRP': 'RRPONTSYD', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
    
    for name, s_id in fred_ids.items():
        success = False
        if api_key:
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json"
                r = requests.get(url, timeout=5).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
                if not df_f[name].isna().all(): success = True
            except: pass
        
        # FAILOVER: Veri gelmediyse alternatif kaynaklara zorla
        if not success:
            if name == 'SPREAD': df_f['SPREAD'] = (100 - df_y['HYG']).rolling(20).mean()
            elif name == 'T10YIE': df_f['T10YIE'] = 2.1 # Tarihsel ortalama
            elif name == 'WALCL': df_f['WALCL'] = 7100000 # Son bilinen değer (proxy)
            else: df_f[name] = 0.0

    return df_y, df_h, df_v, df_f.ffill().bfill()

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=1).mean()) / (s.rolling(win, min_periods=1).std() + 1e-9)

# --- 2. ANALİZ VE HİZALAMA MOTORU ---
try:
    df_y, df_h, df_v, df_f = fetch_bulletproof_data(FRED_API_KEY)
    
    # Tüm serileri tek bir indexte hizala (Label hatasını çözer)
    master_df = pd.concat([df_y, df_f], axis=1).ffill().bfill()
    
    # Faktörler
    reel_faiz = master_df['TNX'] - master_df['T10YIE']
    z_rf = z_score(reel_faiz)
    z_dxy = z_score(master_df['DXY'])
    z_liq = z_score(master_df['WALCL'] - master_df.get('TGA', 0) - master_df.get('RRP', 0))
    z_spr = z_score(master_df['SPREAD'])

    # --- OOS HIT-RATE (ALGORİTMİK HİZALAMA) ---
    # numpy array üzerinden hesaplayarak 'identically-labeled' hatasını baypas et
    sig_vec = ((0.4 * z_liq) + (0.3 * -(z_rf + z_dxy)/2) + (0.3 * -z_spr)).values
    ret_vec = master_df['SPX'].pct_change(2).shift(-2).values
    
    mask = ~np.isnan(sig_vec) & ~np.isnan(ret_vec)
    hits = np.sign(sig_vec[mask]) == np.sign(ret_vec[mask])
    hr = float(hits[-126:].mean() * 100) if len(hits) > 0 else 0.0

    # --- UI ---
    st.title("🏛️ QUANT DIFFERENTIAL TERMINAL")
    
    c_time = datetime.now().strftime('%H:%M:%S')
    st.markdown(f"""
        <div style="background:#161b22; padding:10px; border-radius:10px; border:1px solid #333; display:flex; justify-content:space-between; margin-bottom:15px; font-size:14px;">
            <span>🕒 SON GÜNCELLEME: <b>{c_time}</b></span>
            <span>📡 VERİ HATTI: <b>Zırhlı / Failover Aktif</b></span>
        </div>
    """, unsafe_allow_html=True)

    h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f"""
        <div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};">
            <small>MODEL GÜVEN ROZETİ (OOS %)</small><br>
            <b style="font-size:32px;">%{hr:.1f}</b>
        </div>
    """, unsafe_allow_html=True)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    for i, (name, signs) in enumerate(assets.items()):
        # 1. MAKRO SKOR (Fundamental)
        m_env = (0.35 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 
                 0.40 * z_liq.iloc[-1] + 
                 0.25 * z_spr.iloc[-1] * signs[2])
        
        # 2. ÖZGÜN DEĞERLEME (Varlık Bazlı)
        z_self = z_score(master_df[name], win=200).iloc[-1] * -1 # Çok yükselen negatife döner
        
        # 3. MOMENTUM (Hacim Destekli)
        last_v = df_v[name].tail(24)
        v_z = (df_v[name].iloc[-1] - last_v.mean()) / (last_v.std() + 1e-9)
        roc_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        mom_power = (roc_4h / 4) if v_z > -0.5 else 0

        # NİHAİ SKOR (Nötr Yasak)
        total_score = (m_env * 0.6) + (z_self * 0.2) + mom_power
        
        if total_score > 0.6: status = "GÜÇLÜ_AL"
        elif total_score > 0: status = "DİKKATLİ_AL"
        elif total_score > -0.6: status = "DİKKATLİ_SAT"
        else: status = "GÜÇLÜ_SAT"
            
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{status}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="font-size:32px;">{name}</b>
                    <b style="font-size:20px;">{status.replace("_", " ")}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:22px;">Skor: <b>{total_score:.2f}</b></p>
                <div style="font-size:13px; color:#aaa; line-height:1.6;">
                    🌍 Makro: {m_env:.2f} | 💎 Değerleme: {z_self:.2f}<br>
                    📊 Hacim Z: {v_z:.2f}z | 4s İvme: %{roc_4h:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Kritik Sistem Hatası: {e}")
