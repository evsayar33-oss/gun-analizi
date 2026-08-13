import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME VE AYARLAR ---
st_autorefresh(interval=3600 * 1000, key="macro_forced_direction_v8")
st.set_page_config(page_title="Macro Directional Pro", layout="wide")

# CSS: Canlı ve Agresif Kart Tasarımı
st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .card { padding: 25px; border-radius: 15px; margin-bottom: 20px; border-top: 10px solid; background: #0f121a; box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
    .status-GÜÇLÜ_AL { border-top-color: #00ff00; background: linear-gradient(180deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
    .status-DİKKATLİ_AL { border-top-color: #76ff03; background: linear-gradient(180deg, rgba(118,255,3,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-DİKKATLİ_SAT { border-top-color: #ff9100; background: linear-gradient(180deg, rgba(255,145,0,0.05) 0%, rgba(15,18,26,1) 100%); }
    .status-GÜÇLÜ_SAT { border-top-color: #ff1744; background: linear-gradient(180deg, rgba(255,23,68,0.1) 0%, rgba(15,18,26,1) 100%); }
    .hit-rate-badge { border: 2px solid; padding: 15px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.3); margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MİMARİSİ ---
@st.cache_data(ttl=300)
def get_v8_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'VXN':'^VXN', 'HYG':'HYG'}
    df_y = yf.download(list(syms.values()), period="4y", interval="1d")['Close'].ffill().rename(columns={v: k for k, v in syms.items()})
    
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="7d", interval="1h")
    df_h = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()
    
    df_f = pd.DataFrame(index=df_y.index)
    if api_key:
        fred_ids = {'WALCL': 'WALCL', 'TGA': 'WTREGEN', 'RRP': 'RRPONTSYD', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
        for name, s_id in fred_ids.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={api_key}&file_type=json"
                r = requests.get(url).json()
                obs = pd.DataFrame(r['observations'])[['date', 'value']]
                obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                obs['date'] = pd.to_datetime(obs['date'])
                df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
            except: pass
    return df_y, df_h, df_f.ffill()

def z_roll(s, win=126): return (s - s.rolling(win, min_periods=1).mean()) / (s.rolling(win, min_periods=1).std() + 1e-9)

# --- 2. DİNAMİK REJİM MOTORU ---
try:
    df_y, df_h, df_f = get_v8_data(FRED_API_KEY)
    
    # Faktör Hazırlığı
    breakeven = df_f['T10YIE'] if 'T10YIE' in df_f.columns else 2.1
    reel_faiz = df_y['TNX'] - breakeven
    z_rf, z_dxy = z_roll(reel_faiz), z_roll(df_y['DXY'])
    net_liq = df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)
    z_liq = z_roll(net_liq)
    z_spr = z_roll(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_roll(100 - df_y['HYG'])
    
    # --- REJİM ADAPTÖRÜ (DİNAMİK AĞIRLIKLANDIRMA) ---
    vix_now = df_y['VIX'].iloc[-1]
    spr_now = z_spr.iloc[-1]
    
    # Default Ağırlıklar (DF: 0.40, LIQ: 0.30, SPR: 0.30)
    # Kriz durumunda Spread ve Likiditeye kaç
    if vix_now > 25 or spr_now > 1.5:
        base_w = [0.20, 0.40, 0.40] # Kriz Rejimi
        regime_label = "KRİZ / DEFANSİF"
    elif vix_now < 15 and z_dxy.iloc[-1] < 0:
        base_w = [0.50, 0.30, 0.20] # Genişleme Rejimi
        regime_label = "RİSK-ON / AGRESİF"
    else:
        base_w = [0.40, 0.30, 0.30] # Normal Rejim
        regime_label = "STABİL / DENGELİ"

    # --- VARLIK ANALİZİ ---
    assets = {
        'SPX': {'signs': [-1, 1, -1], 'vol': 'VIX'},
        'NDX': {'signs': [-1, 1, -1], 'vol': 'VXN'},
        'XAU': {'signs': [-1, 1, 1], 'vol': 'VIX'},
        'XAG': {'signs': [-1, 1, -1], 'vol': 'VIX'}
    }
    
    results = {}
    for name, cfg in assets.items():
        # Damped IC Ağırlıklandırma
        fwd_ret = df_y[name].pct_change(5).shift(-5)
        factors = [(z_rf + z_dxy)/2, z_liq, z_spr]
        
        final_weights = []
        for i, f_z in enumerate(factors):
            ic = f_z.rolling(126).corr(fwd_ret).iloc[-1]
            if np.isnan(ic): ic = 0
            # Sönümleme ve Clamp
            ham_w = base_w[i] * np.clip(1 + (ic * cfg['signs'][i]), 0.5, 1.5)
            final_weights.append(ham_w)
        
        w = np.array(final_weights) / sum(final_weights)
        
        # Altın Özel Akut Likidasyon
        spr_sign = cfg['signs'][2]
        if name == 'XAU' and vix_now > 30: spr_sign = -1

        # Ham Makro Skor
        m_skor = (w[0] * (z_rf.iloc[-1] + z_dxy.iloc[-1])/2 * -1 + 
                  w[1] * z_liq.iloc[-1] * 1 + 
                  w[2] * z_spr.iloc[-1] * spr_sign)

        # Momentum Gatekeeper
        roc_4h = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        
        # NİHAİ YÖNÜ ZORLA (Binary + Intensity)
        # Nötr bölge yok; skor 0'ın üzerindeyse AL, altındaysa SAT.
        total_power = m_skor + (roc_4h / 5) # Momentumu skora %20 ekle
        
        if total_power > 1.0: status = "GÜÇLÜ_AL"
        elif total_power > 0: status = "DİKKATLİ_AL"
        elif total_power > -1.0: status = "DİKKATLİ_SAT"
        else: status = "GÜÇLÜ_SAT"
            
        results[name] = {'score': total_power, 'm_skor': m_skor, 'roc': roc_4h, 'status': status, 'weights': w}

    # --- OOS HIT-RATE ---
    prediction = ((z_liq * 0.4) + (-(z_rf + z_dxy)/2 * 0.4)).shift(2)
    hits = (np.sign(prediction) == np.sign(df_y['SPX'].pct_change(2))).tail(126)
    hr = float(hits.mean() * 100)

    # --- UI ---
    st.title("🏛️ MACRO FORCED-DIRECTION TERMINAL")
    
    h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f"""
        <div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};">
            <small>MODEL GÜVEN ROZETİ (OOS %{hr:.1f})</small><br>
            <span style="font-size:20px;">REJİM: <b>{regime_label}</b></span>
        </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (name, data) in enumerate(results.items()):
        s_label = data['status'].replace("_", " ")
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{data['status']}">
                <div style="display:flex; justify-content:space-between;">
                    <b style="font-size:28px;">{name}</b>
                    <b style="font-size:24px;">{s_label}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:5px 0; font-size:16px;">Sinyal Gücü: <b>{data['score']:.2f}</b></p>
                <p style="margin:5px 0; font-size:14px; color:#aaa;">Makro Temel: {data['m_skor']:.2f} | 4s İvme: %{data['roc']:.2f}</p>
                <small style="color:#555;">Dinamik Ağırlıklar: %{int(data['weights'][0]*100)} / %{int(data['weights'][1]*100)} / %{int(data['weights'][2]*100)}</small>
            </div>
            """, unsafe_allow_html=True)

    # MAKRO PANO
    st.sidebar.markdown("### 📊 CANLI MAKRO VERİLER")
    st.sidebar.write(f"10Y Reel Faiz: %{reel_faiz.iloc[-1]:.2f}")
    st.sidebar.write(f"Net Likidite: {l_v if (l_v:=float(df_f['WALCL'].iloc[-1]/1000000)) else 0:.2f}T")
    st.sidebar.write(f"Kredi Spread: %{float(df_f['SPREAD'].iloc[-1]) if 'SPREAD' in df_f.columns else 0:.2f}")

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
