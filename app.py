import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. OTOMATİK YENİLEME ---
st_autorefresh(interval=3600 * 1000, key="macro_master_v1")
st.set_page_config(page_title="Macro Master v1.0", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .card { padding: 18px; border-radius: 12px; margin-bottom: 15px; border-left: 10px solid; background: #161b22; }
    .signal-AL { border-left-color: #00ff00; }
    .signal-SAT { border-left-color: #ff4b4b; }
    .signal-NOTR { border-left-color: #8b949e; }
    .hit-rate-badge { border: 2px solid; padding: 12px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

# --- 1. VERİ MİMARİSİ (3 YILLIK ROLLING BACKFILL) ---
@st.cache_data(ttl=300)
def get_master_v1_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'VXN':'^VXN', 'HYG':'HYG'}
    df_y = yf.download(list(syms.values()), period="4y", interval="1d")['Close'].ffill().rename(columns={v: k for k, v in syms.items()})
    
    # Hacim ve İvme için Saatlik Veri
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F', '^VIX', '^VXN'], period="7d", interval="1h")
    df_h = h_raw['Close'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG', '^VIX':'VIX', '^VXN':'VXN'}).ffill()
    df_v = h_raw['Volume'].rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'}).ffill()

    # FRED Verileri (Tam Plan: WALCL, TGA, RRP)
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
    return df_y, df_h, df_v, df_f.ffill()

def z_roll(s, win=126): return (s - s.rolling(win, min_periods=1).mean()) / (s.rolling(win, min_periods=1).std() + 1e-9)

# --- 2. SÖNÜMLENMİŞ DİNAMİK IC VE REJİM MOTORU ---
try:
    df_y, df_h, df_v, df_f = get_master_v1_data(FRED_API_KEY)
    
    # 1. Göstergeleri Hazırla
    breakeven = df_f['T10YIE'] if 'T10YIE' in df_f.columns else 2.1
    reel_faiz = df_y['TNX'] - breakeven
    z_rf, z_dxy = z_roll(reel_faiz), z_roll(df_y['DXY'])
    
    # [ÇAPA 1]: Dolar-Faiz Kompoziti
    f1_z_dolar_faiz = (z_rf + z_dxy) / 2
    # [ÇAPA 2]: Net Likidite (WALCL - TGA - RRP)
    net_liq = df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)
    f2_z_walcl = z_roll(net_liq)
    # [ÇAPA 3]: Kredi Spread
    f3_z_spread = z_roll(df_f['SPREAD']) if 'SPREAD' in df_f.columns else z_roll(100 - df_y['HYG'])

    # --- VARLIK BAZLI HESAPLAMA ---
    assets = {
        'SPX': {'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1], 'vol': 'VIX'},
        'NDX': {'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1], 'vol': 'VXN'},
        'XAU': {'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, 1], 'vol': 'VIX'}, # XAU Spread Sign normalde +1
        'XAG': {'base': [0.4, 0.3, 0.3], 'signs': [-1, 1, -1], 'vol': 'VIX'}
    }
    
    results = {}
    for name, cfg in assets.items():
        # Rolling IC (126 Gün): Faktörler ile varlık t+5 getirisi Pearson korelasyonu
        fwd_ret = df_y[name].pct_change(5).shift(-5)
        factors = [f1_z_dolar_faiz, f2_z_walcl, f3_z_spread]
        
        damped_weights = []
        for i, f_z in enumerate(factors):
            ic = f_z.rolling(126).corr(fwd_ret).iloc[-1]
            if np.isnan(ic): ic = 0
            ic_multiplier = np.clip(1 + (ic * cfg['signs'][i]), 0.30, 1.70)
            
            # Rejim Adaptörü: Spread > 1.5z veya VIX % > 85 ise Spread ve Likidite 1.8x
            vix_perc = df_y[cfg['vol']].rolling(252).rank(pct=True).iloc[-1]
            regime_mult = 1.8 if (f3_z_spread.iloc[-1] > 1.5 or vix_perc > 0.85) and i > 0 else 1.0
            
            ham_w = cfg['base'][i] * ic_multiplier * regime_mult
            damped_weights.append(ham_w)
            
        # Normalizasyon ve Clamping (%15 - %60)
        w = np.array(damped_weights) / sum(damped_weights)
        w = np.clip(w, 0.15, 0.60)
        w = w / sum(w)
        
        # ALTIN ÖZEL SPREAD REJİMİ
        spr_sign = cfg['signs'][2]
        if name == 'XAU':
            vix_z = z_roll(df_y['VIX']).iloc[-1]
            vix_jump_z = (df_y['VIX'].pct_change() / df_y['VIX'].pct_change().rolling(20).std()).iloc[-1]
            if vix_perc > 0.90 and vix_jump_z > 2.0: spr_sign = -1 # Akut Sıvılaşma

        # --- HAM MAKRO SKOR ---
        ham_makro = (w[0] * f1_z_dolar_faiz.iloc[-1] * cfg['signs'][0] + 
                     w[1] * f2_z_walcl.iloc[-1] * cfg['signs'][1] + 
                     w[2] * f3_z_spread.iloc[-1] * spr_sign)

        # --- GATEKEEPER / MOMENTUM OVERRIDE ---
        p_now, p_prev = df_h[name].iloc[-1], df_h[name].iloc[-5]
        gunluk_degisim = (p_now / p_prev) - 1
        vol_z = z_roll(df_y[cfg['vol']]).iloc[-1]
        vol_change = df_y[cfg['vol']].pct_change().iloc[-1]
        
        gatekeeper, m_adj = "neutral", 0.0
        # Bearish Override
        if vol_z > 1.0 or vol_change > 0.025:
            if gunluk_degisim < -0.0040: gatekeeper, m_adj = "DIVERGING_BEARISH", -0.60
            else: gatekeeper, m_adj = "WATCH_BEARISH", -0.25
        # Bullish Override
        elif vol_z < -0.8 or vol_change < -0.020:
            if gunluk_degisim > 0.0050: gatekeeper, m_adj = "CONFIRMED_BULLISH", 0.60
            
        # NİHAİ SKOR (Z-Toplama ve Clamp)
        z_final = np.clip(ham_makro + m_adj, -3.0, 3.0)
        signal = "AL" if z_final > 0.50 else ("SAT" if z_final < -0.50 else "NOTR")
        
        results[name] = {'skor': z_final, 'ham': ham_makro, 'gate': gatekeeper, 'ret': gunluk_degisim*100, 'weights': w, 'sig': signal}

    # --- OOS HIT-RATE ---
    # SPX Üzerinden 126 günlük OOS Hit-Rate
    prediction = f1_z_dolar_faiz.shift(5) # En basit göstergeyle test
    hits = (np.sign(-prediction) == np.sign(df_y['SPX'].pct_change(5))).tail(126)
    hr = float(hits.mean() * 100)

    # --- UI ---
    st.title("🏛️ TIGHTENED MACRO DIRECTIONAL MATRIX")
    
    h_c = "#00ff00" if hr >= 60 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};"><small>OOS HIT-RATE GÜVEN ROZETİ</small><br><b style="font-size:28px;">%{hr:.1f}</b></div>', unsafe_allow_html=True)

    # Varlıklar
    cols = st.columns(2)
    for i, (name, data) in enumerate(results.items()):
        with cols[i%2]:
            st.markdown(f"""
            <div class="card signal-{data['sig']}">
                <div style="display:flex; justify-content:space-between;">
                    <b style="font-size:24px;">{name}</b>
                    <b style="font-size:24px;">{data['sig']}</b>
                </div>
                <p style="margin:5px 0; font-size:14px;">Composite Z: <b>{data['skor']:.2f}</b> | Ham Makro: {data['ham']:.2f}</p>
                <p style="margin:0; font-size:14px;">Canlı Günlük: <b>%{data['ret']:.2f}</b> | Gatekeeper: <code>{data['gate']}</code></p>
                <hr style="border:0.1px solid #333; margin:10px 0;">
                <small>Ağırlıklar (DF/LQ/SP): %{int(data['weights'][0]*100)} / %{int(data['weights'][1]*100)} / %{int(data['weights'][2]*100)}</small>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
