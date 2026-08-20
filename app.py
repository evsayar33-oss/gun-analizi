import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
from streamlit_autorefresh import st_autorefresh

# --- 0. AYARLAR ---
# 2 dakikada bir otomatik yenileme (Sinyal takibi için agresif hız)
st_autorefresh(interval=120 * 1000, key="sentinel_alert_v20")

st.set_page_config(page_title="Sentinel V20 - Alert Engine", layout="wide")

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

# SECRETS (Telegram bilgilerinin tanımlı olduğu varsayılır)
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", None)
CHAT_ID = st.secrets.get("CHAT_ID", None)
FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

def send_telegram(text):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try: requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=5)
        except: pass

# --- 1. VERİ MOTORU ---
@st.cache_data(ttl=60)
def fetch_synchronized_data(api_key):
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    df_raw = yf.download(list(syms.values()), period="5y", interval="1d")['Close'].ffill().bfill()
    df_y = df_raw.rename(columns={v: k for k, v in syms.items()})
    h_raw = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="2d", interval="15m")['Close'].ffill().bfill()
    df_h = h_raw.rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})
    
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

# --- 2. ANALİZ VE ALARM MOTORU ---
try:
    df_y, df_h, df_f = fetch_synchronized_data(FRED_API_KEY)
    
    # Faktörler
    net_liq = df_f['WALCL'] - df_f.get('TGA', 0) - df_f.get('RRP', 0)
    z_liq_accel = z_score(net_liq.rolling(20).mean(), 126)
    reel_faiz = df_y['TNX'] - df_f['T10YIE']
    z_rf, z_dxy, z_spr = z_score(reel_faiz), z_score(df_y['DXY']), z_score(df_f['SPREAD'])

    # OOS Hit-Rate (Hizalanmış)
    sig_raw = ((0.5 * z_liq_accel) + (0.25 * -(z_rf + z_dxy)/2) + (0.25 * -z_spr)).values
    ret_raw = df_y['SPX'].pct_change(1).shift(-1).values
    mask = ~np.isnan(sig_raw) & ~np.isnan(ret_raw)
    hr = float((np.sign(sig_raw[mask]) == np.sign(ret_raw[mask]))[-60:].mean() * 100)

    # --- UI BAŞLIĞI ---
    st.title("🏛️ ALPHA SENTINEL V20 - ALERTS")
    h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
    st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};"><small>OOS GÜVENİ</small> <b style="font-size:24px;">%{hr:.1f}</b></div>', unsafe_allow_html=True)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    cols = st.columns(2)
    
    # --- HAFIZA SİSTEMİ (ALARM İÇİN) ---
    if 'old_results' not in st.session_state:
        st.session_state.old_results = {}

    for i, (name, signs) in enumerate(assets.items()):
        m_env = (0.50 * z_liq_accel.iloc[-1] + 0.30 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.20 * z_spr.iloc[-1] * signs[2])
        roc_flash = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        total_score = (m_env * 0.3) + (roc_flash * 4.5)
        
        signal = "AL" if total_score > 0.15 else ("SAT" if total_score < -0.15 else "NOTR")
        
        # --- ALARM KONTROLÜ ---
        if name in st.session_state.old_results:
            old_data = st.session_state.old_results[name]
            # 1. Sinyal Değişimi (Örn: SAT -> AL)
            if old_data['signal'] != signal and signal != "NOTR":
                emoji = "🚀" if signal == "AL" else "📉"
                send_telegram(f"{emoji} *SİNYAL DEĞİŞİMİ: {name}*\nEski: `{old_data['signal']}` -> Yeni: *{signal}*\nSkor: `{total_score:.2f}`\n1s İvme: %{roc_flash:.2f}")
            
            # 2. Güçlü Skor Sıçraması (0.50 puandan fazla değişim)
            score_diff = total_score - old_data['score']
            if abs(score_diff) > 0.50:
                direction = "YÜKSELİŞ" if score_diff > 0 else "DÜŞÜŞ"
                send_telegram(f"🚨 *GÜÇLÜ SKOR DEĞİŞİMİ: {name}*\nSkor `{direction}`: `{old_data['score']:.2f}` -> *{total_score:.2f}* (Fark: {score_diff:+.2f})")

        # Hafızayı Güncelle
        st.session_state.old_results[name] = {'signal': signal, 'score': total_score}

        # --- KART RENDER ---
        with cols[i%2]:
            st.markdown(f"""
            <div class="card status-{signal}">
                <div style="display:flex; justify-content:space-between;">
                    <b style="font-size:26px;">{name}</b>
                    <b style="font-size:20px;">{signal}</b>
                </div>
                <hr style="border:0.1px solid rgba(255,255,255,0.1); margin:10px 0;">
                <p style="margin:5px 0; font-size:22px;">Skor: <b>{total_score:.2f}</b></p>
                <div style="font-size:13px; color:#aaa;">
                    💧 Likidite: {z_liq_accel.iloc[-1]:.2f}z | ⚡ 1s İvme: %{roc_flash:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Sistem Hatası: {e}")
