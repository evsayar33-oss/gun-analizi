import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import os
import sys

# --- 0. MOD VE TASARIM AYARLARI ---
is_headless = "--headless" in sys.argv
if not is_headless:
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh
    # Web sitesini her 15 dakikada bir yenile
    st_autorefresh(interval=900 * 1000, key="sentinel_alert_v20")
    st.set_page_config(page_title="Alpha Sentinel V20.1", layout="wide")
    st.markdown("""
        <style>
        .main { background-color: #05070a; color: white; }
        .card { padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 10px solid; background: #0f121a; }
        .status-AL { border-left-color: #00ff00; background: linear-gradient(90deg, rgba(0,255,0,0.1) 0%, rgba(15,18,26,1) 100%); }
        .status-SAT { border-left-color: #ff4b4b; background: linear-gradient(90deg, rgba(255,75,75,0.05) 0%, rgba(15,18,26,1) 100%); }
        .status-DIVERJANZ { border-left-color: #ffcc00; background: linear-gradient(90deg, rgba(255,204,0,0.1) 0%, rgba(15,18,26,1) 100%); }
        .hit-rate-badge { border: 2px solid; padding: 12px; border-radius: 10px; text-align: center; background: rgba(0,0,0,0.2); margin-bottom: 20px; }
        </style>
        """, unsafe_allow_html=True)

# SECRETS
if is_headless:
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
    CHAT_ID = os.environ.get("CHAT_ID")
    FRED_API_KEY = os.environ.get("FRED_API_KEY")
else:
    TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", None)
    CHAT_ID = st.secrets.get("CHAT_ID", None)
    FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)

STATE_FILE = "last_scores.csv"

def send_telegram(text):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try: requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
        except: pass

def z_score(s, win=126):
    return (s - s.rolling(win, min_periods=5).mean()) / (s.rolling(win, min_periods=5).std() + 1e-9)

def fetch_data():
    syms = {'SPX':'ES=F', 'NDX':'NQ=F', 'XAU':'GC=F', 'XAG':'SI=F', 'DXY':'DX-Y.NYB', 'TNX':'^TNX', 'VIX':'^VIX', 'HYG':'HYG'}
    df_y = yf.download(list(syms.values()), period="5y", interval="1d", progress=False)['Close'].ffill().bfill()
    df_y = df_y.rename(columns={v: k for k, v in syms.items()})
    df_h = yf.download(['ES=F', 'NQ=F', 'GC=F', 'SI=F'], period="2d", interval="15m", progress=False)['Close'].ffill().bfill()
    df_h = df_h.rename(columns={'ES=F':'SPX', 'NQ=F':'NDX', 'GC=F':'XAU', 'SI=F':'XAG'})
    
    df_f = pd.DataFrame(index=df_y.index)
    if FRED_API_KEY:
        try:
            fred_ids = {'WALCL': 'WALCL', 'T10YIE': 'T10YIE', 'SPREAD': 'BAMLH0A0HYM2'}
            for name, s_id in fred_ids.items():
                r = requests.get(f"https://api.stlouisfed.org/fred/series/observations?series_id={s_id}&api_key={FRED_API_KEY}&file_type=json", timeout=5).json()
                if 'observations' in r:
                    obs = pd.DataFrame(r['observations'])[['date', 'value']]
                    obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
                    obs['date'] = pd.to_datetime(obs['date'])
                    df_f[name] = obs.set_index('date')['value'].reindex(df_y.index, method='ffill')
        except: pass
    
    if 'SPREAD' not in df_f.columns: df_f['SPREAD'] = (100 - df_y['HYG']).rolling(20).mean()
    if 'T10YIE' not in df_f.columns: df_f['T10YIE'] = 2.1
    if 'WALCL' not in df_f.columns: df_f['WALCL'] = 7150000
    return df_y, df_h, df_f.ffill().bfill()

# --- ANA ANALİZ ---
try:
    df_y, df_h, df_f = fetch_data()
    master = pd.concat([df_y, df_f], axis=1).ffill()
    
    z_rf = z_score(master['TNX'] - master['T10YIE'])
    z_dxy = z_score(master['DXY'])
    z_liq = z_score(master['WALCL'])
    z_spr = z_score(master['SPREAD'])

    # OOS Hit-Rate
    sig_raw = ((0.5 * z_liq) + (0.25 * -(z_rf + z_dxy)/2) + (0.25 * -z_spr)).values
    ret_raw = df_y['SPX'].pct_change(1).shift(-1).values
    mask = ~np.isnan(sig_raw) & ~np.isnan(ret_raw)
    hr = float((np.sign(sig_raw[mask]) == np.sign(ret_raw[mask]))[-60:].mean() * 100)

    assets = {'SPX':[-1,1,-1], 'NDX':[-1,1,-1], 'XAU':[-1,1,1], 'XAG':[-1,1,-1]}
    current_results = {}

    if os.path.exists(STATE_FILE):
        try:
            old_df = pd.read_csv(STATE_FILE, index_col=0)
            old_scores = old_df['score'].to_dict()
        except: old_scores = {}
    else: old_scores = {}

    if not is_headless:
        st.title("🏛️ ALPHA SENTINEL V20.1")
        h_c = "#00ff00" if hr >= 55 else "#ffff00" if hr >= 45 else "#ff4b4b"
        st.markdown(f'<div class="hit-rate-badge" style="border-color:{h_c}; color:{h_c};"><small>OOS GÜVENİ</small> <b style="font-size:24px;">%{hr:.1f}</b></div>', unsafe_allow_html=True)
        cols = st.columns(2)

    for i, (name, signs) in enumerate(assets.items()):
        m_env = (0.5 * z_liq.iloc[-1] + 0.3 * -(z_rf.iloc[-1] + z_dxy.iloc[-1])/2 + 0.2 * z_spr.iloc[-1] * signs[2])
        roc = ((df_h[name].iloc[-1] / df_h[name].iloc[-5]) - 1) * 100
        total_score = round((m_env * 0.4) + (roc * 4.5), 2)
        
        signal = "AL" if total_score > 0.15 else ("SAT" if total_score < -0.15 else "NOTR")
        current_results[name] = total_score

        # ALARM KONTROLÜ
        if name in old_scores:
            old_val = old_scores[name]
            diff = total_score - old_val
            if abs(diff) > 0.60:
                direction = "YÜKSELİŞ" if diff > 0 else "DÜŞÜŞ"
                send_telegram(f"🚨 *GÜÇLÜ SKOR DEĞİŞİMİ: {name}*\nEski: `{old_val}` -> Yeni: *{total_score}*\nFark: `{diff:+.2f}`")
            
            old_sig = "AL" if old_val > 0.15 else ("SAT" if old_val < -0.15 else "NOTR")
            if old_sig != signal and signal != "NOTR":
                emoji = "🚀" if signal == "AL" else "📉"
                send_telegram(f"{emoji} *SİNYAL DÖNÜŞÜ: {name}*\nYön: `{old_sig}` -> *{signal}*\nSkor: `{total_score}` (Eski: {old_val})")

        if not is_headless:
            # Diverjanz Kontrolü (Makro ve Fiyat zıt mı?)
            is_div = np.sign(m_env) != np.sign(roc) and abs(roc) > 0.1
            card_type = "DIVERJANZ" if is_div else signal
            with cols[i%2]:
                st.markdown(f"""
                <div class="card status-{card_type}">
                    <div style="display:flex; justify-content:space-between;">
                        <b style="font-size:24px;">{name}</b>
                        <b style="font-size:18px;">{card_type}</b>
                    </div>
                    <p style="margin:5px 0; font-size:22px;">Skor: <b>{total_score}</b></p>
                    <div style="font-size:12px; color:#aaa;">
                        🌍 Makro: {m_env:.2f} | ⚡ 1s İvme: %{roc:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    pd.DataFrame.from_dict(current_results, orient='index', columns=['score']).to_csv(STATE_FILE)

except Exception as e:
    if is_headless: print(f"Hata: {e}")
    else: st.error(f"Hata: {e}")
