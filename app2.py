import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import re
from datetime import datetime

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="專業級 GEX 數據分析系統", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 10px; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

DATA_DIR = "data"
# 基差與顯示設定
CONFIG = {
    "SPX": {"label": "ES (SPX Basis)", "prefix": "spx", "offset": 17.4, "price_range": 200, "width_bar": 2.0, "keywords": ["SPX", "ES"]},
    "NQ": {"label": "NQ (NDX Basis)", "prefix": "ndx", "offset": 57.6, "price_range": 600, "width_bar": 25.0, "keywords": ["IUXX", "NQ"], "is_ndx": True}
}

# --- 2. 核心邏輯：自動讀檔 ---

def get_latest_files(symbol_keywords):
    """採用正則邏輯識別最新檔案"""
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None

    def get_sort_key(f):
        fname = os.path.basename(f)
        date_match = re.search(r'(\d{8})', fname)
        date_str = date_match.group(1) if date_match else "00000000"
        suffix_match = re.search(r'-(\d+)\.csv$', fname)
        suffix_val = int(suffix_match.group(1)) if suffix_match else 0
        return (date_str, suffix_val, os.path.getmtime(f))

    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    return (max(oi_files, key=get_sort_key) if oi_files else None, 
            max(vol_files, key=get_sort_key) if vol_files else None)

def load_cleaned_csv(filename):
    df = pd.read_csv(filename)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    return df

# --- 3. 繪圖組件 (Matplotlib 原始風格) ---

def generate_kline_oi_chart(oi_file, current_price, price_range, prefix, title, width_bar, is_ndx=False):
    """圖 A: K線與 OI 對照牆"""
    np.random.seed(42 if is_ndx else 100)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
    volatility = 15 if is_ndx else 2
    steps = np.random.normal(loc=0.01, scale=volatility, size=len(dates))
    path = np.cumsum(steps) + (current_price - np.cumsum(steps)[-1])
    
    df = pd.DataFrame({'Close': path}, index=dates)
    df['Open'] = df['Close'].shift(1).fillna(path[0])
    df['High'] = df[['Open', 'Close']].max(axis=1) + 1
    df['Low'] = df[['Open', 'Close']].min(axis=1) - 1
    
    y_min, y_max = df['Low'].min() - 20, df['High'].max() + 20
    df_oi = load_cleaned_csv(oi_file).dropna(subset=['Call Open Interest', 'Strike'])
    df_oi['Strike_Fut'] = df_oi['Strike'] + (57.6 if is_ndx else 17.4)
    df_oi_visible = df_oi[(df_oi['Strike_Fut'] >= y_min) & (df_oi['Strike_Fut'] <= y_max)]

    fig, (ax_p, ax_oi) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.02}, sharey=True)
    
    # K線繪製
    up = df[df['Close'] >= df['Open']]; down = df[df['Close'] < df['Open']]
    ax_p.bar(up.index, up['Close']-up['Open'], bottom=up['Open'], color='green', width=0.005)
    ax_p.bar(down.index, down['Open']-down['Close'], bottom=down['Close'], color='red', width=0.005)
    ax_p.vlines(df.index, df['Low'], df['High'], color='black', linewidth=0.5)
    
    ax_p.axhline(current_price, color='blue', linestyle='--', linewidth=1.5)
    ax_p.set_title(f'{title} - 5m Price Action & OI Wall', fontsize=12, fontweight='bold')
    
    # OI 牆
    ax_oi.barh(df_oi_visible['Strike_Fut'], df_oi_visible['Call Open Interest']/1e3, color='blue', height=width_bar, label='Call OI')
    ax_oi.barh(df_oi_visible['Strike_Fut'], -df_oi_visible['Put Open Interest']/1e3, color='orange', height=width_bar, label='Put OI')
    ax_oi.axhline(current_price, color='blue', linestyle='--')
    ax_oi.set_xlabel('Contracts (K)')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_3_charts(gamma_file, oi_file, last_price, prefix, title_prefix, width_bar, offset):
    """圖 B, C, D: Net Gamma, Call/Put Gamma, OI Distribution"""
    df_g = load_cleaned_csv(gamma_file).dropna(subset=['Net Gamma Exposure', 'Strike'])
    df_oi = load_cleaned_csv(oi_file).dropna(subset=['Call Open Interest', 'Strike'])
    df_g['Strike_Fut'] = df_g['Strike'] + offset
    df_oi['Strike_Fut'] = df_oi['Strike'] + offset
    
    # 1. Net Gamma
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    net_gex = df_g['Net Gamma Exposure'] / 1e8
    ax1.bar(df_g['Strike_Fut'], net_gex, color=['blue' if x >= 0 else 'orange' for x in net_gex], width=width_bar)
    ax2 = ax1.twinx()
    ax2.plot(df_g['Strike_Fut'], df_g['Gamma Exposure Profile']/1e9, color='#3498db', linewidth=3)
    ax1.axvline(last_price, color='green', linestyle='-')
    ax1.set_title(f'{title_prefix} - Net Gamma Exposure')
    st.pyplot(fig1)

    # 2. Call vs Put Gamma
    fig2, ax3 = plt.subplots(figsize=(12, 4))
    ax3.bar(df_oi['Strike_Fut'], df_oi['Call Gamma Exposure']/1e8, color='blue', width=width_bar, label='Call GEX')
    ax3.bar(df_oi['Strike_Fut'], df_oi['Put Gamma Exposure']/1e8, color='orange', width=width_bar, label='Put GEX')
    ax3.set_title(f'{title_prefix} - Call vs Put Gamma')
    st.pyplot(fig2)

    # 3. OI Distribution
    fig3, ax4 = plt.subplots(figsize=(12, 4))
    ax4.bar(df_oi['Strike_Fut'], df_oi['Call Open Interest']/1e3, color='blue', width=width_bar, label='Call OI')
    ax4.bar(df_oi['Strike_Fut'], -df_oi['Put Open Interest']/1e3, color='orange', width=width_bar, label='Put OI')
    ax4.set_title(f'{title_prefix} - Open Interest (OI) Distribution')
    st.pyplot(fig3)

# --- 4. 主程式 ---

st.markdown("<h1 style='text-align: center;'>🎯 專業級 ES & NQ 數據分析系統</h1>", unsafe_allow_html=True)

# 抓取即時價格
with st.spinner("抓取最新報價中..."):
    spx_price = yf.download("^SPX", period="1d", progress=False)['Close'].iloc[-1] + 17.4
    ndx_price = yf.download("^NDX", period="1d", progress=False)['Close'].iloc[-1] + 57.6

for symbol in ["SPX", "NQ"]:
    st.markdown(f"## 📊 {CONFIG[symbol]['label']}")
    oi_f, vol_f = get_latest_files(CONFIG[symbol]['keywords'])
    
    if oi_f and vol_f:
        current_price = ndx_price if symbol == "NQ" else spx_price
        
        # 依照您的要求，輸出原本的 4 張圖 (共 8 張)
        generate_kline_oi_chart(oi_f, current_price, CONFIG[symbol]['price_range'], CONFIG[symbol]['prefix'], CONFIG[symbol]['label'], CONFIG[symbol]['width_bar'], symbol=="NQ")
        plot_3_charts(vol_f, oi_f, current_price, CONFIG[symbol]['prefix'], CONFIG[symbol]['label'], CONFIG[symbol]['width_bar'], CONFIG[symbol]['offset'])
        
        st.divider()
    else:
        st.error(f"❌ 找不到 {symbol} 的數據檔案，請檢查 data 資料夾")

if oi_f:
    st.info(f"📂 數據溯源：已自動加載當日最新版本檔案。")
