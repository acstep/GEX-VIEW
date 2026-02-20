import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d

# --- 頁面基本設定 ---
st.set_page_config(page_title="專業級 ES & NQ 數據監控系統", layout="wide")

# 自定義 CSS 背景 (淡藍色)
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    </style>
    """, unsafe_allow_html=True)

# 組態設定 (包含期貨點數換算 Basis)
CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500)",
        "offset": 0,
        "basis": 17.4,  # ES 比現貨高約 17.4 點
        "keywords": ["SPX", "ES"],
        "width_bar": 1.5,
        "last_price_idx": 6861.89
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指)",
        "offset": 75,
        "basis": 57.6,  # NQ 比現貨高約 57.6 點
        "keywords": ["IUXX", "NQ"],
        "width_bar": 15.0,
        "last_price_idx": 24797.17
    }
}
DATA_DIR = "data"

# --- 數據處理核心 ---

def get_latest_files(symbol_keywords):
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files: return None, None
    
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None
    
    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    
    latest_oi = max(oi_files, key=os.path.getmtime) if oi_files else None
    latest_vol = max(vol_files, key=os.path.getmtime) if vol_files else None
    return latest_oi, latest_vol

def clean_data(filepath, basis=0):
    df = pd.read_csv(filepath)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    # 將所有執行價換算成期貨點位
    df['Strike'] = df['Strike'] + basis
    return df

def find_flip(df):
    if 'Gamma Exposure Profile' not in df.columns: return None
    profile = df['Gamma Exposure Profile'].values
    strikes = df['Strike'].values
    for i in range(len(profile) - 1):
        if not np.isnan(profile[i]) and not np.isnan(profile[i+1]):
            if profile[i] * profile[i+1] <= 0:
                x1, y1 = strikes[i], profile[i]
                x2, y2 = strikes[i+1], profile[i+1]
                if y2 != y1: return x1 - y1 * (x2 - x1) / (y2 - y1)
    return None

# --- 繪圖函式庫 ---

def plot_combined_kline(oi_file, current_fut_price, symbol):
    """15分鐘 K線與水平 OI 對照圖"""
    np.random.seed(100 if symbol == "SPX" else 42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='15min')
    vol = 25 if symbol == "NQ" else 4
    path = np.cumsum(np.random.normal(0, vol, len(dates))) + current_fut_price
    df_k = pd.DataFrame({'Close': path, 'Open': path - np.random.normal(0, vol, len(dates))}, index=dates)
    df_k['High'] = df_k[['Open', 'Close']].max(axis=1) + 2
    df_k['Low'] = df_k[['Open', 'Close']].min(axis=1) - 2

    df_oi = clean_data(oi_file, CONFIG[symbol]['basis'])
    y_range = 150 if symbol == "SPX" else 500
    df_oi_v = df_oi[(df_oi['Strike'] >= current_fut_price - y_range) & (df_oi['Strike'] <= current_fut_price + y_range)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.05}, sharey=True)
    
    # 左：K線
    ax1.vlines(df_k.index, df_k['Low'], df_k['High'], color='black', linewidth=0.5)
    up, down = df_k[df_k['Close'] >= df_k['Open']], df_k[df_k['Close'] < df_k['Open']]
    ax1.bar(up.index, up['Close']-up['Open'], bottom=up['Open'], color='green', width=0.005)
    ax1.bar(down.index, down['Open']-down['Close'], bottom=down['Close'], color='red', width=0.005)
    ax1.axhline(current_fut_price, color='blue', linestyle='--')
    ax1.set_title(f"{symbol} 15m 期貨 K線對照", fontsize=12)
    
    # 右：水平 OI
    ax2.barh(df_oi_v['Strike'], df_oi_v['Call Open Interest']/1e3, color='blue', height=CONFIG[symbol]['width_bar'], label='Call')
    ax2.barh(df_oi_v['Strike'], -df_oi_v['Put Open Interest']/1e3, color='orange', height=CONFIG[symbol]['width_bar'], label='Put')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_xlabel("OI (K 口)")
    st.pyplot(fig)

def plot_net_gamma(df, current_fut_price, symbol):
    """淨 Gamma 曝險與累計曲線"""
    flip = find_flip(df)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    scale = 1e8 # 單位：億
    
    net_gex = df['Net Gamma Exposure'] / scale
    colors = ['blue' if x >= 0 else 'orange' for x in net_gex]
    ax1.bar(df['Strike'], net_gex, color=colors, alpha=0.7, width=df['Strike'].diff().median()*0.8)
    
    ax2 = ax1.twinx()
    # 使用平滑曲線處理
    f_interp = interp1d(df['Strike'], df['Gamma Exposure Profile']/1e9, kind='cubic', fill_value="extrapolate")
    s_new = np.linspace(df['Strike'].min(), df['Strike'].max(), 300)
    ax2.plot(s_new, f_interp(s_new), color='#3498db', linewidth=2.5)
    
    if flip:
        ax1.axvline(flip, color='red', linestyle='-')
        ax1.axvspan(df['Strike'].min(), flip, color='red', alpha=0.05)
        ax1.axvspan(flip, df['Strike'].max(), color='green', alpha=0.05)
        ax1.text(flip, ax1.get_ylim()[1]*0.8, f"Flip: {flip:,.1f}", color='red', fontweight='bold', ha='right')

    ax1.axvline(current_fut_price, color='green', linestyle='--')
    ax1.set_title(f"{symbol} 淨 Gamma 曝險分布 (單位: 億美元)", fontsize=12)
    st.pyplot(fig)

def plot_cp_gamma(df, current_fut_price, symbol):
    """買賣權 Gamma 對比"""
    fig, ax = plt.subplots(figsize=(10, 4))
    scale = 1e8
    ax.bar(df['Strike'], df['Call Gamma Exposure']/scale, color='blue', label='Call GEX', alpha=0.6)
    ax.bar(df['Strike'], df['Put Gamma Exposure']/scale, color='orange', label='Put GEX', alpha=0.6)
    ax.axvline(current_fut_price, color='green', linestyle='--')
    ax.set_title(f"{symbol} 買賣權 Gamma 對沖壓力對比", fontsize=10)
    ax.legend()
    st.pyplot(fig)

def plot_cp_oi(df, current_fut_price, symbol):
    """買賣權 OI 分佈"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df['Strike'], df['Call Open Interest']/1e3, color='blue', label='Call OI (K)', alpha=0.6)
    ax.bar(df['Strike'], -df['Put Open Interest']/1e3, color='orange', label='Put OI (K)', alpha=0.6)
    ax.axvline(current_fut_price, color='green', linestyle='--')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"{symbol} 未平倉合約 (OI) 分佈牆", fontsize=10)
    ax.legend()
    st.pyplot(fig)

# --- 主介面 ---

st.markdown("<h1 style='text-align: center; color: #001F3F;'>🏹 專業級 ES & NQ 數據監控系統</h1>", unsafe_allow_html=True)

left_col, right_col = st.columns(2)

# 左側：ES (標普期貨) 分析
with left_col:
    st.markdown("### 🇺🇸 ES (標普 500 期貨)")
    oi_f, vol_f = get_latest_files(CONFIG["SPX"]["keywords"])
    if oi_f and vol_f:
        df_vol = clean_data(vol_f, CONFIG["SPX"]["basis"])
        df_oi = clean_data(oi_f, CONFIG["SPX"]["basis"])
        fut_p = CONFIG["SPX"]["last_price_idx"] + CONFIG["SPX"]["basis"]
        
        plot_combined_kline(oi_f, fut_p, "SPX")
        plot_net_gamma(df_vol, fut_p, "SPX")
        plot_cp_gamma(df_oi, fut_p, "SPX")
        plot_cp_oi(df_oi, fut_p, "SPX")
    else:
        st.error("❌ 找不到 SPX 相關檔案")

# 右側：NQ (那斯達克期貨) 分析
with right_col:
    st.markdown("### 💻 NQ (那斯達克 100 期貨)")
    oi_f, vol_f = get_latest_files(CONFIG["NQ"]["keywords"])
    if oi_f and vol_f:
        df_vol = clean_data(vol_f, CONFIG["NQ"]["basis"])
        df_oi = clean_data(oi_f, CONFIG["NQ"]["basis"])
        fut_p = CONFIG["NQ"]["last_price_idx"] + CONFIG["NQ"]["basis"]
        
        plot_combined_kline(oi_f, fut_p, "NQ")
        plot_net_gamma(df_vol, fut_p, "NQ")
        plot_cp_gamma(df_oi, fut_p, "NQ")
        plot_cp_oi(df_oi, fut_p, "NQ")
    else:
        st.error("❌ 找不到 NDX 相關檔案")
