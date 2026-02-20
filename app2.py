import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
from scipy.interpolate import interp1d

# --- 頁面設定 ---
st.set_page_config(layout="wide", page_title="Market Gamma & OI Analysis")
st.title("市場籌碼區間分析 (SPX/ES vs NDX/NQ)")

# --- 數據處理函數 ---
def load_cleaned_csv(filepath):
    df = pd.read_csv(filepath)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            except:
                pass
    return df

def get_latest_files(data_dir="DATA"):
    """從 DATA 資料夾中獲取最新的 SPX 與 NDX 相關檔案"""
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        st.error(f"在 {data_dir} 資料夾中找不到 CSV 檔案！")
        return None
    
    # 依修改時間排序
    files.sort(key=os.path.getmtime, reverse=True)
    
    latest_files = {
        "spx_gamma": None, "spx_oi": None,
        "ndx_gamma": None, "ndx_oi": None
    }
    
    for f in files:
        fname = os.path.basename(f).upper()
        if "SPX" in fname:
            if "OPEN-INTEREST" in fname and not latest_files["spx_oi"]:
                latest_files["spx_oi"] = f
            elif "OPEN-INTEREST" not in fname and not latest_files["spx_gamma"]:
                latest_files["spx_gamma"] = f
        elif "IUXX" in fname or "NDX" in fname:
            if "OPEN-INTEREST" in fname and not latest_files["ndx_oi"]:
                latest_files["ndx_oi"] = f
            elif "OPEN-INTEREST" not in fname and not latest_files["ndx_gamma"]:
                latest_files["ndx_gamma"] = f
                
    return latest_files

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

# --- 繪圖函數 (整合至 Streamlit) ---

def plot_combined_kline(oi_file, current_price, price_range, title, width_bar, is_ndx=False):
    np.random.seed(42 if is_ndx else 100)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min') # 縮短點數以利顯示
    volatility = 15 if is_ndx else 2
    steps = np.random.normal(loc=0.005, scale=volatility, size=len(dates))
    path = np.cumsum(steps) + (current_price - np.cumsum(steps)[-1])
    df = pd.DataFrame({'Close': path, 'Open': path - steps}, index=dates)
    df['High'] = df[['Open', 'Close']].max(axis=1) + 2
    df['Low'] = df[['Open', 'Close']].min(axis=1) - 2

    df_oi = load_cleaned_csv(oi_file)
    y_min, y_max = current_price - price_range, current_price + price_range
    df_oi_v = df_oi[(df_oi['Strike'] >= y_min) & (df_oi['Strike'] <= y_max)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05}, sharey=True)
    
    # K-Line
    ax1.vlines(df.index, df['Low'], df['High'], color='black', linewidth=0.5)
    up = df[df['Close'] >= df['Open']]
    down = df[df['Close'] < df['Open']]
    ax1.bar(up.index, up['Close']-up['Open'], bottom=up['Open'], color='green', width=0.005)
    ax1.bar(down.index, down['Open']-down['Close'], bottom=down['Close'], color='red', width=0.005)
    ax1.axhline(current_price, color='blue', linestyle='--')
    ax1.set_title(f"{title} 15m K-Line", fontsize=10)
    
    # OI
    ax2.barh(df_oi_v['Strike'], df_oi_v['Call Open Interest']/1e3, color='blue', height=width_bar, label='Call')
    ax2.barh(df_oi_v['Strike'], -df_oi_v['Put Open Interest']/1e3, color='orange', height=width_bar, label='Put')
    ax2.axhline(current_price, color='blue', linestyle='--')
    ax2.set_xlabel("OI (K)")
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_gamma_strike(gamma_file, last_price, prefix):
    df = load_cleaned_csv(gamma_file)
    flip = find_flip(df)
    scale = 1e9 if prefix == 'es' else 1e6
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    colors = ['blue' if x >= 0 else 'orange' for x in df['Net Gamma Exposure']]
    ax1.bar(df['Strike'], df['Net Gamma Exposure']/scale, color=colors, alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.plot(df['Strike'], df['Gamma Exposure Profile']/1e9, color='#3498db', linewidth=2)
    
    if flip:
        ax1.axvline(flip, color='red', linestyle='-')
        ax1.axvspan(df['Strike'].min(), flip, color='red', alpha=0.05)
        ax1.axvspan(flip, df['Strike'].max(), color='green', alpha=0.05)
    ax1.axvline(last_price, color='green', linestyle='--')
    ax1.set_title(f"Net Gamma Exposure ({prefix.upper()})")
    st.pyplot(fig)

def plot_cp_gamma(oi_file, last_price, prefix):
    df = load_cleaned_csv(oi_file)
    scale = 1e9 if prefix == 'es' else 1e6
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df['Strike'], df['Call Gamma Exposure']/scale, color='blue', label='Call')
    ax.bar(df['Strike'], df['Put Gamma Exposure']/scale, color='orange', label='Put')
    ax.axvline(last_price, color='green', linestyle='--')
    ax.set_title(f"Call vs Put Gamma ({prefix.upper()})")
    ax.legend()
    st.pyplot(fig)

def plot_cp_oi(oi_file, last_price, prefix):
    df = load_cleaned_csv(oi_file)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df['Strike'], df['Call Open Interest']/1e3, color='blue', label='Call')
    ax.bar(df['Strike'], -df['Put Open Interest']/1e3, color='orange', label='Put')
    ax.axvline(last_price, color='green', linestyle='--')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title(f"Open Interest Distribution ({prefix.upper()})")
    ax.legend()
    st.pyplot(fig)

# --- 執行流程 ---

latest = get_latest_files("DATA")

if latest:
    col_left, col_right = st.columns(2)
    
    # 左側：SPX / ES
    with col_left:
        st.header("🇺🇸 ES (SPX Basis)")
        if latest["spx_oi"] and latest["spx_gamma"]:
            plot_combined_kline(latest["spx_oi"], 6861.89, 200, "ES Combined", 1.5)
            plot_gamma_strike(latest["spx_gamma"], 6861.89, 'es')
            plot_cp_gamma(latest["spx_oi"], 6861.89, 'es')
            plot_cp_oi(latest["spx_oi"], 6861.89, 'es')
        else:
            st.warning("缺少 SPX 相關檔案")

    # 右側：NDX / NQ
    with col_right:
        st.header("💻 NQ (NDX Basis)")
        if latest["ndx_oi"] and latest["ndx_gamma"]:
            plot_combined_kline(latest["ndx_oi"], 24797.17, 600, "NQ Combined", 15.0, is_ndx=True)
            plot_gamma_strike(latest["ndx_gamma"], 24797.17, 'nq')
            plot_cp_gamma(latest["ndx_oi"], 24797.17, 'nq')
            plot_cp_oi(latest["ndx_oi"], 24797.17, 'nq')
        else:
            st.warning("缺少 NDX 相關檔案")
else:
    st.info("請確認 DATA 資料夾路徑正確且包含檔案。")
