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

# 自定義背景顏色 (淡藍色)
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    </style>
    """, unsafe_allow_html=True)

# 組態設定
CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500)",
        "offset": 0,
        "keywords": ["SPX", "ES"],
        "width_bar": 1.5,
        "last_price": 6861.89
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指)",
        "offset": 75,
        "keywords": ["IUXX", "NQ"],
        "width_bar": 15.0,
        "last_price": 24797.17
    }
}
DATA_DIR = "data"

# --- 數據處理核心 ---

def get_latest_files(symbol_keywords):
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files: return None, None
    
    # 篩選對應代號的檔案
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None
    
    # 區分 OI 檔案與 Gamma 檔案 (Vol 檔案)
    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    
    latest_oi = max(oi_files, key=os.path.getmtime) if oi_files else None
    latest_vol = max(vol_files, key=os.path.getmtime) if vol_files else None
    return latest_oi, latest_vol

def clean_data(filepath, offset=0):
    df = pd.read_csv(filepath)
    # 轉換數值並處理逗號
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Strike'] = df['Strike'] + offset
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

# --- 繪圖函數 (八張圖) ---

def plot_net_gamma(df, last_price, title):
    flip = find_flip(df)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    scale = 1e8 # 億美元
    net_gex = df['Net Gamma Exposure'] / scale
    colors = ['blue' if x >= 0 else 'orange' for x in net_gex]
    ax1.bar(df['Strike'], net_gex, color=colors, alpha=0.7, width=df['Strike'].diff().median()*0.8)
    
    ax2 = ax1.twinx()
    agg_gex = df['Gamma Exposure Profile'] / 1e9 # 累計用十億
    ax2.plot(df['Strike'], agg_gex, color='#3498db', linewidth=3)
    
    if flip:
        ax1.axvline(flip, color='red', linestyle='-', linewidth=1.5)
        ax1.axvspan(df['Strike'].min(), flip, color='red', alpha=0.05)
        ax1.axvspan(flip, df['Strike'].max(), color='green', alpha=0.05)
        ax1.text(flip, ax1.get_ylim()[1]*0.8, f"Flip: {flip:,.0f}", color='red', fontweight='bold', ha='right')
        
    ax1.axvline(last_price, color='green', linestyle='--', linewidth=1.5)
    ax1.set_title(f"{title} - 淨 Gamma 曝險 (億)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

def plot_cp_gamma(df, last_price, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    scale = 1e8
    ax.bar(df['Strike'], df['Call Gamma Exposure']/scale, color='blue', label='Call GEX', alpha=0.7)
    ax.bar(df['Strike'], df['Put Gamma Exposure']/scale, color='orange', label='Put GEX', alpha=0.7)
    ax.axvline(last_price, color='green', linestyle='--')
    ax.set_title(f"{title} - 買賣權 Gamma 對比 (億)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

def plot_cp_oi(df, last_
