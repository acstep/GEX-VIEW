import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 網頁頁面設定
st.set_page_config(page_title="NQ Gamma Trade Map", layout="wide")

# 常數設定
OFFSET = 75

def clean_data(df):
    cols = ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['NQ_Strike'] = df['Strike'] + OFFSET
    return df

def get_levels(df):
    if df.empty: return None, None, None
    cw = df.loc[df['Net Gamma Exposure'].idxmax(), 'NQ_Strike']
    pw = df.loc[df['Net Gamma Exposure'].idxmin(), 'NQ_Strike']
    flip = None
    for i in range(len(df)-1):
        y1 = df.iloc[i]['Net Gamma Exposure']
        y2 = df.iloc[i+1]['Net Gamma Exposure']
        if pd.isna(y1) or pd.isna(y2): continue
        if y1 * y2 <= 0:
            x1, x2 = df.iloc[i]['NQ_Strike'], df.iloc[i+1]['NQ_Strike']
            if y2 != y1:
                flip = x1 - y1 * (x2 - x1) / (y2 - y1)
                break
    return cw, pw, flip

# 網頁標題
st.title("📊 NQ Integrated Trade Map")
st.markdown("### Volume vs. OI Gamma Analysis")

# 讀取檔案
try:
    df_oi = pd.read_csv('$IUXX-gamma-levels-exp-20260220-monthly-open-interest.csv')
    df_vol = pd.read_csv('$IUXX-gamma-levels-exp-20260220-monthly.csv')
    
    df_oi = clean_data(df_oi)
    df_vol = clean_data(df_vol)

    # 計算數值
    oi_cw, oi_pw, oi_flip = get_levels(df_oi)
    vol_cw, vol_pw, vol_flip = get_levels(df_vol)

    # 顯示關鍵指標卡片
    col1, col2, col3 = st.columns(3)
    col1.metric("Daily Pivot (NQ)", f"{vol_flip:.2f}")
    col2.metric("Call Wall (NQ)", f"{oi_cw:.0f}")
    col3.metric("Put Wall (NQ)", f"{oi_pw:.0f}")

    # 繪圖邏輯
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 主要軸：Open Interest 柱狀圖
    width = 20
    ax1.bar(df_oi['NQ_Strike'], df_oi['Call Open Interest'], width=width, color='teal', alpha=0.4, label='Call OI (Resistance)')
    ax1.bar(df_oi['NQ_Strike'], -df_oi['Put Open Interest'], width=width, color='crimson', alpha=0.4, label='Put OI (Support)')
    ax1.set_ylabel('Open Interest (Contracts)', fontsize=12)
    ax1.set_xlabel('NQ Price (NDX + 75)', fontsize=12)
    ax1.axhline(0, color='black', linewidth=1)

    # 次要軸：Net Gamma 曲線
    ax2 = ax1.twinx()
    ax2.plot(df_oi['NQ_Strike'], df_oi['Net Gamma Exposure'], color='blue', linewidth=2.5, label='OI Net Gamma (Trend)')
    ax2.plot(df_vol['NQ_Strike'], df_vol['Net Gamma Exposure'], color='orange', linewidth=2, linestyle='--', label='Vol Net Gamma (Active)')
    ax2.set_ylabel('Net Gamma Exposure', fontsize=12, color='blue')

    # 關鍵水平線
    if oi_cw: ax1.axvline(oi_cw, color='darkgreen', linestyle=':', linewidth=2, label=f'Call Wall: {oi_cw:.0f}')
    if oi_pw: ax1.axvline(oi_pw, color='red', linestyle=':', linewidth=2, label=f'Put Wall: {oi_pw:.0f}')
    if vol_flip: ax1.axvline(vol_flip, color='orange', linestyle='-', linewidth=3, label=f'Daily Pivot: {vol_flip:.0f}')

    # 背景著色
    ax2.fill_between(df_oi['NQ_Strike'], 0, df_oi['Net Gamma Exposure'], where=(df_oi['Net Gamma Exposure'] > 0), color='green', alpha=0.05)
    ax2.fill_between(df_oi['NQ_Strike'], 0, df_oi['Net Gamma Exposure'], where=(df_oi['Net Gamma Exposure'] < 0), color='red', alpha=0.05)

    # 設定顯示範圍 (Pivot 前後 600 點)
    if vol_flip:
        ax1.set_xlim(vol_flip - 600, vol_flip + 600)

    plt.title('NQ Integrated Trade Map: Volume vs. OI Gamma Analysis', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax2.legend(loc='upper right', frameon=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 將圖表秀在網頁上
    st.pyplot(fig)

    # 顯示原始數據 (可選)
    with st.expander("查看原始數據"):
        st.write("Open Interest Data", df_oi)
        st.write("Volume Gamma Data", df_vol)

except FileNotFoundError:
    st.error("找不到 CSV 檔案，請確保檔案位於程式相同目錄下。")
except Exception as e:
    st.error(f"發生錯誤: {e}")
