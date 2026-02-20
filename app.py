import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 頁面設定
st.set_page_config(page_title="Index Gamma Auto-Detector", layout="wide")

CONFIG = {
    "NQ": {"offset": 75, "range": 600, "color": "teal", "keywords": ["IUXX", "NQ"]},
    "SPX": {"offset": 0, "range": 150, "color": "blue", "keywords": ["SPX"]}
}

def get_latest_files(symbol_keywords):
    """自動偵測資料夾內最新的 OI 與 Vol 檔案"""
    all_files = glob.glob("*.csv")
    
    # 過濾出符合指數關鍵字的檔案
    symbol_files = [f for f in all_files if any(k.upper() in f.upper() for k in symbol_keywords)]
    
    # 區分 OI 檔案與 Vol 檔案
    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    
    # 根據檔案修改時間排序，取最新的一個
    latest_oi = max(oi_files, key=os.path.getmtime) if oi_files else None
    latest_vol = max(vol_files, key=os.path.getmtime) if vol_files else None
    
    return latest_oi, latest_vol

def clean_data(df, offset):
    cols = ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Adjusted_Strike'] = df['Strike'] + offset
    return df

def get_levels(df):
    if df is None or df.empty: return None, None, None
    cw = df.loc[df['Net Gamma Exposure'].idxmax(), 'Adjusted_Strike']
    pw = df.loc[df['Net Gamma Exposure'].idxmin(), 'Adjusted_Strike']
    flip = None
    for i in range(len(df)-1):
        y1, y2 = df.iloc[i]['Net Gamma Exposure'], df.iloc[i+1]['Net Gamma Exposure']
        if pd.isna(y1) or pd.isna(y2): continue
        if y1 * y2 <= 0:
            x1, x2 = df.iloc[i]['Adjusted_Strike'], df.iloc[i+1]['Adjusted_Strike']
            if y2 != y1:
                flip = x1 - y1 * (x2 - x1) / (y2 - y1)
                break
    return cw, pw, flip

def draw_plot(df_oi, df_vol, symbol, oi_name, vol_name):
    conf = CONFIG[symbol]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    oi_cw, oi_pw, _ = get_levels(df_oi)
    _, _, vol_flip = get_levels(df_vol)

    # 繪製 OI 柱狀圖
    width = 20 if symbol == "NQ" else 5
    ax1.bar(df_oi['Adjusted_Strike'], df_oi['Call Open Interest'], width=width, color=conf['color'], alpha=0.3, label='Call OI')
    ax1.bar(df_oi['Adjusted_Strike'], -df_oi['Put Open Interest'], width=width, color='crimson', alpha=0.3, label='Put OI')
    
    # 繪製 Gamma 曲線
    ax2 = ax1.twinx()
    ax2.plot(df_oi['Adjusted_Strike'], df_oi['Net Gamma Exposure'], color='blue', linewidth=2, label='OI Gamma (Trend)')
    ax2.plot(df_vol['Adjusted_Strike'], df_vol['Net Gamma Exposure'], color='orange', linestyle='--', label='Vol Gamma (Active)')

    # 標註關鍵位
    if oi_cw: ax1.axvline(oi_cw, color='green', linestyle=':', label=f'Call Wall: {oi_cw:.0f}')
    if oi_pw: ax1.axvline(oi_pw, color='red', linestyle=':', label=f'Put Wall: {oi_pw:.0f}')
    if vol_flip: ax1.axvline(vol_flip, color='orange', linewidth=2, label=f'Pivot: {vol_flip:.1f}')

    if vol_flip:
        ax1.set_xlim(vol_flip - conf['range'], vol_flip + conf['range'])

    plt.title(f"{symbol} Gamma Map (Auto-Detected)")
    ax1.legend(loc='upper left', fontsize='small')
    ax2.legend(loc='upper right', fontsize='small')
    
    # 在圖表下方註記使用的檔案名稱
    st.caption(f"數據來源: {vol_name} / {oi_name}")
    return fig

# --- 主程式 ---
st.title("🚀 自動偵測：NQ & SPX 交易地圖")
st.markdown("程式會自動抓取資料夾內最新的 `.csv` 檔案進行分析。")

for symbol in ["NQ", "SPX"]:
    st.header(f"📈 {symbol} 即時分析")
    
    # 自動偵測最新檔案
    oi_file, vol_file = get_latest_files(CONFIG[symbol]['keywords'])
    
    if oi_file and vol_file:
        try:
            df_oi = clean_data(pd.read_csv(oi_file), CONFIG[symbol]['offset'])
            df_vol = clean_data(pd.read_csv(vol_file), CONFIG[symbol]['offset'])
            
            # 顯示資訊卡片
            cw, pw, _ = get_levels(df_oi)
            _, _, flip = get_levels(df_vol)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Pivot", f"{flip:.1f}" if flip else "N/A")
            c2.metric("Call Wall", f"{cw:.0f}" if cw else "N/A")
            c3.metric("Put Wall", f"{pw:.0f}" if pw else "N/A")
            
            # 畫圖
            st.pyplot(draw_plot(df_oi, df_vol, symbol, oi_file, vol_file))
            st.divider()
        except Exception as e:
            st.error(f"解析 {symbol} 數據時發生錯誤: {e}")
    else:
        st.warning(f"找不到符合 {symbol} 關鍵字的最新檔案。")

# 顯示目前資料夾內的所有 CSV (除錯用)
with st.expander("📁 查看資料夾內所有檔案"):
    st.write(os.listdir("."))
