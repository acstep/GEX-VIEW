import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 頁面設定
st.set_page_config(page_title="ES & NQ Gamma Map", layout="wide")

# 設定不同指數的參數 (順序調整：SPX 放在前面)
CONFIG = {
    "SPX": {
        "label": "ES / SPX",
        "offset": 0, 
        "range": 150, 
        "color": "blue", 
        "keywords": ["SPX", "ES"]  # 同時支援 SPX 或 ES 的檔名
    },
    "NQ": {
        "label": "NQ / NASDAQ",
        "offset": 75, 
        "range": 600, 
        "color": "teal", 
        "keywords": ["IUXX", "NQ"]
    }
}

DATA_DIR = "data"  # 指定子目錄名稱

def get_latest_files(symbol_keywords):
    """在 data 子目錄內自動偵測最新的 OI 與 Vol 檔案"""
    search_path = os.path.join(DATA_DIR, "*.csv")
    all_files = glob.glob(search_path)
    
    if not all_files:
        return None, None
    
    # 過濾出符合指數關鍵字的檔案
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    
    if not symbol_files:
        return None, None

    # 區分 OI 檔案（檔名包含 open-interest）與 Vol 檔案
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

def draw_plot(df_oi, df_vol, symbol, oi_path, vol_path):
    conf = CONFIG[symbol]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    oi_cw, oi_pw, _ = get_levels(df_oi)
    _, _, vol_flip = get_levels(df_vol)

    # 繪製 OI 柱狀圖
    width = 20 if "NQ" in symbol else 5
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

    plt.title(f"{conf['label']} Integrated Gamma Map")
    ax1.legend(loc='upper left', fontsize='small')
    ax2.legend(loc='upper right', fontsize='small')
    
    # 顯示檔案資訊
    st.caption(f"📂 數據來源：{os.path.basename(vol_path)} / {os.path.basename(oi_path)}")
    return fig

# --- 主介面 ---
st.title("📊 自動偵測：ES & NQ 交易地圖")
st.markdown(f"目前搜尋目錄：`/{DATA_DIR}`")

# 檢查 data 目錄是否存在
if not os.path.exists(DATA_DIR):
    st.error(f"找不到 `{DATA_DIR}` 資料夾！請建立目錄並上傳 CSV。")
else:
    # 關鍵點：修改迴圈順序，先巡覽 SPX 再巡覽 NQ
    for symbol in ["SPX", "NQ"]:
        st.header(f"📈 {CONFIG[symbol]['label']} 分析")
        
        # 從 data 子目錄偵測最新檔案
        oi_file, vol_file = get_latest_files(CONFIG[symbol]['keywords'])
        
        if oi_file and vol_file:
            try:
                df_oi = clean_data(pd.read_csv(oi_file), CONFIG[symbol]['offset'])
                df_vol = clean_data(pd.read_csv(vol_file), CONFIG[symbol]['offset'])
                
                # 指標卡片
                cw, pw, _ = get_levels(df_oi)
                _, _, flip = get_levels(df_vol)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Pivot", f"{flip:.1f}" if flip else "N/A")
                c2.metric("Call Wall", f"{cw:.0f}" if cw else "N/A")
                c3.metric("Put Wall", f"{pw:.0f}" if pw else "N/A")
                
                # 繪圖
                st.pyplot(draw_plot(df_oi, df_vol, symbol, oi_file, vol_file))
                st.divider()
            except Exception as e:
                st.error(f"解析 {symbol} 數據時發生錯誤: {e}")
        else:
            st.warning(f"在 `{DATA_DIR}` 中找不到符合 {symbol} 關鍵字的最新檔案。")

# 除錯用：顯示 data 目錄內容
with st.expander("📁 查看 /data 資料夾內的所有檔案"):
    if os.path.exists(DATA_DIR):
        st.write(os.listdir(DATA_DIR))
    else:
        st.write("目錄不存在")
