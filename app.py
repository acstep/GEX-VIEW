import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 頁面設定 (強制使用寬螢幕模式)
st.set_page_config(page_title="ES & NQ Gamma Map", layout="wide")

# 設定不同指數的參數 (SPX 在前)
CONFIG = {
    "SPX": {
        "label": "ES / SPX",
        "offset": 0, 
        "range": 150, 
        "color": "blue", 
        "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "NQ / NASDAQ",
        "offset": 75, 
        "range": 600, 
        "color": "teal", 
        "keywords": ["IUXX", "NQ"]
    }
}

DATA_DIR = "data"

def get_latest_files(symbol_keywords):
    search_path = os.path.join(DATA_DIR, "*.csv")
    all_files = glob.glob(search_path)
    if not all_files: return None, None
    
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None

    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    
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
    # 縮小圖表尺寸以利並排 (高度縮減)
    fig, ax1 = plt.subplots(figsize=(8, 5)) 
    
    oi_cw, oi_pw, _ = get_levels(df_oi)
    _, _, vol_flip = get_levels(df_vol)

    width = 20 if "NQ" in symbol else 5
    ax1.bar(df_oi['Adjusted_Strike'], df_oi['Call Open Interest'], width=width, color=conf['color'], alpha=0.3, label='Call OI')
    ax1.bar(df_oi['Adjusted_Strike'], -df_oi['Put Open Interest'], width=width, color='crimson', alpha=0.3, label='Put OI')
    
    ax2 = ax1.twinx()
    ax2.plot(df_oi['Adjusted_Strike'], df_oi['Net Gamma Exposure'], color='blue', linewidth=1.5, label='OI Gamma')
    ax2.plot(df_vol['Adjusted_Strike'], df_vol['Net Gamma Exposure'], color='orange', linestyle='--', linewidth=1.2, label='Vol Gamma')

    if oi_cw: ax1.axvline(oi_cw, color='green', linestyle=':', linewidth=1, label=f'CW:{oi_cw:.0f}')
    if oi_pw: ax1.axvline(oi_pw, color='red', linestyle=':', linewidth=1, label=f'PW:{oi_pw:.0f}')
    if vol_flip: ax1.axvline(vol_flip, color='orange', linewidth=1.5, label=f'Piv:{vol_flip:.1f}')

    if vol_flip:
        ax1.set_xlim(vol_flip - conf['range'], vol_flip + conf['range'])

    # 縮小字體以防並排時擠壓
    plt.title(f"{conf['label']} Gamma Map", fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.legend(loc='upper left', fontsize='x-small', framealpha=0.5)
    ax2.legend(loc='upper right', fontsize='x-small', framealpha=0.5)
    
    return fig

# --- 主介面 ---
st.title("📊 ES & NQ 交易地圖 (並排模式)")

if not os.path.exists(DATA_DIR):
    st.error(f"找不到 `{DATA_DIR}` 資料夾！")
else:
    # 建立左右兩欄
    col_left, col_right = st.columns(2)
    cols = {"SPX": col_left, "NQ": col_right}

    for symbol in ["SPX", "NQ"]:
        with cols[symbol]:
            st.subheader(f"📈 {CONFIG[symbol]['label']}")
            
            oi_file, vol_file = get_latest_files(CONFIG[symbol]['keywords'])
            
            if oi_file and vol_file:
                try:
                    df_oi = clean_data(pd.read_csv(oi_file), CONFIG[symbol]['offset'])
                    df_vol = clean_data(pd.read_csv(vol_file), CONFIG[symbol]['offset'])
                    
                    # 關鍵指標 (用較小的小卡片顯示)
                    cw, pw, _ = get_levels(df_oi)
                    _, _, flip = get_levels(df_vol)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.caption(f"Pivot: **{flip:.1f}**" if flip else "Pivot: N/A")
                    m2.caption(f"Call Wall: **{cw:.0f}**" if cw else "CW: N/A")
                    m3.caption(f"Put Wall: **{pw:.0f}**" if pw else "PW: N/A")
                    
                    # 繪圖
                    fig = draw_plot(df_oi, df_vol, symbol, oi_file, vol_file)
                    st.pyplot(fig, use_container_width=True)
                    
                    st.caption(f"📄 {os.path.basename(vol_file)}")
                except Exception as e:
                    st.error(f"錯誤: {e}")
            else:
                st.warning(f"缺少 {symbol} 檔案")

# 底部檔案清單 (收納起來)
with st.expander("📁 檔案管理"):
    if os.path.exists(DATA_DIR):
        st.write(os.listdir(DATA_DIR))
