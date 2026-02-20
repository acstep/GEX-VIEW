import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面設定
st.set_page_config(page_title="Interactive Gamma Map", layout="wide")

CONFIG = {
    "SPX": {"label": "ES / SPX", "offset": 0, "range": 150, "color": "#1f77b4", "keywords": ["SPX", "ES"]},
    "NQ": {"label": "NQ / NASDAQ", "offset": 75, "range": 600, "color": "#008080", "keywords": ["IUXX", "NQ"]}
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
            x1 = df.iloc[i]['Adjusted_Strike']
            flip = x1 # 簡化計算
            break
    return cw, pw, flip

def create_interactive_plot(df_oi, df_vol, symbol):
    conf = CONFIG[symbol]
    cw, pw, _ = get_levels(df_oi)
    _, _, flip = get_levels(df_vol)

    # 建立雙 Y 軸圖表
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Call OI (正柱狀圖)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=df_oi['Call Open Interest'],
        name='Call OI', marker_color=conf['color'], opacity=0.4,
        hovertemplate='Strike: %{x}<br>Call OI: %{y}<extra></extra>'
    ), secondary_y=False)

    # 2. Put OI (負柱狀圖)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=-df_oi['Put Open Interest'],
        name='Put OI', marker_color='crimson', opacity=0.4,
        hovertemplate='Strike: %{x}<br>Put OI: %{y}<extra></extra>'
    ), secondary_y=False)

    # 3. OI Net Gamma (藍色實線)
    fig.add_trace(go.Scatter(
        x=df_oi['Adjusted_Strike'], y=df_oi['Net Gamma Exposure'],
        name='OI Gamma', line=dict(color='blue', width=2),
        hovertemplate='Strike: %{x}<br>Net GEX: %{y:.2f}<extra></extra>'
    ), secondary_y=True)

    # 4. Vol Net Gamma (橘色虛線)
    fig.add_trace(go.Scatter(
        x=df_vol['Adjusted_Strike'], y=df_vol['Net Gamma Exposure'],
        name='Vol Gamma', line=dict(color='orange', width=1.5, dash='dash'),
        hovertemplate='Strike: %{x}<br>Vol GEX: %{y:.2f}<extra></extra>'
    ), secondary_y=True)

    # 標註關鍵位 (垂直線)
    if cw: fig.add_vline(x=cw, line_dash="dot", line_color="green", annotation_text=f"CW:{cw:.0f}")
    if pw: fig.add_vline(x=pw, line_dash="dot", line_color="red", annotation_text=f"PW:{pw:.0f}")
    if flip: fig.add_vline(x=flip, line_width=2, line_color="orange", annotation_text=f"Pivot:{flip:.0f}")

    # 版面設定
    fig.update_layout(
        title=f"{conf['label']} Interactive Map",
        hovermode="x unified", # 同一 X 軸數值全部顯示在一個 Tip
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(range=[flip-conf['range'], flip+conf['range']] if flip else None)
    )
    return fig

# --- 主介面 ---
st.title("💡 互動式 ES & NQ 交易地圖")

if not os.path.exists(DATA_DIR):
    st.error("請建立 /data 資料夾")
else:
    col_left, col_right = st.columns(2)
    cols = {"SPX": col_left, "NQ": col_right}

    for symbol in ["SPX", "NQ"]:
        with cols[symbol]:
            oi_file, vol_file = get_latest_files(CONFIG[symbol]['keywords'])
            if oi_file and vol_file:
                df_oi = clean_data(pd.read_csv(oi_file), CONFIG[symbol]['offset'])
                df_vol = clean_data(pd.read_csv(vol_file), CONFIG[symbol]['offset'])
                
                # 指標顯示
                cw, pw, _ = get_levels(df_oi)
                _, _, flip = get_levels(df_vol)
                st.markdown(f"**{CONFIG[symbol]['label']}** | Pivot: `{flip:.0f}` | CW: `{cw:.0f}` | PW: `{pw:.0f}`")
                
                # 顯示互動圖表
                fig = create_interactive_plot(df_oi, df_vol, symbol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"缺少 {symbol} 檔案")
