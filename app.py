import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面設定
st.set_page_config(page_title="Gamma Map Vertical", layout="wide")

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
            flip = df.iloc[i]['Adjusted_Strike']
            break
    return cw, pw, flip

def create_interactive_plot(df_oi, df_vol, symbol):
    conf = CONFIG[symbol]
    cw, pw, _ = get_levels(df_oi)
    _, _, flip = get_levels(df_vol)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Call OI 柱狀圖
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=df_oi['Call Open Interest'],
        name='Call OI', marker_color=conf['color'], opacity=0.4,
        hovertemplate='Price: %{x}<br>Call OI: %{y}<extra></extra>'
    ), secondary_y=False)

    # Put OI 柱狀圖
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=-df_oi['Put Open Interest'],
        name='Put OI', marker_color='crimson', opacity=0.4,
        hovertemplate='Price: %{x}<br>Put OI: %{y}<extra></extra>'
    ), secondary_y=False)

    # OI Gamma 曲線
    fig.add_trace(go.Scatter(
        x=df_oi['Adjusted_Strike'], y=df_oi['Net Gamma Exposure'],
        name='OI Gamma', line=dict(color='blue', width=2.5),
        hovertemplate='Price: %{x}<br>OI GEX: %{y:.2f}<extra></extra>'
    ), secondary_y=True)

    # Vol Gamma 曲線
    fig.add_trace(go.Scatter(
        x=df_vol['Adjusted_Strike'], y=df_vol['Net Gamma Exposure'],
        name='Vol Gamma', line=dict(color='orange', width=2, dash='dash'),
        hovertemplate='Price: %{x}<br>Vol GEX: %{y:.2f}<extra></extra>'
    ), secondary_y=True)

    # 關鍵位標註
    if cw: fig.add_vline(x=cw, line_dash="dot", line_color="green", annotation_text=f"CW:{cw:.0f}", annotation_position="top left")
    if pw: fig.add_vline(x=pw, line_dash="dot", line_color="red", annotation_text=f"PW:{pw:.0f}", annotation_position="top left")
    if flip: fig.add_vline(x=flip, line_width=2.5, line_color="orange", annotation_text=f"Pivot:{flip:.0f}", annotation_position="bottom right")

    # 版面設定 - 增加高度讓上下看更清晰
    fig.update_layout(
        title=f"<b>{conf['label']} Integrated Gamma Map</b>",
        hovermode="x unified",
        height=550, # 增加高度
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Price Level", range=[flip-conf['range'], flip+conf['range']] if flip else None),
        yaxis=dict(title="Open Interest (Contracts)"),
        yaxis2=dict(title="Net Gamma Exposure", overlaying='y', side='right')
    )
    return fig

# --- 主介面 ---
st.title("📊 市場交易地圖 (上下佈局)")

if not os.path.exists(DATA_DIR):
    st.error(f"找不到目錄: {DATA_DIR}")
else:
    # 按照 SPX -> NQ 的順序垂直排列
    for symbol in ["SPX", "NQ"]:
        oi_file, vol_file = get_latest_files(CONFIG[symbol]['keywords'])
        
        if oi_file and vol_file:
            st.subheader(f"📈 {CONFIG[symbol]['label']} 分析")
            
            df_oi = clean_data(pd.read_csv(oi_file), CONFIG[symbol]['offset'])
            df_vol = clean_data(pd.read_csv(vol_file), CONFIG[symbol]['offset'])
            
            # 指標數據卡片
            cw, pw, _ = get_levels(df_oi)
            _, _, flip = get_levels(df_vol)
            
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            c1.metric("Pivot", f"{flip:.0f}" if flip else "N/A")
            c2.metric("Call Wall", f"{cw:.0f}" if cw else "N/A")
            c3.metric("Put Wall", f"{pw:.0f}" if pw else "N/A")
            c4.caption(f"📅 數據源: {os.path.basename(vol_file)}")

            # 顯示圖表
            fig = create_interactive_plot(df_oi, df_vol, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider() # 加入分隔線
        else:
            st.warning(f"缺少 {symbol} 最新檔案，請檢查 /data 資料夾。")
