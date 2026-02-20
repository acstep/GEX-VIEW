import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gamma Map - Advanced View", layout="wide")

# 設定參數
CONFIG = {
    "SPX": {"label": "ES / SPX", "offset": 0, "default_range": 300, "color": "#1f77b4", "keywords": ["SPX", "ES"]},
    "NQ": {"label": "NQ / NASDAQ", "offset": 75, "default_range": 800, "color": "#008080", "keywords": ["IUXX", "NQ"]}
}
DATA_DIR = "data"

# --- 側邊欄控制 ---
st.sidebar.header("圖表設定")
view_range_spx = st.sidebar.slider("SPX 顯示範圍 (+/-)", 100, 1500, 500)
view_range_nq = st.sidebar.slider("NQ 顯示範圍 (+/-)", 200, 2000, 800)
range_map = {"SPX": view_range_spx, "NQ": view_range_nq}

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
    cw = df.loc[df['Call Open Interest'].idxmax(), 'Adjusted_Strike'] # 改用最大 OI 找牆
    pw = df.loc[df['Put Open Interest'].idxmax(), 'Adjusted_Strike']
    
    # 計算 Pivot (Gamma Flip)
    flip = None
    for i in range(len(df)-1):
        y1, y2 = df.iloc[i]['Net Gamma Exposure'], df.iloc[i+1]['Net Gamma Exposure']
        if pd.isna(y1) or pd.isna(y2): continue
        if y1 * y2 <= 0:
            flip = df.iloc[i]['Adjusted_Strike']
            break
    return cw, pw, flip

def create_plot(df_oi, df_vol, symbol):
    conf = CONFIG[symbol]
    cw, pw, flip = get_levels(df_oi)
    _, _, vol_flip = get_levels(df_vol)
    target_flip = vol_flip if vol_flip else flip

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Call OI 柱狀圖 (加上精確數字提示)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=df_oi['Call Open Interest'],
        name='Call OI', marker_color=conf['color'], opacity=0.5,
        hovertemplate='<b>價格: %{x}</b><br>看漲口數: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)

    # Put OI 柱狀圖
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=-df_oi['Put Open Interest'],
        name='Put OI', marker_color='crimson', opacity=0.5,
        hovertemplate='<b>價格: %{x}</b><br>看跌口數: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)

    # Net Gamma 曲線
    fig.add_trace(go.Scatter(
        x=df_oi['Adjusted_Strike'], y=df_oi['Net Gamma Exposure'],
        name='Net Gamma', line=dict(color='yellow', width=2),
        hovertemplate='價格: %{x}<br>Gamma值: %{y:.2f}<extra></extra>'
    ), secondary_y=True)

    # 標註 Call Wall
    if cw:
        fig.add_vline(x=cw, line_dash="dash", line_color="lime", line_width=2)
        fig.add_annotation(x=cw, y=1, yref="paper", text=f"主力牆(Call): {cw:.0f}", showarrow=False, bgcolor="green", font=dict(color="white"))

    # 設定顯示範圍 (使用側邊欄的滑桿值)
    current_range = range_map[symbol]
    fig.update_layout(
        title=f"<b>{conf['label']} 互動交易地圖</b>",
        hovermode="x unified",
        height=600,
        xaxis=dict(range=[target_flip - current_range, target_flip + current_range] if target_flip else None),
        template="plotly_dark" # 使用深色模式讓顏色更顯眼
    )
    return fig

# --- 主程式 ---
st.title("📈 專業 Gamma 牆監測站")

for symbol in ["SPX", "NQ"]:
    oi_file, vol_file = get_latest_files(CONFIG[symbol]['keywords'])
    if oi_file and vol_file:
        df_oi = clean_data(pd.read_csv(oi_file), CONFIG[symbol]['offset'])
        df_vol = clean_data(pd.read_csv(vol_file), CONFIG[symbol]['offset'])
        
        cw, pw, flip = get_levels(df_oi)
        _, _, v_flip = get_levels(df_vol)

        st.subheader(f"{CONFIG[symbol]['label']}")
        
        # 指標欄位
        c1, c2, c3 = st.columns(3)
        c1.metric("當前轉折 (Pivot)", f"{v_flip:.0f}")
        c2.metric("最大阻力 (Call Wall)", f"{cw:.0f}")
        c3.metric("最大支撐 (Put Wall)", f"{pw:.0f}")

        st.plotly_chart(create_plot(df_oi, df_vol, symbol), use_container_width=True)
        st.divider()
