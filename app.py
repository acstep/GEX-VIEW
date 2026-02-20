import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面設定
st.set_page_config(page_title="High-Precision Gamma Map", layout="wide")

# 配置設定
CONFIG = {
    "SPX": {
        "label": "ES / SPX (S&P 500)",
        "offset": 0,
        "color": "#1f77b4",
        "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "NQ / NASDAQ 100",
        "offset": 75,
        "color": "#008080",
        "keywords": ["IUXX", "NQ"]
    }
}
DATA_DIR = "data"

# --- 側邊欄：手動調整價格顯示範圍 ---
st.sidebar.header("🔍 顯示範圍設定")
st.sidebar.markdown("若要觀察遠處的買/賣權牆 (如 6900)，請拉大範圍。")
range_spx = st.sidebar.slider("SPX 價格範圍 (+/-)", 50, 2000, 300)
range_nq = st.sidebar.slider("NQ 價格範圍 (+/-)", 100, 3000, 800)
RANGE_MAP = {"SPX": range_spx, "NQ": range_nq}

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
    # 改用最大 OI 尋找牆的位置，這比 GEX 找牆更直觀
    cw = df.loc[df['Call Open Interest'].idxmax(), 'Adjusted_Strike']
    pw = df.loc[df['Put Open Interest'].idxmax(), 'Adjusted_Strike']
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

    # 1. Call OI 柱狀圖 (綠色)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=df_oi['Call Open Interest'],
        name='Call OI (Resistance)', marker_color='#00CC96', opacity=0.5,
        hovertemplate='<b>Call OI: %{y:,.0f}</b><extra></extra>'
    ), secondary_y=False)

    # 2. Put OI 柱狀圖 (紅色)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=-df_oi['Put Open Interest'],
        name='Put OI (Support)', marker_color='#EF553B', opacity=0.5,
        hovertemplate='<b>Put OI: %{y:,.0f}</b><extra></extra>'
    ), secondary_y=False)

    # 3. OI Gamma 曲線 (亮青色，加粗)
    fig.add_trace(go.Scatter(
        x=df_oi['Adjusted_Strike'], y=df_oi['Net Gamma Exposure'],
        name='Net Gamma', line=dict(color='#00FFFF', width=3),
        hovertemplate='Net Gamma: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)

    # 4. Vol Gamma 曲線 (亮橘色虛線)
    fig.add_trace(go.Scatter(
        x=df_vol['Adjusted_Strike'], y=df_vol['Net Gamma Exposure'],
        name='Vol Gamma', line=dict(color='#FFA500', width=2, dash='dash'),
        hovertemplate='Vol Gamma: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)

    # 關鍵位標註 (垂直線)
    if cw: fig.add_vline(x=cw, line_dash="dash", line_color="#00FF00", annotation_text=f"Call Wall: {cw:.0f}")
    if pw: fig.add_vline(x=pw, line_dash="dash", line_color="#FF0000", annotation_text=f"Put Wall: {pw:.0f}")
    if flip: fig.add_vline(x=flip, line_width=3, line_color="#FFFFFF", annotation_text=f"Pivot: {flip:.0f}")

    # 版面設定
    fig.update_layout(
        template="plotly_dark",
        title=f"<b>{conf['label']} 深度數據分析圖</b>",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.9)", font_size=15),
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Price Level", 
            range=[flip-RANGE_MAP[symbol], flip+RANGE_MAP[symbol]] if flip else None,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(title="Open Interest (Contracts)", gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title="Net Gamma Exposure", overlaying='y', side='right', showgrid=False)
    )
    return fig

# --- 主介面 ---
st.title("🏹 專業交易者：市場牆與 Gamma 分佈")

if not os.path.exists(DATA_DIR):
    st.error(f"找不到資料夾: {DATA_DIR}")
else:
    for symbol in ["SPX", "NQ"]:
        oi_file, vol_file = get_latest_files(CONFIG[symbol]['keywords'])
        
        if oi_file and vol_file:
            st.subheader(f"📈 {CONFIG[symbol]['label']}")
            
            df_oi = clean_data(pd.read_csv(oi_file), CONFIG[symbol]['offset'])
            df_vol = clean_data(pd.read_csv(vol_file), CONFIG[symbol]['offset'])
            
            # 指標數據卡片
            cw, pw, _ = get_levels(df_oi)
            _, _, flip = get_levels(df_vol)
            
            m1, m2, m3, m4 = st.columns([1, 1, 1, 2])
            m1.metric("當前轉折 (Pivot)", f"{flip:.0f}")
            m2.metric("阻力牆 (Call Wall)", f"{cw:.0f}")
            m3.metric("支撐牆 (Put Wall)", f"{pw:.0f}")
            m4.caption(f"📅 數據源: {os.path.basename(vol_file)}")

            fig = create_interactive_plot(df_oi, df_vol, symbol)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
        else:
            st.warning(f"缺少 {symbol} 最新檔案。")
