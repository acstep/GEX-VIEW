import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面基本設定
st.set_page_config(page_title="High-Contrast Gamma Map", layout="wide")

# 組態與配色 (使用您要求的鮮豔色系)
CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500)",
        "offset": 0,
        "call_color": "#00FF66", # 鮮豔螢光綠
        "put_color": "#FF007F",  # 鮮豔亮粉紅
        "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指)",
        "offset": 75,
        "call_color": "#00FFFF", # 亮青色
        "put_color": "#FF3131",  # 螢光紅
        "keywords": ["IUXX", "NQ"]
    }
}
DATA_DIR = "data"

# --- 側邊欄控制 ---
st.sidebar.header("🔍 顯示範圍設定")
range_spx = st.sidebar.slider("SPX 觀察範圍 (+/-)", 50, 2000, 400, step=50)
range_nq = st.sidebar.slider("NQ 觀察範圍 (+/-)", 100, 3000, 1000, step=100)
RANGE_MAP = {"SPX": range_spx, "NQ": range_nq}

def get_latest_files(symbol_keywords):
    if not os.path.exists(DATA_DIR): return None, None
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
    cols = ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure', 'Absolute Gamma Exposure']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Adjusted_Strike'] = df['Strike'] + offset
    return df

def get_levels(df):
    if df is None or df.empty: return None, None, None
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

def create_vivid_plot(df_oi, df_vol, symbol):
    conf = CONFIG[symbol]
    cw, pw, _ = get_levels(df_oi)
    _, _, v_flip = get_levels(df_vol)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 看漲 OI (Call) - 加上白色細邊框增加發光感
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=df_oi['Call Open Interest'],
        name='看漲 (Call) OI', 
        marker=dict(color=conf['call_color'], line=dict(width=1, color='white')),
        opacity=0.7,
        hovertemplate='<b>價格: %{x}</b><br>看漲口數: %{y:,.0f}<br><extra></extra>'
    ), secondary_y=False)

    # 2. 看跌 OI (Put)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=-df_oi['Put Open Interest'],
        name='看跌 (Put) OI', 
        marker=dict(color=conf['put_color'], line=dict(width=1, color='white')),
        opacity=0.7,
        hovertemplate='看跌口數: %{y:,.0f}<br><extra></extra>'
    ), secondary_y=False)

    # 3. 淨 Gamma 曲線 (亮青色加粗實線)
    fig.add_trace(go.Scatter(
        x=df_oi['Adjusted_Strike'], y=df_oi['Net Gamma Exposure'],
        name='淨 GEX (趨勢)', 
        line=dict(color='#00FFFF', width=4), 
        hovertemplate='淨 Gamma 值: %{y:,.0f}<br><extra></extra>'
    ), secondary_y=True)

    # 4. 波動 Gamma 曲線 (亮橘色虛線)
    fig.add_trace(go.Scatter(
        x=df_vol['Adjusted_Strike'], y=df_vol['Net Gamma Exposure'],
        name='波動 GEX (動態)', 
        line=dict(color='#FFA500', width=2, dash='dash'), 
        hovertemplate='波動 Gamma: %{y:,.0f}<br><extra></extra>'
    ), secondary_y=True)

    # 關鍵位標註 (垂直線)
    if cw: fig.add_vline(x=cw, line_dash="dash", line_color="#00FF66", line_width=2, annotation_text=f"買權牆: {cw:.0f}")
    if pw: fig.add_vline(x=pw, line_dash="dash", line_color="#FF007F", line_width=2, annotation_text=f"賣權牆: {pw:.0f}")
    if v_flip: fig.add_vline(x=v_flip, line_width=3, line_color="#FFFFFF", annotation_text=f"多空轉折: {v_flip:.0f}")

    # Layout 設定
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified", # 讓滑鼠移到 X 軸時顯示該位置所有數據
        title_text=f"<b>{conf['label']} 詳細 Gamma 數據圖</b>",
        height=600,
        xaxis=dict(
            title="執行價 (Strike)",
            gridcolor='rgba(255,255,255,0.05)',
            range=[v_flip - RANGE_MAP[symbol], v_flip + RANGE_MAP[symbol]] if v_flip else None
        ),
        yaxis=dict(title="未平倉合約口數 (OI)", gridcolor='rgba(255,255,255,0.05)'),
        yaxis2=dict(title="Gamma 曝險值", overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    return fig

# --- 主介面 ---
st.title("🏹 專業交易者分析系統 (ES & NQ)")

if not os.path.exists(DATA_DIR):
    st.error(f"❌ 找不到目錄: {DATA_DIR}")
else:
    for symbol in ["SPX", "NQ"]:
        oi_f, vol_f = get_latest_files(CONFIG[symbol]['keywords'])
        if oi_f and vol_f:
            st.markdown(f"### 📉 {CONFIG[symbol]['label']}")
            df_oi = clean_data(pd.read_csv(oi_f), CONFIG[symbol]['offset'])
            df_vol = clean_data(pd.read_csv(vol_f), CONFIG[symbol]['offset'])
            
            cw, pw, _ = get_levels(df_oi)
            _, _, v_flip = get_levels(df_vol)

            # 指標卡片
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            c1.metric("多空轉折", f"{v_flip:.0f}")
            c2.metric("買權牆 (阻力)", f"{cw:.0f}")
            c3.metric("賣權牆 (支撐)", f"{pw:.0f}")
            c4.info(f"📄 最新檔案: {os.path.basename(vol_f)}")

            # 渲染圖表
            fig = create_vivid_plot(df_oi, df_vol, symbol)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
        else:
            st.warning(f"找不到 {symbol} 的數據。")
