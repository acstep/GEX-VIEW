import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面基本設定
st.set_page_config(page_title="專業級期權分析系統", layout="wide")

# 配置與配色 (鮮艷高對比)
CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500)",
        "offset": 0,
        "call_color": "#00FF66", # 螢光綠
        "put_color": "#FF0066",  # 亮粉紅
        "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指)",
        "offset": 75,
        "call_color": "#00FFFF", # 亮青色
        "put_color": "#FF3333",  # 鮮紅色
        "keywords": ["IUXX", "NQ"]
    }
}
DATA_DIR = "data"

# --- 側邊欄設定 ---
st.sidebar.header("📊 圖表控制面板")
st.sidebar.markdown("---")
range_spx = st.sidebar.slider("SPX 價格觀察範圍", 100, 2000, 500, step=50)
range_nq = st.sidebar.slider("NQ 價格觀察範圍", 200, 3000, 1000, step=100)
RANGE_MAP = {"SPX": range_spx, "NQ": range_nq}
st.sidebar.markdown("---")
st.sidebar.info("💡 提示：滑鼠移至圖表可看詳細數據，雙擊圖表可恢復預設縮放。")

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
    # 確保包含所有需要的資訊欄位
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

    # 1. 看漲 OI (Call)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], 
        y=df_oi['Call Open Interest'],
        name='看漲未平倉 (Call)', 
        marker=dict(color=conf['call_color'], line=dict(width=1.5, color='white')),
        opacity=0.75,
        customdata=df_oi['Net Gamma Exposure'],
        hovertemplate='<b>執行價: %{x}</b><br>' +
                      '看漲口數: %{y:,.0f}<br>' +
                      '<extra></extra>'
    ), secondary_y=False)

    # 2. 看跌 OI (Put)
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], 
        y=-df_oi['Put Open Interest'],
        name='看跌未平倉 (Put)', 
        marker=dict(color=conf['put_color'], line=dict(width=1.5, color='white')),
        opacity=0.75,
        hovertemplate='看跌口數: %{y:,.0f}<br>' +
                      '<extra></extra>'
    ), secondary_y=False)

    # 3. 淨 Gamma 曲線 (亮青色加粗)
    fig.add_trace(go.Scatter(
        x=df_oi['Adjusted_Strike'], 
        y=df_oi['Net Gamma Exposure'],
        name='淨 Gamma 曝險 (Net GEX)', 
        line=dict(color='#00FFFF', width=4), 
        customdata=df_oi['Absolute Gamma Exposure'],
        hovertemplate='淨 Gamma 值: %{y:,.0f}<br>' +
                      '總曝險 (Abs GEX): %{customdata:,.0f}<br>' +
                      '<extra></extra>'
    ), secondary_y=True)

    # 4. 波動 Gamma 曲線 (紫色虛線)
    fig.add_trace(go.Scatter(
        x=df_vol['Adjusted_Strike'], 
        y=df_vol['Net Gamma Exposure'],
        name='動態波動 Gamma', 
        line=dict(color='#CC00FF', width=2, dash='dash'), 
        hovertemplate='動態 Gamma: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)

    # 垂直標註線 (中文化)
    if cw: fig.add_vline(x=cw, line_dash="dash", line_color="#00FF00", line_width=2, 
                         annotation_text=f"看漲牆: {cw:.0f}", annotation_font_color="#00FF00")
    if pw: fig.add_vline(x=pw, line_dash="dash", line_color="#FF0066", line_width=2, 
                         annotation_text=f"看跌牆: {pw:.0f}", annotation_font_color="#FF0066")
    if v_flip: fig.add_vline(x=v_flip, line_width=3, line_color="#FFFFFF", 
                            annotation_text=f"多空轉折: {v_flip:.0f}")

    # 圖表版面配置
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(20,20,20,0.9)", font_size=16, font_family="Microsoft JhengHei"),
        height=600,
        margin=dict(l=60, r=60, t=100, b=60),
        xaxis=dict(
            title="指數價格 / 執行價", 
            titlefont=dict(size=14),
            range=[v_flip - RANGE_MAP[symbol], v_flip + RANGE_MAP[symbol]] if v_flip else None,
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(title="未平倉合約口數 (OI)", titlefont=dict(size=14)),
        yaxis2=dict(title="Gamma 曝險強度", overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=12))
    )
    return fig

# --- 主程式介面 ---
st.title("🏹 專業交易監測系統 (ES & NQ)")

if not os.path.exists(DATA_DIR):
    st.error(f"❌ 找不到數據目錄: {DATA_DIR}")
else:
    # 按照順序垂直繪製
    for symbol in ["SPX", "NQ"]:
        oi_f, vol_f = get_latest_files(CONFIG[symbol]['keywords'])
        if oi_f and vol_f:
            st.markdown(f"### 📈 {CONFIG[symbol]['label']}")
            
            df_oi = clean_data(pd.read_csv(oi_f), CONFIG[symbol]['offset'])
            df_vol = clean_data(pd.read_csv(vol_f), CONFIG[symbol]['offset'])
            
            cw, pw, _ = get_levels(df_oi)
            _, _, v_flip = get_levels(df_vol)

            # 頂部數據看板 (中文化)
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            c1.metric("多空分界 (Pivot)", f"{v_flip:.0f}")
            c2.metric("看漲壓力牆 (CW)", f"{cw:.0f}", delta="阻力區", delta_color="inverse")
            c3.metric("看跌支撐牆 (PW)", f"{pw:.0f}", delta="支撐區")
            c4.info(f"📅 數據同步時間: {pd.to_datetime(os.path.getmtime(vol_f), unit='s').strftime('%Y-%m-%d %H:%M')}\n檔案: {os.path.basename(vol_f)}")

            # 渲染圖表
            st.plotly_chart(create_vivid_plot(df_oi, df_vol, symbol), use_container_width=True)
            st.divider()
        else:
            st.warning(f"⚠️ 找不到 {symbol} 的最新數據，請檢查 /data 資料夾。")
