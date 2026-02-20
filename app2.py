import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 頁面基本設定 ---
st.set_page_config(page_title="ES & NQ 籌碼監控系統", layout="wide")

# 背景與字體優化
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 2px solid #001F3F; }
    </style>
    """, unsafe_allow_html=True)

# 完全複製 Barchart 色彩代碼
COLORS = {
    "positive_bar": "#0000FF",      # 正 Gamma 藍
    "negative_bar": "#FFA500",      # 負 Gamma 橘
    "aggregate_line": "#3498db",    # 累計曲線亮藍
    "flip_line": "#FF0000",         # Flip 紅線
    "last_price_line": "#008000",   # 現價綠線
    "bg_green": "rgba(0, 255, 0, 0.05)", # 正 Gamma 綠區背景
    "bg_red": "rgba(255, 0, 0, 0.05)",   # 負 Gamma 紅區背景
}

# 基差設定 (2026-02-20)
CONFIG = {
    "SPX": {"label": "ES / SPX (標普 500)", "basis": 17.4, "keywords": ["SPX", "ES"], "last_idx": 6861.89},
    "NQ": {"label": "NQ / NASDAQ 100 (那指)", "basis": 57.6, "keywords": ["IUXX", "NQ"], "last_idx": 24797.17}
}
DATA_DIR = "data"

# --- 數據自動化讀取 ---

def get_latest_files(symbol_keywords):
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files: return None, None
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None
    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    return (max(oi_files, key=os.path.getmtime) if oi_files else None, 
            max(vol_files, key=os.path.getmtime) if vol_files else None)

def clean_data(filepath, basis=0):
    df = pd.read_csv(filepath)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Strike_Fut'] = df['Strike'] + basis
    return df

def find_flip(df):
    if 'Gamma Exposure Profile' not in df.columns: return None
    profile = df['Gamma Exposure Profile'].values
    strikes = df['Strike_Fut'].values
    for i in range(len(profile) - 1):
        if not np.isnan(profile[i]) and not np.isnan(profile[i+1]):
            if profile[i] * profile[i+1] <= 0:
                return strikes[i]
    return None

# --- 繪圖函式 (內建 TIP 提示功能) ---

def draw_kline_oi(oi_file, fut_price, symbol):
    """15m K線與水平 OI 城牆"""
    np.random.seed(100 if symbol=="SPX" else 42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
    path = np.cumsum(np.random.normal(0, 5 if symbol=="SPX" else 20, 100)) + fut_price
    
    df_oi = clean_data(oi_file, CONFIG[symbol]['basis'])
    y_min, y_max = fut_price * 0.98, fut_price * 1.02
    df_oi_v = df_oi[(df_oi['Strike_Fut'] >= y_min) & (df_oi['Strike_Fut'] <= y_max)]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.75, 0.25])

    # K線
    fig.add_trace(go.Candlestick(x=dates, open=path-2, high=path+4, low=path-4, close=path, name="K線"), row=1, col=1)
    
    # 水平 OI 牆 (TIP 顯示精確口數)
    fig.add_trace(go.Bar(y=df_oi_v['Strike_Fut'], x=df_oi_v['Call Open Interest']/1e3, orientation='h', 
                         name="Call OI(K)", marker_color=COLORS['positive_bar'], 
                         hovertemplate="執行價: %{y}<br>Call OI: %{x:.1f}K"), row=1, col=2)
    fig.add_trace(go.Bar(y=df_oi_v['Strike_Fut'], x=-df_oi_v['Put Open Interest']/1e3, orientation='h', 
                         name="Put OI(K)", marker_color=COLORS['negative_bar'],
                         hovertemplate="執行價: %{y}<br>Put OI: %{x:.1f}K"), row=1, col=2)

    fig.update_layout(height=500, showlegend=False, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def draw_gex_main(gamma_df, fut_price, symbol):
    """淨 Gamma 曝險圖 (完全複製 Barchart 配色)"""
    flip = find_flip(gamma_df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. 柱狀圖 (GEX 億美元)
    fig.add_trace(go.Bar(x=gamma_df['Strike_Fut'], y=gamma_df['Net Gamma Exposure']/1e8, 
                         name="Net GEX", marker_color=np.where(gamma_df['Net Gamma Exposure']>=0, COLORS['positive_bar'], COLORS['negative_bar']),
                         hovertemplate="執行價: %{x}<br>淨曝險: %{y:.2f} 億"), secondary_y=False)
    
    # 2. 累計曲線 (S 曲線)
    fig.add_trace(go.Scatter(x=gamma_df['Strike_Fut'], y=gamma_df['Gamma Exposure Profile']/1e9, 
                             name="Aggregate", line=dict(color=COLORS['aggregate_line'], width=4),
                             hovertemplate="移動至此曝險: %{y:.2f}B"), secondary_y=True)

    # 3. 背景與線條
    if flip:
        fig.add_vline(x=flip, line_color=COLORS['flip_line'], line_width=2)
        fig.add_vrect(x0=gamma_df['Strike_Fut'].min(), x1=flip, fillcolor=COLORS['bg_red'], opacity=1, layer="below", line_width=0)
        fig.add_vrect(x0=flip, x1=gamma_df['Strike_Fut'].max(), fillcolor=COLORS['bg_green'], opacity=1, layer="below", line_width=0)

    fig.add_vline(x=fut_price, line_color=COLORS['last_price_line'], line_dash="dash")
    fig.update_layout(title=f"<b>{symbol} 淨 Gamma 曝險 (完全複製 Barchart 風格)</b>", height=500, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def draw_details(oi_df, fut_price, symbol, mode="Gamma"):
    """買賣權對比圖 (維持顏色不變)"""
    scale = 1e8 if mode == "Gamma" else 1e3
    col_c = "Call Gamma Exposure" if mode == "Gamma" else "Call Open Interest"
    col_p = "Put Gamma Exposure" if mode == "Gamma" else "Put Open Interest"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=oi_df['Strike_Fut'], y=oi_df[col_c]/scale, name="Call", marker_color=COLORS['positive_bar']))
    fig.add_trace(go.Bar(x=oi_df['Strike_Fut'], y=oi_df[col_p]/scale if mode=="Gamma" else -oi_df[col_p]/scale, 
                         name="Put", marker_color=COLORS['negative_bar']))
    
    fig.add_vline(x=fut_price, line_color=COLORS['last_price_line'], line_dash="dash")
    fig.update_layout(title=f"{symbol} {mode} 買賣權細節對比", height=400, barmode='relative', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- 執行流程 ---

st.markdown("<h1 style='text-align: center;'>🎯 專業期貨籌碼實時監控系統</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"## 📈 {CONFIG[asset]['label']} 分析區塊")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        fut_p = CONFIG[asset]['last_idx'] + CONFIG[asset]['basis']
        df_oi = clean_data(oi_f, CONFIG[asset]['basis'])
        df_vol = clean_data(vol_f, CONFIG[asset]['basis'])
        
        # 垂直呈現 4 張圖
        draw_kline_oi(oi_f, fut_p, asset)
        draw_gex_main(df_vol, fut_p, asset)
        draw_details(df_oi, fut_p, asset, mode="Gamma")
        draw_details(df_oi, fut_p, asset, mode="OI")
    else:
        st.error(f"❌ DATA 子目錄中缺少 {asset} 的 CSV 數據")
    st.divider()
