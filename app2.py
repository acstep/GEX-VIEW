import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 頁面基本設定 ---
st.set_page_config(page_title="ES & NQ 籌碼監控系統", layout="wide")

# 背景淡藍色
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 2px solid #001F3F; margin-top: 40px; }
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

# 基差與 Yahoo 代號設定
CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500 期貨)",
        "ticker": "^SPX", 
        "basis": 17.4, 
        "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指期貨)",
        "ticker": "^NDX", 
        "basis": 57.6, 
        "keywords": ["IUXX", "NQ"]
    }
}
DATA_DIR = "data"

# --- 數據自動化讀取與 Yahoo K 線抓取 ---

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

def fetch_real_kline(ticker, basis):
    """從 Yahoo Finance 抓取真實 15 分鐘 K 線並加上基差"""
    try:
        # 抓取最近 1 個月的 15 分鐘數據 (Yahoo 15m 限制最多抓 60 天)
        data = yf.download(ticker, period="1mo", interval="15m", progress=False)
        if data.empty: return None
        # 換算為期貨點位
        data['Open'] = data['Open'] + basis
        data['High'] = data['High'] + basis
        data['Low'] = data['Low'] + basis
        data['Close'] = data['Close'] + basis
        return data
    except:
        return None

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

# --- 繪圖函式庫 (Plotly 內建互動 TIP) ---

def draw_kline_oi(oi_file, symbol):
    """圖表 1: 真實 15m K線 + 水平 OI 城牆"""
    conf = CONFIG[symbol]
    df_k = fetch_real_kline(conf['ticker'], conf['basis'])
    
    if df_k is None:
        st.warning(f"無法從 Yahoo Finance 抓取 {symbol} 真實數據，請檢查網路。")
        return

    df_oi = clean_data(oi_file, conf['basis'])
    last_p = float(df_k['Close'].iloc[-1])
    
    # 過濾顯示範圍 (以現價上下 2% 為主)
    y_min, y_max = last_p * 0.98, last_p * 1.02
    df_oi_v = df_oi[(df_oi['Strike_Fut'] >= y_min) & (df_oi['Strike_Fut'] <= y_max)]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, 
                        column_widths=[0.75, 0.25], subplot_titles=("Yahoo 15m 真實 K線", "OI 籌碼牆"))

    # 左：K線
    fig.add_trace(go.Candlestick(x=df_k.index, open=df_k['Open'], high=df_k['High'], 
                                 low=df_k['Low'], close=df_k['Close'], name="K線"), row=1, col=1)
    
    # 右：水平 OI (TIP 顯示精確口數)
    fig.add_trace(go.Bar(y=df_oi_v['Strike_Fut'], x=df_oi_v['Call Open Interest']/1e3, orientation='h', 
                         name="Call OI(K)", marker_color=COLORS['positive_bar'], 
                         hovertemplate="執行價: %{y}<br>Call OI: %{x:.1f}K"), row=1, col=2)
    fig.add_trace(go.Bar(y=df_oi_v['Strike_Fut'], x=-df_oi_v['Put Open Interest']/1e3, orientation='h', 
                         name="Put OI(K)", marker_color=COLORS['negative_bar'],
                         hovertemplate="執行價: %{y}<br>Put OI: %{x:.1f}K"), row=1, col=2)

    fig.add_hline(y=last_p, line_dash="dash", line_color=COLORS['last_price_line'], annotation_text=f"期貨現價:{last_p:,.1f}")
    fig.update_layout(height=650, showlegend=False, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def draw_gex_main(gamma_df, symbol):
    """圖表 2: 淨 Gamma 曝險圖 (Barchart 配色)"""
    # 獲取現價
    conf = CONFIG[symbol]
    temp_k = fetch_real_kline(conf['ticker'], conf['basis'])
    last_p = float(temp_k['Close'].iloc[-1]) if temp_k is not None else 0

    flip = find_flip(gamma_df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(x=gamma_df['Strike_Fut'], y=gamma_df['Net Gamma Exposure']/1e8, 
                         name="Net GEX", marker_color=np.where(gamma_df['Net Gamma Exposure']>=0, COLORS['positive_bar'], COLORS['negative_bar']),
                         hovertemplate="執行價: %{x}<br>淨曝險: %{y:.2f} 億"), secondary_y=False)
    
    fig.add_trace(go.Scatter(x=gamma_df['Strike_Fut'], y=gamma_df['Gamma Exposure Profile']/1e9, 
                             name="Aggregate", line=dict(color=COLORS['aggregate_line'], width=4),
                             hovertemplate="移動至此曝險: %{y:.2f}B"), secondary_y=True)

    if flip:
        fig.add_vline(x=flip, line_color=COLORS['flip_line'], line_width=2)
        fig.add_vrect(x0=gamma_df['Strike_Fut'].min(), x1=flip, fillcolor=COLORS['bg_red'], opacity=1, layer="below", line_width=0)
        fig.add_vrect(x0=flip, x1=gamma_df['Strike_Fut'].max(), fillcolor=COLORS['bg_green'], opacity=1, layer="below", line_width=0)

    fig.add_vline(x=last_p, line_color=COLORS['last_price_line'], line_dash="dash")
    fig.update_layout(title=f"<b>{symbol} 淨 Gamma 曝險分佈 (單位：億美元)</b>", height=500, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def draw_details(oi_df, symbol, mode="Gamma"):
    """圖表 3 & 4: 買賣權對比圖"""
    scale = 1e8 if mode == "Gamma" else 1e3
    col_c = "Call Gamma Exposure" if mode == "Gamma" else "Call Open Interest"
    col_p = "Put Gamma Exposure" if mode == "Gamma" else "Put Open Interest"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=oi_df['Strike_Fut'], y=oi_df[col_c]/scale, name="Call", marker_color=COLORS['positive_bar']))
    fig.add_trace(go.Bar(x=oi_df['Strike_Fut'], y=oi_df[col_p]/scale if mode=="Gamma" else -oi_df[col_p]/scale, 
                         name="Put", marker_color=COLORS['negative_bar']))
    
    fig.update_layout(title=f"{symbol} {mode} 買賣權細節對比", height=400, barmode='relative', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- 主程式執行 ---

st.markdown("<h1 style='text-align: center;'>🏹 ES & NQ 真實數據即時監控系統</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"## 📈 {CONFIG[asset]['label']} 分析區塊")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        df_oi = clean_data(oi_f, CONFIG[asset]['basis'])
        df_vol = clean_data(vol_f, CONFIG[asset]['basis'])
        
        # 垂直呈現 4 張圖
        draw_kline_oi(oi_f, asset)
        draw_gex_main(df_vol, asset)
        draw_details(df_oi, asset, mode="Gamma")
        draw_details(df_oi, asset, mode="Open Interest")
    else:
        st.error(f"❌ 請在 data 資料夾中放入 {asset} 的 CSV 數據")
    st.divider()
