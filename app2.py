import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="專業級期貨籌碼全方位監控", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 10px; margin-top: 50px; }
    .metric-card { text-align:center; background:white; padding:15px; border-radius:15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 專業配色
COLORS = {
    "pos_bar": "#0000FF", "neg_bar": "#FFA500", "agg_line": "#3498db",
    "flip_line": "#FF0000", "price_line": "#008000",
}

CONFIG = {
    "SPX": {"label": "🇺🇸 ES / SPX (標普 500)", "ticker": "^SPX", "offset": 17.4, "keywords": ["SPX", "ES"]},
    "NQ": {"label": "💻 NQ / NASDAQ 100 (那指)", "ticker": "^NDX", "offset": 57.6, "keywords": ["IUXX", "NQ"]}
}
DATA_DIR = "data"
loaded_log = []

# --- 2. 數據核心函數 ---

def get_latest_files(symbol_keywords):
    """正則識別最新檔案邏輯：日期 > 版本號 > 時間"""
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None

    def get_sort_key(f):
        fname = os.path.basename(f)
        date_match = re.search(r'(\d{8})', fname)
        date_str = date_match.group(1) if date_match else "00000000"
        suffix_match = re.search(r'-(\d+)\.csv$', fname)
        suffix_val = int(suffix_match.group(1)) if suffix_match else 0
        return (date_str, suffix_val, os.path.getmtime(f))

    oi_f = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_f = [f for f in symbol_files if "open-interest" not in f.lower()]
    
    latest_oi = max(oi_f, key=get_sort_key) if oi_f else None
    latest_vol = max(vol_f, key=get_sort_key) if vol_f else None
    return latest_oi, latest_vol

@st.cache_data(ttl=300)
def fetch_60d_5m_kline(ticker, offset):
    """抓取 60 天 5 分鐘線極限數據"""
    try:
        df = yf.download(ticker, period="60d", interval="5m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df + offset
        df['time_label'] = df.index.strftime('%m-%d %H:%M')
        return df
    except: return None

def clean_csv(filepath, offset):
    df = pd.read_csv(filepath)
    for col in ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure', 'Call Gamma Exposure', 'Put Gamma Exposure', 'Gamma Exposure Profile']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Adjusted_Strike'] = df['Strike'] + offset
    return df

# --- 3. 繪圖組件 (8張圖的核心函數) ---

def draw_chart_1_kline(df_k, df_oi, symbol):
    """圖 1: 60日 5m 連續 K 線 + OI 牆"""
    last_p = df_k['Close'].iloc[-1]
    y_range = 100 if symbol == "SPX" else 350
    oi_v = df_oi[(df_oi['Adjusted_Strike'] >= last_p - y_range) & (df_oi['Adjusted_Strike'] <= last_p + y_range)]
    bar_w = (oi_v['Adjusted_Strike'].diff().median() if not oi_v.empty else 5) * 0.7
    
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=df_k['time_label'], open=df_k['Open'], high=df_k['High'], low=df_k['Low'], close=df_k['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Bar(y=oi_v['Adjusted_Strike'], x=oi_v['Call Open Interest']/1e3, orientation='h', marker_color="#0000FF", width=bar_w, hovertemplate="點數: %{y}<br>Call OI: %{x:.1f}K"), row=1, col=2)
    fig.add_trace(go.Bar(y=oi_v['Adjusted_Strike'], x=-oi_v['Put Open Interest']/1e3, orientation='h', marker_color="#FFA500", width=bar_w, hovertemplate="點數: %{y}<br>Put OI: %{x:.1f}K"), row=1, col=2)
    
    total_bars = len(df_k)
    fig.update_xaxes(type='category', nticks=15, range=[total_bars-200, total_bars-1], row=1, col=1)
    fig.update_layout(height=650, template="plotly_white", showlegend=False, xaxis_rangeslider_visible=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def draw_chart_2_gex(df_vol, last_p, symbol):
    """圖 2: 淨 Gamma 曝險與累計曲線"""
    bar_w = (df_vol['Adjusted_Strike'].diff().median() if not df_vol.empty else 5) * 0.7
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_vol['Adjusted_Strike'], y=df_vol['Net Gamma Exposure']/1e8, marker_color=np.where(df_vol['Net Gamma Exposure']>=0, "#0000FF", "#FFA500"), width=bar_w, hovertemplate="點數: %{x}<br>淨GEX: %{y:.2f}億"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_vol['Adjusted_Strike'], y=df_vol['Gamma Exposure Profile']/1e9, line=dict(color="#3498db", width=4), hovertemplate="點數: %{x}<br>累計: %{y:.2f}B"), secondary_y=True)
    fig.add_vline(x=last_p, line_dash="dash", line_color="#008000")
    fig.update_layout(height=450, template="plotly_white", title=f"<b>{symbol} 淨 Gamma 與累計曲線</b>", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def draw_chart_3_details(df_oi, last_p, symbol, mode="Gamma"):
    """圖 3 & 4: 買賣權細節對比 (Gamma 或 OI)"""
    scale = 1e8 if mode == "Gamma" else 1e3
    unit = "億" if mode == "Gamma" else "K"
    col_c = f"Call {mode} Exposure" if mode == "Gamma" else "Call Open Interest"
    col_p = f"Put {mode} Exposure" if mode == "Gamma" else "Put Open Interest"
    
    bar_w = (df_oi['Adjusted_Strike'].diff().median() if not df_oi.empty else 5) * 0.7
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_oi['Adjusted_Strike'], y=df_oi[col_c]/scale, name="Call", marker_color="#0000FF", width=bar_w, hovertemplate=f"點數: %{{x}}<br>Call {mode}: %{{y:.2f}}{unit}"))
    fig.add_trace(go.Bar(x=df_oi['Adjusted_Strike'], y=df_oi[col_p]/scale if mode=="Gamma" else -df_oi[col_p]/scale, name="Put", marker_color="#FFA500", width=bar_w, hovertemplate=f"點數: %{{x}}<br>Put {mode}: %{{y:.2f}}{unit}"))
    fig.add_vline(x=last_p, line_dash="dash", line_color="#008000")
    fig.update_layout(height=400, template="plotly_white", barmode='relative', title=f"<b>{symbol} {mode} 買賣權對照</b>", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- 4. 主程式執行 ---

st.markdown("<h1 style='text-align: center;'>🎯 ES & NQ 全方位 8 圖監控系統 (60日 5m)</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"---")
    st.markdown(f"## {CONFIG[asset]['label']}")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        loaded_log.append(os.path.basename(oi_f))
        loaded_log.append(os.path.basename(vol_f))
        
        df_oi = clean_csv(oi_f, CONFIG[asset]['offset'])
        df_vol = clean_csv(vol_f, CONFIG[asset]['offset'])
        df_k = fetch_60d_5m_kline(CONFIG[asset]['ticker'], CONFIG[asset]['offset'])
        
        if df_k is not None:
            last_p = df_k['Close'].iloc[-1]
            # 1. K線牆
            draw_chart_1_kline(df_k, df_oi, asset)
            # 2. 淨 GEX
            draw_chart_2_gex(df_vol, last_p, asset)
            # 3. Gamma 細節
            draw_chart_3_details(df_oi, last_p, asset, mode="Gamma")
            # 4. OI 細節
            draw_chart_3_details(df_oi, last_p, asset, mode="Open Interest")
    else:
        st.error(f"❌ 找不到 {asset} 的數據檔案")

# 底部溯源
if loaded_log:
    st.markdown("### 📂 本次數據源明細：")
    for f in sorted(list(set(loaded_log))): st.code(f)
