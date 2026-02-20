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
st.set_page_config(page_title="ES & NQ 數據同步監控", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 5px; margin-top: 10px; margin-bottom: 5px; }
    .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

CONFIG = {
    "SPX": {"label": "🇺🇸 ES / SPX (標普 500)", "ticker": "^SPX", "offset": 17.4, "keywords": ["SPX", "ES"], "default_width": 5},
    "NQ": {"label": "💻 NQ / NASDAQ 100 (那指)", "ticker": "^NDX", "offset": 57.6, "keywords": ["IUXX", "NQ"], "default_width": 25}
}
DATA_DIR = "data"

# --- 2. 數據核心函數 ---

def get_latest_files(symbol_keywords):
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
    return (max(oi_f, key=get_sort_key) if oi_f else None, max(vol_f, key=get_sort_key) if vol_f else None)

@st.cache_data(ttl=300)
def fetch_kline_data(ticker, offset):
    try:
        df = yf.download(ticker, period="60d", interval="15m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        # 關鍵：先轉 float 再加基差，避免 Series 運算問題
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col].astype(float) + offset
        df['time_label'] = df.index.strftime('%m-%d %H:%M')
        return df
    except: return None

def clean_csv(filepath, offset):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    cols_to_fix = ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure', 
                   'Call Gamma Exposure', 'Put Gamma Exposure', 'Gamma Exposure Profile']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Adjusted_Strike'] = df['Strike'] + offset
    return df

# --- 3. 繪圖組件 (修正：SPX 視角同步優化) ---

def draw_chart_1_kline(df_k, df_oi, symbol):
    """圖 1: 修正 SPX 點位對齊與縮放，使其與 NDX 視覺同步"""
    last_p = float(df_k['Close'].iloc[-1])
    
    # 重新計算 Y 軸範圍：以現價為中心，參考最大 OI 牆
    cw_idx = df_oi['Call Open Interest'].idxmax()
    pw_idx = df_oi['Put Open Interest'].idxmax()
    wall_min = df_oi.loc[pw_idx, 'Adjusted_Strike']
    wall_max = df_oi.loc[cw_idx, 'Adjusted_Strike']
    
    # 計算視覺區間，確保現價與兩大牆都在視線內
    v_min = min(wall_min, last_p)
    v_max = max(wall_max, last_p)
    
    # 增加 15% 呼吸空間 (使 K 線自然不擁擠)
    padding = (v_max - v_min) * 0.15
    y_min, y_max = v_min - padding, v_max + padding
    
    # 過濾右側 OI 數據
    oi_v = df_oi[(df_oi['Adjusted_Strike'] >= y_min) & (df_oi['Adjusted_Strike'] <= y_max)]
    diff = oi_v['Adjusted_Strike'].diff().median()
    bar_w = (diff if not pd.isna(diff) and diff > 0 else CONFIG[symbol]['default_width']) * 0.75

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0, column_widths=[0.82, 0.18])
    
    # 1. K線圖 (同步配色)
    fig.add_trace(go.Candlestick(
        x=df_k['time_label'], open=df_k['Open'], high=df_k['High'], low=df_k['Low'], close=df_k['Close'],
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
        increasing_fillcolor='#26A69A', decreasing_fillcolor='#EF5350',
        line_width=1, name="15m K線"
    ), row=1, col=1)
    
    # 2. OI 牆 (同步點位)
    fig.add_trace(go.Bar(y=oi_v['Adjusted_Strike'], x=oi_v['Call Open Interest']/1e3, orientation='h', marker_color="#0000FF", width=bar_w), row=1, col=2)
    fig.add_trace(go.Bar(y=oi_v['Adjusted_Strike'], x=-oi_v['Put Open Interest']/1e3, orientation='h', marker_color="#FFA500", width=bar_w), row=1, col=2)
    
    # 設定 Y 軸 (強制同步)
    fig.update_yaxes(range=[y_min, y_max], gridcolor='#E1E1E1', row=1, col=1)
    fig.update_yaxes(range=[y_min, y_max], row=1, col=2)

    total_bars = len(df_k)
    fig.update_xaxes(type='category', range=[max(0, total_bars-150), total_bars-1], row=1, col=1)
    
    fig.update_layout(
        height=620, margin=dict(t=30, b=10), template="plotly_white", 
        showlegend=False, xaxis_rangeslider_visible=False, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- 其餘 6 張圖表保持不變 -----------------

def draw_chart_2_gex(df_vol, last_p, symbol):
    target_col = 'Net Gamma Exposure'
    if target_col not in df_vol.columns: return
    diff = df_vol['Adjusted_Strike'].diff().median()
    bar_w = (diff if not pd.isna(diff) and diff > 0 else CONFIG[symbol]['default_width']) * 0.8
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_vol['Adjusted_Strike'], y=df_vol[target_col]/1e8, marker_color=np.where(df_vol[target_col]>=0, "#0000FF", "#FFA500"), width=bar_w), secondary_y=False)
    if 'Gamma Exposure Profile' in df_vol.columns:
        fig.add_trace(go.Scatter(x=df_vol['Adjusted_Strike'], y=df_vol['Gamma Exposure Profile']/1e9, line=dict(color="#3498db", width=4)), secondary_y=True)
    fig.add_vline(x=last_p, line_dash="dash", line_color="#008000")
    fig.update_layout(height=450, margin=dict(t=50, b=30), template="plotly_white", title=f"<b>{symbol} 淨 Gamma 分佈</b>", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def draw_chart_3_details(df_data, last_p, symbol, mode="Gamma"):
    col_c = f"Call {mode} Exposure" if mode == "Gamma" else "Call Open Interest"
    col_p = f"Put {mode} Exposure" if mode == "Gamma" else "Put Open Interest"
    if col_c not in df_data.columns: return
    scale = 1e8 if mode == "Gamma" else 1e3
    diff = df_data['Adjusted_Strike'].diff().median()
    bar_w = (diff if not pd.isna(diff) and diff > 0 else CONFIG[symbol]['default_width']) * 0.8
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_data['Adjusted_Strike'], y=df_data[col_c]/scale, name="Call", marker_color="#0000FF", width=bar_w))
    fig.add_trace(go.Bar(x=df_data['Adjusted_Strike'], y=df_data[col_p]/scale if mode=="Gamma" else -df_data[col_p]/scale, name="Put", marker_color="#FFA500", width=bar_w))
    fig.add_vline(x=last_p, line_dash="dash", line_color="#008000")
    fig.update_layout(height=400, margin=dict(t=50, b=30), template="plotly_white", barmode='relative', title=f"<b>{symbol} {mode} 買賣權對照</b>", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- 4. 主程式 ---

st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>🎯 ES & NQ 同步監控系統 (15m)</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"## {CONFIG[asset]['label']}")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        df_oi = clean_csv(oi_f, CONFIG[asset]['offset'])
        df_vol = clean_csv(vol_f, CONFIG[asset]['offset'])
        df_k = fetch_kline_data(CONFIG[asset]['ticker'], CONFIG[asset]['offset'])
        
        if df_k is not None:
            last_p = float(df_k['Close'].iloc[-1])
            draw_chart_1_kline(df_k, df_oi, asset)
            draw_chart_2_gex(df_vol, last_p, asset)
            draw_chart_3_details(df_vol, last_p, asset, mode="Gamma")
            draw_chart_3_details(df_oi, last_p, asset, mode="Open Interest")
            st.divider()
