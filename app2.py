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

# --- 1. 頁面設定 ---
st.set_page_config(page_title="ES & NQ 60日監控系統", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 10px; margin-top: 50px; }
    .metric-card { text-align:center; background:white; padding:15px; border-radius:15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500)", "ticker": "^SPX", "offset": 17.4,
        "call_color": "#008000", "put_color": "#B22222", "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指)", "ticker": "^NDX", "offset": 57.6,
        "call_color": "#000080", "put_color": "#FF4500", "keywords": ["IUXX", "NQ"]
    }
}
DATA_DIR = "data"

# --- 2. 數據核心函數 ---

def get_latest_files(symbol_keywords):
    """正則識別最新檔案邏輯"""
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
    return (max(oi_f, key=get_sort_key) if oi_f else None, 
            max(vol_f, key=get_sort_key) if vol_f else None)

@st.cache_data(ttl=300)
def fetch_60d_5m_kline(ticker, offset):
    """抓取 Yahoo 5分鐘線的 60 天極限數據"""
    try:
        # 5m 數據在 Yahoo 最多只能抓 60 天
        df = yf.download(ticker, period="60d", interval="5m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df + offset
        # 建立連續標籤
        df['time_label'] = df.index.strftime('%m-%d %H:%M')
        return df
    except:
        return None

def clean_data(filepath, offset):
    df = pd.read_csv(filepath)
    for col in ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Adjusted_Strike'] = df['Strike'] + offset
    return df

# --- 3. 繪圖組件 ---

def draw_60d_kline(df_k, df_oi, symbol):
    """繪製 60 天連續 5m K 線與 OI 牆"""
    last_p = df_k['Close'].iloc[-1]
    y_range = 150 if symbol == "SPX" else 450
    oi_v = df_oi[(df_oi['Adjusted_Strike'] >= last_p - y_range) & (df_oi['Adjusted_Strike'] <= last_p + y_range)]
    
    # 計算適當柱狀寬度
    diff = oi_v['Adjusted_Strike'].diff().median()
    bar_w = (diff if not pd.isna(diff) else 5) * 0.7

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.85, 0.15])
    
    # 5m K線 (Category 軸確保連續)
    fig.add_trace(go.Candlestick(
        x=df_k['time_label'], open=df_k['Open'], high=df_k['High'], 
        low=df_k['Low'], close=df_k['Close'], name="60日 5m K線"
    ), row=1, col=1)
    
    # OI 水平牆
    fig.add_trace(go.Bar(
        y=oi_v['Adjusted_Strike'], x=oi_v['Call Open Interest']/1e3, orientation='h', 
        marker_color=CONFIG[symbol]['call_color'], width=bar_w, name="Call OI",
        hovertemplate="<b>點數: %{y}</b><br>Call OI: %{x:.2f}K<extra></extra>"
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        y=oi_v['Adjusted_Strike'], x=-oi_v['Put Open Interest']/1e3, orientation='h', 
        marker_color=CONFIG[symbol]['put_color'], width=bar_w, name="Put OI",
        hovertemplate="<b>點數: %{y}</b><br>Put OI: %{x:.2f}K<extra></extra>"
    ), row=1, col=2)

    fig.add_hline(y=last_p, line_dash="dash", line_color="#008000", annotation_text="現價")
    
    # 設定 X 軸為分類軸並預設縮放到最近 2 天 (使用者可自行縮放回 60 天)
    total_bars = len(df_k)
    fig.update_xaxes(type='category', nticks=20, range=[total_bars-300, total_bars-1], row=1, col=1)
    
    fig.update_layout(height=800, template="plotly_white", showlegend=False, xaxis_rangeslider_visible=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- 4. 主程式 ---

st.markdown("<h1 style='text-align: center;'>🎯 ES & NQ 60日連續 5m 數據監控</h1>", unsafe_allow_html=True)

for symbol in ["SPX", "NQ"]:
    st.markdown(f"## 📈 {CONFIG[symbol]['label']}")
    oi_f, vol_f = get_latest_files(CONFIG[symbol]['keywords'])
    
    if oi_f:
        df_oi = clean_data(oi_f, CONFIG[symbol]['offset'])
        df_k = fetch_60d_5m_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['offset'])
        
        if df_k is not None:
            # 顯示關鍵點位卡片
            cw_val = df_oi.loc[df_oi['Call Open Interest'].idxmax(), 'Adjusted_Strike']
            pw_val = df_oi.loc[df_oi['Put Open Interest'].idxmax(), 'Adjusted_Strike']
            
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-card'>目前現價<br><b style='font-size:30px;'>{df_k['Close'].iloc[-1]:.2f}</b></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-card'>買權牆 (Call Wall)<br><b style='font-size:30px; color:green;'>{cw_val:.0f}</b></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-card'>賣權牆 (Put Wall)<br><b style='font-size:30px; color:red;'>{pw_val:.0f}</b></div>", unsafe_allow_html=True)
            
            # 繪圖
            draw_60d_kline(df_k, df_oi, symbol)
            st.divider()
        else:
            st.warning(f"正在載入 {symbol} 的 60 天數據中，請稍候...")
    else:
        st.error(f"❌ 找不到 {symbol} 的數據檔案")
