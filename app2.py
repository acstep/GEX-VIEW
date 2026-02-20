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
st.set_page_config(page_title="專業級 GEX 監測系統", layout="wide")

# 背景淡藍色 CSS
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 10px; margin-top: 50px; }
    .metric-card { text-align:center; background:white; padding:15px; border-radius:15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

CONFIG = {
    "SPX": {
        "label": "🇺🇸 ES / SPX (標普 500)",
        "ticker": "^SPX",
        "offset": 0,
        "call_color": "#008000", 
        "put_color": "#B22222",  
        "bar_width": 4,          
        "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "💻 NQ / NASDAQ 100 (那指)",
        "ticker": "^NDX",
        "offset": 75,
        "call_color": "#000080", 
        "put_color": "#FF4500",  
        "bar_width": 40,         
        "keywords": ["IUXX", "NQ"]
    }
}
DATA_DIR = "data"
read_files_list = []

# --- 2. 核心邏輯：正則表達式讀檔 ---

def get_latest_files(symbol_keywords):
    """
    採用您提供的邏輯：日期 (YYYYMMDD) > 版本編號 (-1, -2) > 修改時間
    """
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files: return None, None
    
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None

    def get_sort_key(f):
        fname = os.path.basename(f)
        # 1. 抓取日期 (YYYYMMDD)
        date_match = re.search(r'(\d{8})', fname)
        date_str = date_match.group(1) if date_match else "00000000"
        # 2. 抓取版本編號 (例如 -2)
        suffix_match = re.search(r'-(\d+)\.csv$', fname)
        suffix_val = int(suffix_match.group(1)) if suffix_match else 0
        return (date_str, suffix_val, os.path.getmtime(f))

    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    
    latest_oi = max(oi_files, key=get_sort_key) if oi_files else None
    latest_vol = max(vol_files, key=get_sort_key) if vol_files else None
    
    return latest_oi, latest_vol

@st.cache_data(ttl=60)
def fetch_yahoo_kline(ticker, offset):
    """抓取 Yahoo 5m 連續數據"""
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty: return None
        if df.columns.nlevels > 1: df.columns = df.columns.get_level_values(0)
        df = df + offset
        df['time_label'] = df.index.strftime('%m-%d %H:%M')
        return df
    except: return None

def clean_data(filepath, offset):
    df = pd.read_csv(filepath)
    cols = ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure', 'Gamma Exposure Profile']
    for col in cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Adjusted_Strike'] = df['Strike'] + offset
    if 'Net Gamma Exposure' in df.columns:
        df['Net_GEX_Yi'] = df['Net Gamma Exposure'] / 1e8
    return df

# --- 3. 繪圖組件 ---

def draw_kline_with_oi(df_k, df_oi, symbol):
    """圖 1: 連續 K 線 + OI 水平牆 (無空隙版)"""
    last_p = df_k['Close'].iloc[-1]
    y_range = 150 if symbol == "SPX" else 450
    oi_v = df_oi[(df_oi['Adjusted_Strike'] >= last_p - y_range) & (df_oi['Adjusted_Strike'] <= last_p + y_range)]
    
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=df_k['time_label'], open=df_k['Open'], high=df_k['High'], low=df_k['Low'], close=df_k['Close'], name="5m K線"), row=1, col=1)
    
    fig.add_trace(go.Bar(y=oi_v['Adjusted_Strike'], x=oi_v['Call Open Interest']/1e3, orientation='h', name='Call OI', marker_color=CONFIG[symbol]['call_color'], width=CONFIG[symbol]['bar_width']/2), row=1, col=2)
    fig.add_trace(go.Bar(y=oi_v['Adjusted_Strike'], x=-oi_v['Put Open Interest']/1e3, orientation='h', name='Put OI', marker_color=CONFIG[symbol]['put_color'], width=CONFIG[symbol]['bar_width']/2), row=1, col=2)
    
    fig.update_xaxes(type='category', nticks=15, row=1, col=1)
    fig.update_layout(height=600, template="plotly_white", showlegend=False, xaxis_rangeslider_visible=False, hovermode="x unified")
    return fig

def draw_vivid_plot(df_oi, df_vol, symbol, v_flip):
    """圖 2: 您提供的 OI 與 GEX 綜合對照圖 (Vivid Plot)"""
    conf = CONFIG[symbol]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 這裡加入您提供的 Bar 與 Scatter 邏輯
    fig.add_trace(go.Bar(x=df_oi['Adjusted_Strike'], y=df_oi['Call Open Interest'], name='看漲 OI', marker_color=conf['call_color'], opacity=0.6, width=conf['bar_width']), secondary_y=False)
    fig.add_trace(go.Bar(x=df_oi['Adjusted_Strike'], y=-df_oi['Put Open Interest'], name='看跌 OI', marker_color=conf['put_color'], opacity=0.6, width=conf['bar_width']), secondary_y=False)
    
    if 'Net_GEX_Yi' in df_vol.columns:
        fig.add_trace(go.Scatter(x=df_vol['Adjusted_Strike'], y=df_vol['Net_GEX_Yi'], name='淨 GEX (億)', line=dict(color='#00008B', width=5)), secondary_y=True)

    if v_flip: fig.add_vline(x=v_flip, line_width=4, line_color="black", annotation_text=f"轉折:{v_flip:.0f}")

    fig.update_layout(height=500, template="plotly_white", hovermode="x unified", title=f"<b>{conf['label']} GEX 強度對照</b>")
    return fig

# --- 4. 主程式 ---

st.markdown("<h1 style='text-align: center; font-size: 45px; color: #001F3F;'>🏹 專業級 ES & NQ 數據系統</h1>", unsafe_allow_html=True)

for symbol in ["SPX", "NQ"]:
    oi_f, vol_f = get_latest_files(CONFIG[symbol]['keywords'])
    if oi_f and vol_f:
        read_files_list.append(os.path.basename(oi_f))
        read_files_list.append(os.path.basename(vol_f))
        
        df_oi = clean_data(pd.read_csv(oi_f), CONFIG[symbol]['offset'])
        df_vol = clean_data(pd.read_csv(vol_f), CONFIG[symbol]['offset'])
        df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['offset'])
        
        # 計算關鍵指標
        cw_val = df_oi.loc[df_oi['Call Open Interest'].idxmax(), 'Adjusted_Strike']
        pw_val = df_oi.loc[df_oi['Put Open Interest'].idxmax(), 'Adjusted_Strike']
        v_flip = None
        if not df_vol.empty:
            for i in range(len(df_vol)-1):
                if df_vol.iloc[i]['Net Gamma Exposure'] * df_vol.iloc[i+1]['Net Gamma Exposure'] <= 0:
                    v_flip = df_vol.iloc[i]['Adjusted_Strike']; break

        # 顯示卡片
        st.markdown(f"## 📈 {CONFIG[symbol]['label']}")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='metric-card'>多空分界 (Pivot)<br><b style='font-size:35px; color:black;'>{v_flip:.0f if v_flip else 'N/A'}</b></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'>買權牆 (Call Wall)<br><b style='font-size:35px; color:green;'>{cw_val:.0f}</b></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'>賣權牆 (Put Wall)<br><b style='font-size:35px; color:red;'>{pw_val:.0f}</b></div>", unsafe_allow_html=True)

        # 繪製兩張核心圖表
        if df_k is not None:
            st.plotly_chart(draw_kline_with_oi(df_k, df_oi, symbol), use_container_width=True)
        st.plotly_chart(draw_vivid_plot(df_oi, df_vol, symbol, v_flip), use_container_width=True)
        st.divider()

# 底部解讀
with st.expander("📖 數據解讀說明", expanded=True):
    st.markdown("### 🔵 淨 GEX 代表法人長線佈局；🟠 波動 GEX 代表當日動態資金。")

# 數據溯源清單
if read_files_list:
    st.markdown("### 📂 本次讀取的數據檔案：")
    for f in sorted(list(set(read_files_list))):
        st.code(f)
