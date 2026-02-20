import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="專業級 ES & NQ 數據監控系統", layout="wide")

# 自定義 CSS (底色淡藍)
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { 
        color: #001F3F; 
        border-bottom: 3px solid #001F3F; 
        padding-bottom: 10px; 
        margin-top: 50px; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 組態與色彩設定 ---
# 顏色完全複製 Barchart 配色
COLORS = {
    "pos_bar": "#0000FF",        # 正值：藍色
    "neg_bar": "#FFA500",        # 負值：橘色
    "agg_line": "#3498db",       # 累計曲線：亮藍色
    "flip_line": "#FF0000",      # Flip：紅色
    "price_line": "#008000",     # 現價：深綠色
    "bg_green": "rgba(0, 255, 0, 0.05)", 
    "bg_red": "rgba(255, 0, 0, 0.05)"
}

CONFIG = {
    "SPX": {
        "label": "🇺🇸 ES / SPX (標普 500 期貨基準)",
        "ticker": "^SPX", 
        "basis": 17.4, 
        "keywords": ["SPX", "ES"],
        "width_bar": 2
    },
    "NQ": {
        "label": "💻 NQ / NASDAQ 100 (那指期貨基準)",
        "ticker": "^NDX", 
        "basis": 57.6, 
        "keywords": ["IUXX", "NQ"],
        "width_bar": 20
    }
}
DATA_DIR = "data"

# --- 3. 數據核心函數 ---

@st.cache_data(ttl=300)
def fetch_yahoo_kline(ticker, basis):
    """抓取 Yahoo 真實 15 分鐘數據並修正基差"""
    try:
        # 抓取最近 1 個月的 15m 數據
        df = yf.download(ticker, period="1mo", interval="15m", progress=False)
        if df.empty: return None
        # 換算為期貨點位
        return df + basis
    except:
        return None

def get_latest_files(keywords):
    """自動從 data 資料夾找最新檔案"""
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in keywords)]
    if not symbol_files: return None, None
    
    oi_f = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_f = [f for f in symbol_files if "open-interest" not in f.lower()]
    
    return (max(oi_f, key=os.path.getmtime) if oi_f else None, 
            max(vol_f, key=os.path.getmtime) if vol_f else None)

def clean_csv(filepath, basis):
    """讀取並清洗 CSV 數據"""
    df = pd.read_csv(filepath)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Strike_Fut'] = df['Strike'] + basis
    return df

def find_gamma_flip(df):
    """計算 Gamma Flip 點位"""
    if 'Gamma Exposure Profile' not in df.columns: return None
    profile, strikes = df['Gamma Exposure Profile'].values, df['Strike_Fut'].values
    for i in range(len(profile) - 1):
        if not np.isnan(profile[i]) and not np.isnan(profile[i+1]):
            if profile[i] * profile[i+1] <= 0:
                return strikes[i]
    return None

# --- 4. 繪圖組件 (互動式 Plotly) ---

def draw_kline_profile(oi_df, symbol):
    """圖 1: 真實 15m K線 + 水平 OI 牆"""
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    if df_k is None: 
        st.warning("無法抓取 Yahoo 數據")
        return

    last_p = float(df_k['Close'].iloc[-1])
    y_min, y_max = last_p * 0.98, last_p * 1.02
    oi_v = oi_df[(oi_df['Strike_Fut'] >= y_min) & (oi_df['Strike_Fut'] <= y_max)]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.75, 0.25])
    # K線
    fig.add_trace(go.Candlestick(x=df_k.index, open=df_k['Open'], high=df_k['High'], low=df_k['Low'], close=df_k['Close'], name="K線"), row=1, col=1)
    # OI 牆 (TIP 顯示)
    fig.add_trace(go.Bar(y=oi_v['Strike_Fut'], x=oi_v['Call Open Interest']/1e3, orientation='h', name="Call OI(K)", marker_color=COLORS['pos_bar'], hovertemplate="Strike: %{y}<br>Call OI: %{x:.1f}K"), row=1, col=2)
    fig.add_trace(go.Bar(y=oi_v['Strike_Fut'], x=-oi_v['Put Open Interest']/1e3, orientation='h', name="Put OI(K)", marker_color=COLORS['neg_bar'], hovertemplate="Strike: %{y}<br>Put OI: %{x:.1f}K"), row=1, col=2)

    fig.add_hline(y=last_p, line_dash="dash", line_color=COLORS['price_line'], annotation_text=f"期貨現價:{last_p:,.1f}")
    fig.update_layout(height=600, template="plotly_white", showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def draw_gex_main(gamma_df, symbol):
    """圖 2: 淨 Gamma 曝險圖 (完全複製 Barchart 配色)"""
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    last_p = float(df_k['Close'].iloc[-1]) if df_k is not None else 0
    flip = find_gamma_flip(gamma_df)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # 柱狀圖
    fig.add_trace(go.Bar(x=gamma_df['Strike_Fut'], y=gamma_df['Net Gamma Exposure']/1e8, name="Net GEX", 
                         marker_color=np.where(gamma_df['Net Gamma Exposure']>=0, COLORS['pos_bar'], COLORS['neg_bar']),
                         hovertemplate="Strike: %{x}<br>GEX: %{y:.2f} 億"), secondary_y=False)
    # 累計曲線
    fig.add_trace(go.Scatter(x=gamma_df['Strike_Fut'], y=gamma_df['Gamma Exposure Profile']/1e9, name="Aggregate", 
                             line=dict(color=COLORS['agg_line'], width=4), hovertemplate="累計曝險: %{y:.2f}B"), secondary_y=True)
    
    if flip:
        fig.add_vline(x=flip, line_color=COLORS['flip_line'], line_width=2)
        fig.add_vrect(x0=gamma_df['Strike_Fut'].min(), x1=flip, fillcolor=COLORS['bg_red'], opacity=1, layer="below", line_width=0)
        fig.add_vrect(x0=flip, x1=gamma_df['Strike_Fut'].max(), fillcolor=COLORS['bg_green'], opacity=1, layer="below", line_width=0)

    fig.add_vline(x=last_p, line_color=COLORS['price_line'], line_dash="dash")
    fig.update_layout(title=f"<b>{symbol} 淨 Gamma 曝險 (單位：億美元)</b>", height=500, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def draw_detail_bars(oi_df, symbol, mode="Gamma"):
    """圖 3 & 4: Call/Put 對比圖"""
    scale = 1e8 if mode == "Gamma" else 1e3
    col_c = "Call Gamma Exposure" if mode == "Gamma" else "Call Open Interest"
    col_p = "Put Gamma Exposure" if mode == "Gamma" else "Put Open Interest"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=oi_df['Strike_Fut'], y=oi_df[col_c]/scale, name="Call", marker_color=COLORS['pos_bar']))
    fig.add_trace(go.Bar(x=oi_df['Strike_Fut'], y=oi_df[col_p]/scale if mode=="Gamma" else -oi_df[col_p]/scale, 
                         name="Put", marker_color=COLORS['neg_bar']))
    fig.update_layout(title=f"{symbol} {mode} 買賣權細節", height=400, barmode='relative', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- 5. 主程式 ---

st.markdown("<h1 style='text-align: center;'>🎯 ES & NQ 真實籌碼監控系統</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"## {CONFIG[asset]['label']}")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        df_oi = clean_csv(oi_f, CONFIG[asset]['basis'])
        df_vol = clean_csv(vol_f, CONFIG[asset]['basis'])
        
        # 依序呈現 4 張圖 (上下排版)
        draw_kline_profile(df_oi, asset)
        draw_gex_main(df_vol, asset)
        draw_detail_bars(df_oi, asset, mode="Gamma")
        draw_detail_bars(df_oi, asset, mode="Open Interest")
    else:
        st.error(f"❌ 請確認 DATA 資料夾內有 {asset} 的最新 CSV 檔案")

st.info("💡 系統自動抓取 Yahoo 15m 真實數據，並已自動換算期貨基差點位。")
