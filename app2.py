import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="ES & NQ 籌碼監控 (無空隙連續版)", layout="wide")

# 背景淡藍色
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 10px; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# 完全複製 Barchart 色彩
COLORS = {
    "pos_bar": "#0000FF", "neg_bar": "#FFA500", "agg_line": "#3498db",
    "flip_line": "#FF0000", "price_line": "#008000",
    "bg_green": "rgba(0, 255, 0, 0.05)", "bg_red": "rgba(255, 0, 0, 0.05)"
}

CONFIG = {
    "SPX": {"label": "🇺🇸 ES / SPX (標普 500)", "ticker": "^SPX", "basis": 17.4, "keywords": ["SPX", "ES"], "width_bar": 2},
    "NQ": {"label": "💻 NQ / NASDAQ 100 (那指)", "ticker": "^NDX", "basis": 57.6, "keywords": ["IUXX", "NQ"], "width_bar": 20}
}
DATA_DIR = "data"

# --- 2. 數據核心函數 ---

@st.cache_data(ttl=60)
def fetch_yahoo_kline(ticker, basis):
    """抓取 Yahoo 真實 5 分鐘數據（近 5 天）"""
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty: return None
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df + basis
        # 關鍵：將索引轉為字串格式，用於分類軸
        df['time_label'] = df.index.strftime('%m-%d %H:%M')
        return df
    except:
        return None

def get_latest_files(keywords):
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in keywords)]
    if not symbol_files: return None, None
    oi_f = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_f = [f for f in symbol_files if "open-interest" not in f.lower()]
    return (max(oi_f, key=os.path.getmtime) if oi_f else None, 
            max(vol_f, key=os.path.getmtime) if vol_f else None)

def clean_csv(filepath, basis):
    df = pd.read_csv(filepath)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Strike_Fut'] = df['Strike'] + basis
    return df

def get_safe_float(series):
    val = series.iloc[-1]
    return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)

# --- 3. 繪圖組件 (互動式 Plotly) ---

def draw_kline_profile(oi_df, symbol):
    """圖 1: 完全連續的 5m K線 + 水平 OI 牆"""
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    if df_k is None: return

    last_p = get_safe_float(df_k['Close'])
    y_range = 100 if symbol == "SPX" else 350
    oi_v = oi_df[(oi_df['Strike_Fut'] >= last_p - y_range) & (oi_df['Strike_Fut'] <= last_p + y_range)]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.8, 0.2])
    
    # K線：使用 'time_label' 作為 X 軸，並設定 type='category'
    fig.add_trace(go.Candlestick(
        x=df_k['time_label'], 
        open=df_k['Open'], high=df_k['High'], low=df_k['Low'], close=df_k['Close'], 
        name="5m K線"
    ), row=1, col=1)
    
    # 水平 OI 牆
    fig.add_trace(go.Bar(
        y=oi_v['Strike_Fut'], x=oi_v['Call Open Interest']/1e3, 
        orientation='h', name="Call OI", marker_color=COLORS['pos_bar'], 
        hovertemplate="Strike: %{y}<br>Call OI: %{x:.1f}K"
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=oi_v['Strike_Fut'], x=-oi_v['Put Open Interest']/1e3, 
        orientation='h', name="Put OI", marker_color=COLORS['neg_bar'], 
        hovertemplate="Strike: %{y}<br>Put OI: %{x:.1f}K"
    ), row=1, col=2)

    fig.add_hline(y=last_p, line_dash="dash", line_color=COLORS['price_line'], annotation_text=f"期貨現價:{last_p:,.1f}")

    # --- 核心設定：讓 K 線圖連續 ---
    fig.update_xaxes(
        type='category',  # 強制使用分類軸，消除時間空隙
        nticks=20,        # 限制標籤數量避免過擠
        row=1, col=1
    )

    fig.update_layout(
        height=700, 
        template="plotly_white", 
        showlegend=False, 
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    st.plotly_chart(fig, width='stretch')

def draw_gex_main(gamma_df, symbol):
    """圖 2: 淨 Gamma 曝險圖 (Barchart 配色)"""
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    last_p = get_safe_float(df_k['Close']) if df_k is not None else 0
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=gamma_df['Strike_Fut'], y=gamma_df['Net Gamma Exposure']/1e8, name="Net GEX", 
                         marker_color=np.where(gamma_df['Net Gamma Exposure']>=0, COLORS['pos_bar'], COLORS['neg_bar'])), secondary_y=False)
    fig.add_trace(go.Scatter(x=gamma_df['Strike_Fut'], y=gamma_df['Gamma Exposure Profile']/1e9, name="Aggregate", 
                             line=dict(color=COLORS['agg_line'], width=4)), secondary_y=True)
    
    fig.add_vline(x=last_p, line_color=COLORS['price_line'], line_dash="dash")
    fig.update_layout(title=f"<b>{symbol} 淨 Gamma 曝險</b>", height=500, template="plotly_white")
    st.plotly_chart(fig, width='stretch')

def draw_details(df, symbol, mode="Gamma"):
    """圖 3 & 4: 買賣權對比"""
    scale = 1e8 if mode == "Gamma" else 1e3
    col_c = "Call Gamma Exposure" if mode == "Gamma" else "Call Open Interest"
    col_p = "Put Gamma Exposure" if mode == "Gamma" else "Put Open Interest"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike_Fut'], y=df[col_c]/scale, name="Call", marker_color=COLORS['pos_bar']))
    fig.add_trace(go.Bar(x=df['Strike_Fut'], y=df[col_p]/scale if mode=="Gamma" else -df[col_p]/scale, name="Put", marker_color=COLORS['neg_bar']))
    fig.update_layout(title=f"{symbol} {mode} 對比", height=400, barmode='relative', template="plotly_white")
    st.plotly_chart(fig, width='stretch')

# --- 4. 主程式執行 ---

st.markdown("<h1 style='text-align: center;'>🎯 ES & NQ 連續無空隙 K 線監控系統</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"---")
    st.markdown(f"## {CONFIG[asset]['label']}")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        df_oi = clean_csv(oi_f, CONFIG[asset]['basis'])
        df_vol = clean_csv(vol_f, CONFIG[asset]['basis'])
        draw_kline_profile(df_oi, asset)
        draw_gex_main(df_vol, asset)
        draw_details(df_oi, asset, mode="Gamma")
        draw_details(df_oi, asset, mode="Open Interest")
    else:
        st.error(f"❌ 找不到 {asset} 的 CSV 檔案")
