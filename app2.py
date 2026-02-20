import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="專業級 ES & NQ 數據監控系統", layout="wide")

# 自定義 CSS
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 10px; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

COLORS = {
    "pos_bar": "#0000FF", "neg_bar": "#FFA500", "agg_line": "#3498db",
    "flip_line": "#FF0000", "price_line": "#008000",
    "bg_green": "rgba(0, 255, 0, 0.05)", "bg_red": "rgba(255, 0, 0, 0.05)"
}

CONFIG = {
    "SPX": {"label": "🇺🇸 ES / SPX (標普 500)", "ticker": "^SPX", "basis": 17.4, "keywords": ["SPX", "ES"]},
    "NQ": {"label": "💻 NQ / NASDAQ 100", "ticker": "^NDX", "basis": 57.6, "keywords": ["IUXX", "NQ"]}
}
DATA_DIR = "data"

# --- 2. 數據核心函數 (修正 Yahoo 60天限制) ---

@st.cache_data(ttl=300)
def fetch_yahoo_kline(ticker, basis):
    """
    抓取 Yahoo 真實 15 分鐘數據。
    注意：Yahoo 限制 15m 數據只能抓取最近 60 天。
    """
    # 嘗試 60 天（極限），若失敗則嘗試 1 個月
    for p in ["60d", "1mo"]:
        try:
            df = yf.download(ticker, period=p, interval="15m", progress=False)
            if not df.empty:
                if df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)
                return df + basis
        except Exception:
            continue
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

# --- 3. 繪圖組件 ---

def draw_kline_profile(oi_df, symbol):
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    if df_k is None: 
        st.error(f"❌ 無法獲取 {symbol} 數據。原因：Yahoo 限制 15m 數據僅限最近 60 天內。")
        return

    last_p = get_safe_float(df_k['Close'])
    y_range = 150 if symbol == "SPX" else 500
    oi_v = oi_df[(oi_df['Strike_Fut'] >= last_p - y_range) & (oi_df['Strike_Fut'] <= last_p + y_range)]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.8, 0.2])
    
    # K線
    fig.add_trace(go.Candlestick(x=df_k.index, open=df_k['Open'], high=df_k['High'], low=df_k['Low'], close=df_k['Close'], name="K線"), row=1, col=1)
    
    # OI 牆
    fig.add_trace(go.Bar(y=oi_v['Strike_Fut'], x=oi_v['Call Open Interest']/1e3, orientation='h', name="Call OI", marker_color=COLORS['pos_bar'], hovertemplate="Strike: %{y}<br>Call: %{x:.1f}K"), row=1, col=2)
    fig.add_trace(go.Bar(y=oi_v['Strike_Fut'], x=-oi_v['Put Open Interest']/1e3, orientation='h', name="Put OI", marker_color=COLORS['neg_bar'], hovertemplate="Strike: %{y}<br>Put: %{x:.1f}K"), row=1, col=2)

    fig.add_hline(y=last_p, line_dash="dash", line_color=COLORS['price_line'], annotation_text=f"現價:{last_p:,.1f}")
    fig.update_xaxes(range=[df_k.index[-150], df_k.index[-1]], row=1, col=1)
    fig.update_layout(height=700, template="plotly_white", showlegend=False, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, width='stretch')

def draw_gex_main(gamma_df, symbol):
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    last_p = get_safe_float(df_k['Close']) if df_k is not None else 0
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=gamma_df['Strike_Fut'], y=gamma_df['Net Gamma Exposure']/1e8, name="Net GEX", 
                         marker_color=np.where(gamma_df['Net Gamma Exposure']>=0, COLORS['pos_bar'], COLORS['neg_bar'])), secondary_y=False)
    fig.add_trace(go.Scatter(x=gamma_df['Strike_Fut'], y=gamma_df['Gamma Exposure Profile']/1e9, name="Agg", line=dict(color=COLORS['agg_line'], width=4)), secondary_y=True)
    
    fig.add_vline(x=last_p, line_color=COLORS['price_line'], line_dash="dash")
    fig.update_layout(title=f"<b>{symbol} 淨 Gamma 曝險 (億美元)</b>", height=500, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

# --- 4. 主程式 ---

st.markdown("<h1 style='text-align: center;'>🎯 ES & NQ 真實籌碼監控系統</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"## {CONFIG[asset]['label']}")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        df_oi = clean_csv(oi_f, CONFIG[asset]['basis'])
        df_vol = clean_csv(vol_f, CONFIG[asset]['basis'])
        draw_kline_profile(df_oi, asset)
        draw_gex_main(df_vol, asset)
    else:
        st.error(f"❌ 找不到 {asset} 的 CSV 檔案")
