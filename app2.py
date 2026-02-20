import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 頁面基本設定 ---
st.set_page_config(page_title="ES & NQ 籌碼監控 (數據溯源版)", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    .stMarkdown h2 { color: #001F3F; border-bottom: 3px solid #001F3F; padding-bottom: 10px; margin-top: 50px; }
    .file-info { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #001F3F; }
    </style>
    """, unsafe_allow_html=True)

COLORS = {
    "pos_bar": "#0000FF", "neg_bar": "#FFA500", "agg_line": "#3498db",
    "flip_line": "#FF0000", "price_line": "#008000",
    "bg_green": "rgba(0, 255, 0, 0.05)", "bg_red": "rgba(255, 0, 0, 0.05)"
}

CONFIG = {
    "SPX": {"label": "🇺🇸 ES / SPX (標普 500)", "ticker": "^SPX", "basis": 17.4, "keywords": ["SPX", "ES"]},
    "NQ": {"label": "💻 NQ / NASDAQ 100 (那斯達克)", "ticker": "^NDX", "basis": 57.6, "keywords": ["IUXX", "NQ"]}
}
DATA_DIR = "data"

# 用於儲存本次讀取的檔案名稱
used_files = []

# --- 2. 數據核心函數 ---

def get_latest_files(keywords):
    """識別序號最大 (如 -2) 或最新的檔案"""
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in keywords)]
    if not symbol_files: return None, None
    
    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    
    def pick_latest(file_list):
        if not file_list: return None
        # 排序確保 "-2" 在 "-1" 之後
        sorted_list = sorted(file_list, key=lambda x: (x, os.path.getmtime(x)))
        latest = sorted_list[-1]
        if latest not in used_files: used_files.append(latest)
        return latest

    return pick_latest(oi_files), pick_latest(vol_files)

@st.cache_data(ttl=60)
def fetch_yahoo_kline(ticker, basis):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty: return None
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df + basis
        df['time_label'] = df.index.strftime('%m-%d %H:%M')
        return df
    except: return None

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

# --- 3. 繪圖組件 (含精確點位 TIP) ---

def draw_kline_profile(oi_df, symbol):
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    if df_k is None: return
    last_p = get_safe_float(df_k['Close'])
    y_range = 100 if symbol == "SPX" else 350 
    oi_v = oi_df[(oi_df['Strike_Fut'] >= last_p - y_range) & (oi_df['Strike_Fut'] <= last_p + y_range)].copy()
    diff = oi_v['Strike_Fut'].diff().median()
    bar_w = (diff if not pd.isna(diff) else 5) * 0.7

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.01, column_widths=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=df_k['time_label'], open=df_k['Open'], high=df_k['High'], low=df_k['Low'], close=df_k['Close'], name="K線"), row=1, col=1)
    
    # TIP 顯示履約點數
    fig.add_trace(go.Bar(y=oi_v['Strike_Fut'], x=oi_v['Call Open Interest']/1e3, orientation='h', name="Call OI", 
                         marker_color=COLORS['pos_bar'], width=bar_w,
                         hovertemplate="<b>履約點數: %{y}</b><br>看漲 OI: %{x:.2f} K口<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(y=oi_v['Strike_Fut'], x=-oi_v['Put Open Interest']/1e3, orientation='h', name="Put OI", 
                         marker_color=COLORS['neg_bar'], width=bar_w,
                         hovertemplate="<b>履約點數: %{y}</b><br>看跌 OI: %{x:.2f} K口<extra></extra>"), row=1, col=2)

    fig.add_hline(y=last_p, line_dash="dash", line_color=COLORS['price_line'], annotation_text=f"現價:{last_p:,.1f}")
    fig.update_xaxes(type='category', nticks=15, row=1, col=1)
    fig.update_layout(height=750, template="plotly_white", showlegend=False, xaxis_rangeslider_visible=False, hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

def draw_gex_main(gamma_df, symbol):
    df_k = fetch_yahoo_kline(CONFIG[symbol]['ticker'], CONFIG[symbol]['basis'])
    last_p = get_safe_float(df_k['Close']) if df_k is not None else 0
    diff = gamma_df['Strike_Fut'].diff().median()
    bar_w = (diff if not pd.isna(diff) else 5) * 0.7

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=gamma_df['Strike_Fut'], y=gamma_df['Net Gamma Exposure']/1e8, name="淨 GEX", width=bar_w,
                         marker_color=np.where(gamma_df['Net Gamma Exposure']>=0, COLORS['pos_bar'], COLORS['neg_bar']),
                         hovertemplate="<b>點數: %{x}</b><br>淨曝險: %{y:.2f} 億美元<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=gamma_df['Strike_Fut'], y=gamma_df['Gamma Exposure Profile']/1e9, name="累計 GEX", 
                             line=dict(color=COLORS['agg_line'], width=4),
                             hovertemplate="<b>價格點數: %{x}</b><br>總曝險: %{y:.2f} B<extra></extra>"), secondary_y=True)
    
    fig.add_vline(x=last_p, line_color=COLORS['price_line'], line_dash="dash")
    fig.update_layout(title=f"<b>{symbol} 淨 Gamma 曝險與累計曲線</b>", height=500, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

def draw_details(df, symbol, mode="Gamma"):
    diff = df['Strike_Fut'].diff().median()
    bar_w = (diff if not pd.isna(diff) else 5) * 0.7
    scale = 1e8 if mode == "Gamma" else 1e3
    unit = "億美元" if mode == "Gamma" else "K口"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike_Fut'], y=df[f"Call {mode} Exposure" if mode=="Gamma" else "Call Open Interest"]/scale, 
                         name="買權 (Call)", marker_color=COLORS['pos_bar'], width=bar_w,
                         hovertemplate=f"<b>點數: %{{x}}</b><br>買權{mode}: %{{y:.2f}} {unit}<extra></extra>"))
    fig.add_trace(go.Bar(x=df['Strike_Fut'], y=df[f"Put {mode} Exposure" if mode=="Gamma" else "Put Open Interest"]/scale if mode=="Gamma" else -df["Put Open Interest"]/scale, 
                         name="賣權 (Put)", marker_color=COLORS['neg_bar'], width=bar_w,
                         hovertemplate=f"<b>點數: %{{x}}</b><br>賣權{mode}: %{{y:.2f}} {unit}<extra></extra>"))
    fig.update_layout(title=f"{symbol} {mode} 細節對比", height=400, barmode='relative', template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

# --- 4. 主程式執行 ---

st.markdown("<h1 style='text-align: center;'>🎯 ES & NQ 真實籌碼監控系統</h1>", unsafe_allow_html=True)

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
        st.error(f"❌ 找不到 {asset} 的數據檔案")

# --- 5. 底部數據源資訊 (您要求的新增功能) ---
st.markdown("<br><br>", unsafe_allow_html=True)
if used_files:
    st.markdown("### 📂 本次分析使用的數據檔案：")
    info_html = "<div class='file-info'><ul>"
    for f in sorted(list(set(used_files))):
        info_html += f"<li>{os.path.basename(f)}</li>"
    info_html += "</ul></div>"
    st.markdown(info_html, unsafe_allow_html=True)
