import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 頁面基本設定 ---
st.set_page_config(page_title="專業級 ES & NQ 數據監控系統", layout="wide")

# 背景淡藍色
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    </style>
    """, unsafe_allow_html=True)

# 組態設定 (含 Basis 價差換算)
CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500 期貨)",
        "basis": 17.4,  # ES 比現貨高約 17.4 點
        "keywords": ["SPX", "ES"],
        "color_call": "#1f77b4", # 專業藍
        "color_put": "#ff7f0e",  # 專業橘
        "last_price_idx": 6861.89
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指期貨)",
        "basis": 57.6,  # NQ 比現貨高約 57.6 點
        "keywords": ["IUXX", "NQ"],
        "color_call": "#000080", # 深藍
        "color_put": "#FF4500",  # 橘紅
        "last_price_idx": 24797.17
    }
}
DATA_DIR = "data"

# --- 數據自動讀取與清洗 ---
def get_latest_files(symbol_keywords):
    if not os.path.exists(DATA_DIR): return None, None
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files: return None, None
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None
    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    latest_oi = max(oi_files, key=os.path.getmtime) if oi_files else None
    latest_vol = max(vol_files, key=os.path.getmtime) if vol_files else None
    return latest_oi, latest_vol

def clean_data(filepath, basis=0):
    df = pd.read_csv(filepath)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Strike'] = df['Strike'] + basis # 換算為期貨點數
    return df

# --- Plotly 繪圖核心 (內建 Tooltip 與中文支援) ---

def draw_kline_oi_chart(oi_file, fut_price, symbol):
    """圖表 1: 15分K線 + 水平 OI 牆 (上下編排的第一張)"""
    # 模擬 15 分鐘數據
    np.random.seed(100 if symbol == "SPX" else 42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=150, freq='15min')
    vol = 25 if symbol == "NQ" else 4
    path = np.cumsum(np.random.normal(0, vol, len(dates))) + fut_price
    
    df_oi = clean_data(oi_file, CONFIG[symbol]['basis'])
    y_range = 150 if symbol == "SPX" else 500
    df_oi_v = df_oi[(df_oi['Strike'] >= fut_price - y_range) & (df_oi['Strike'] <= fut_price + y_range)]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02, 
                        column_widths=[0.75, 0.25], subplot_titles=(f"{symbol} 15m K線", "OI 籌碼牆"))

    # 左：K線
    fig.add_trace(go.Candlestick(x=dates, open=path-2, high=path+5, low=path-5, close=path, name="K線"), row=1, col=1)
    
    # 右：水平 OI (支援 Tooltip)
    fig.add_trace(go.Bar(y=df_oi_v['Strike'], x=df_oi_v['Call Open Interest']/1e3, orientation='h', 
                         name="Call OI (K)", marker_color='blue', hovertemplate="履約價: %{y}<br>Call OI: %{x:,.0f}K"), row=1, col=2)
    fig.add_trace(go.Bar(y=df_oi_v['Strike'], x=-df_oi_v['Put Open Interest']/1e3, orientation='h', 
                         name="Put OI (K)", marker_color='orange', hovertemplate="履約價: %{y}<br>Put OI: %{x:,.0f}K"), row=1, col=2)

    fig.add_hline(y=fut_price, line_dash="dash", line_color="green", annotation_text=f"期貨現價:{fut_price:,.1f}")
    fig.update_layout(height=600, showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def draw_gamma_analysis(gamma_df, fut_price, symbol):
    """圖表 2: 淨 Gamma 曝險圖"""
    scale = 1e8 # 單位：億美元
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 柱狀圖
    fig.add_trace(go.Bar(x=gamma_df['Strike'], y=gamma_df['Net Gamma Exposure']/scale, 
                         name="淨 GEX (億)", marker_color='blue', opacity=0.5,
                         hovertemplate="履約價: %{x}<br>淨曝險: %{y:.2f} 億"), secondary_y=False)
    
    # 累計曲線
    fig.add_trace(go.Scatter(x=gamma_df['Strike'], y=gamma_df['Gamma Exposure Profile']/1e9, 
                             name="累計 GEX (B)", line=dict(color='dodgerblue', width=3),
                             hovertemplate="價格移動至此<br>總曝險: %{y:.2f} B"), secondary_y=True)
    
    fig.add_vline(x=fut_price, line_dash="dash", line_color="green")
    fig.update_layout(title=f"{symbol} 淨 Gamma 分佈與累計曲線 (單位：億美元)", height=450)
    st.plotly_chart(fig, use_container_width=True)

def draw_cp_details(oi_df, fut_price, symbol, mode="Gamma"):
    """圖表 3 & 4: 買賣權對比圖"""
    scale = 1e8 if mode == "Gamma" else 1e3
    unit = "億" if mode == "Gamma" else "K"
    col_c = f"Call {mode} Exposure" if mode == "Gamma" else "Call Open Interest"
    col_p = f"Put {mode} Exposure" if mode == "Gamma" else "Put Open Interest"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=oi_df['Strike'], y=oi_df[col_c]/scale, name=f"Call {mode}", marker_color='blue'))
    fig.add_trace(go.Bar(x=oi_df['Strike'], y=oi_df[col_p]/scale if mode=="Gamma" else -oi_df[col_p]/scale, 
                         name=f"Put {mode}", marker_color='orange'))
    
    fig.add_vline(x=fut_price, line_dash="dash", line_color="green")
    fig.update_layout(title=f"{symbol} 買賣權 {mode} 對比 (單位：{unit})", height=400, barmode='relative')
    st.plotly_chart(fig, use_container_width=True)

# --- 主程式介面 ---
st.markdown("<h1 style='text-align: center; color: #001F3F;'>🏹 ES & NQ 期貨籌碼動態監控系統</h1>", unsafe_allow_html=True)

for asset in ["SPX", "NQ"]:
    st.markdown(f"## 📊 {CONFIG[asset]['label']} 分析區塊")
    oi_f, vol_f = get_latest_files(CONFIG[asset]['keywords'])
    
    if oi_f and vol_f:
        # 計算期貨價差換算
        basis = CONFIG[asset]['basis']
        fut_p = CONFIG[asset]['last_price_idx'] + basis # 換算為期貨點位
        
        df_oi = clean_data(oi_f, basis)
        df_vol = clean_data(vol_f, basis)
        
        # 垂直編排四張圖
        draw_kline_oi_chart(oi_f, fut_p, asset)
        draw_gamma_analysis(df_vol, fut_p, asset)
        draw_cp_details(df_oi, fut_p, asset, mode="Gamma")
        draw_cp_details(df_oi, fut_p, asset, mode="Open Interest")
    else:
        st.error(f"❌ 找不到 {asset} 的數據檔案")
    st.markdown("---")
