import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面設定
st.set_page_config(page_title="專業級 GEX 監測系統", layout="wide")

# 背景淡藍色 CSS
st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    </style>
    """, unsafe_allow_html=True)

CONFIG = {
    "SPX": {
        "label": "ES / SPX (標普 500)",
        "offset": 0,
        "call_color": "#008000", 
        "put_color": "#B22222",  
        "bar_width": 4,          
        "keywords": ["SPX", "ES"]
    },
    "NQ": {
        "label": "NQ / NASDAQ 100 (那指)",
        "offset": 75,
        "call_color": "#000080", 
        "put_color": "#FF4500",  
        "bar_width": 20,         
        "keywords": ["IUXX", "NQ"]
    }
}
DATA_DIR = "data"
read_files_list = []

# --- 側邊欄 ---
st.sidebar.markdown("### 🔍 觀察範圍控制")
range_spx = st.sidebar.slider("SPX 範圍", 50, 2000, 500, step=50)
range_nq = st.sidebar.slider("NQ 範圍", 100, 3000, 1000, step=100)
RANGE_MAP = {"SPX": range_spx, "NQ": range_nq}

def get_latest_files(symbol_keywords):
    if not os.path.exists(DATA_DIR): return None, None
    search_path = os.path.join(DATA_DIR, "*.csv")
    all_files = glob.glob(search_path)
    if not all_files: return None, None
    symbol_files = [f for f in all_files if any(k.upper() in os.path.basename(f).upper() for k in symbol_keywords)]
    if not symbol_files: return None, None
    oi_files = [f for f in symbol_files if "open-interest" in f.lower()]
    vol_files = [f for f in symbol_files if "open-interest" not in f.lower()]
    latest_oi = max(oi_files, key=os.path.getmtime) if oi_files else None
    latest_vol = max(vol_files, key=os.path.getmtime) if vol_files else None
    return latest_oi, latest_vol

def clean_data(df, offset):
    cols = ['Strike', 'Call Open Interest', 'Put Open Interest', 'Net Gamma Exposure']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Strike']).sort_values('Strike')
    df['Adjusted_Strike'] = df['Strike'] + offset
    if 'Net Gamma Exposure' in df.columns:
        df['Net_GEX_Yi'] = df['Net Gamma Exposure'] / 1e8
    return df

def create_vivid_plot(df_oi, df_vol, symbol, v_flip):
    conf = CONFIG[symbol]
    cw_idx = df_oi['Call Open Interest'].idxmax()
    pw_idx = df_oi['Put Open Interest'].idxmax()
    cw, pw = df_oi.loc[cw_idx, 'Adjusted_Strike'], df_oi.loc[pw_idx, 'Adjusted_Strike']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=df_oi['Call Open Interest'],
        name='看漲 OI', marker=dict(color=conf['call_color'], line=dict(width=1, color='white')),
        opacity=0.6, width=conf['bar_width'],
        hovertemplate='<b>價格: %{x}</b><br>看漲口數: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=df_oi['Adjusted_Strike'], y=-df_oi['Put Open Interest'],
        name='看跌 OI', marker=dict(color=conf['put_color'], line=dict(width=1, color='white')),
        opacity=0.6, width=conf['bar_width'],
        hovertemplate='<b>價格: %{x}</b><br>看跌口數: %{y:,.0f}<extra></extra>'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df_oi['Adjusted_Strike'], y=df_oi['Net_GEX_Yi'],
        name='淨 GEX (億)', line=dict(color='#00008B', width=5), 
        hovertemplate='淨 Gamma: %{y:,.2f} 億<extra></extra>'
    ), secondary_y=True)

    line_font = dict(size=18, color="black", family="Arial Black")
    if cw: fig.add_vline(x=cw, line_dash="dash", line_color="green", line_width=3, annotation_text=f"買權牆:{cw:.0f}", annotation_font=line_font)
    if pw: fig.add_vline(x=pw, line_dash="dash", line_color="red", line_width=3, annotation_text=f"賣權牆:{pw:.0f}", annotation_font=line_font)
    if v_flip: fig.add_vline(x=v_flip, line_width=4, line_color="black", annotation_text=f"轉折:{v_flip:.0f}", annotation_font=line_font)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F0F8FF',
        hovermode="x unified", height=650,
        title=dict(text=f"<b>{conf['label']} 數據監測</b>", font=dict(size=28, color='black')),
        xaxis=dict(
            title=dict(text="執行價 (Strike)", font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black'), gridcolor='white',
            range=[v_flip - RANGE_MAP[symbol], v_flip + RANGE_MAP[symbol]] if v_flip else None
        ),
        yaxis=dict(title=dict(text="未平倉合約 (OI)", font=dict(size=20, color='black')), tickfont=dict(size=16, color='black')),
        yaxis2=dict(title=dict(text="GEX 強度 (億美元)", font=dict(size=20, color='black')), tickfont=dict(size=16, color='black'), overlaying='y', side='right', showgrid=False),
        hoverlabel=dict(bgcolor="#001F3F", font_size=20, font_color="white", font_family="Arial Black"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=18, color='black')),
        margin=dict(l=80, r=80, t=120, b=80),
        bargap=0.05
    )
    return fig

# --- 主程式 ---
st.markdown("<h1 style='text-align: center; font-size: 45px; color: #001F3F;'>🏹 專業級 ES & NQ 數據系統</h1>", unsafe_allow_html=True)

if not os.path.exists(DATA_DIR):
    st.error(f"❌ 找不到目錄: {DATA_DIR}")
else:
    for symbol in ["SPX", "NQ"]:
        oi_f, vol_f = get_latest_files(CONFIG[symbol]['keywords'])
        if oi_f and vol_f:
            read_files_list.append(os.path.basename(oi_f))
            read_files_list.append(os.path.basename(vol_f))
            
            df_oi = clean_data(pd.read_csv(oi_f), CONFIG[symbol]['offset'])
            df_vol = clean_data(pd.read_csv(vol_f), CONFIG[symbol]['offset'])
            
            # --- 修正處：提早計算數值 ---
            cw_idx = df_oi['Call Open Interest'].idxmax()
            pw_idx = df_oi['Put Open Interest'].idxmax()
            cw_val = df_oi.loc[cw_idx, 'Adjusted_Strike']
            pw_val = df_oi.loc[pw_idx, 'Adjusted_Strike']
            
            v_flip = None
            if not df_vol.empty:
                for i in range(len(df_vol)-1):
                    if df_vol.iloc[i]['Net Gamma Exposure'] * df_vol.iloc[i+1]['Net Gamma Exposure'] <= 0:
                        v_flip = df_vol.iloc[i]['Adjusted_Strike']
                        break
            
            # 轉為字串避免 f-string 內判斷出錯
            piv_text = f"{v_flip:.0f}" if v_flip is not None else "N/A"
            cw_text = f"{cw_val:.0f}"
            pw_text = f"{pw_val:.0f}"

            st.markdown(f"<h2 style='color: #004080; font-size: 35px;'>📈 {CONFIG[symbol]['label']}</h2>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            # 現在 f-string 內容非常單純，不會報錯
            with c1: st.markdown(f"<div style='text-align:center; background:white; padding:15px; border-radius:15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>多空分界 (Pivot)<br><b style='font-size:35px; color:black;'>{piv_text}</b></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div style='text-align:center; background:white; padding:15px; border-radius:15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>買權牆 (Call Wall)<br><b style='font-size:35px; color:green;'>{cw_text}</b></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div style='text-align:center; background:white; padding:15px; border-radius:15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>賣權牆 (Put Wall)<br><b style='font-size:35px; color:red;'>{pw_text}</b></div>", unsafe_allow_html=True)

            st.plotly_chart(create_vivid_plot(df_oi, df_vol, symbol, v_flip), use_container_width=True)
            st.divider()

# 底部解讀說明
with st.expander("📖 數據解讀說明 (GEX 概念指南)", expanded=True):
    st.markdown("""
    ### 🔵 淨 GEX (Net Gamma Exposure) —— 「結構性資金」
    * **計算來源**：依據 **未平倉合約 (Open Interest, OI)**。
    * **單位解讀**：代表市場的底層結構，反映的是大戶、法人長線佈局。
    ### 🟠 波動 GEX (Vol Gamma Exposure) —— 「動態資金」
    * **計算來源**：依據 **當日成交量 (Volume)**。
    """, unsafe_allow_html=True)

if read_files_list:
    st.markdown("--- ")
    st.markdown("### 📂 本次讀取的數據檔案：")
    for f in sorted(list(set(read_files_list))):
        st.code(f)
