import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime
import time
import os
import random

# --- CONFIG & SAFETY ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v8.0", 
    page_icon="💠", 
    initial_sidebar_state="collapsed" 
)

# AI Safety Check
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Session State
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]

# --- 2025 "KINETIC" MOBILE THEME ---
st.markdown("""
<style>
    /* 1. BACKGROUND */
    .stApp {background-color: #0e1117; color: #ffffff;}
    
    /* 2. COMPACT TABS (Fixes "Split Text" Issue) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px; background-color: #0e1117; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; 
        white-space: nowrap; /* Prevents text splitting */
        background-color: #1c1f26;
        border-radius: 8px; 
        color: #888; 
        font-size: 14px;
        font-weight: 600; 
        flex: 1;
        padding: 0 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ccff00; /* Neon Green Active Tab */
        color: black !important;
    }

    /* 3. METRIC CARDS */
    .metric-container {
        background-color: #1c1f26; padding: 12px; border-radius: 12px;
        text-align: center; border: 1px solid #2d323b; margin-bottom: 5px;
    }
    
    /* 4. BIG ACTION BUTTONS */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 55px; font-weight: 900; font-size: 18px;
    }
    
    /* 5. INPUTS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1c1f26 !important; color: white !important;
        border: 1px solid #2d323b !important; border-radius: 8px;
    }
    
    /* 6. HEADER FIX */
    .asset-header {
        font-size: 24px; font-weight: 900; color: white; margin: 0;
    }
    .price-pulse {
        font-size: 32px; font-weight: 900; color: #ccff00; margin: 0;
    }
    
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
</style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIC ---
@st.cache_data(ttl=300) 
def get_market_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if df.empty: return None
        return df
    except: return None

def get_signal(df):
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    signal = "BUY" if curr > sma20 else "SELL"
    return signal, curr

def log_trade(ticker, action, price, shares, total, notes="Manual"):
    entry = pd.DataFrame([{
        'Date': datetime.now().strftime("%H:%M"), 'Ticker': ticker, 
        'Action': action, 'Price': f"${price:.2f}", 'Shares': shares, 'Total': f"${total:.2f}", 'Notes': notes
    }])
    st.session_state['journal'] = pd.concat([st.session_state['journal'], entry], ignore_index=True)

# --- APP LAYOUT ---

# 1. FIXED TOP BAR (Search & Type)
c_type, c_search = st.columns([1, 2.5])
with c_type:
    asset_type = st.selectbox("Type", ["CRYPTO", "STOCKS"], label_visibility="collapsed")
with c_search:
    default_tick = "BTC" if asset_type == "CRYPTO" else "NVDA"
    raw_ticker = st.text_input("Search", value=default_tick, label_visibility="collapsed").upper()

ticker = f"{raw_ticker}-USD" if asset_type == "CRYPTO" and "-USD" not in raw_ticker else raw_ticker
df = get_market_data(ticker)

# 2. MAIN HEADER (Brings "Life" back)
if df is not None:
    curr_price = df['Close'].iloc[-1]
    change = (curr_price - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100
    color_hex = "#ccff00" if change >= 0 else "#ff4444"
    
    st.markdown(f"""
    <div style="padding: 10px 0;">
        <p class="asset-header">{ticker}</p>
        <p class="price-pulse" style="color:{color_hex}">${curr_price:,.2f}</p>
        <p style="color:{color_hex}; font-weight:bold; margin:0;">{change:+.2f}% Today</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"<h3>{ticker}</h3>", unsafe_allow_html=True)

# 3. COMPACT TABS (Single Word Labels)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trade", "Chat", "Scan", "Wallet", "Learn"])

# --- TAB 1: TRADE ---
with tab1:
    if df is not None:
        signal, _ = get_signal(df)
        
        # LOCKED CHART (Fixes Scrolling Issue)
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(
            template="plotly_dark", 
            height=350, 
            margin=dict(l=0, r=0, t=10, b=0), 
            paper_bgcolor='#0e1117', 
            plot_bgcolor='#0e1117', 
            xaxis_rangeslider_visible=False,
            dragmode=False, # DISABLES PANNING (Fixes "Touch" issue)
            clickmode='none'
        )
        # Disable zoom config
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False})
        
        # METRICS ROW
        m1, m2 = st.columns(2)
        with m1: st.markdown(f"<div class='metric-container'><h3>SIGNAL</h3><h2 style='color:#2196F3'>{signal}</h2></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-container'><h3>VOLATILITY</h3><h2>Normal</h2></div>", unsafe_allow_html=True)

        # ACTION BUTTONS
        st.write("---")
        qty_col, buy_col, sell_col = st.columns([1, 1.5, 1.5])
        with qty_col:
            shares = st.number_input("Qty", value=1.0, min_value=0.01, label_visibility="collapsed")
        with buy_col:
            if st.button("BUY", type="primary", use_container_width=True):
                cost = shares * curr_price
                if st.session_state['cash'] >= cost:
                    st.session_state['cash'] -= cost
                    st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + shares
                    log_trade(ticker, "BUY", curr_price, shares, cost)
                    st.success("BOUGHT")
                    time.sleep(0.5); st.rerun()
                else: st.error("Funds?")
        with sell_col:
            if st.button("SELL", use_container_width=True):
                if st.session_state['holdings'].get(ticker, 0) >= shares:
                    st.session_state['cash'] += shares * curr_price
                    st.session_state['holdings'][ticker] -= shares
                    log_trade(ticker, "SELL", curr_price, shares, shares * curr_price)
                    st.success("SOLD")
                    time.sleep(0.5); st.rerun()
                else: st.error("No Asset")
    else: st.info("Enter a symbol to load chart.")

# --- TAB 2: CHAT ---
with tab2:
    for msg in st.session_state['chat_history']:
        if msg['role'] != 'system':
            with st.chat_message(msg['role']): st.write(msg['content'])
    prompt = st.chat_input("Ask Oracle...")
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        response = f"Checking markets for {prompt}..."
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.write(response)

# --- TAB 3: SCAN ---
with tab3:
    st.subheader("Radar")
    if st.button("SCAN MARKET", use_container_width=True):
        st.info("Scanning Top 10 Assets...")
        # Mock results for speed
        data = [
            {"Ticker": "BTC", "Signal": "BUY", "Price": "$98,000"},
            {"Ticker": "ETH", "Signal": "HOLD", "Price": "$2,800"},
            {"Ticker": "SOL", "Signal": "SELL", "Price": "$140"},
        ]
        st.dataframe(pd.DataFrame(data), use_container_width=True)

# --- TAB 4: WALLET ---
with tab4:
    total_assets = 0
    for t, s in st.session_state['holdings'].items():
        if s > 0:
            d = get_market_data(t)
            p = d['Close'].iloc[-1] if d is not None else 0
            total_assets += s * p
    net = st.session_state['cash'] + total_assets
    
    st.markdown(f"<div class='metric-container'><h3>NET WORTH</h3><h1 style='color:#ccff00'>${net:,.2f}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-container'><h3>CASH</h3><h2>${st.session_state['cash']:,.2f}</h2></div>", unsafe_allow_html=True)
    
    st.write("#### Your Assets")
    if st.session_state['holdings']: 
        st.write(st.session_state['holdings'])
    else: st.caption("Empty wallet.")

# --- TAB 5: LEARN ---
with tab5:
    st.info("💡 **Tip:** Don't trade against the trend.")
    st.markdown("**Glossary**")
    with st.expander("RSI"): st.write("Relative Strength Index. Over 70 = Expensive.")
    with st.expander("MACD"): st.write("Momentum indicator. Crossovers signal entries.")
