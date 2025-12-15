import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

# --- CONFIG & SAFETY ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v7.0", 
    page_icon="💠", 
    initial_sidebar_state="collapsed" 
)

# Session State
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]

# --- 2025 "PRO TRADER" THEME ---
st.markdown("""
<style>
    /* 1. BACKGROUND - Dark Grey/Black */
    .stApp {background-color: #0e1117; color: #ffffff;}
    
    /* 2. TABS - Mimics Bottom Nav */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0e1117;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1c1f26;
        border-radius: 10px;
        color: #888;
        font-weight: 600;
        flex: 1; /* Equal width */
    }
    .stTabs [aria-selected="true"] {
        background-color: #2196F3;
        color: white;
    }

    /* 3. METRIC CARDS */
    .metric-container {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2d323b;
        margin-bottom: 10px;
    }
    
    /* 4. BUY/SELL ACTION BAR */
    .trade-btn-buy {
        background-color: #00C805 !important;
        color: white !important;
        font-weight: bold;
        width: 100%;
        border-radius: 8px;
        height: 50px;
        border: none;
    }
    .trade-btn-sell {
        background-color: #FF3B30 !important;
        color: white !important;
        font-weight: bold;
        width: 100%;
        border-radius: 8px;
        height: 50px;
        border: none;
    }
    
    /* 5. INPUTS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1c1f26 !important;
        color: white !important;
        border: 1px solid #2d323b !important;
        border-radius: 8px;
    }
    
    /* Hide Default Header */
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA ENGINE ---
@st.cache_data(ttl=300) 
def get_market_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="3mo", interval="1d") # Longer history for candlesticks
        if df.empty: return None
        return df
    except: return None

# --- CORE LOGIC ---
def get_signal(df):
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    signal = "BUY" if curr > sma20 else "SELL"
    return signal, curr

def log_trade(ticker, action, price, shares, total):
    entry = pd.DataFrame([{
        'Date': datetime.now().strftime("%H:%M"), 'Ticker': ticker, 
        'Action': action, 'Price': f"${price:.2f}", 'Shares': shares, 'Total': f"${total:.2f}", 'Notes': 'Manual'
    }])
    st.session_state['journal'] = pd.concat([st.session_state['journal'], entry], ignore_index=True)

# --- APP LAYOUT ---
# TOP SEARCH BAR (Always Visible)
c_search, c_type = st.columns([3, 1])
with c_type:
    asset_type = st.selectbox("Type", ["CRYPTO", "STOCKS"], label_visibility="collapsed")
with c_search:
    default_tick = "BTC" if asset_type == "CRYPTO" else "NVDA"
    raw_ticker = st.text_input("Search", value=default_tick, label_visibility="collapsed").upper()

ticker = f"{raw_ticker}-USD" if asset_type == "CRYPTO" and "-USD" not in raw_ticker else raw_ticker
df = get_market_data(ticker)

# MAIN TABS (The "One Screen" Feel)
tab1, tab2, tab3, tab4 = st.tabs(["📈 TRADE", "🤖 ORACLE", "⚡ SCANNER", "💼 WALLET"])

# --- TAB 1: TRADE (Charts & Buttons) ---
with tab1:
    if df is not None:
        curr_price = df['Close'].iloc[-1]
        signal, _ = get_signal(df)
        change = (curr_price - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100
        color = "green" if change >= 0 else "red"
        
        # 1. PRICE HEADER
        st.markdown(f"""
        <div style="text-align:center; padding:10px;">
            <h1 style="margin:0; font-size:36px;">${curr_price:,.2f}</h1>
            <p style="color:{color}; margin:0; font-weight:bold;">{change:+.2f}% Today</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. CANDLESTICK CHART (Like Image 1039)
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'])])
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. ACTION ZONE (Like Image 1038)
        st.write("---")
        qty_col, buy_col, sell_col = st.columns([1, 1, 1])
        
        with qty_col:
            shares = st.number_input("Qty", value=1.0, min_value=0.01, label_visibility="collapsed")
        
        with buy_col:
            # Custom styled button via markdown hack isn't clickable in Streamlit, 
            # so we use standard buttons but rely on the CSS block above to color them green/red.
            if st.button("BUY", key="buy_btn", use_container_width=True):
                cost = shares * curr_price
                if st.session_state['cash'] >= cost:
                    st.session_state['cash'] -= cost
                    st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + shares
                    log_trade(ticker, "BUY", curr_price, shares, cost)
                    st.success(f"Bought {shares} {ticker}")
                    time.sleep(1)
                    st.rerun()
                else: st.error("Insufficient Funds")
                
        with sell_col:
            if st.button("SELL", key="sell_btn", use_container_width=True):
                if st.session_state['holdings'].get(ticker, 0) >= shares:
                    st.session_state['cash'] += shares * curr_price
                    st.session_state['holdings'][ticker] -= shares
                    log_trade(ticker, "SELL", curr_price, shares, shares * curr_price)
                    st.success(f"Sold {shares} {ticker}")
                    time.sleep(1)
                    st.rerun()
                else: st.error("No Assets")

    else:
        st.warning("Loading data or invalid ticker...")

# --- TAB 2: ORACLE (Chat) ---
with tab2:
    st.subheader("Oracle AI")
    for msg in st.session_state['chat_history']:
        if msg['role'] != 'system':
            with st.chat_message(msg['role']):
                st.write(msg['content'])
                
    prompt = st.chat_input("Ask about the market...")
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # Simple Response Logic
        response = f"I am analyzing {prompt}..."
        if "BUY" in prompt.upper(): response = "Check the RSI on the daily chart. Ideally wait for a dip."
        elif "BTC" in prompt.upper(): response = "Bitcoin is showing volatility. Check the 4H candle."
        
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.write(response)

# --- TAB 3: SCANNER ---
with tab3:
    st.subheader("Market Scanner")
    watch = ["BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL"]
    if st.button("SCAN NOW", use_container_width=True):
        res = []
        for t in watch:
            d = get_market_data(t)
            if d is not None:
                s, p = get_signal(d)
                res.append({"Ticker": t, "Price": f"${p:.2f}", "Signal": s})
        st.dataframe(pd.DataFrame(res), use_container_width=True)

# --- TAB 4: WALLET ---
with tab4:
    st.subheader("Portfolio")
    
    # Net Worth Calculation
    total_assets = 0
    for t, s in st.session_state['holdings'].items():
        if s > 0:
            d = get_market_data(t)
            p = d['Close'].iloc[-1] if d is not None else 0
            total_assets += s * p
    
    net_worth = st.session_state['cash'] + total_assets
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='metric-container'><h3>CASH</h3><h2>${st.session_state['cash']:,.2f}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-container'><h3>NET WORTH</h3><h2>${net_worth:,.2f}</h2></div>", unsafe_allow_html=True)
        
    st.write("### Holdings")
    if st.session_state['holdings']:
        st.write(st.session_state['holdings'])
    else:
        st.info("No active trades.")
        
    st.write("### Journal")
    st.dataframe(st.session_state['journal'], use_container_width=True)
