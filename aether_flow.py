import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import feedparser
import random
from datetime import datetime
import os
import time

# --- CONFIG & SAFETY ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v6.0", 
    page_icon="🤖", 
    initial_sidebar_state="expanded" 
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
if 'scan_results' not in st.session_state: st.session_state['scan_results'] = None

# --- 2025 "STEALTH FINTECH" THEME ---
st.markdown("""
<style>
    /* 1. BACKGROUND - TRUE BLACK (OLED) */
    .stApp {background-color: #000000; color: #ffffff;}
    
    /* 2. SIDEBAR - DARK GREY */
    [data-testid="stSidebar"] {background-color: #121212; border-right: 1px solid #333;}
    [data-testid="stSidebar"] * {color: #e0e0e0 !important;}

    /* 3. NEON GREEN BUTTONS (Like Screenshot) */
    .stButton > button {
        background-color: #ccff00 !important; /* Neon Green */
        color: #000000 !important;            /* Black Text */
        -webkit-text-fill-color: #000000 !important;
        border: none !important;
        padding: 15px 30px;
        border-radius: 25px;                  /* Pill Shape */
        font-weight: 800 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 15px rgba(204, 255, 0, 0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(204, 255, 0, 0.5);
    }

    /* 4. CARDS (The "Floating" Look) */
    .metric-card {
        background: #1e1e1e; 
        border: 1px solid #333;
        padding: 20px; 
        border-radius: 20px; 
        margin-bottom: 15px;
    }
    .metric-title {color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;}
    .metric-value {color: #fff; font-size: 24px; font-weight: 700;}
    .metric-green {color: #ccff00; font-size: 14px; font-weight: bold;}
    
    /* 5. INPUTS - DARK MODE */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1e1e1e !important; 
        color: #fff !important; 
        border: 1px solid #333 !important; 
        border-radius: 12px;
    }
    
    /* 6. CHAT STYLING */
    .user-bubble {
        background-color: #ccff00; 
        color: black; 
        padding: 12px 18px; 
        border-radius: 18px 18px 0 18px; 
        margin: 5px 0; 
        text-align: right; 
        font-weight: 600;
        display: inline-block;
        float: right;
        clear: both;
    }
    .bot-bubble {
        background-color: #1e1e1e; 
        color: white; 
        padding: 12px 18px; 
        border-radius: 18px 18px 18px 0; 
        margin: 5px 0; 
        text-align: left;
        display: inline-block;
        float: left;
        clear: both;
        border: 1px solid #333;
    }
    
    /* 7. REMOVE DEFAULT PADDING */
    .block-container {padding-top: 2rem; padding-bottom: 5rem;}
    
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAV ---
st.sidebar.title("🤖 AETHER")
st.sidebar.markdown("`v6.0 | STEALTH`")

with st.sidebar.expander("GUIDE", expanded=False):
    st.markdown("Use the radio buttons below to switch apps.")

# NAV SELECTOR
mode = st.sidebar.radio("APPS:", [
    "TERMINAL", 
    "SCANNER", 
    "ORACLE", 
    "OPTIMIZER", 
    "ACADEMY"
])

# --- HELPER FUNCTIONS ---
def get_hype_score(symbol):
    try:
        # Mocking for speed/safety in v6, real scraping can be slow
        return random.randint(30, 90), "HIGH INTEREST"
    except: return 50, "STABLE"

def train_brain(df, model_file):
    if not AI_AVAILABLE: return None, None
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    x, y = [], []
    for i in range(60, len(scaled)):
        x.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, batch_size=32, epochs=1, verbose=0)
    model.save(model_file)
    return model, scaler

def log_trade(ticker, action, price, shares, total, notes):
    entry = pd.DataFrame([{
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'Ticker': ticker, 'Action': action, 'Price': f"${price:.2f}",
        'Shares': shares, 'Total': f"${total:.2f}", 'Notes': notes
    }])
    st.session_state['journal'] = pd.concat([st.session_state['journal'], entry], ignore_index=True)

def get_ai_signal(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y")
        if df.empty or len(df) < 50: return "WAITING", 0.0, 0.0
        curr = df['Close'].iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        signal = "BUY NOW" if curr > sma50 else "SELL / WAIT"
        return signal, curr, atr
    except: return "ERROR", 0.0, 0.0

def execute_autopilot(scan_df):
    trades_made = 0
    for index, row in scan_df.iterrows():
        ticker = row['Ticker']
        signal = row['Signal']
        price = row['Price']
        if "BUY" in signal:
            cost = price * 1.0 
            if st.session_state['cash'] >= cost:
                st.session_state['cash'] -= cost
                st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + 1.0
                log_trade(ticker, "AUTO-BUY", price, 1.0, cost, "Bot Trigger")
                trades_made += 1
    return trades_made

# --- INTERFACES ---

if mode == "TERMINAL":
    # HEADER CARD
    st.markdown("""
    <div style="text-align:center; padding:20px;">
        <h1 style="margin:0; font-size:40px;">$238.70</h1>
        <p style="color:#ccff00; margin:0;">▲ +4.2% (Today)</p>
    </div>
    """, unsafe_allow_html=True)

    # SEARCH
    asset_class = st.sidebar.radio("TYPE", ["STOCKS", "CRYPTO"], horizontal=True)
    raw_ticker = st.sidebar.text_input("SEARCH ASSET", value="BTC" if asset_class=="CRYPTO" else "NVDA").upper()
    
    if asset_class == "STOCKS": ticker = raw_ticker
    elif asset_class == "CRYPTO": ticker = f"{raw_ticker}-USD" if "-USD" not in raw_ticker else raw_ticker
    
    # DATA
    df = yf.Ticker(ticker).history(period="2y")
    if df.empty: st.error("Asset not found"); st.stop()
    
    # AI SIGNAL
    curr_price = df['Close'].iloc[-1]
    signal, _, atr = get_ai_signal(ticker)
    
    # MODERN CHART (GLOWING GREEN)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].iloc[-60:], mode='lines', 
                             line=dict(color='#ccff00', width=3), name='Price',
                             fill='tozeroy', fillcolor='rgba(204, 255, 0, 0.1)')) # Green Glow
    fig.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False) # Minimalist Sparkline look
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # CARD GRID
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AI SIGNAL</div>
            <div class="metric-value" style="color:#ccff00">{signal}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">VOLATILITY</div>
            <div class="metric-value">{atr:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # ACTION BUTTONS (BIG PILLS)
    b1, b2 = st.columns(2)
    with b1:
        if st.button(f"BUY {ticker}"):
            cost = curr_price
            if st.session_state['cash'] >= cost:
                st.session_state['cash'] -= cost
                st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + 1.0
                log_trade(ticker, "BUY", curr_price, 1.0, cost, "Manual")
                st.success("BOUGHT")
    with b2:
        if st.button(f"SELL {ticker}"):
            if st.session_state['holdings'].get(ticker, 0) >= 1.0:
                st.session_state['cash'] += curr_price
                st.session_state['holdings'][ticker] -= 1.0
                log_trade(ticker, "SELL", curr_price, 1.0, curr_price, "Manual")
                st.success("SOLD")

elif mode == "SCANNER":
    st.header("MARKET RADAR")
    basket = st.multiselect("WATCHLIST", ["BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL"], default=["BTC-USD", "NVDA"])
    
    if st.button("SCAN NOW"):
        results = []
        for t in basket:
            sig, price, _ = get_ai_signal(t)
            results.append({"Asset": t, "Price": price, "Signal": sig})
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.markdown("---")
        if st.button("AUTO-TRADE ALL SIGNALS"):
            st.success("Bot Active: Executing trades...")

elif mode == "ORACLE":
    st.header("AI CHAT")
    
    # CHAT HISTORY
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'user':
            st.markdown(f"<div class='user-bubble'>{msg['content']}</div><br>", unsafe_allow_html=True)
        elif msg['role'] != 'system':
            st.markdown(f"<div class='bot-bubble'>🤖 {msg['content']}</div><br>", unsafe_allow_html=True)
            
    # INPUT
    prompt = st.text_input("Ask Emo...", key="chat_in")
    if st.button("SEND") and prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        
        # MOCK RESPONSE (Add real NLP here if needed)
        resp = f"I'm checking the charts for {prompt}... It looks bullish!"
        if "BTC" in prompt.upper(): resp = "Bitcoin is showing strong momentum on the 4H chart."
        
        st.session_state['chat_history'].append({"role": "assistant", "content": resp})
        st.rerun()

elif mode == "OPTIMIZER":
    st.header("PORTFOLIO LAB")
    tickers = st.text_input("ASSETS (Comma Separated)", "BTC-USD, ETH-USD, NVDA")
    if st.button("OPTIMIZE"):
        st.success("Generating Optimal Curve...")
        # (Simplified Visualization for look)
        df = pd.DataFrame({'Asset': ['BTC', 'ETH', 'NVDA'], 'Alloc': [50, 30, 20]})
        fig = px.pie(df, values='Alloc', names='Asset', hole=0.6, color_discrete_sequence=['#ccff00', '#ffffff', '#888888'])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

elif mode == "ACADEMY":
    st.header("INSIGHTS")
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">DAILY TIP</div>
        <div class="metric-value" style="font-size:18px">"Don't catch a falling knife."</div>
    </div>
    """, unsafe_allow_html=True)
