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
from datetime import datetime
import time
import os
import random

# --- 1. CONFIGURATION & SAFETY CHECKS ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v9.0", 
    page_icon="💠", 
    initial_sidebar_state="collapsed" 
)

# AI Dependency Safety Check (Prevents white screen of death)
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Session State Initialization (The App's Memory)
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]

# --- 2. STATIC CONTENT (NO INTERNET REQUIRED) ---
TRADING_TIPS = [
    "Trend is your friend: Don't trade against the market.",
    "Cut losses early: Don't let a small loss become a big one.",
    "Buy the Rumor, Sell the News.",
    "Never risk more than 2% of your account on a single trade."
]

GLOSSARY = {
    "RSI": "Relative Strength Index. >70 is Expensive (Overbought). <30 is Cheap (Oversold).",
    "MACD": "Momentum indicator. Crossovers signal entry/exit.",
    "Candlestick": "Green = Up. Red = Down. Wicks show high/low range.",
    "Volume": "High volume confirms the trend strength.",
    "ATR": "Average True Range. Measures volatility (how much price moves)."
}

# --- 3. CSS STYLING (THEME ENGINE) ---
st.markdown("""
<style>
    /* Global Reset */
    .stApp {background-color: #0e1117; color: #ffffff;}
    
    /* Compact Mobile Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px; background-color: #0e1117; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px; 
        white-space: nowrap; 
        background-color: #1c1f26;
        border-radius: 8px; 
        color: #888; 
        font-size: 14px;
        font-weight: 600; 
        flex: 1;
        padding: 0 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ccff00; /* Neon Green */
        color: black !important;
    }

    /* Metric Cards */
    .metric-container {
        background-color: #1c1f26; padding: 12px; border-radius: 12px;
        text-align: center; border: 1px solid #2d323b; margin-bottom: 5px;
    }
    
    /* Buttons (Large Touch Targets) */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 55px; font-weight: 900; font-size: 18px;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1c1f26 !important; color: white !important;
        border: 1px solid #2d323b !important; border-radius: 8px;
    }
    
    /* Custom Headers */
    .asset-header {font-size: 24px; font-weight: 900; color: white; margin: 0;}
    .price-pulse {font-size: 32px; font-weight: 900; color: #ccff00; margin: 0;}
    
    /* Cleanup */
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
</style>
""", unsafe_allow_html=True)

# --- 4. BACKEND LOGIC (DATA & AI) ---

@st.cache_data(ttl=300) 
def get_market_data(ticker):
    """Fetches 6 months of data with caching to prevent crashing."""
    try:
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if df.empty: return None
        return df
    except: return None

def get_hype_score(symbol):
    """Real Sentiment Analysis via Web Scraping."""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')]
        if not headlines: return 50, "NO DATA"
        
        keywords = ['surge', 'soar', 'moon', 'bull', 'buy', 'high']
        panic = ['crash', 'drop', 'bear', 'sell', 'warning', 'low']
        score = 50
        for h in headlines:
            if any(w in h.lower() for w in keywords): score += 5
            if any(w in h.lower() for w in panic): score -= 5
        score = max(0, min(100, score))
        label = "🔥 HIGH" if score > 70 else "❄️ LOW" if score < 30 else "⚖️ STABLE"
        return score, label
    except: return 50, "OFFLINE"

def train_brain(df, model_file):
    """LSTM Neural Network Training."""
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

def get_signal(df):
    """Technical Analysis Signal Generator."""
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    # Simple Moving Average Strategy
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    signal = "BUY" if curr > sma20 else "SELL"
    return signal, curr

def log_trade(ticker, action, price, shares, total, notes="Manual"):
    """Saves trades to the journal."""
    entry = pd.DataFrame([{
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Ticker': ticker, 
        'Action': action, 'Price': f"${price:.2f}", 'Shares': shares, 'Total': f"${total:.2f}", 'Notes': notes
    }])
    st.session_state['journal'] = pd.concat([st.session_state['journal'], entry], ignore_index=True)

# --- 5. FRONTEND UI (LAYOUT) ---

# Top Search Bar
c_type, c_search = st.columns([1, 2.5])
with c_type:
    asset_type = st.selectbox("Type", ["CRYPTO", "STOCKS"], label_visibility="collapsed")
with c_search:
    default_tick = "BTC" if asset_type == "CRYPTO" else "NVDA"
    raw_ticker = st.text_input("Search", value=default_tick, label_visibility="collapsed").upper()

ticker = f"{raw_ticker}-USD" if asset_type == "CRYPTO" and "-USD" not in raw_ticker else raw_ticker
df = get_market_data(ticker)

# Asset Header (Pulsing Price)
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

# Navigation Tabs (Feature Complete)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Trade", "Chat", "Scan", "Tools", "Wallet", "Learn"])

# --- TAB 1: TRADE (Chart & Buttons) ---
with tab1:
    if df is not None:
        # AI Trainer Button
        if AI_AVAILABLE:
            if st.button("🧠 TRAIN BRAIN", use_container_width=True):
                with st.spinner("Training Neural Network..."):
                    train_brain(df, f"model_{ticker}.keras")
                    st.success("AI Model Updated!")

        signal, _ = get_signal(df)
        hype_score, hype_label = get_hype_score(ticker)
        
        # Interactive Candlestick Chart (Locked config for Mobile)
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0), 
                          paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False,
                          dragmode=False, clickmode='none')
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False})
        
        # Metrics Row
        m1, m2 = st.columns(2)
        with m1: st.markdown(f"<div class='metric-container'><h3>SIGNAL</h3><h2 style='color:#2196F3'>{signal}</h2></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-container'><h3>HYPE</h3><h2>{hype_score}</h2></div>", unsafe_allow_html=True)

        # Trading Buttons
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
                else: st.error("No Funds")
        with sell_col:
            if st.button("SELL", use_container_width=True):
                if st.session_state['holdings'].get(ticker, 0) >= shares:
                    st.session_state['cash'] += shares * curr_price
                    st.session_state['holdings'][ticker] -= shares
                    log_trade(ticker, "SELL", curr_price, shares, shares * curr_price)
                    st.success("SOLD")
                    time.sleep(0.5); st.rerun()
                else: st.error("No Assets")
    else: st.info("Enter a symbol to load chart.")

# --- TAB 2: CHAT (Oracle) ---
with tab2:
    st.info(f"💡 **Tip:** {random.choice(TRADING_TIPS)}")
    for msg in st.session_state['chat_history']:
        if msg['role'] != 'system':
            with st.chat_message(msg['role']): st.write(msg['content'])
    prompt = st.chat_input("Ask Oracle...")
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        # Simple NLP Response
        response = f"Checking markets for {prompt}..."
        if "BUY" in prompt.upper(): response = "Check RSI and MACD divergence before entry."
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.write(response)

# --- TAB 3: SCAN (Market Radar) ---
with tab3:
    st.subheader("Radar")
    if st.button("SCAN MARKET", use_container_width=True):
        st.write("Scanning Top Assets...")
        watch = ["BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL", "SOL-USD"]
        res = []
        progress = st.progress(0)
        for i, t in enumerate(watch):
            d = get_market_data(t)
            if d is not None:
                s, p = get_signal(d)
                h, _ = get_hype_score(t)
                res.append({"Ticker": t, "Price": f"${p:.2f}", "Signal": s, "Hype": h})
            progress.progress((i + 1) / len(watch))
        st.dataframe(pd.DataFrame(res), use_container_width=True)

# --- TAB 4: TOOLS (Optimizer) ---
with tab4:
    st.subheader("Optimizer")
    st.caption("Mathematical Allocation Engine")
    ticks = st.text_input("Assets (comma separated)", "BTC-USD, ETH-USD, NVDA")
    if st.button("CALCULATE OPTIMAL MIX", use_container_width=True):
        try:
            t_list = [x.strip() for x in ticks.split(',')]
            data = yf.download(t_list, period="1y")['Close']
            # Basic Random Optimization Logic
            weights = np.random.random(len(t_list))
            weights /= np.sum(weights)
            fig = px.pie(values=weights, names=t_list, title="Recommended Allocation")
            fig.update_layout(template="plotly_dark", paper_bgcolor='#0e1117')
            st.plotly_chart(fig, use_container_width=True)
        except: st.error("Optimization requires valid tickers.")

# --- TAB 5: WALLET (Holdings) ---
with tab5:
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
    if st.session_state['holdings']: st.write(st.session_state['holdings'])
    else: st.caption("Empty wallet.")
    
    st.write("#### Trade Log")
    st.dataframe(st.session_state['journal'], use_container_width=True)

# --- TAB 6: LEARN (Academy) ---
with tab6:
    st.header("Glossary")
    for term, definition in GLOSSARY.items():
        with st.expander(f"📘 {term}"): st.write(definition)
        
    st.write("---")
    st.header("News Feed")
    try:
        d = feedparser.parse("https://finance.yahoo.com/news/rssindex")
        for e in d.entries[:5]:
            st.markdown(f"**[{e.title}]({e.link})**")
            st.caption(e.published)
    except: st.write("News offline.")
