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
from datetime import datetime, timedelta
import time
import os
import random
import sqlite3

# --- 1. CONFIGURATION & DATABASE SETUP ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v17.2", 
    page_icon=" ", 
    initial_sidebar_state="collapsed" 
)

# Initialize Database (SQLite)
def init_db():
    conn = sqlite3.connect('aether.db')
    c = conn.cursor()
    # User Table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, cash REAL)''')
    # Portfolio Table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio 
                 (username TEXT, ticker TEXT, shares REAL, 
                 UNIQUE(username, ticker))''')
    # Journal Table
    c.execute('''CREATE TABLE IF NOT EXISTS journal 
                 (username TEXT, date TEXT, ticker TEXT, action TEXT, 
                 price REAL, shares REAL, total REAL, notes TEXT)''')
    conn.commit()
    conn.close()

# Database Functions
def get_user_cash(username):
    conn = sqlite3.connect('aether.db')
    c = conn.cursor()
    c.execute("SELECT cash FROM users WHERE username=?", (username,))
    res = c.fetchone()
    conn.close()
    if res: return res[0]
    else: return None

def create_user(username):
    conn = sqlite3.connect('aether.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", (username, 100000.00))
    conn.commit()
    conn.close()

def update_cash(username, amount):
    conn = sqlite3.connect('aether.db')
    c = conn.cursor()
    c.execute("UPDATE users SET cash = ? WHERE username = ?", (amount, username))
    conn.commit()
    conn.close()

def get_portfolio(username):
    conn = sqlite3.connect('aether.db')
    df = pd.read_sql_query("SELECT ticker, shares FROM portfolio WHERE username = ?", conn, params=(username,))
    conn.close()
    return dict(zip(df.ticker, df.shares))

def update_portfolio(username, ticker, shares):
    conn = sqlite3.connect('aether.db')
    c = conn.cursor()
    if shares == 0:
        c.execute("DELETE FROM portfolio WHERE username=? AND ticker=?", (username, ticker))
    else:
        c.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?)", (username, ticker, shares))
    conn.commit()
    conn.close()

def log_trade_db(username, ticker, action, price, shares, total, notes="Manual"):
    conn = sqlite3.connect('aether.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO journal VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
              (username, date_str, ticker, action, price, shares, total, notes))
    conn.commit()
    conn.close()

def get_journal_db(username):
    conn = sqlite3.connect('aether.db')
    df = pd.read_sql_query("SELECT date, ticker, action, price, shares, total, notes FROM journal WHERE username = ?", conn, params=(username,))
    conn.close()
    return df

# Initialize System
if 'db_init' not in st.session_state:
    init_db()
    st.session_state['db_init'] = True

# Session State Defaults
if 'username' not in st.session_state: st.session_state['username'] = 'Guest'
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]
if 'ai_memory' not in st.session_state: st.session_state['ai_memory'] = {"sentiment": 50, "last_scan": "Never"}
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "BTC-USD"

# AI Dependency Safety Check
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# --- 2. DATA SOURCES ---
MARKET_OPTIONS = {
    "CRYPTO": {
        "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD", 
        "XRP (Ripple)": "XRP-USD", "Dogecoin (DOGE)": "DOGE-USD"
    },
    "STOCKS": {
        "Nvidia (NVDA)": "NVDA", "Tesla (TSLA)": "TSLA", "Apple (AAPL)": "AAPL", 
        "Microsoft (MSFT)": "MSFT", "Amazon (AMZN)": "AMZN"
    },
    "FOREX": {
        "Euro / USD": "EURUSD=X", "GBP / USD": "GBPUSD=X", "USD / JPY": "JPY=X"
    },
    "COMMODITIES": {
        "Gold": "GC=X", "Silver": "SI=X", "Crude Oil": "CL=X"
    },
    "INDICES": {
        "S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC"
    }
}

NEWS_SOURCES = [
    "https://finance.yahoo.com/news/rssindex",
    "http://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.investing.com/rss/news.rss",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://seekingalpha.com/feed.xml",
    "https://feeds.content.dowjones.io/public/rss/mw/topstories",
    "https://www.wsj.com/xml/rss/3_3008.xml"
]

TRADING_TIPS = [
    "Trend is your friend: Don't trade against the market.",
    "Cut losses early: Don't let a small loss become a big one.",
    "Buy the Rumor, Sell the News.",
    "Never risk more than 2% of your account on a single trade."
]

GLOSSARY = {
    "RSI": "Relative Strength Index. >70 is Expensive (Overbought). <30 is Cheap (Oversold).",
    "MACD": "Momentum indicator. Line crossing Signal Line upwards is Bullish.",
    "Candlestick": "Green = Up. Red = Down.",
    "Volume": "High volume confirms the trend strength.",
    "ATR": "Average True Range. Measures volatility."
}

# --- 3. CSS STYLING & TICKER TAPE ---
st.markdown("""
<style>
    /* Global Reset */
    .stApp {background-color: #0e1117; color: #ffffff;}
    
    /* Ticker Tape */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #1c1f26; 
        border-bottom: 1px solid #00d2ff; padding: 5px; white-space: nowrap;
    }
    .ticker-move { display: inline-block; animation: ticker 30s linear infinite; }
    @keyframes ticker { 0% { transform: translate3d(100%, 0, 0); } 100% { transform: translate3d(-100%, 0, 0); } }
    .ticker-item { display: inline-block; padding: 0 2rem; font-weight: bold; color: #ccff00; }

    /* App Title */
    .main-title {
        font-size: 30px; font-weight: 900; 
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: #0e1117; padding: 5px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; white-space: nowrap; background-color: #1c1f26;
        border-radius: 8px; color: #888; font-size: 14px; font-weight: 600; flex: 1; padding: 0 5px;
    }
    .stTabs [aria-selected="true"] { background-color: #ccff00; color: black !important; }

    /* Cards & Inputs */
    .metric-container {
        background-color: #1c1f26; padding: 12px; border-radius: 12px;
        text-align: center; border: 1px solid #2d323b; margin-bottom: 5px;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {
        background-color: #1c1f26 !important; color: white !important;
        border: 1px solid #2d323b !important; border-radius: 8px;
    }

    /* Button Visibility */
    div.stButton > button {
        background-color: #00d2ff !important; color: #000000 !important; border: none !important;
        width: 100%; border-radius: 12px; height: 55px; font-weight: 900 !important; font-size: 18px !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00aaff !important; color: #ffffff !important; transform: scale(1.02);
    }
    
    .cc-form {
        background-color: #1c1f26; padding: 20px; border-radius: 15px; 
        border: 1px solid #00d2ff; margin-bottom: 20px;
    }
    .asset-header {font-size: 24px; font-weight: 900; color: white; margin: 0;}
    .price-pulse {font-size: 32px; font-weight: 900; color: #ccff00; margin: 0;}
    
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
</style>
""", unsafe_allow_html=True)

# --- 4. BACKEND LOGIC ---

@st.cache_data(ttl=600) 
def get_market_data(ticker):
    try:
        time.sleep(0.05)
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if df.empty: return None
        return df
    except Exception:
        return None

def calculate_technicals(df):
    if df is None or len(df) < 26: return 50, 0, 0  
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    return rsi.iloc[-1], macd.iloc[-1], signal.iloc[-1]

def deep_scan_web():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    bullish_words = ['surge', 'soar', 'jump', 'gain', 'bull', 'buy', 'record', 'profit', 'up', 'growth', 'rally', 'positive', 'high', 'moon', 'breakout']
    bearish_words = ['crash', 'drop', 'fall', 'loss', 'bear', 'sell', 'warning', 'down', 'recession', 'dump', 'negative', 'low', 'crisis', 'inflation', 'panic']
    
    score = 50
    
    for i, url in enumerate(NEWS_SOURCES):
        status_text.text(f"Scanning Source {i+1}/{len(NEWS_SOURCES)}...")
        progress_bar.progress((i + 1) / len(NEWS_SOURCES))
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]: 
                title = entry.title.lower()
                if any(w in title for w in bullish_words): score += 1.5
                if any(w in title for w in bearish_words): score -= 1.5
        except: continue
            
    score = max(0, min(100, score))
    
    st.session_state['ai_memory']['sentiment'] = score
    st.session_state['ai_memory']['last_scan'] = datetime.now().strftime("%H:%M")
    
    status_text.empty()
    progress_bar.empty()
    return score

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
    model.add(Dropout(0.2)) 
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, batch_size=32, epochs=3, verbose=0)
    model.save(model_file)
    return model, scaler

def get_signal(df):
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    
    rsi, macd, macd_signal = calculate_technicals(df)
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    
    tech_score = 0
    if curr > sma20: tech_score += 1
    if rsi < 30: tech_score += 2 
    elif rsi > 70: tech_score -= 2 
    if macd > macd_signal: tech_score += 1
    
    sentiment = st.session_state['ai_memory']['sentiment']
    final_signal = "HOLD"
    
    if tech_score > 1 and sentiment > 55:
        final_signal = "STRONG BUY"
    elif tech_score > 0 and sentiment > 50:
        final_signal = "BUY"
    elif tech_score < -1 and sentiment < 45:
        final_signal = "STRONG SELL"
    elif tech_score < 0 and sentiment < 50:
        final_signal = "SELL"
    
    return final_signal, curr

# --- 5. FRONTEND UI ---

# Ticker Tape Implementation
st.markdown("""
<div class="ticker-wrap">
<div class="ticker-move">
<div class="ticker-item">BTC: $95,432</div>
<div class="ticker-item">ETH: $3,421</div>
<div class="ticker-item">SOL: $145</div>
<div class="ticker-item">NVDA: $135</div>
<div class="ticker-item">TSLA: $250</div>
<div class="ticker-item">AAPL: $220</div>
<div class="ticker-item">SPY: $560</div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">  AETHER APEX v17.2</div>', unsafe_allow_html=True)

# --- V17.2 UPGRADE: MAIN SCREEN LOGIN ---
with st.expander("👤 User Login / Switch Account", expanded=True):
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        input_user = st.text_input("Username (Enter to Create or Login)", value="Guest")
    with col_u2:
        if st.button("LOAD / CREATE", use_container_width=True):
            st.session_state['username'] = input_user
            cash_db = get_user_cash(input_user)
            if cash_db is None:
                create_user(input_user)
                st.session_state['cash'] = 100000.00
                st.session_state['holdings'] = {}
                st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])
                st.success(f"Created New User: {input_user}")
            else:
                st.session_state['cash'] = cash_db
                st.session_state['holdings'] = get_portfolio(input_user)
                st.session_state['journal'] = get_journal_db(input_user)
                st.success(f"Loaded: {input_user}")

st.write("---")

# --- SEARCH BAR ---
c_type, c_search = st.columns([1, 2.5])
with c_type:
    market_cats = list(MARKET_OPTIONS.keys())
    asset_type = st.selectbox("Market", market_cats, label_visibility="collapsed")
with c_search:
    category_assets = MARKET_OPTIONS[asset_type]
    friendly_names = list(category_assets.keys())
    friendly_names.append("Other (Type Custom)")
    selected_friendly = st.selectbox("Select Asset", friendly_names, label_visibility="collapsed")

    if selected_friendly == "Other (Type Custom)":
        raw_ticker = st.text_input("Type Symbol (e.g., COIN)", value="BTC", label_visibility="collapsed").upper()
        if asset_type == "CRYPTO" and "-USD" not in raw_ticker: ticker = f"{raw_ticker}-USD"
        elif asset_type == "FOREX" and "=X" not in raw_ticker: ticker = f"{raw_ticker}=X"
        else: ticker = raw_ticker
    else:
        ticker = category_assets[selected_friendly]
        
st.session_state['selected_ticker'] = ticker

# DATA FETCH
df = get_market_data(ticker)

# Asset Header
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
    st.warning(f"   Market Data Paused for {ticker} (Rate Limit). Try again in 10 minutes.")
    st.stop()

# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Trade", "Chat", "Scan", "Portfolio", "Account", "Learn"])

# --- TAB 1: TRADE (Risk Calc & Comparison) ---
with tab1:
    # 1. Comparison Feature
    compare_ticker = st.text_input("Compare Against (e.g. GLD, ETH-USD)", placeholder="Type symbol to overlay...")
    
    # Pro Chart
    df['SMA_50'] = df['Close'].rolling(50).mean()
    fig = go.Figure()
    # Main Asset
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', line=dict(color='#2196F3', width=1.5), name='Trend (50d)'))
    
    # Comparison Asset Logic
    if compare_ticker:
        comp_df = get_market_data(compare_ticker.upper())
        if comp_df is not None:
            norm_factor = df['Close'].iloc[0] / comp_df['Close'].iloc[0]
            fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Close'] * norm_factor, mode='lines', line=dict(color='#ccff00', width=1.5, dash='dot'), name=f"{compare_ticker} (Norm)"))
    
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        xaxis=dict(showgrid=True, gridcolor='#222', rangeslider=dict(visible=False), title="Date"),
        yaxis=dict(showgrid=True, gridcolor='#222', side='right', tickprefix="$"),
        title=dict(text=f"{ticker} Daily Chart", x=0.5, font=dict(size=14, color='#888')), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    if AI_AVAILABLE:
        if st.button("  DEEP LEARN (SCAN WEB & TRAIN)", use_container_width=True):
            st.toast("Phase 1: Scanning Global News...")
            new_sentiment = deep_scan_web()
            st.toast("Phase 2: Training Deep Neural Network (2-Layer)...")
            train_brain(df, f"model_{ticker}.keras")
            st.success(f"Brain Updated! Market Mood: {new_sentiment}/100")
            time.sleep(1)
            st.rerun()

    signal, _ = get_signal(df)
    mood_score = st.session_state['ai_memory']['sentiment']
    
    m1, m2 = st.columns(2)
    sig_color = "#ccff00" if "BUY" in signal else "#ff4444" if "SELL" in signal else "#ffffff"
    with m1: st.markdown(f"<div class='metric-container'><h3>AI HYBRID SIGNAL</h3><h2 style='color:{sig_color}'>{signal}</h2></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='metric-container'><h3>MARKET MOOD</h3><h2>{mood_score}/100</h2></div>", unsafe_allow_html=True)

    st.write("---")
    
    # 2. Safety Net: Risk Calculator
    st.markdown("#### Transaction & Risk Analysis")
    qty_col, buy_col, sell_col = st.columns([1, 1.5, 1.5])
    with qty_col:
        shares = st.number_input("Qty", value=1.0, min_value=0.01, label_visibility="collapsed")
    
    # Risk Calculation Display
    potential_loss = (shares * curr_price) * 0.10
    st.warning(f"⚠️ **Safety Net:** If {ticker} drops 10%, you will lose **${potential_loss:,.2f}**. Trade wisely.")
    
    with buy_col:
        if st.button("BUY", type="primary", use_container_width=True):
            cost = shares * curr_price
            if st.session_state['cash'] >= cost:
                new_cash = st.session_state['cash'] - cost
                st.session_state['cash'] = new_cash
                
                curr_shares = st.session_state['holdings'].get(ticker, 0)
                st.session_state['holdings'][ticker] = curr_shares + shares
                
                user = st.session_state['username']
                update_cash(user, new_cash)
                update_portfolio(user, ticker, st.session_state['holdings'][ticker])
                log_trade_db(user, ticker, "BUY", curr_price, shares, cost)
                
                st.success("BOUGHT")
                time.sleep(0.5); st.rerun()
            else: st.error("No Funds")
    with sell_col:
        if st.button("SELL", use_container_width=True):
            if st.session_state['holdings'].get(ticker, 0) >= shares:
                revenue = shares * curr_price
                new_cash = st.session_state['cash'] + revenue
                st.session_state['cash'] = new_cash
                
                st.session_state['holdings'][ticker] -= shares
                
                user = st.session_state['username']
                update_cash(user, new_cash)
                update_portfolio(user, ticker, st.session_state['holdings'][ticker])
                log_trade_db(user, ticker, "SELL", curr_price, shares, revenue)
                
                st.success("SOLD")
                time.sleep(0.5); st.rerun()
            else: st.error("No Assets")

# --- TAB 2: CHAT ---
with tab2:
    st.info(f"  **Tip:** {random.choice(TRADING_TIPS)}")
    q1, q2, q3, q4 = st.columns(4)
    user_clicked_chip = None
    if q1.button("Predict Price", use_container_width=True): user_clicked_chip = f"What is the prediction for {st.session_state['selected_ticker']}?"
    if q2.button("Analyze Trend", use_container_width=True): user_clicked_chip = f"Analyze the trend for {st.session_state['selected_ticker']}"
    if q3.button("Explain Indicators", use_container_width=True): user_clicked_chip = "Explain RSI and MACD for this asset."
    if q4.button("Reset Memory", type="primary", use_container_width=True): 
        st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Memory cleared."}]
        st.rerun()
        
    for msg in st.session_state['chat_history']:
        if msg['role'] != 'system':
            with st.chat_message(msg['role']): st.write(msg['content'])

    prompt = st.chat_input("Ask Oracle...")
    if user_clicked_chip: prompt = user_clicked_chip
    
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        current_ticker = st.session_state['selected_ticker']
        d_chat = get_market_data(current_ticker)
        rsi_val, macd_val, _ = calculate_technicals(d_chat) if d_chat is not None else (50, 0, 0)
        curr_price_chat = d_chat['Close'].iloc[-1] if d_chat is not None else 0
        sentiment_chat = st.session_state['ai_memory']['sentiment']
        
        response = ""
        prompt_lower = prompt.lower()
        
        if "predict" in prompt_lower or "prediction" in prompt_lower:
            response = f"I cannot predict the future, but for **{current_ticker}** at **${curr_price_chat:,.2f}**: The trend is {'UP' if curr_price_chat > d_chat['Open'].iloc[-1] else 'DOWN'}. Watch the resistance levels carefully."
        elif "rsi" in prompt_lower or "indicator" in prompt_lower:
            status = "Oversold (Cheap)" if rsi_val < 30 else "Overbought (Expensive)" if rsi_val > 70 else "Neutral"
            response = f"**Technical Analysis for {current_ticker}:**\n- **RSI:** {rsi_val:.2f} ({status})\n- **MACD:** {'Bullish' if macd_val > 0 else 'Bearish'}\n*Beginner Tip: RSI below 30 often suggests a good time to buy.*"
        elif "analyze" in prompt_lower or "trend" in prompt_lower:
            response = f"**Analysis for {current_ticker}:**\nPrice is ${curr_price_chat:,.2f}. The AI Sentiment score is **{sentiment_chat}/100**. {'⚠️ Be careful, sentiment is low.' if sentiment_chat < 40 else '🚀 Sentiment is strong!'}"
        else:
            response = f"I am tracking **{current_ticker}**. You can ask me about its Price, RSI, or Trend."

        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.write(response)

# --- TAB 3: SCAN ---
with tab3:
    st.subheader("Deep Radar (Top 20)")
    if st.button("SCAN MARKET (TOP 20)", use_container_width=True):
        st.write("Initializing Deep Scan...")
        watch = [
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", 
            "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD", "SHIB-USD",
            "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", 
            "GOOG", "META", "AMD", "PLTR", "COIN"
        ]
        res = []
        progress = st.progress(0)
        
        for i, t in enumerate(watch):
            try:
                d = get_market_data(t)
                if d is not None:
                    s, p = get_signal(d)
                    h = st.session_state['ai_memory']['sentiment']
                    h_label = "HIGH" if h > 60 else "LOW"
                    res.append({"Ticker": t, "Price": f"${p:.2f}", "Signal": s, "Sentiment": h_label})
            except: pass
            progress.progress((i + 1) / len(watch))
            
        st.dataframe(pd.DataFrame(res), use_container_width=True)

# --- TAB 4: PORTFOLIO ---
with tab4:
    st.header(f"Portfolio: {st.session_state['username']}")
    
    total_assets = 0
    for t, s in st.session_state['holdings'].items():
        if s > 0:
            d = get_market_data(t)
            if d is not None:
                p = d['Close'].iloc[-1]
                total_assets += s * p
    net = st.session_state['cash'] + total_assets
    
    st.markdown(f"<div class='metric-container'><h3>NET WORTH</h3><h1 style='color:#ccff00'>${net:,.2f}</h1></div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: st.markdown(f"<div class='metric-container'><h3>CASH BALANCE</h3><h2>${st.session_state['cash']:,.2f}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-container'><h3>ACTIVE POSITIONS</h3><h2>{len(st.session_state['holdings'])}</h2></div>", unsafe_allow_html=True)

    st.write("### Your Holdings")
    if st.session_state['holdings']: st.write(st.session_state['holdings'])
    else: st.caption("No assets owned yet.")
    
    st.write("---")

    st.subheader("Future Simulator & Optimizer")
    selected_assets = st.multiselect("Select Assets (Min 3 for best results)", 
                                     ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOG", "AMD"],
                                     default=["BTC-USD", "NVDA", "TSLA", "AAPL"])
    
    if st.button("RUN QUANT ANALYSIS", use_container_width=True):
        if len(selected_assets) < 2:
            st.error("Select at least 2 assets.")
        else:
            try:
                st.write("Fetching Historical Data...")
                data = yf.download(selected_assets, period="2y")['Close']
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                returns = data.pct_change()
                
                st.write("#### 1. Asset Correlation (Risk Matrix)")
                corr_matrix = returns.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap (Red = Moves Together)")
                fig_corr.update_layout(template="plotly_dark", paper_bgcolor='#0e1117')
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.write("#### 2. Efficient Frontier (Risk vs Return)")
                num_portfolios = 5000
                all_weights = np.zeros((num_portfolios, len(selected_assets)))
                ret_arr = np.zeros(num_portfolios)
                vol_arr = np.zeros(num_portfolios)
                sharpe_arr = np.zeros(num_portfolios)
                
                mean_returns = returns.mean() * 252
                cov_matrix = returns.cov() * 252
                
                for ind in range(num_portfolios):
                    weights = np.array(np.random.random(len(selected_assets)))
                    weights = weights / np.sum(weights)
                    all_weights[ind,:] = weights
                    ret_arr[ind] = np.sum(mean_returns * weights)
                    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]
                
                max_sharpe_idx = sharpe_arr.argmax()
                max_sharpe_ret = ret_arr[max_sharpe_idx]
                max_sharpe_vol = vol_arr[max_sharpe_idx]
                
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=vol_arr, y=ret_arr, mode='markers', marker=dict(color=sharpe_arr, colorscale='Viridis', showscale=True), name='Portfolios'))
                fig_ef.add_trace(go.Scatter(x=[max_sharpe_vol], y=[max_sharpe_ret], mode='markers', marker=dict(color='red', size=14, symbol='star'), name='Max Sharpe (Optimal)'))
                fig_ef.update_layout(template="plotly_dark", paper_bgcolor='#0e1117', title="Efficient Frontier", xaxis_title="Volatility (Risk)", yaxis_title="Annual Return")
                st.plotly_chart(fig_ef, use_container_width=True)
                
                st.success("Optimal Portfolio Weights:")
                best_weights = all_weights[max_sharpe_idx,:]
                fig_pie = px.pie(values=best_weights, names=selected_assets, title="Max Sharpe Allocation")
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='#0e1117')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.write("#### 3. Monte Carlo Simulation (Next 30 Days)")
                simulations = 200 
                days = 30
                last_price = 100 
                
                simulation_df = pd.DataFrame()
                
                for x in range(simulations):
                    count = 0
                    daily_vol = max_sharpe_vol / np.sqrt(252)
                    price_series = []
                    price = last_price * (1 + np.random.normal(0, daily_vol))
                    price_series.append(price)
                    
                    for y in range(days):
                        price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                        price_series.append(price)
                        count += 1
                    
                    simulation_df[f"Sim {x}"] = price_series
                    
                fig_mc = go.Figure()
                for col in simulation_df.columns[:50]:
                    fig_mc.add_trace(go.Scatter(
                        x=list(range(days + 1)), 
                        y=simulation_df[col],
                        mode='lines',
                        line=dict(color='#2196F3', width=1),
                        opacity=0.1,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig_mc.update_layout(template="plotly_dark", paper_bgcolor='#0e1117', title="Future Scenarios (30 Days)", xaxis_title="Days", yaxis_title="Portfolio Value ($100 Start)")
                st.plotly_chart(fig_mc, use_container_width=True)
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# --- TAB 5: ACCOUNT ---
with tab5:
    st.header(f"Settings for {st.session_state['username']}")
    st.write("To switch users, use the Login Panel at the top of the app.")

    st.subheader("  Deposit Funds")
    st.markdown('<div class="cc-form">', unsafe_allow_html=True)
    
    dc1, dc2 = st.columns([2, 1])
    with dc1:
        st.text_input("Card Number", placeholder="0000 0000 0000 0000")
    with dc2:
        st.text_input("MM/YY", placeholder="12/26")
        
    dc3, dc4 = st.columns([1, 2])
    with dc3:
        st.text_input("CVC", placeholder="123")
    with dc4:
        deposit_amount = st.number_input("Amount ($)", min_value=100.0, step=100.0, value=1000.0)
        
    if st.button("CONFIRM DEPOSIT", use_container_width=True):
        new_cash = st.session_state['cash'] + deposit_amount
        st.session_state['cash'] = new_cash
        update_cash(st.session_state['username'], new_cash)
        st.success(f"Successfully deposited ${deposit_amount:,.2f}!")
        time.sleep(1)
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("#### Transaction History (Saved to Database)")
    if 'journal' in st.session_state:
        st.dataframe(st.session_state['journal'], use_container_width=True)

# --- TAB 6: LEARN ---
with tab6:
    st.header("Glossary")
    for term, definition in GLOSSARY.items():
        with st.expander(f"  {term}"): st.write(definition)
        
    st.write("---")
    st.header("News Feed")
    try:
        d = feedparser.parse("https://finance.yahoo.com/news/rssindex")
        for e in d.entries[:5]:
            st.markdown(f"**[{e.title}]({e.link})**")
            st.caption(e.published)
    except: st.write("News offline.")
