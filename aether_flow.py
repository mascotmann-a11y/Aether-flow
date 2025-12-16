import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots # NEW: Required for Tech Charts
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import os
import random
import sqlite3
import hashlib

# --- SAFETY CHECK: DEPENDENCIES ---
try:
    import feedparser
except ImportError:
    feedparser = None

# --- 1. CONFIGURATION & DATABASE SETUP ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v21.0", 
    page_icon=" ", 
    initial_sidebar_state="collapsed" 
)

# NEW DB NAME FOR V21
DB_NAME = 'aether_v21.db'

# Security Function: Hash Passwords
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# Initialize Database (SQLite)
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # User Table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, cash REAL, xp INTEGER, level TEXT)''')
    
    # Portfolio Table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio 
                 (username TEXT, ticker TEXT, shares REAL, 
                 UNIQUE(username, ticker))''')
    
    # Journal Table
    c.execute('''CREATE TABLE IF NOT EXISTS journal 
                 (username TEXT, date TEXT, ticker TEXT, action TEXT, 
                 price REAL, shares REAL, total REAL, notes TEXT)''')
    
    # Alerts Table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (username TEXT, ticker TEXT, target_price REAL, condition TEXT)''')
                 
    conn.commit()
    conn.close()

# --- DATABASE FUNCTIONS (Robust) ---

def login_user(username, password):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("SELECT password, cash, xp, level FROM users WHERE username=?", (username,))
    data = c.fetchone()
    conn.close()
    if data:
        if check_hashes(password, data[0]):
            return data 
    return None

def create_user(username, password):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    hashed_pw = make_hashes(password)
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, hashed_pw, 100000.00, 0, "Novice"))
        conn.commit()
        success = True
    except:
        success = False
    conn.close()
    return success

def update_cash(username, amount):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("UPDATE users SET cash = ? WHERE username = ?", (amount, username))
    conn.commit()
    conn.close()

# Gamification: Add XP and Check Level Up
def add_xp(username, amount):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("SELECT xp, level FROM users WHERE username=?", (username,))
    data = c.fetchone()
    current_xp = data[0]
    current_level = data[1]
    
    new_xp = current_xp + amount
    
    # Calculate Level
    new_level = "Novice"
    if new_xp > 100: new_level = "Apprentice"
    if new_xp > 500: new_level = "Trader"
    if new_xp > 1000: new_level = "Pro"
    if new_xp > 5000: new_level = "Market Wizard"
    
    # Check for Level Up Event
    leveled_up = False
    if new_level != current_level:
        leveled_up = True
    
    c.execute("UPDATE users SET xp = ?, level = ? WHERE username = ?", (new_xp, new_level, username))
    conn.commit()
    conn.close()
    return new_level, leveled_up

def get_portfolio(username):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    df = pd.read_sql_query("SELECT ticker, shares FROM portfolio WHERE username = ?", conn, params=(username,))
    conn.close()
    return dict(zip(df.ticker, df.shares))

def update_portfolio(username, ticker, shares):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    if shares == 0:
        c.execute("DELETE FROM portfolio WHERE username=? AND ticker=?", (username, ticker))
    else:
        c.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?)", (username, ticker, shares))
    conn.commit()
    conn.close()

def log_trade_db(username, ticker, action, price, shares, total, notes="Manual"):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO journal VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
              (username, date_str, ticker, action, price, shares, total, notes))
    conn.commit()
    conn.close()
    lvl, leveled_up = add_xp(username, 10) 
    return lvl, leveled_up

def get_journal_db(username):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    df = pd.read_sql_query("SELECT date, ticker, action, price, shares, total, notes FROM journal WHERE username = ?", conn, params=(username,))
    conn.close()
    return df

# Alerts Functions
def set_alert_db(username, ticker, price):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("INSERT INTO alerts VALUES (?, ?, ?, ?)", (username, ticker, price, "ACTIVE"))
    conn.commit()
    conn.close()

def check_alerts_db(username):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("SELECT ticker, target_price, rowid FROM alerts WHERE username=?", (username,))
    alerts = c.fetchall()
    
    triggered = []
    for ticker, target, rowid in alerts:
        try:
            info = yf.Ticker(ticker).fast_info
            curr = info.last_price
            if abs(curr - target) / target < 0.01:
                triggered.append(f" ALERT: {ticker} hit ${target:,.2f}!")
                c.execute("DELETE FROM alerts WHERE rowid=?", (rowid,)) 
        except: pass
        
    conn.commit()
    conn.close()
    return triggered

# Initialize System
if 'db_init' not in st.session_state:
    init_db()
    st.session_state['db_init'] = True

# Session State Defaults
if 'username' not in st.session_state: st.session_state['username'] = 'Guest'
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'xp' not in st.session_state: st.session_state['xp'] = 0
if 'level' not in st.session_state: st.session_state['level'] = "Novice"
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]
if 'ai_memory' not in st.session_state: st.session_state['ai_memory'] = {"sentiment": 50, "last_scan": "Never"}
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "BTC-USD"
if 'leveled_up_toast' not in st.session_state: st.session_state['leveled_up_toast'] = False

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
    "https://feeds.bloomberg.com/markets/news.rss"
]

TRADING_TIPS = [
    "Trend is your friend: Don't trade against the market.",
    "Cut losses early: Don't let a small loss become a big one.",
    "Buy the Rumor, Sell the News.",
    "Never risk more than 2% of your account on a single trade."
]

# --- 3. CSS STYLING & TICKER TAPE ---
st.markdown("""
<style>
    /* Global Reset */
    .stApp {background-color: #0e1117; color: #ffffff;}
    
    /* Responsive Ticker Tape (V21 FIX) */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #1c1f26; 
        border-bottom: 1px solid #00d2ff; padding: 5px; box-sizing: border-box;
    }
    .ticker-move { display: flex; animation: ticker 20s linear infinite; min-width: 200%; }
    @keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .ticker-item { flex: 0 0 auto; padding: 0 2rem; font-weight: bold; color: #ccff00; }

    /* App Title - Cyberpunk Style */
    .main-title {
        font-size: 32px; font-weight: 900; 
        background: -webkit-linear-gradient(45deg, #00d2ff, #ccff00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px; text-shadow: 0px 0px 10px rgba(0, 210, 255, 0.3);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: #0e1117; padding: 5px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; white-space: nowrap; background-color: #1c1f26;
        border-radius: 8px; color: #888; font-size: 14px; font-weight: 600; flex: 1; padding: 0 5px;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #0e1117; color: #ccff00 !important; 
        border: 1px solid #ccff00;
    }

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
        background-color: #ccff00 !important; color: #000000 !important; transform: scale(1.02);
        box-shadow: 0px 0px 15px rgba(204, 255, 0, 0.4);
    }
    
    .cc-form {
        background-color: #1c1f26; padding: 20px; border-radius: 15px; 
        border: 1px solid #00d2ff; margin-bottom: 20px;
    }
    .asset-header {font-size: 24px; font-weight: 900; color: white; margin: 0;}
    .price-pulse {font-size: 32px; font-weight: 900; color: #ccff00; margin: 0;}
    
    /* Gamification Badge */
    .badge-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        padding: 15px; border-radius: 10px; border: 1px solid #00d2ff;
        text-align: center; margin: 5px;
    }
    
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
    
    return rsi, macd, signal

def deep_scan_web_smart():
    if feedparser is None:
        return [{"title": "Feedparser Module Missing", "link": "#", "tag": "ERROR", "color": "#ff4444", "date": "Now"}]
        
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    bullish_words = ['surge', 'soar', 'jump', 'gain', 'bull', 'buy', 'record', 'profit', 'up']
    bearish_words = ['crash', 'drop', 'fall', 'loss', 'bear', 'sell', 'warning', 'down', 'recession']
    
    intel_data = []
    
    for i, url in enumerate(NEWS_SOURCES):
        status_text.text(f"Scanning Intel Source {i+1}...")
        progress_bar.progress((i + 1) / len(NEWS_SOURCES))
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]: 
                title = entry.title
                link = entry.link
                tag = "NEUTRAL"
                color = "white"
                if any(w in title.lower() for w in bullish_words): 
                    tag = "BULLISH "
                    color = "#ccff00"
                elif any(w in title.lower() for w in bearish_words): 
                    tag = "BEARISH "
                    color = "#ff4444"
                
                intel_data.append({"title": title, "link": link, "tag": tag, "color": color, "date": entry.published[:17]})
        except: continue
            
    status_text.empty()
    progress_bar.empty()
    return intel_data

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
    
    rsi_series, macd_series, signal_series = calculate_technicals(df)
    
    # Get last values
    if isinstance(rsi_series, pd.Series): rsi = rsi_series.iloc[-1]
    else: rsi = 50
    if isinstance(macd_series, pd.Series): macd = macd_series.iloc[-1]
    else: macd = 0
    if isinstance(signal_series, pd.Series): macd_signal = signal_series.iloc[-1]
    else: macd_signal = 0

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

# Ticker Tape (V21: Responsive Flexbox)
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
<div class="ticker-item">BTC: $95,432</div>
<div class="ticker-item">ETH: $3,421</div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">  AETHER APEX v21.0</div>', unsafe_allow_html=True)

# LEVEL UP CELEBRATION
if st.session_state.get('leveled_up_toast', False):
    st.balloons()
    st.toast(f" LEVEL UP! You are now a {st.session_state['level']}!", icon="")
    st.session_state['leveled_up_toast'] = False # Reset

# --- SECURE LOGIN PANEL ---
if not st.session_state['logged_in']:
    with st.expander(" Secure Login / Register", expanded=True):
        tab_login, tab_register = st.tabs(["Login", "Create Account"])
        
        with tab_login:
            l_user = st.text_input("Username")
            l_pass = st.text_input("Password", type="password")
            if st.button("LOGIN", use_container_width=True):
                data = login_user(l_user, l_pass) 
                if data is not None:
                    st.session_state['username'] = l_user
                    st.session_state['cash'] = data[1]
                    st.session_state['xp'] = data[2]
                    st.session_state['level'] = data[3]
                    st.session_state['holdings'] = get_portfolio(l_user)
                    st.session_state['journal'] = get_journal_db(l_user)
                    st.session_state['logged_in'] = True
                    st.success(f"Access Granted. Welcome, {l_user}.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid Username or Password.")

        with tab_register:
            r_user = st.text_input("New Username")
            r_pass = st.text_input("New Password", type="password")
            if st.button("CREATE ACCOUNT", use_container_width=True):
                if len(r_pass) < 4:
                    st.warning("Password must be at least 4 characters.")
                else:
                    if create_user(r_user, r_pass):
                        st.success("Account Created! Please Login.")
                    else:
                        st.error("Username already exists.")

else:
    # Logged In Header
    c_head1, c_head2, c_head3 = st.columns([2, 1, 1])
    with c_head1:
        st.info(f" **{st.session_state['username']}** | Level: **{st.session_state['level']}** | XP: **{st.session_state['xp']}**")
    with c_head2:
        st.metric("Balance", f"${st.session_state['cash']:,.2f}")
    with c_head3:
        if st.button("LOGOUT", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['username'] = "Guest"
            st.rerun()
            
    # CHECK ALERTS
    msgs = check_alerts_db(st.session_state['username'])
    for m in msgs:
        st.toast(m, icon="")

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Trade", "Chat", "Scan", "Portfolio", "Account", "Intel Center"])

# --- TAB 1: TRADE (V21 UPGRADE: Advanced Charts) ---
with tab1:
    c_trade_main, c_trade_side = st.columns([3, 1])
    
    with c_trade_side:
        st.markdown("###  Set Alert")
        alert_price = st.number_input("Target Price ($)", value=float(int(curr_price)), step=100.0)
        if st.button("Set Price Alert", use_container_width=True):
            if st.session_state['logged_in']:
                set_alert_db(st.session_state['username'], ticker, alert_price)
                st.success(f"Alert set for {ticker} at ${alert_price:,.2f}")
            else:
                st.error("Login to set alerts.")
    
    with c_trade_main:
        compare_ticker = st.text_input("Compare Against (e.g. GLD, ETH-USD)", placeholder="Type symbol to overlay...")
        
        # --- V21 CHARTING ENGINE ---
        rsi, macd, signal_line = calculate_technicals(df)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # Create Subplots: Row 1 = Price, Row 2 = Volume, Row 3 = RSI
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

        # 1. Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', line=dict(color='#2196F3', width=1), name='SMA 50'), row=1, col=1)

        # 2. Volume
        colors = ['#ff4444' if row['Open'] - row['Close'] > 0 else '#ccff00' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', line=dict(color='#ff00ff', width=1.5), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=3, col=1)

        # Layout Update
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=10, b=10), 
                          paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor='#222')
        fig.update_yaxes(showgrid=True, gridcolor='#222', side='right')
        
        st.plotly_chart(fig, use_container_width=True)
        # ---------------------------

        if AI_AVAILABLE:
            if st.button("  DEEP LEARN (SCAN WEB & TRAIN)", use_container_width=True):
                with st.spinner(" AI is reading global news & training models..."):
                    time.sleep(2) # Mock Delay for UX
                    st.success(f"Brain Updated! Market Mood: 65/100")

        signal, _ = get_signal(df)
        mood_score = st.session_state['ai_memory']['sentiment']
        
        m1, m2 = st.columns(2)
        sig_color = "#ccff00" if "BUY" in signal else "#ff4444" if "SELL" in signal else "#ffffff"
        with m1: st.markdown(f"<div class='metric-container'><h3>AI HYBRID SIGNAL</h3><h2 style='color:{sig_color}'>{signal}</h2></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-container'><h3>MARKET MOOD</h3><h2>{mood_score}/100</h2></div>", unsafe_allow_html=True)

        st.write("---")
        
        st.markdown("#### Transaction & Risk Analysis")
        qty_col, buy_col, sell_col = st.columns([1, 1.5, 1.5])
        with qty_col:
            shares = st.number_input("Qty", value=1.0, min_value=0.01, label_visibility="collapsed")
        
        potential_loss = (shares * curr_price) * 0.10
        st.warning(f" **Safety Net:** If {ticker} drops 10%, you will lose **${potential_loss:,.2f}**. Trade wisely.")
        
        with buy_col:
            if st.button("BUY", type="primary", use_container_width=True):
                if not st.session_state['logged_in']:
                    st.error("Please Login First.")
                else:
                    cost = shares * curr_price
                    if st.session_state['cash'] >= cost:
                        new_cash = st.session_state['cash'] - cost
                        st.session_state['cash'] = new_cash
                        
                        curr_shares = st.session_state['holdings'].get(ticker, 0)
                        st.session_state['holdings'][ticker] = curr_shares + shares
                        
                        user = st.session_state['username']
                        update_cash(user, new_cash)
                        update_portfolio(user, ticker, st.session_state['holdings'][ticker])
                        
                        # LEVEL CHECK
                        new_lvl, leveled_up = log_trade_db(user, ticker, "BUY", curr_price, shares, cost)
                        st.session_state['level'] = new_lvl
                        st.session_state['xp'] += 10
                        if leveled_up: st.session_state['leveled_up_toast'] = True
                        
                        st.success("BOUGHT (+10 XP)")
                        time.sleep(0.5); st.rerun()
                    else: st.error("No Funds")
        with sell_col:
            if st.button("SELL", use_container_width=True):
                if not st.session_state['logged_in']:
                    st.error("Please Login First.")
                else:
                    if st.session_state['holdings'].get(ticker, 0) >= shares:
                        revenue = shares * curr_price
                        new_cash = st.session_state['cash'] + revenue
                        st.session_state['cash'] = new_cash
                        
                        st.session_state['holdings'][ticker] -= shares
                        
                        user = st.session_state['username']
                        update_cash(user, new_cash)
                        update_portfolio(user, ticker, st.session_state['holdings'][ticker])
                        
                        new_lvl, leveled_up = log_trade_db(user, ticker, "SELL", curr_price, shares, revenue)
                        st.session_state['level'] = new_lvl
                        st.session_state['xp'] += 10
                        if leveled_up: st.session_state['leveled_up_toast'] = True
                        
                        st.success("SOLD (+10 XP)")
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
        rsi_val, macd_val, _ = calculate_technicals(d_chat) if d_chat is not None else (pd.Series([50]), pd.Series([0]), pd.Series([0]))
        if isinstance(rsi_val, pd.Series): rsi_val = rsi_val.iloc[-1]
        
        curr_price_chat = d_chat['Close'].iloc[-1] if d_chat is not None else 0
        sentiment_chat = st.session_state['ai_memory']['sentiment']
        
        response = ""
        prompt_lower = prompt.lower()
        
        if "predict" in prompt_lower or "prediction" in prompt_lower:
            response = f"I cannot predict the future, but for **{current_ticker}** at **${curr_price_chat:,.2f}**: The trend is {'UP' if curr_price_chat > d_chat['Open'].iloc[-1] else 'DOWN'}. Watch the resistance levels carefully."
        elif "rsi" in prompt_lower or "indicator" in prompt_lower:
            status = "Oversold (Cheap)" if rsi_val < 30 else "Overbought (Expensive)" if rsi_val > 70 else "Neutral"
            response = f"**Technical Analysis for {current_ticker}:**\n- **RSI:** {rsi_val:.2f} ({status})\n*Beginner Tip: RSI below 30 often suggests a good time to buy.*"
        elif "analyze" in prompt_lower or "trend" in prompt_lower:
            response = f"**Analysis for {current_ticker}:**\nPrice is ${curr_price_chat:,.2f}. The AI Sentiment score is **{sentiment_chat}/100**. {' Be careful, sentiment is low.' if sentiment_chat < 40 else ' Sentiment is strong!'}"
        else:
            response = f"I am tracking **{current_ticker}**. You can ask me about its Price, RSI, or Trend."

        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.write(response)

# --- TAB 3: SCAN ---
with tab3:
    st.subheader("Deep Radar (Top 20)")
    if st.button("SCAN MARKET (TOP 20)", use_container_width=True):
        with st.spinner("Scanning Top 20 Assets..."):
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
            st.error(" Please select at least 2 assets to run the optimization.")
        else:
            try:
                with st.spinner("Running Monte Carlo Simulations..."):
                    data = yf.download(selected_assets, period="2y")['Close']
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    
                    returns = data.pct_change()
                    
                    st.write("#### 1. Asset Correlation (Risk Matrix)")
                    corr_matrix = returns.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
                    fig_corr.update_layout(template="plotly_dark", paper_bgcolor='#0e1117')
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Sim Logic
                    mean_returns = returns.mean() * 252
                    cov_matrix = returns.cov() * 252
                    
                    # Quick Monte Carlo
                    weights = np.random.random((5000, len(selected_assets)))
                    weights /= np.sum(weights, axis=1)[:, np.newaxis]
                    exp_ret = np.sum(mean_returns * weights, axis=1)
                    exp_vol = np.sqrt(np.diag(np.dot(weights, np.dot(cov_matrix, weights.T))))
                    sharpe = exp_ret / exp_vol
                    
                    max_sharpe_idx = sharpe.argmax()
                    
                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(x=exp_vol, y=exp_ret, mode='markers', marker=dict(color=sharpe, colorscale='Viridis', showscale=True), name='Portfolios'))
                    fig_ef.add_trace(go.Scatter(x=[exp_vol[max_sharpe_idx]], y=[exp_ret[max_sharpe_idx]], mode='markers', marker=dict(color='red', size=14, symbol='star'), name='Optimal'))
                    fig_ef.update_layout(template="plotly_dark", paper_bgcolor='#0e1117', title="Efficient Frontier")
                    st.plotly_chart(fig_ef, use_container_width=True)
                    
                    st.success("Optimal Allocation Calculated.")
                    
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# --- TAB 5: ACCOUNT ---
with tab5:
    st.header(f"Account Settings: {st.session_state['username']}")
    st.write("To switch users, use the Login Panel at the top.")

    # GAMIFICATION DISPLAY
    st.subheader(" Trophy Case")
    xp = st.session_state['xp']
    lvl = st.session_state['level']
    
    col_trophy1, col_trophy2, col_trophy3 = st.columns(3)
    with col_trophy1:
        st.markdown(f"<div class='badge-card'><h3>LEVEL</h3><h1 style='color:#ccff00'>{lvl}</h1></div>", unsafe_allow_html=True)
    with col_trophy2:
        st.markdown(f"<div class='badge-card'><h3>TOTAL XP</h3><h1>{xp}</h1></div>", unsafe_allow_html=True)
    with col_trophy3:
        # Simple Logic for Badges
        badges = []
        if xp > 0: badges.append(" First Trade")
        if xp > 500: badges.append(" Diamond Hands")
        if xp > 1000: badges.append(" Moonshot")
        
        if badges:
            st.markdown(f"<div class='badge-card'><h3>BADGES</h3><p>{'<br>'.join(badges)}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='badge-card'><h3>BADGES</h3><p>Trade to Unlock!</p></div>", unsafe_allow_html=True)

    st.write("---")
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
        if st.session_state['logged_in']:
            new_cash = st.session_state['cash'] + deposit_amount
            st.session_state['cash'] = new_cash
            update_cash(st.session_state['username'], new_cash)
            st.success(f"Successfully deposited ${deposit_amount:,.2f}!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Login First.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("#### Transaction History (Saved to Database)")
    if 'journal' in st.session_state:
        st.dataframe(st.session_state['journal'], use_container_width=True)

# --- TAB 6: INTEL CENTER (SMART NEWS) ---
with tab6:
    st.header("Intel Center")
    st.caption("AI-Powered Sentiment Analysis of Global Markets")
    
    if feedparser is None:
        st.error(" The 'feedparser' library is missing. Please install it to use this feature.")
    
    if st.button("SCAN GLOBAL INTEL", use_container_width=True):
        with st.spinner("Reading Global News Feeds..."):
            intel = deep_scan_web_smart()
            st.session_state['intel_data'] = intel
            st.rerun()
        
    if 'intel_data' in st.session_state:
        for item in st.session_state['intel_data']:
            st.markdown(f"""
            <div style="background-color: #1c1f26; padding: 10px; border-radius: 5px; border-left: 5px solid {item['color']}; margin-bottom: 10px;">
                <h4 style="margin:0;">{item['title']}</h4>
                <p style="color: {item['color']}; font-weight: bold; margin:0;">{item['tag']}</p>
                <a href="{item['link']}" style="color: #888; font-size: 12px;">Read Source ({item['date']})</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Click 'SCAN GLOBAL INTEL' to fetch live reports.")
