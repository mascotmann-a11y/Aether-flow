import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    page_title="AETHER APEX v22.0", 
    page_icon=" ", 
    initial_sidebar_state="collapsed" 
)

# DB NAME FOR V22
DB_NAME = 'aether_v22.db'

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
                triggered.append(f"🔔 ALERT: {ticker} hit ${target:,.2f}!")
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
if 'ai_memory' not in st.session_state: st.session_state['ai_memory'] = {"sentiment": 50, "last_scan": "Never", "neural_prediction": None}
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
    
    /* Responsive Ticker Tape */
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
                    tag = "BULLISH 🚀"
                    color = "#ccff00"
                elif any(w in title.lower() for w in bearish_words): 
                    tag = "BEARISH ⚠️"
                    color = "#ff4444"
                
                intel_data.append({"title": title, "link": link, "tag": tag, "color": color, "date": entry.published[:17]})
        except: continue
            
    status_text.empty()
    progress_bar.empty()
    return intel_data

# --- V22 AI ENGINE ---
def train_brain_advanced(df, epochs=5, batch_size=32, look_back=60):
    if not AI_AVAILABLE: return None, None
    
    # Preprocessing
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    
    if len(scaled_data) <= look_back:
        return None, None # Not enough data
        
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # LSTM Architecture
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1)) # Prediction
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    return model, scaler

def predict_next_day(model, scaler, df, look_back=60):
    if model is None or len(df) < look_back: return None
    
    data = df.filter(['Close']).values
    # Get last 'look_back' days
    last_days = data[-look_back:]
    scaled_last = scaler.transform(last_days)
    
    X_test = []
    X_test.append(scaled_last)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Predict
    pred_price = model.predict(X_test, verbose=0)
    pred_price = scaler.inverse_transform(pred_price)
    
    return float(pred_price[0][0])

def get_signal(df):
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    
    rsi_series, macd_series, signal_series = calculate_technicals(df)
    
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
    neural_pred = st.session_state['ai_memory'].get('neural_prediction')
    
    # V22: Integrate Neural Prediction into Signal
    if neural_pred:
        if neural_pred > curr * 1.01: tech_score += 1 # Predicted Up > 1%
        if neural_pred < curr * 0.99: tech_score -= 1 # Predicted Down > 1%

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

st.markdown('<div class="main-title">  AETHER APEX v22.0</div>', unsafe_allow_html=True)

# LEVEL UP CELEBRATION
if st.session_state.get('leveled_up_toast', False):
    st.balloons()
    st.toast(f"🎉 LEVEL UP! You are now a {st.session_state['level']}!", icon="🏆")
    st.session_state['leveled_up_toast'] = False # Reset

# --- SECURE LOGIN PANEL ---
if not st.session_state['logged_in']:
    with st.expander("🔐 Secure Login / Register", expanded=True):
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
