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

# --- 0. ROBUST IMPORT SAFETY CHECK ---
# We wrap every optional library in a try-except block.
# If a library is missing, the app disables that feature instead of crashing.

# 1. News Parsing
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# 2. Advanced Sentiment (NLP)
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# 3. Time Series AI (Prophet)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# 4. Standard Math AI (Fallback)
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 5. Deep Learning (TensorFlow)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


# --- 1. CONFIGURATION & DATABASE ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v23.0", 
    page_icon="⚡", 
    initial_sidebar_state="collapsed" 
)

DB_NAME = 'aether_v23_stable.db'

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, cash REAL, xp INTEGER, level TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio (username TEXT, ticker TEXT, shares REAL, UNIQUE(username, ticker))''')
    c.execute('''CREATE TABLE IF NOT EXISTS journal (username TEXT, date TEXT, ticker TEXT, action TEXT, price REAL, shares REAL, total REAL, notes TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (username TEXT, ticker TEXT, target_price REAL, condition TEXT)''')
    conn.commit()
    conn.close()

# --- DATABASE FUNCTIONS (Safe Mode) ---
def login_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        c = conn.cursor()
        c.execute("SELECT password, cash, xp, level FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        if data and check_hashes(password, data[0]): return data
    except: return None
    return None

def create_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        c = conn.cursor()
        hashed_pw = make_hashes(password)
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, hashed_pw, 100000.00, 0, "Novice"))
        conn.commit()
        conn.close()
        return True
    except: return False

def update_cash(username, amount):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("UPDATE users SET cash = ? WHERE username = ?", (amount, username))
    conn.commit()
    conn.close()

def log_trade_db(username, ticker, action, price, shares, total, notes="Manual"):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO journal VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, date_str, ticker, action, price, shares, total, notes))
    
    # XP Logic
    c.execute("SELECT xp FROM users WHERE username=?", (username,))
    current_xp = c.fetchone()[0]
    new_xp = current_xp + 10
    
    new_level = "Novice"
    if new_xp > 100: new_level = "Apprentice"
    if new_xp > 500: new_level = "Trader"
    if new_xp > 1000: new_level = "Pro"
    
    c.execute("UPDATE users SET xp = ?, level = ? WHERE username = ?", (new_xp, new_level, username))
    conn.commit()
    conn.close()
    return new_level

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

def set_alert_db(username, ticker, price):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("INSERT INTO alerts VALUES (?, ?, ?, ?)", (username, ticker, price, "ACTIVE"))
    conn.commit()
    conn.close()

# --- INITIALIZATION ---
if 'db_init' not in st.session_state:
    init_db()
    st.session_state['db_init'] = True
if 'username' not in st.session_state: st.session_state.update({'username': 'Guest', 'logged_in': False, 'cash': 100000.00, 'xp': 0, 'level': 'Novice', 'holdings': {}, 'journal': pd.DataFrame()})
if 'ai_memory' not in st.session_state: st.session_state['ai_memory'] = {'sentiment': 50, 'neural_prediction': None, 'battle_prediction': None}
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "BTC-USD"


# --- 2. CORE LOGIC ---
MARKET_OPTIONS = {
    "CRYPTO": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"},
    "STOCKS": {"Nvidia": "NVDA", "Tesla": "TSLA", "Apple": "AAPL", "Microsoft": "MSFT"},
    "INDICES": {"S&P 500": "^GSPC", "Nasdaq": "^IXIC"}
}

NEWS_SOURCES = ["https://finance.yahoo.com/news/rssindex", "http://feeds.marketwatch.com/marketwatch/topstories/"]

@st.cache_data(ttl=300)
def get_market_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        return df if not df.empty else None
    except: return None

def calculate_technicals(df):
    if df is None or len(df) < 26: return 50, 0, 0
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return rsi, macd, signal

def get_signal(df):
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    rsi_s, macd_s, sig_s = calculate_technicals(df)
    rsi = rsi_s.iloc[-1]
    macd = macd_s.iloc[-1]
    sig = sig_s.iloc[-1]
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    
    score = 0
    if curr > sma20: score += 1
    if rsi < 30: score += 2 
    elif rsi > 70: score -= 2
    if macd > sig: score += 1
    
    # AI Influence
    neural = st.session_state['ai_memory'].get('neural_prediction')
    if neural:
        if neural > curr * 1.01: score += 1
        elif neural < curr * 0.99: score -= 1
        
    sentiment = st.session_state['ai_memory']['sentiment']
    
    final = "HOLD"
    if score > 1 and sentiment > 55: final = "STRONG BUY"
    elif score > 0 and sentiment > 50: final = "BUY"
    elif score < -1 and sentiment < 45: final = "STRONG SELL"
    elif score < 0 and sentiment < 50: final = "SELL"
    return final, curr

# --- AI FUNCTIONS (Robust) ---
def train_lstm_safe(df):
    if not AI_AVAILABLE: return None, None
    if not SKLEARN_AVAILABLE: return None, None # Scaler needs sklearn
    
    try:
        data = df.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data)
        
        x, y = [], []
        lookback = 60
        if len(scaled) <= lookback: return None, None
        
        for i in range(lookback, len(scaled)):
            x.append(scaled[i-lookback:i, 0])
            y.append(scaled[i, 0])
            
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
        model.add(Dropout(0.2)); model.add(LSTM(50)); model.add(Dropout(0.2)); model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(x, y, batch_size=32, epochs=2, verbose=0)
        return model, scaler
    except: return None, None

def run_battle_model(df):
    # This runs the "Opponent" model (Prophet or Linear Reg)
    if not SKLEARN_AVAILABLE: return None
    
    # Prepare Data
    df_r = df.reset_index()
    df_r['date_ordinal'] = df_r['Date'].map(datetime.toordinal)
    
    # Fallback to Linear Regression (Very Safe)
    try:
        lr = LinearRegression()
        lr.fit(df_r[['date_ordinal']], df_r['Close'])
        # Predict tomorrow
        tomorrow = df_r['date_ordinal'].iloc[-1] + 1
        pred = lr.predict([[tomorrow]])[0]
        return pred
    except: return None

def scan_sentiment_safe():
    if not FEEDPARSER_AVAILABLE:
        return 50, [{"title": "Feedparser Missing", "color": "#ff4444", "tag": "ERR"}]
    
    intel = []
    total_score = 0
    count = 0
    
    bull_words = ['up', 'growth', 'high', 'gain', 'bull', 'soar']
    bear_words = ['down', 'loss', 'drop', 'crash', 'bear', 'fall']
    
    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                title = entry.title
                
                # Try NLP, fallback to Keyword
                score = 0
                if TEXTBLOB_AVAILABLE:
                    score = TextBlob(title).sentiment.polarity
                else:
                    if any(w in title.lower() for w in bull_words): score = 0.5
                    if any(w in title.lower() for w in bear_words): score = -0.5
                
                total_score += score
                count += 1
                
                # Tagging
                tag = "NEUTRAL"
                col = "white"
                if score > 0.1: tag, col = "BULLISH", "#ccff00"
                if score < -0.1: tag, col = "BEARISH", "#ff4444"
                
                intel.append({'title': title, 'color': col, 'tag': tag, 'link': entry.link})
        except: continue
        
    final_sentiment = 50
    if count > 0:
        avg = total_score / count
        final_sentiment = int((avg + 1) * 50) # Scale -1..1 to 0..100
        
    return final_sentiment, intel

# --- 3. UI LAYOUT ---
st.markdown("""
<style>
    .stApp {background-color: #0e1117; color: white;}
    .ticker-wrap {width: 100%; background: #1c1f26; border-bottom: 1px solid #00d2ff; white-space: nowrap; overflow: hidden; padding: 5px;}
    .main-title {font-size: 32px; font-weight: 900; background: -webkit-linear-gradient(45deg, #00d2ff, #ccff00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0px 0px 15px rgba(0, 210, 255, 0.4);}
    .metric-box {background: #1c1f26; border: 1px solid #333; padding: 15px; border-radius: 10px; text-align: center;}
    div.stButton > button {background-color: #00d2ff !important; color: black !important; font-weight: bold; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ticker-wrap">BTC: $95,000 | ETH: $3,400 | SOL: $145 | NVDA: $135 | TSLA: $250</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">  AETHER APEX v23.0 (Stable)</div>', unsafe_allow_html=True)

# LOGIN
if not st.session_state['logged_in']:
    with st.expander("🔐 Login / Register", expanded=True):
        t1, t2 = st.tabs(["Login", "Register"])
        with t1:
            u = st.text_input("User")
            p = st.text_input("Pass", type="password")
            if st.button("LOGIN"):
                d = login_user(u, p)
                if d:
                    st.session_state.update({'logged_in':True, 'username':u, 'cash':d[1], 'xp':d[2], 'level':d[3], 'holdings':get_portfolio(u)})
                    st.rerun()
                else: st.error("Invalid")
        with t2:
            nu = st.text_input("New User")
            npw = st.text_input("New Pass", type="password")
            if st.button("CREATE"):
                if create_user(nu, npw): st.success("Created!")
                else: st.error("Taken")
else:
    c1, c2, c3 = st.columns(3)
    c1.info(f"User: {st.session_state['username']}")
    c2.metric("Cash", f"${st.session_state['cash']:,.2f}")
    if c3.button("LOGOUT"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    # Alerts
    alerts = check_alerts_db(st.session_state['username'])
    for a in alerts: st.toast(a, icon="🔔")

st.write("---")

# SEARCH
c_type, c_search = st.columns([1, 2.5])
with c_type: cat = st.selectbox("Market", list(MARKET_OPTIONS.keys()))
with c_search:
    opts = list(MARKET_OPTIONS[cat].keys()) + ["Other"]
    sel = st.selectbox("Asset", opts)
    ticker = MARKET_OPTIONS[cat][sel] if sel in MARKET_OPTIONS[cat] else st.text_input("Symbol", "BTC-USD").upper()

st.session_state['selected_ticker'] = ticker
df = get_market_data(ticker)

if df is None:
    st.error("Market Data Unavailable. Try another ticker.")
    st.stop()

# MAIN TABS
t1, t2, t3, t4, t5 = st.tabs(["Trade", "AI Lab", "Portfolio", "Account", "Intel"])

# --- TRADE TAB ---
with t1:
    curr = df['Close'].iloc[-1]
    sig, _ = get_signal(df)
    
    # Top Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Price", f"${curr:,.2f}", f"{(curr - df['Open'].iloc[-1]):.2f}")
    m2.metric("AI Signal", sig)
    m3.metric("Sentiment", f"{st.session_state['ai_memory']['sentiment']}/100")
    
    # Chart
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#0e1117')
    st.plotly_chart(fig, use_container_width=True)
    
    # MANUAL TRADE
    c_qty, c_buy, c_sell = st.columns(3)
    qty = c_qty.number_input("Shares", 1.0)
    
    if c_buy.button("BUY", use_container_width=True):
        cost = qty * curr
        if st.session_state['cash'] >= cost:
            st.session_state['cash'] -= cost
            st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + qty
            update_cash(st.session_state['username'], st.session_state['cash'])
            update_portfolio(st.session_state['username'], ticker, st.session_state['holdings'][ticker])
            log_trade_db(st.session_state['username'], ticker, "BUY", curr, qty, cost)
            st.success("Bought!")
            st.rerun()
        else: st.error("No Funds")
        
    if c_sell.button("SELL", use_container_width=True):
        if st.session_state['holdings'].get(ticker, 0) >= qty:
            rev = qty * curr
            st.session_state['cash'] += rev
            st.session_state['holdings'][ticker] -= qty
            update_cash(st.session_state['username'], st.session_state['cash'])
            update_portfolio(st.session_state['username'], ticker, st.session_state['holdings'][ticker])
            log_trade_db(st.session_state['username'], ticker, "SELL", curr, qty, rev)
            st.success("Sold!")
            st.rerun()
        else: st.error("No Assets")
        
    st.write("---")
    
    # AUTO-PILOT (SAFE VERSION)
    with st.expander("🤖 Auto-Pilot (Simulation Mode)", expanded=False):
        st.info("Click 'Step' to simulate 1 AI decision. No loops, no freezing.")
        if st.button("STEP FORWARD (1 Decision)"):
            with st.spinner("AI Thinking..."):
                time.sleep(0.5)
                decision = "HOLD"
                if "BUY" in sig: decision = "BUY"
                if "SELL" in sig: decision = "SELL"
                
                st.write(f"**AI Decision:** {decision}")
                if decision == "BUY" and st.session_state['cash'] > curr:
                    st.success(f"Simulation: Would have bought 1 share of {ticker}")
                elif decision == "SELL":
                    st.warning(f"Simulation: Would have sold 1 share of {ticker}")
                else:
                    st.write("Simulation: Holding position.")

# --- AI LAB (BATTLE ARENA) ---
with t2:
    st.subheader("⚔️ Neural Battle Arena")
    st.caption("Compare Deep Learning (LSTM) vs Math (Linear Regression)")
    
    if st.button("FIGHT MODELS"):
        with st.spinner("Training Contenders..."):
            # 1. LSTM
            model_lstm, scaler = train_lstm_safe(df)
            pred_lstm = 0
            if model_lstm:
                last_60 = df['Close'].values[-60:].reshape(-1, 1)
                last_60_s = scaler.transform(last_60)
                pred_raw = model_lstm.predict(np.array([last_60_s]), verbose=0)
                pred_lstm = scaler.inverse_transform(pred_raw)[0][0]
                st.session_state['ai_memory']['neural_prediction'] = pred_lstm
            
            # 2. Opponent (Linear Reg)
            pred_opp = run_battle_model(df)
            
            # Display
            c1, c2 = st.columns(2)
            c1.metric("LSTM (Neural Net)", f"${pred_lstm:,.2f}" if pred_lstm else "N/A")
            c2.metric("Math Model", f"${pred_opp:,.2f}" if pred_opp else "N/A")
            
            if pred_lstm and pred_opp:
                avg = (pred_lstm + pred_opp) / 2
                st.info(f"🏆 Consensus Target: ${avg:,.2f}")

# --- INTEL TAB ---
with t5:
    st.subheader("Global Intel Scanner")
    if st.button("SCAN WEB"):
        sent, intel = scan_sentiment_safe()
        st.session_state['ai_memory']['sentiment'] = sent
        st.session_state['intel_data'] = intel
        st.rerun()
        
    if 'intel_data' in st.session_state:
        for item in st.session_state['intel_data']:
             st.markdown(f"<div style='border-left:4px solid {item['color']}; padding:5px; margin:5px; background:#222;'>{item['title']}<br><b>{item['tag']}</b></div>", unsafe_allow_html=True)

# --- PORTFOLIO & ACCOUNT ---
with t3:
    st.subheader("Holdings")
    st.write(st.session_state['holdings'])

with t4:
    st.subheader("Journal")
    try:
        j = pd.read_sql_query("SELECT * FROM journal WHERE username=?", sqlite3.connect(DB_NAME), params=(st.session_state['username'],))
        st.dataframe(j)
    except: st.write("No history.")
