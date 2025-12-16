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
# 1. Feedparser (News)
try:
    import feedparser
except ImportError:
    feedparser = None

# 2. TextBlob (Sentiment)
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

# 3. Prophet (Time-Series) - Fallback to Sklearn if missing
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURATION & DATABASE SETUP ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v23.0", 
    page_icon=" ", 
    initial_sidebar_state="collapsed" 
)

# NEW DB NAME FOR V23
DB_NAME = 'aether_v23.db'

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
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, cash REAL, xp INTEGER, level TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio (username TEXT, ticker TEXT, shares REAL, UNIQUE(username, ticker))''')
    c.execute('''CREATE TABLE IF NOT EXISTS journal (username TEXT, date TEXT, ticker TEXT, action TEXT, price REAL, shares REAL, total REAL, notes TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (username TEXT, ticker TEXT, target_price REAL, condition TEXT)''')
    conn.commit()
    conn.close()

# --- DATABASE FUNCTIONS ---
def login_user(username, password):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("SELECT password, cash, xp, level FROM users WHERE username=?", (username,))
    data = c.fetchone()
    conn.close()
    if data and check_hashes(password, data[0]): return data 
    return None

def create_user(username, password):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    hashed_pw = make_hashes(password)
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, hashed_pw, 100000.00, 0, "Novice"))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

def update_cash(username, amount):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("UPDATE users SET cash = ? WHERE username = ?", (amount, username))
    conn.commit()
    conn.close()

def add_xp(username, amount):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("SELECT xp, level FROM users WHERE username=?", (username,))
    data = c.fetchone()
    current_xp, current_level = data[0], data[1]
    new_xp = current_xp + amount
    new_level = "Novice"
    if new_xp > 100: new_level = "Apprentice"
    if new_xp > 500: new_level = "Trader"
    if new_xp > 1000: new_level = "Pro"
    if new_xp > 5000: new_level = "Market Wizard"
    leveled_up = new_level != current_level
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
    if shares == 0: c.execute("DELETE FROM portfolio WHERE username=? AND ticker=?", (username, ticker))
    else: c.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?)", (username, ticker, shares))
    conn.commit()
    conn.close()

def log_trade_db(username, ticker, action, price, shares, total, notes="Manual"):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO journal VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (username, date_str, ticker, action, price, shares, total, notes))
    conn.commit()
    conn.close()
    return add_xp(username, 10)

def get_journal_db(username):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    df = pd.read_sql_query("SELECT date, ticker, action, price, shares, total, notes FROM journal WHERE username = ? ORDER BY date DESC", conn, params=(username,))
    conn.close()
    return df

def check_alerts_db(username):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("SELECT ticker, target_price, rowid FROM alerts WHERE username=?", (username,))
    alerts = c.fetchall()
    triggered = []
    for ticker, target, rowid in alerts:
        try:
            info = yf.Ticker(ticker).fast_info
            if abs(info.last_price - target) / target < 0.01:
                triggered.append(f" ALERT: {ticker} hit ${target:,.2f}!")
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

# Initialize System
if 'db_init' not in st.session_state:
    init_db()
    st.session_state['db_init'] = True
if 'username' not in st.session_state: st.session_state['username'] = 'Guest'
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'xp' not in st.session_state: st.session_state['xp'] = 0
if 'level' not in st.session_state: st.session_state['level'] = "Novice"
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]
if 'ai_memory' not in st.session_state: st.session_state['ai_memory'] = {"sentiment": 50, "last_scan": "Never", "neural_prediction": None, "prophet_prediction": None}
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "BTC-USD"
if 'leveled_up_toast' not in st.session_state: st.session_state['leveled_up_toast'] = False

# AI Dependency Safety Check
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# --- 2. DATA SOURCES ---
MARKET_OPTIONS = {
    "CRYPTO": {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Solana (SOL)": "SOL-USD", "XRP (Ripple)": "XRP-USD", "Dogecoin (DOGE)": "DOGE-USD"},
    "STOCKS": {"Nvidia (NVDA)": "NVDA", "Tesla (TSLA)": "TSLA", "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "Amazon (AMZN)": "AMZN"},
    "FOREX": {"Euro / USD": "EURUSD=X", "GBP / USD": "GBPUSD=X", "USD / JPY": "JPY=X"},
    "COMMODITIES": {"Gold": "GC=X", "Silver": "SI=X", "Crude Oil": "CL=X"},
    "INDICES": {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC"}
}
NEWS_SOURCES = ["https://finance.yahoo.com/news/rssindex", "http://feeds.marketwatch.com/marketwatch/topstories/"]
TRADING_TIPS = ["Trend is your friend.", "Cut losses early.", "Buy the Rumor, Sell the News.", "Risk management is key."]

# --- 3. CSS STYLING & TICKER TAPE ---
st.markdown("""
<style>
    .stApp {background-color: #0e1117; color: #ffffff;}
    .ticker-wrap {width: 100%; overflow: hidden; background-color: #1c1f26; border-bottom: 1px solid #00d2ff; padding: 5px; box-sizing: border-box;}
    .ticker-move { display: flex; animation: ticker 20s linear infinite; min-width: 200%; }
    @keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .ticker-item { flex: 0 0 auto; padding: 0 2rem; font-weight: bold; color: #ccff00; }
    .main-title {font-size: 32px; font-weight: 900; background: -webkit-linear-gradient(45deg, #00d2ff, #ccff00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; text-shadow: 0px 0px 10px rgba(0, 210, 255, 0.3);}
    .metric-container {background-color: #1c1f26; padding: 12px; border-radius: 12px; text-align: center; border: 1px solid #2d323b; margin-bottom: 5px;}
    div.stButton > button {background-color: #00d2ff !important; color: #000000 !important; border-radius: 12px; height: 55px; font-weight: 900 !important;}
    div.stButton > button:hover {background-color: #ccff00 !important; transform: scale(1.02); box-shadow: 0px 0px 15px rgba(204, 255, 0, 0.4);}
    .cc-form {background-color: #1c1f26; padding: 20px; border-radius: 15px; border: 1px solid #00d2ff; margin-bottom: 20px;}
    .asset-header {font-size: 24px; font-weight: 900; color: white; margin: 0;}
    .price-pulse {font-size: 32px; font-weight: 900; color: #ccff00; margin: 0;}
    .badge-card {background: linear-gradient(135deg, #1e1e2f, #2a2a40); padding: 15px; border-radius: 10px; border: 1px solid #00d2ff; text-align: center; margin: 5px;}
</style>
""", unsafe_allow_html=True)

# --- 4. BACKEND LOGIC ---
@st.cache_data(ttl=600) 
def get_market_data(ticker):
    try:
        time.sleep(0.05)
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        return None if df.empty else df
    except: return None

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
    if feedparser is None: return [{"title": "Feedparser Module Missing", "color": "#ff4444", "tag": "ERROR"}]
    
    intel_data = []
    bullish_words = ['surge', 'soar', 'jump', 'gain', 'bull', 'buy', 'record', 'profit', 'up']
    bearish_words = ['crash', 'drop', 'fall', 'loss', 'bear', 'sell', 'warning', 'down', 'recession']
    
    total_polarity = 0
    count = 0

    for url in NEWS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]: 
                title = entry.title
                tag = "NEUTRAL"
                color = "white"
                
                # V23: TextBlob Sentiment
                score = 0
                if TextBlob:
                    blob = TextBlob(title)
                    score = blob.sentiment.polarity
                    total_polarity += score
                    count += 1
                
                if score > 0.1 or any(w in title.lower() for w in bullish_words): 
                    tag = "BULLISH "
                    color = "#ccff00"
                elif score < -0.1 or any(w in title.lower() for w in bearish_words): 
                    tag = "BEARISH "
                    color = "#ff4444"
                
                intel_data.append({"title": title, "link": entry.link, "tag": tag, "color": color, "date": entry.published[:17]})
        except: continue
    
    # Update AI Memory with "Social Hype"
    if count > 0:
        avg_score = total_polarity / count # -1 to 1
        normalized_sentiment = int((avg_score + 1) * 50) # Convert to 0-100
        st.session_state['ai_memory']['sentiment'] = normalized_sentiment
        
    return intel_data

# --- V22/V23 AI ENGINE ---
def train_lstm(df, epochs=5, lookback=60):
    if not AI_AVAILABLE: return None, None
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    if len(scaled_data) <= lookback: return None, None
        
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2)); model.add(LSTM(50, return_sequences=False)); model.add(Dropout(0.2)); model.add(Dense(25)); model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    return model, scaler

def predict_lstm(model, scaler, df, lookback=60):
    if not model or len(df) < lookback: return None
    data = df.filter(['Close']).values
    last_days = data[-lookback:]
    scaled_last = scaler.transform(last_days)
    X_test = np.array([scaled_last])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred = model.predict(X_test, verbose=0)
    return float(scaler.inverse_transform(pred)[0][0])

def run_prophet_or_linear(df):
    # V23: Battle Arena Opponent
    df_p = df.reset_index()[['Date', 'Close']]
    df_p.columns = ['ds', 'y']
    df_p['ds'] = df_p['ds'].dt.tz_localize(None) # Remove timezone for Prophet
    
    if PROPHET_AVAILABLE:
        m = Prophet(daily_seasonality=True)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        return forecast.iloc[-1]['yhat']
    else:
        # Fallback: Linear Regression on simple integers
        df_p['num'] = range(len(df_p))
        lr = LinearRegression()
        lr.fit(df_p[['num']], df_p['y'])
        return lr.predict([[len(df_p)]])[0]

def get_signal(df):
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    rsi_s, macd_s, sig_s = calculate_technicals(df)
    rsi = rsi_s.iloc[-1] if isinstance(rsi_s, pd.Series) else 50
    macd = macd_s.iloc[-1] if isinstance(macd_s, pd.Series) else 0
    sig = sig_s.iloc[-1] if isinstance(sig_s, pd.Series) else 0
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    
    score = 0
    if curr > sma20: score += 1
    if rsi < 30: score += 2 
    elif rsi > 70: score -= 2 
    if macd > sig: score += 1
    
    # V23: Neural Consensus
    lstm_p = st.session_state['ai_memory'].get('neural_prediction')
    prophet_p = st.session_state['ai_memory'].get('prophet_prediction')
    
    if lstm_p and lstm_p > curr: score += 1
    if prophet_p and prophet_p > curr: score += 1
    
    sentiment = st.session_state['ai_memory']['sentiment']
    
    final = "HOLD"
    if score > 1 and sentiment > 55: final = "STRONG BUY"
    elif score > 0 and sentiment > 50: final = "BUY"
    elif score < -1 and sentiment < 45: final = "STRONG SELL"
    elif score < 0 and sentiment < 50: final = "SELL"
    return final, curr

# --- 5. FRONTEND UI ---
st.markdown("""<div class="ticker-wrap"><div class="ticker-move">
<div class="ticker-item">BTC: $95,432</div><div class="ticker-item">ETH: $3,421</div><div class="ticker-item">SOL: $145</div>
<div class="ticker-item">NVDA: $135</div><div class="ticker-item">TSLA: $250</div><div class="ticker-item">AAPL: $220</div>
</div></div>""", unsafe_allow_html=True)

st.markdown('<div class="main-title">  AETHER APEX v23.0</div>', unsafe_allow_html=True)

# Level Up Toast
if st.session_state.get('leveled_up_toast', False):
    st.balloons()
    st.toast(f" LEVEL UP! {st.session_state['level']}!", icon="")
    st.session_state['leveled_up_toast'] = False

# LOGIN SYSTEM
if not st.session_state['logged_in']:
    with st.expander(" Secure Login / Register", expanded=True):
        t1, t2 = st.tabs(["Login", "Register"])
        with t1:
            u, p = st.text_input("Username"), st.text_input("Password", type="password")
            if st.button("LOGIN", use_container_width=True):
                d = login_user(u, p)
                if d:
                    st.session_state.update({'username':u, 'cash':d[1], 'xp':d[2], 'level':d[3], 'holdings':get_portfolio(u), 'journal':get_journal_db(u), 'logged_in':True})
                    st.rerun()
                else: st.error("Fail")
        with t2:
            nu, npw = st.text_input("New User"), st.text_input("New Pass", type="password")
            if st.button("CREATE", use_container_width=True):
                if create_user(nu, npw): st.success("Created! Login now.")
                else: st.error("Taken.")
else:
    c1, c2, c3 = st.columns([2,1,1])
    with c1: st.info(f" **{st.session_state['username']}** | Lvl: **{st.session_state['level']}**")
    with c2: st.metric("Balance", f"${st.session_state['cash']:,.2f}")
    with c3: 
        if st.button("LOGOUT"): 
            st.session_state.update({'logged_in':False, 'username':'Guest'})
            st.rerun()
    for m in check_alerts_db(st.session_state['username']): st.toast(m, icon="")

st.write("---")

# SEARCH
c_type, c_search = st.columns([1, 2.5])
with c_type: asset_type = st.selectbox("Market", list(MARKET_OPTIONS.keys()))
with c_search:
    opts = list(MARKET_OPTIONS[asset_type].keys()) + ["Other (Type Custom)"]
    sel = st.selectbox("Asset", opts)
    ticker = MARKET_OPTIONS[asset_type][sel] if sel in MARKET_OPTIONS[asset_type] else st.text_input("Symbol", "BTC-USD").upper()
st.session_state['selected_ticker'] = ticker

df = get_market_data(ticker)
if df is not None:
    curr = df['Close'].iloc[-1]
    chg = (curr - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100
    color = "#ccff00" if chg >= 0 else "#ff4444"
    st.markdown(f"<div style='padding:10px;'><p class='asset-header'>{ticker}</p><p class='price-pulse' style='color:{color}'>${curr:,.2f}</p><p style='color:{color}'>{chg:+.2f}% Today</p></div>", unsafe_allow_html=True)
else: st.stop()

# TABS
t1, t2, t3, t4, t5, t6 = st.tabs(["Trade", "Chat", "Scan", "Portfolio", "Account", "Intel"])

# --- TAB 1: TRADE (V23 UPDATES) ---
with t1:
    cm, cs = st.columns([3, 1])
    with cs:
        st.markdown("###  Alert")
        if st.button("Set Alert"): set_alert_db(st.session_state['username'], ticker, curr)
        
        st.write("---")
        # --- V23: NEURAL BATTLE ARENA ---
        with st.expander(" Neural Battle Arena", expanded=True):
            if AI_AVAILABLE:
                if st.button("FIGHT: LSTM vs PROPHET", type="primary"):
                    with st.spinner("Training Models..."):
                        # Model A: LSTM
                        m_lstm, sc = train_lstm(df)
                        if m_lstm:
                            pred_a = predict_lstm(m_lstm, sc, df)
                            st.session_state['ai_memory']['neural_prediction'] = pred_a
                        
                        # Model B: Prophet
                        pred_b = run_prophet_or_linear(df)
                        st.session_state['ai_memory']['prophet_prediction'] = pred_b
                        
                        # Display Results
                        col_a, col_b = st.columns(2)
                        with col_a: 
                            st.metric("LSTM (Deep)", f"${pred_a:,.2f}", delta=f"{pred_a-curr:.2f}")
                        with col_b:
                            lbl = "Prophet" if PROPHET_AVAILABLE else "Linear Reg"
                            st.metric(f"{lbl} (Math)", f"${pred_b:,.2f}", delta=f"{pred_b-curr:.2f}")
                            
                        # Consensus
                        avg_pred = (pred_a + pred_b) / 2
                        st.info(f" Consensus Target: ${avg_pred:,.2f}")

    with cm:
        # --- V23: AUTO-PILOT ---
        with st.expander(" AUTO-PILOT (Active Trading Bot)", expanded=False):
            st.caption("Auto-Pilot will run for 30 seconds, attempting to trade every 3 seconds based on AI signals.")
            if st.button("ACTIVATE AUTO-PILOT (30s SIMULATION)", type="primary"):
                sim_log = st.empty()
                progress = st.progress(0)
                
                for i in range(10): # 10 iterations * 3 sec = 30 sec
                    # Refresh Data
                    d_sim = get_market_data(ticker)
                    sig_sim, p_sim = get_signal(d_sim)
                    
                    # Logic
                    action = "HOLD"
                    if "BUY" in sig_sim: action = "BUY"
                    elif "SELL" in sig_sim: action = "SELL"
                    
                    # Fake Trade Execution for Demo
                    msg = f" T+{i*3}s | Price: ${p_sim:,.2f} | Signal: {sig_sim} | Action: {action}"
                    if action != "HOLD":
                        # Actually Log to DB if logged in
                        if st.session_state['logged_in']:
                            log_trade_db(st.session_state['username'], ticker, action, p_sim, 1, p_sim)
                            msg += "  EXECUTED"
                    
                    sim_log.code(msg)
                    progress.progress((i+1)/10)
                    time.sleep(3)
                st.success("Auto-Pilot Sequence Complete.")

        # Charts
        rsi, macd, sig_l = calculate_technicals(df)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=['#ccff00' if c>=o else '#ff4444' for c,o in zip(df['Close'], df['Open'])]), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='#ff00ff')), row=3, col=1)
        fig.update_layout(template="plotly_dark", height=600, margin=dict(t=0,b=0,l=0,r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Signal Display
        sig, _ = get_signal(df)
        s_col = "#ccff00" if "BUY" in sig else "#ff4444" if "SELL" in sig else "white"
        c1, c2 = st.columns(2)
        with c1: st.markdown(f"<div class='metric-container'><h3>AI SIGNAL</h3><h2 style='color:{s_col}'>{sig}</h2></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-container'><h3>SENTIMENT</h3><h2>{st.session_state['ai_memory']['sentiment']}/100</h2></div>", unsafe_allow_html=True)

        # Manual Trade
        q_c, b_c, s_c = st.columns([1, 1.5, 1.5])
        shares = q_c.number_input("Qty", 1.0)
        if b_c.button("BUY", type="primary"):
            cost = shares * curr
            if st.session_state['cash'] >= cost:
                st.session_state['cash'] -= cost
                st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + shares
                update_cash(st.session_state['username'], st.session_state['cash'])
                update_portfolio(st.session_state['username'], ticker, st.session_state['holdings'][ticker])
                nl, lu = log_trade_db(st.session_state['username'], ticker, "BUY", curr, shares, cost)
                st.session_state['level'] = nl
                if lu: st.session_state['leveled_up_toast'] = True
                st.success("BOUGHT")
                time.sleep(1); st.rerun()
            else: st.error("Funds?")
        
        if s_c.button("SELL"):
            if st.session_state['holdings'].get(ticker, 0) >= shares:
                rev = shares * curr
                st.session_state['cash'] += rev
                st.session_state['holdings'][ticker] -= shares
                update_cash(st.session_state['username'], st.session_state['cash'])
                update_portfolio(st.session_state['username'], ticker, st.session_state['holdings'][ticker])
                nl, lu = log_trade_db(st.session_state['username'], ticker, "SELL", curr, shares, rev)
                st.session_state['level'] = nl
                if lu: st.session_state['leveled_up_toast'] = True
                st.success("SOLD")
                time.sleep(1); st.rerun()
            else: st.error("Assets?")

# --- TAB 6: INTEL (V23 UPGRADE) ---
with t6:
    st.header("Global Intel & Social Sentiment")
    if st.button("SCAN NEWS & SENTIMENT"):
        with st.spinner("Analyzing text polarity..."):
            d = deep_scan_web_smart()
            st.session_state['intel_data'] = d
            st.rerun()
    
    if 'intel_data' in st.session_state:
        for i in st.session_state['intel_data']:
            st.markdown(f"<div style='background:#1c1f26; padding:10px; border-left:4px solid {i['color']}; margin:5px;'><b>{i['title']}</b><br><span style='color:{i['color']}'>{i['tag']}</span></div>", unsafe_allow_html=True)

# TAB 4 (Portfolio) & 5 (Account) - Standard
with t4:
    st.subheader("Holdings")
    st.write(st.session_state['holdings'])
    st.metric("Net Worth", f"${st.session_state['cash'] + sum(st.session_state['holdings'].get(t,0)*get_market_data(t)['Close'].iloc[-1] for t in st.session_state['holdings'] if get_market_data(t) is not None):,.2f}")

with t5:
    st.subheader("Account")
    st.dataframe(st.session_state['journal'], use_container_width=True)

# Other tabs (Chat, Scan) kept simple for brevity as focus is on new features.
