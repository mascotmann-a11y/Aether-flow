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
import hashlib
import io

# --- 1. CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER APEX v22.0", 
    page_icon=" ", 
    initial_sidebar_state="collapsed" 
)

# NEW DB NAME FOR STABILITY
DB_NAME = 'aether_v22.db'

# --- 2. SAFE START ENGINE (CRASH GUARD) ---
# This ensures all variables exist instantly so KeyError is impossible.
if 'init_done' not in st.session_state:
    st.session_state['username'] = 'Guest'
    st.session_state['logged_in'] = False
    st.session_state['cash'] = 100000.00
    st.session_state['xp'] = 0
    st.session_state['level'] = 'Novice'
    st.session_state['holdings'] = {}
    st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])
    st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]
    st.session_state['ai_memory'] = {"sentiment": 50, "last_scan": "Never"}
    st.session_state['selected_ticker'] = "BTC-USD"
    st.session_state['intel_data'] = []
    st.session_state['db_init'] = False
    st.session_state['init_done'] = True

# --- 3. DATABASE SYSTEM ---
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users 
                     (username TEXT PRIMARY KEY, password TEXT, cash REAL, xp INTEGER, level TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio 
                     (username TEXT, ticker TEXT, shares REAL, 
                     UNIQUE(username, ticker))''')
        c.execute('''CREATE TABLE IF NOT EXISTS journal 
                     (username TEXT, date TEXT, ticker TEXT, action TEXT, 
                     price REAL, shares REAL, total REAL, notes TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS alerts
                     (username TEXT, ticker TEXT, target_price REAL, condition TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database Error: {e}")

if not st.session_state['db_init']:
    init_db()
    st.session_state['db_init'] = True

# Security & Logic Functions
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def login_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT password, cash, xp, level FROM users WHERE username=?", (username,))
        data = c.fetchone()
        conn.close()
        if data and check_hashes(password, data[0]):
            return data 
    except: pass
    return None

def create_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        hashed_pw = make_hashes(password)
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (username, hashed_pw, 100000.00, 0, "Novice"))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def update_cash(username, amount):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET cash = ? WHERE username = ?", (amount, username))
    conn.commit()
    conn.close()

def add_xp(username, amount):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT xp FROM users WHERE username=?", (username,))
    res = c.fetchone()
    if res:
        current_xp = res[0]
        new_xp = current_xp + amount
        
        new_level = "Novice"
        if new_xp > 100: new_level = "Apprentice"
        if new_xp > 500: new_level = "Trader"
        if new_xp > 1000: new_level = "Pro"
        if new_xp > 5000: new_level = "Market Wizard"
        
        c.execute("UPDATE users SET xp = ?, level = ? WHERE username = ?", (new_xp, new_level, username))
        conn.commit()
        conn.close()
        return new_level
    conn.close()
    return "Novice"

def get_portfolio(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT ticker, shares FROM portfolio WHERE username = ?", conn, params=(username,))
    conn.close()
    return dict(zip(df.ticker, df.shares))

def update_portfolio(username, ticker, shares):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if shares == 0:
        c.execute("DELETE FROM portfolio WHERE username=? AND ticker=?", (username, ticker))
    else:
        c.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?)", (username, ticker, shares))
    conn.commit()
    conn.close()

def log_trade_db(username, ticker, action, price, shares, total, notes="Manual"):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO journal VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
              (username, date_str, ticker, action, price, shares, total, notes))
    conn.commit()
    conn.close()
    return add_xp(username, 10)

def get_journal_db(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT date, ticker, action, price, shares, total, notes FROM journal WHERE username = ?", conn, params=(username,))
    conn.close()
    return df

def get_leaderboard_data():
    conn = sqlite3.connect(DB_NAME)
    df_cash = pd.read_sql_query("SELECT username, cash, level FROM users ORDER BY cash DESC LIMIT 5", conn)
    df_xp = pd.read_sql_query("SELECT username, xp, level FROM users ORDER BY xp DESC LIMIT 5", conn)
    conn.close()
    return df_cash, df_xp

def set_alert_db(username, ticker, price):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO alerts VALUES (?, ?, ?, ?)", (username, ticker, price, "ACTIVE"))
    conn.commit()
    conn.close()

def check_alerts_db(username):
    conn = sqlite3.connect(DB_NAME)
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

# --- 4. BACKEND LOGIC ---

@st.cache_data(ttl=600) 
def get_market_data(ticker):
    try:
        time.sleep(0.05)
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if df.empty: return None
        return df
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
    return rsi.iloc[-1], macd.iloc[-1], signal.iloc[-1]

def deep_scan_web_smart():
    status_text = st.empty()
    progress_bar = st.progress(0)
    bullish_words = ['surge', 'soar', 'jump', 'gain', 'bull', 'buy', 'record', 'profit', 'up']
    bearish_words = ['crash', 'drop', 'fall', 'loss', 'bear', 'sell', 'warning', 'down', 'recession']
    intel_data = []
    
    # Reduced list for stability
    sources = ["https://finance.yahoo.com/news/rssindex", "http://feeds.marketwatch.com/marketwatch/topstories/"]
    
    for i, url in enumerate(sources):
        status_text.text(f"Scanning Intel Source {i+1}...")
        progress_bar.progress((i + 1) / len(sources))
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]: 
                title = entry.title
                link = entry.link
                tag = "NEUTRAL"
                color = "white"
                if any(w in title.lower() for w in bullish_words): 
                    tag = "BULLISH "; color = "#ccff00"
                elif any(w in title.lower() for w in bearish_words): 
                    tag = "BEARISH "; color = "#ff4444"
                intel_data.append({"title": title, "link": link, "tag": tag, "color": color, "date": entry.published[:17]})
        except: continue
    status_text.empty(); progress_bar.empty()
    return intel_data

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
    if tech_score > 1 and sentiment > 55: final_signal = "STRONG BUY"
    elif tech_score > 0 and sentiment > 50: final_signal = "BUY"
    elif tech_score < -1 and sentiment < 45: final_signal = "STRONG SELL"
    elif tech_score < 0 and sentiment < 50: final_signal = "SELL"
    return final_signal, curr

def generate_report_html(username, cash, net_worth, holdings, journal):
    # Crash Guard: Handle empty journal
    journal_html = "No trades yet."
    if journal is not None and not journal.empty:
        journal_html = journal.to_html(index=False)
        
    html = f"""
    <html>
    <head><style>body{{font-family:sans-serif;}} table{{width:100%; border-collapse: collapse;}} th, td{{border: 1px solid #ddd; padding: 8px; text-align: left;}} th{{background-color: #00d2ff;}}</style></head>
    <body>
        <h1>AETHER APEX Portfolio Report</h1>
        <h3>User: {username}</h3>
        <p>Date: {datetime.now().strftime("%Y-%m-%d")}</p>
        <hr>
        <h2>Summary</h2>
        <ul>
            <li><b>Cash Balance:</b> ${cash:,.2f}</li>
            <li><b>Net Worth:</b> ${net_worth:,.2f}</li>
        </ul>
        <h2>Current Holdings</h2>
        <p>{holdings}</p>
        <h2>Recent Transactions</h2>
        {journal_html}
    </body>
    </html>
    """
    return html

def diagnose_portfolio(holdings):
    if not holdings: return "F", " Portfolio Empty. Start trading."
    crypto_count = sum(1 for t in holdings if "-USD" in t)
    total = len(holdings)
    ratio = crypto_count / total if total > 0 else 0
    if ratio > 0.8: return "C-", " High Risk! Too much Crypto."
    elif ratio < 0.2 and total < 3: return "B", "Safe but boring."
    elif total >= 5: return "A+", " Excellent Diversification!"
    else: return "B+", "Good start."

# --- 5. FRONTEND UI ---

# Ticker Tape
st.markdown("""
<div class="ticker-wrap">
<div class="ticker-move">
<div class="ticker-item">BTC: $95,432</div><div class="ticker-item">ETH: $3,421</div><div class="ticker-item">SOL: $145</div>
<div class="ticker-item">NVDA: $135</div><div class="ticker-item">TSLA: $250</div><div class="ticker-item">SPY: $560</div>
</div></div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">  AETHER APEX v22.0 (Stable)</div>', unsafe_allow_html=True)

# LOGIN
if not st.session_state['logged_in']:
    with st.expander(" Secure Login / Register", expanded=True):
        t1, t2 = st.tabs(["Login", "Create Account"])
        with t1:
            l_user = st.text_input("Username")
            l_pass = st.text_input("Password", type="password")
            if st.button("LOGIN", use_container_width=True):
                data = login_user(l_user, l_pass) 
                if data:
                    st.session_state.update({'username':l_user, 'cash':data[1], 'xp':data[2], 'level':data[3], 'logged_in':True})
                    st.session_state['holdings'] = get_portfolio(l_user)
                    st.session_state['journal'] = get_journal_db(l_user)
                    st.success("Access Granted."); time.sleep(0.5); st.rerun()
                else: st.error("Invalid Creds.")
        with t2:
            r_user = st.text_input("New Username")
            r_pass = st.text_input("New Password", type="password")
            if st.button("CREATE", use_container_width=True):
                if len(r_pass)<4: st.warning("Pass > 3 chars")
                elif create_user(r_user, r_pass): st.success("Created! Login now.")
                else: st.error("User exists.")
else:
    c1, c2, c3 = st.columns([2,1,1])
    with c1: st.info(f" **{st.session_state['username']}** | Lvl: {st.session_state['level']} | XP: {st.session_state['xp']}")
    with c2: st.metric("Balance", f"${st.session_state['cash']:,.2f}")
    with c3: 
        if st.button("LOGOUT"):
            for k in ['logged_in','username','cash','holdings','journal']: del st.session_state[k]
            st.rerun()
            
    msgs = check_alerts_db(st.session_state['username'])
    for m in msgs: st.toast(m, icon="")

st.write("---")

# ASSET SELECTOR
c_type, c_search = st.columns([1, 2.5])
with c_type:
    # Basic Asset List
    MARKET_OPTIONS = {
        "CRYPTO": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"},
        "STOCKS": {"Nvidia": "NVDA", "Tesla": "TSLA", "Apple": "AAPL"},
        "FOREX": {"Euro": "EURUSD=X"}
    }
    asset_type = st.selectbox("Market", list(MARKET_OPTIONS.keys()), label_visibility="collapsed")
with c_search:
    category_assets = MARKET_OPTIONS[asset_type]
    friendly_names = list(category_assets.keys()) + ["Other (Type Custom)"]
    selected_friendly = st.selectbox("Select Asset", friendly_names, label_visibility="collapsed")
    if "Other" in selected_friendly:
        ticker = st.text_input("Symbol", value="BTC-USD").upper()
    else:
        ticker = category_assets[selected_friendly]

st.session_state['selected_ticker'] = ticker
df = get_market_data(ticker)

if df is not None:
    curr_price = df['Close'].iloc[-1]
    change = (curr_price - df['Open'].iloc[-1]) / df['Open'].iloc[-1] * 100
    st.markdown(f"### {ticker} : ${curr_price:,.2f} ({change:+.2f}%)")
else:
    st.warning("Market Data Offline. Try again.")
    st.stop()

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Trade", "Chat", "Scan", "Portfolio", "Rankings", "Account", "Intel"])

# TAB 1: TRADE
with tab1:
    col_main, col_alert = st.columns([3, 1])
    with col_alert:
        alert_price = st.number_input("Target ($)", value=float(int(curr_price)), step=100.0)
        if st.button("Set Alert"):
            if st.session_state['logged_in']:
                set_alert_db(st.session_state['username'], ticker, alert_price)
                st.success("Alert Set!")
            else: st.error("Login first.")
            
    with col_main:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    signal, _ = get_signal(df)
    st.metric("AI Signal", signal, delta=st.session_state['ai_memory']['sentiment'])
    
    c_qty, c_buy, c_sell = st.columns([1,1,1])
    with c_qty: qty = st.number_input("Qty", 1.0)
    with c_buy:
        if st.button("BUY", type="primary", use_container_width=True):
            if st.session_state['logged_in']:
                cost = qty * curr_price
                if st.session_state['cash'] >= cost:
                    new_cash = st.session_state['cash'] - cost
                    st.session_state['cash'] = new_cash
                    st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + qty
                    update_cash(st.session_state['username'], new_cash)
                    update_portfolio(st.session_state['username'], ticker, st.session_state['holdings'][ticker])
                    lvl = log_trade_db(st.session_state['username'], ticker, "BUY", curr_price, qty, cost)
                    st.session_state['level'] = lvl
                    st.success("Bought!")
                    time.sleep(0.5); st.rerun()
                else: st.error("No Cash.")
            else: st.error("Login.")
    with c_sell:
        if st.button("SELL", use_container_width=True):
            if st.session_state['logged_in']:
                if st.session_state['holdings'].get(ticker, 0) >= qty:
                    rev = qty * curr_price
                    new_cash = st.session_state['cash'] + rev
                    st.session_state['cash'] = new_cash
                    st.session_state['holdings'][ticker] -= qty
                    update_cash(st.session_state['username'], new_cash)
                    update_portfolio(st.session_state['username'], ticker, st.session_state['holdings'][ticker])
                    lvl = log_trade_db(st.session_state['username'], ticker, "SELL", curr_price, qty, rev)
                    st.session_state['level'] = lvl
                    st.success("Sold!")
                    time.sleep(0.5); st.rerun()
                else: st.error("No Shares.")
            else: st.error("Login.")

# TAB 2: CHAT
with tab2:
    st.info(f"Tip: {random.choice(TRADING_TIPS)}")
    audio_val = st.audio_input("Voice Command")
    prompt = st.chat_input("Ask Oracle...")
    if audio_val: prompt = f"Analyze {ticker}"
    
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        # Simple Response Logic
        response = f"I am analyzing **{ticker}**. Price: ${curr_price:,.2f}. Sentiment: {st.session_state['ai_memory']['sentiment']}."
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
    
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']): st.write(msg['content'])

# TAB 3: SCAN
with tab3:
    if st.button("SCAN MARKET (Heatmap)"):
        data = [{"Ticker": "BTC-USD", "Sector": "Crypto", "Change": random.uniform(-5, 5)},
                {"Ticker": "NVDA", "Sector": "Tech", "Change": random.uniform(-5, 5)},
                {"Ticker": "TSLA", "Sector": "Auto", "Change": random.uniform(-5, 5)}]
        df_h = pd.DataFrame(data)
        df_h['Abs'] = df_h['Change'].abs()
        fig = px.treemap(df_h, path=['Sector', 'Ticker'], values='Abs', color='Change', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: PORTFOLIO
with tab4:
    st.subheader("Holdings")
    st.write(st.session_state['holdings'])
    
    if st.button("Download Report"):
        # CRASH GUARD: Ensure journal exists and handle key errors
        j = st.session_state.get('journal', pd.DataFrame())
        html = generate_report_html(st.session_state['username'], st.session_state['cash'], 0, str(st.session_state['holdings']), j)
        st.download_button("Get PDF/HTML", html, "report.html", "text/html")

    if st.button("Diagnose"):
        g, m = diagnose_portfolio(st.session_state['holdings'])
        st.info(f"Grade: {g} - {m}")
        
    st.write("### Monte Carlo (Beta)")
    if st.button("Run Simulation"):
        # PLOTLY FIX: No 'data=' property
        sim_df = pd.DataFrame()
        for i in range(10): 
            sim_df[i] = [100 * (1 + random.uniform(-0.02, 0.02)) for _ in range(30)]
        
        fig = go.Figure()
        for c in sim_df.columns:
            fig.add_trace(go.Scatter(y=sim_df[c], mode='lines', line=dict(width=1), opacity=0.5))
        st.plotly_chart(fig)

# TAB 5: RANKINGS
with tab5:
    dc, dx = get_leaderboard_data()
    c1, c2 = st.columns(2)
    with c1: st.dataframe(dc)
    with c2: st.dataframe(dx)

# TAB 6: ACCOUNT
with tab6:
    st.metric("Total XP", st.session_state['xp'])
    if st.session_state['journal'] is not None:
        st.dataframe(st.session_state['journal'])

# TAB 7: INTEL
with tab7:
    if st.button("Scan News"):
        intel = deep_scan_web_smart()
        st.session_state['intel_data'] = intel
        
    for i in st.session_state['intel_data']:
        st.markdown(f"**{i['tag']}**: [{i['title']}]({i['link']})")
