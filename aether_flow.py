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
    page_title="AETHER APEX v9.3", 
    page_icon="💠", 
    initial_sidebar_state="collapsed" 
)

# AI Dependency Safety Check
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Session State Initialization
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]
if 'user_profile' not in st.session_state: st.session_state['user_profile'] = {"name": "Trader", "email": "trader@aether.com"}

# --- 2. ASSET LISTS ---
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

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    /* Global Reset */
    .stApp {background-color: #0e1117; color: #ffffff;}
    
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
    .stButton>button {
        width: 100%; border-radius: 12px; height: 55px; font-weight: 900; font-size: 18px;
    }
    
    /* Credit Card Form Styling */
    .cc-form {
        background-color: #1c1f26; padding: 20px; border-radius: 15px; 
        border: 1px solid #00d2ff; margin-bottom: 20px;
    }

    /* Custom Headers */
    .asset-header {font-size: 24px; font-weight: 900; color: white; margin: 0;}
    .price-pulse {font-size: 32px; font-weight: 900; color: #ccff00; margin: 0;}
    
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 5rem;}
</style>
""", unsafe_allow_html=True)

# --- 4. BACKEND LOGIC ---

@st.cache_data(ttl=300) 
def get_market_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if df.empty: return None
        return df
    except: return None

def get_hype_score(symbol):
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
    if df is None: return "WAITING", 0
    curr = df['Close'].iloc[-1]
    sma20 = df['Close'].rolling(20).mean().iloc[-1]
    signal = "BUY" if curr > sma20 else "SELL"
    return signal, curr

def log_trade(ticker, action, price, shares, total, notes="Manual"):
    entry = pd.DataFrame([{
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"), 'Ticker': ticker, 
        'Action': action, 'Price': f"${price:.2f}", 'Shares': shares, 'Total': f"${total:.2f}", 'Notes': notes
    }])
    st.session_state['journal'] = pd.concat([st.session_state['journal'], entry], ignore_index=True)

# --- 5. FRONTEND UI ---

st.markdown('<div class="main-title">💠 AETHER APEX</div>', unsafe_allow_html=True)

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
    st.markdown(f"<h3>{ticker}</h3>", unsafe_allow_html=True)

# TABS (Renamed Wallet -> Account)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Trade", "Chat", "Scan", "Tools", "Account", "Learn"])

# --- TAB 1: TRADE ---
with tab1:
    if df is not None:
        if AI_AVAILABLE:
            if st.button("🧠 TRAIN BRAIN", use_container_width=True):
                with st.spinner("Training Neural Network..."):
                    train_brain(df, f"model_{ticker}.keras")
                    st.success("AI Model Updated!")

        signal, _ = get_signal(df)
        hype_score, hype_label = get_hype_score(ticker)
        
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0), 
                          paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False,
                          dragmode=False, clickmode='none')
        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': False, 'scrollZoom': False, 'displayModeBar': False})
        
        m1, m2 = st.columns(2)
        with m1: st.markdown(f"<div class='metric-container'><h3>SIGNAL</h3><h2 style='color:#2196F3'>{signal}</h2></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-container'><h3>HYPE</h3><h2>{hype_score}</h2></div>", unsafe_allow_html=True)

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
    else: st.info("Select an asset to load chart.")

# --- TAB 2: CHAT ---
with tab2:
    st.info(f"💡 **Tip:** {random.choice(TRADING_TIPS)}")
    for msg in st.session_state['chat_history']:
        if msg['role'] != 'system':
            with st.chat_message(msg['role']): st.write(msg['content'])
    prompt = st.chat_input("Ask Oracle...")
    if prompt:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        response = f"Checking markets for {prompt}..."
        if "BUY" in prompt.upper(): response = "Check RSI and MACD divergence before entry."
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.write(response)

# --- TAB 3: SCAN ---
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

# --- TAB 4: TOOLS ---
with tab4:
    st.subheader("Optimizer")
    st.caption("Select assets to build a balanced portfolio:")
    selected_assets = st.multiselect("Select Assets", 
                                     ["BTC-USD", "ETH-USD", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOG"],
                                     default=["BTC-USD", "NVDA", "TSLA"])
    if st.button("CALCULATE OPTIMAL MIX", use_container_width=True):
        try:
            if not selected_assets:
                st.error("Please select at least one asset.")
            else:
                data = yf.download(selected_assets, period="1y")['Close']
                weights = np.random.random(len(selected_assets))
                weights /= np.sum(weights)
                fig = px.pie(values=weights, names=selected_assets, title="Recommended Allocation")
                fig.update_layout(template="plotly_dark", paper_bgcolor='#0e1117')
                st.plotly_chart(fig, use_container_width=True)
        except: st.error("Optimization error. Try different assets.")

# --- TAB 5: ACCOUNT (My Account + Payments) ---
with tab5:
    st.header("My Account")
    
    # Profile Section
    with st.expander("👤 User Profile", expanded=False):
        new_name = st.text_input("Display Name", st.session_state['user_profile']['name'])
        new_email = st.text_input("Email", st.session_state['user_profile']['email'])
        if st.button("Save Profile"):
            st.session_state['user_profile']['name'] = new_name
            st.session_state['user_profile']['email'] = new_email
            st.success("Profile Updated!")
    
    # Financial Overview
    total_assets = 0
    for t, s in st.session_state['holdings'].items():
        if s > 0:
            d = get_market_data(t)
            p = d['Close'].iloc[-1] if d is not None else 0
            total_assets += s * p
    net = st.session_state['cash'] + total_assets
    
    st.markdown(f"<div class='metric-container'><h3>NET WORTH</h3><h1 style='color:#ccff00'>${net:,.2f}</h1></div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: st.markdown(f"<div class='metric-container'><h3>CASH BALANCE</h3><h2>${st.session_state['cash']:,.2f}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-container'><h3>ACTIVE POSITIONS</h3><h2>{len(st.session_state['holdings'])}</h2></div>", unsafe_allow_html=True)

    st.write("---")
    
    # DEPOSIT CENTER (The Requested Feature)
    st.subheader("💳 Deposit Funds")
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
        st.session_state['cash'] += deposit_amount
        st.success(f"Successfully deposited ${deposit_amount:,.2f}!")
        time.sleep(1)
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Holdings & Logs
    st.write("#### Your Assets")
    if st.session_state['holdings']: st.write(st.session_state['holdings'])
    else: st.caption("No assets owned yet.")
    
    st.write("#### Transaction History")
    st.dataframe(st.session_state['journal'], use_container_width=True)

# --- TAB 6: LEARN ---
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
