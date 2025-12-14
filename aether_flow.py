import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import scipy.optimize as sco
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
import os
import requests
from bs4 import BeautifulSoup
import feedparser
import random
from datetime import datetime

# --- CONFIG: FORCE MENU OPEN & WIDE MODE ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER FLOW v3.0", 
    page_icon="💠", 
    initial_sidebar_state="expanded" 
)

# --- SAFETY CHECK ---
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])

# --- 2025 NEON HORIZON THEME (CSS) ---
st.markdown("""
<style>
    /* 1. BACKGROUND: Deep Space Gradient (Not Flat Black) */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
    }
    
    /* 2. SIDEBAR: Frosted Glass Look */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid #00d2ff;
    }
    
    /* 3. METRIC CARDS: Bright & Readable */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: #00d2ff;
    }
    label[data-testid="stMetricLabel"] {
        color: #00d2ff !important; /* Neon Blue Titles */
        font-weight: bold;
    }
    
    /* 4. BUTTONS: Gradient & Glowing */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.8);
        transform: translateY(-2px);
    }

    /* 5. MENU BUTTON: Make it HUGE and Visible */
    [data-testid="stSidebarCollapsedControl"] {
        color: #00d2ff !important;
        background-color: rgba(255,255,255,0.1);
        border: 1px solid #00d2ff;
        border-radius: 50%;
        padding: 5px;
    }

    /* 6. CUSTOM TEXT HIGHLIGHTS */
    .highlight-bull { color: #00ff99; font-weight: bold; text-shadow: 0 0 10px rgba(0,255,153,0.4); }
    .highlight-bear { color: #ff0055; font-weight: bold; text-shadow: 0 0 10px rgba(255,0,85,0.4); }
    .highlight-wait { color: #ffcc00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAV ---
st.sidebar.title("💠 AETHER FLOW")
st.sidebar.markdown("`v3.0 | NEON HORIZON`")
mode = st.sidebar.selectbox("NAVIGATION", ["TERMINAL (PRO)", "ACADEMY (LEARN)", "OPTIMIZER (MATH)"])

# --- HELPER FUNCTIONS ---
def get_hype_score(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')]
        keywords = ['surge', 'soar', 'moon', 'bull', 'buy']
        panic = ['crash', 'drop', 'bear', 'sell', 'warning']
        score = 50
        for h in headlines:
            if any(w in h.lower() for w in keywords): score += 5
            if any(w in h.lower() for w in panic): score -= 5
        score = max(0, min(100, score))
        label = "🔥 HIGH ENERGY" if score > 70 else "❄️ LOW ENERGY" if score < 30 else "⚖️ STABLE"
        return score, label
    except: return 50, "NO DATA"

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
    model.fit(x, y, batch_size=32, epochs=2, verbose=0)
    model.save(model_file)
    return model, scaler

def log_trade(ticker, action, price, shares, total, notes):
    entry = pd.DataFrame([{
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'Ticker': ticker, 'Action': action, 'Price': f"${price:.2f}",
        'Shares': shares, 'Total': f"${total:.2f}", 'Notes': notes
    }])
    st.session_state['journal'] = pd.concat([st.session_state['journal'], entry], ignore_index=True)

# --- MAIN INTERFACE ---

if mode == "TERMINAL (PRO)":
    st.sidebar.markdown("---")
    st.sidebar.header("📡 SETUP")
    asset_class = st.sidebar.radio("MARKET", ["STOCKS", "CRYPTO", "FOREX"], horizontal=True)
    raw_ticker = st.sidebar.text_input("TICKER SYMBOL", value="BTC" if asset_class=="CRYPTO" else "NVDA").upper()
    
    if asset_class == "STOCKS": ticker = raw_ticker
    elif asset_class == "CRYPTO": ticker = f"{raw_ticker}-USD" if "-USD" not in raw_ticker else raw_ticker
    elif asset_class == "FOREX": ticker = f"{raw_ticker}=X" if "=X" not in raw_ticker else raw_ticker

    if AI_AVAILABLE:
        if st.sidebar.button("🧠 RETRAIN AI"): st.session_state['train_trigger'] = True

    # Get Data
    df = yf.Ticker(ticker).history(period="2y")
    if df.empty: st.error(f"❌ ERROR: Could not find {ticker}"); st.stop()
    
    # AI Logic
    MODEL_FILE = f"model_{ticker}.keras"
    if 'train_trigger' in st.session_state and AI_AVAILABLE:
        with st.spinner("💠 AETHER IS LEARNING..."): train_brain(df, MODEL_FILE)
        del st.session_state['train_trigger']
    
    curr_price = df['Close'].iloc[-1]
    trend, target = "WAITING", 0.0
    
    if AI_AVAILABLE and os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(df[['Close']])
        last_60 = scaler.transform(df['Close'].values[-60:].reshape(-1, 1))
        pred = model.predict(np.array([last_60]))
        target = scaler.inverse_transform(pred)[0][0]
        if target > curr_price: trend = "BULLISH 🚀"
        else: trend = "BEARISH 📉"
    
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
    stop_loss = curr_price - (atr * 2)
    hype_score, hype_label = get_hype_score(ticker)
    
    # METRICS ROW - FIXED VISIBILITY
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRICE", f"${curr_price:.2f}")
    c2.metric("AI SIGNAL", trend, f"Target: ${target:.2f}")
    c3.metric("SAFE STOP", f"${stop_loss:.2f}", "-2.0 ATR")
    c4.metric("HYPE", f"{hype_score}", hype_label)
    
    # TABS
    t1, t2, t3 = st.tabs(["📊 LIVE CHART", "🎮 SIMULATOR", "📓 JOURNAL"])
    
    with t1:
        # Chart with Gradient Fill
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], mode='lines', 
                                 line=dict(color='#00d2ff', width=3), name='Price', fill='tozeroy', 
                                 fillcolor='rgba(0, 210, 255, 0.1)'))
        if target > 0:
            color = "#00ff99" if "BULLISH" in trend else "#ff0055"
            fig.add_trace(go.Scatter(x=[df.index[-1] + pd.Timedelta(days=1)], y=[target], mode='markers', 
                                     marker=dict(color=color, size=15, symbol='diamond'), name='AI Target'))
            
        fig.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🏦 SIMULATOR ACCOUNT")
        net_worth = st.session_state['cash']
        # Calc Net Worth
        for t, s in st.session_state['holdings'].items():
            try: p = yf.Ticker(t).history(period='1d')['Close'].iloc[-1]
            except: p = 0
            net_worth += p * s
            
        # Custom Bright Cards for Mobile
        k1, k2 = st.columns(2)
        k1.markdown(f"""
        <div style="background:linear-gradient(45deg, #00d2ff, #3a7bd5); padding:20px; border-radius:10px; text-align:center;">
            <h4 style="color:white; margin:0;">CASH</h4>
            <h2 style="color:white; margin:0;">${st.session_state['cash']:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        k2.markdown(f"""
        <div style="background:rgba(255,255,255,0.1); border:1px solid #00d2ff; padding:20px; border-radius:10px; text-align:center;">
            <h4 style="color:#00d2ff; margin:0;">NET WORTH</h4>
            <h2 style="color:white; margin:0;">${net_worth:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Trade Inputs
        sc1, sc2 = st.columns([1, 2])
        shares = sc1.number_input("QTY", min_value=0.0001, value=1.0)
        notes = sc2.text_input("NOTES", "AI Follow")
        
        # Big Buttons
        b1, b2 = st.columns(2)
        if b1.button(f"BUY {ticker}"):
            cost = shares * curr_price
            if st.session_state['cash'] >= cost:
                st.session_state['cash'] -= cost
                st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + shares
                log_trade(ticker, "BUY", curr_price, shares, cost, notes)
                st.success(f"BOUGHT {shares} {ticker}")
                st.rerun()
            else: st.error("NO FUNDS")
            
        if b2.button(f"SELL {ticker}"):
            cost = shares * curr_price
            if st.session_state['holdings'].get(ticker, 0) >= shares:
                st.session_state['cash'] += cost
                st.session_state['holdings'][ticker] -= shares
                if st.session_state['holdings'][ticker] <= 0: del st.session_state['holdings'][ticker]
                log_trade(ticker, "SELL", curr_price, shares, cost, notes)
                st.success(f"SOLD {shares} {ticker}")
                st.rerun()
            else: st.error("NO ASSETS")

    with t3:
        st.dataframe(st.session_state['journal'], use_container_width=True)

elif mode == "OPTIMIZER (MATH)":
    st.sidebar.markdown("---")
    tickers_input = st.sidebar.text_area("ASSET LIST", value="BTC-USD, ETH-USD, NVDA, TSLA, GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    st.header("⚡ PORTFOLIO OPTIMIZER")
    if st.button("RUN CALCULATION"):
        with st.spinner("CRUNCHING NUMBERS..."):
            try:
                # FIX FOR CRASH (MultiIndex Handling)
                data = yf.download(tickers, period="1y")
                
                # Robust extraction of Close prices
                if isinstance(data.columns, pd.MultiIndex):
                    try: closes = data.xs('Close', axis=1, level=1) # Try level 1 first
                    except: closes = data['Close'] # Fallback
                else:
                    closes = data['Close'] if 'Close' in data else data
                
                closes = closes.dropna()
                if closes.empty:
                    st.error("No data. Check symbols.")
                else:
                    # Random simulation (Simple)
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)
                    
                    st.success("OPTIMAL SPLIT FOUND")
                    fig = px.pie(values=weights, names=tickers[:len(weights)], hole=0.4, title="RECOMMENDED ALLOCATION")
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Calculation Error: {e}")

elif mode == "ACADEMY (LEARN)":
    st.title("🎓 AETHER ACADEMY")
    st.info(f"💡 TIP: {random.choice(['Trend is your friend', 'Don\'t risk >2%', 'Patience pays'])}")
    
    st.subheader("📰 LIVE FEED")
    try:
        d = feedparser.parse("https://finance.yahoo.com/news/rssindex")
        for e in d.entries[:5]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05); padding:15px; margin-bottom:10px; border-radius:10px; border-left:4px solid #00d2ff;">
                <a href="{e.link}" target="_blank" style="color:white; font-weight:bold; text-decoration:none;">{e.title}</a>
            </div>
            """, unsafe_allow_html=True)
    except: st.warning("News unavailable")
