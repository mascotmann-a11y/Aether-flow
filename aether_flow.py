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
    page_title="AETHER APEX v5.4", 
    page_icon="💠", 
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

# Session State Initialization
if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = [{"role": "system", "content": "I am Oracle. Ask me about the market."}]
if 'scan_results' not in st.session_state: st.session_state['scan_results'] = None

# --- 2025 "BEGINNER VISION" THEME (HIGH VISIBILITY) ---
st.markdown("""
<style>
    /* 1. BACKGROUND */
    .stApp {background-color: #f4f6f9; color: #000000;}
    
    /* 2. SIDEBAR - BLACK TEXT ON WHITE */
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 2px solid #00d2ff;}
    [data-testid="stSidebar"] * {color: #000000 !important;} /* FORCE BLACK TEXT */
    
    /* 3. BUTTONS - FORCE BLACK TEXT ON CYAN */
    .stButton > button {
        background-color: #00d2ff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important; /* Forces Phone Browsers to obey */
        border: 2px solid #005f73 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background-color: #00b4d8 !important;
        transform: scale(1.02);
    }

    /* 4. MENUS & INPUTS - SOLID WHITE BACKGROUND, BLACK TEXT */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #00d2ff !important;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        border: 2px solid #ccc !important;
    }
    
    /* 5. METRIC CARDS */
    .metric-card {
        background: white; border-left: 10px solid #00d2ff;
        padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* 6. CHAT BUBBLES */
    .user-msg {background-color: #00d2ff; color: black; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right; font-weight: bold;}
    .bot-msg {background-color: #e0e0e0; color: black; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: left;}

</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAV ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=50)
st.sidebar.title("AETHER APEX")
st.sidebar.markdown("**Beginner Mode: Active**")

with st.sidebar.expander("❓ HOW TO USE (CLICK ME)", expanded=True):
    st.markdown("""
    **1. TERMINAL:** View charts and buy stocks manually.
    **2. SCANNER:** Find the best stocks automatically.
    **3. ORACLE:** Chat with AI for advice.
    **4. OPTIMIZER:** Calculate portfolio math.
    **5. ACADEMY:** Learn trading basics.
    """)

st.sidebar.markdown("---")
st.sidebar.header("📍 SELECT MENU")
# CHANGED FROM DROPDOWN TO RADIO LIST FOR VISIBILITY
mode = st.sidebar.radio("Choose a Tool:", [
    "TERMINAL (PRO)", 
    "OMNI-SCANNER (AUTO)", 
    "ORACLE CHAT (AI)", 
    "OPTIMIZER (MATH)", 
    "ACADEMY"
])

# --- CORE ENGINES ---
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
        label = "🔥 HIGH" if score > 70 else "❄️ LOW" if score < 30 else "⚖️ STABLE"
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
        if df.empty or len(df) < 50: return "NEUTRAL", 0.0, 0.0
        curr = df['Close'].iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
        signal = "BULLISH 🚀" if curr > sma50 else "BEARISH 📉"
        return signal, curr, atr
    except: return "ERROR", 0.0, 0.0

def execute_autopilot(scan_df):
    trades_made = 0
    for index, row in scan_df.iterrows():
        ticker = row['Ticker']
        signal = row['Signal']
        price = row['Price']
        if "BULLISH" in signal and "HIGH" in row['Hype']:
            cost = price * 1.0 
            if st.session_state['cash'] >= cost:
                st.session_state['cash'] -= cost
                st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + 1.0
                log_trade(ticker, "AUTO-BUY", price, 1.0, cost, "Auto-Pilot Trigger")
                trades_made += 1
    return trades_made

# --- INTERFACE MODES ---

if mode == "TERMINAL (PRO)":
    st.header("📈 TRADING TERMINAL")
    st.info("Step 1: Choose a Market. Step 2: Type a Symbol (like 'NVDA'). Step 3: Check the AI Signal.")
    
    # 1. SETUP
    st.sidebar.markdown("---")
    st.sidebar.header("📡 SIGNAL INPUT")
    asset_class = st.sidebar.radio("Market Type:", ["STOCKS", "CRYPTO", "FOREX"], horizontal=True)
    raw_ticker = st.sidebar.text_input("Type Symbol:", value="BTC" if asset_class=="CRYPTO" else "NVDA").upper()
    
    if asset_class == "STOCKS": ticker = raw_ticker
    elif asset_class == "CRYPTO": ticker = f"{raw_ticker}-USD" if "-USD" not in raw_ticker else raw_ticker
    elif asset_class == "FOREX": ticker = f"{raw_ticker}=X" if "=X" not in raw_ticker else raw_ticker

    if AI_AVAILABLE:
        if st.sidebar.button("🧠 CLICK TO RETRAIN AI"): st.session_state['train_trigger'] = True

    # 2. DATA
    df = yf.Ticker(ticker).history(period="2y")
    if df.empty: st.error(f"❌ ERROR: Could not find {ticker}"); st.stop()
    
    # 3. AI PROCESSING
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
        trend = "BULLISH 🚀" if target > curr_price else "BEARISH 📉"
    
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
    stop_loss = curr_price - (atr * 2)
    hype_score, hype_label = get_hype_score(ticker)
    
    # 4. METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${curr_price:.2f}")
    c2.metric("AI Prediction", trend, f"Target: ${target:.2f}")
    c3.metric("Safety Stop Loss", f"${stop_loss:.2f}", "-2.0 ATR")
    c4.metric("Social Hype", f"{hype_score}", hype_label)
    
    # 5. TABS
    t1, t2, t3 = st.tabs(["📊 LIVE CHART", "🎮 SIMULATOR", "📓 JOURNAL"])
    
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], mode='lines', 
                                 line=dict(color='#00d2ff', width=3), name='Price', 
                                 fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.1)'))
        if target > 0:
            color = "#00b894" if "BULLISH" in trend else "#d63031"
            fig.add_trace(go.Scatter(x=[df.index[-1] + pd.Timedelta(days=1)], y=[target], mode='markers', 
                                     marker=dict(color=color, size=18, symbol='diamond', line=dict(color='white', width=2)), name='AI Target'))
        fig.update_layout(
            template="plotly_white", 
            height=500, 
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='rgba(0,0,0,0)', 
            xaxis=dict(showgrid=False), 
            yaxis=dict(showgrid=True, gridcolor='#eee')
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🏦 SIMULATOR (Fake Money)")
        net_worth = st.session_state['cash']
        for t, s in st.session_state['holdings'].items():
            try: p = yf.Ticker(t).history(period='1d')['Close'].iloc[-1]
            except: p = 0
            net_worth += p * s
            
        k1, k2 = st.columns(2)
        k1.markdown(f"<div class='metric-card'><h4>CASH AVAILABLE</h4><h2>${st.session_state['cash']:,.2f}</h2></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='metric-card'><h4>TOTAL NET WORTH</h4><h2>${net_worth:,.2f}</h2></div>", unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        shares = sc1.number_input("Shares to Buy/Sell:", min_value=0.0001, value=1.0)
        notes = sc2.text_input("Trade Notes:", "Manual Trade")
        
        b1, b2 = st.columns(2)
        if b1.button(f"BUY {ticker}"):
            cost = shares * curr_price
            if st.session_state['cash'] >= cost:
                st.session_state['cash'] -= cost
                st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + shares
                log_trade(ticker, "BUY", curr_price, shares, cost, notes)
                st.success("ORDER FILLED")
                st.rerun()
            else: st.error("NO FUNDS")
            
        if b2.button(f"SELL {ticker}"):
            cost = shares * curr_price
            if st.session_state['holdings'].get(ticker, 0) >= shares:
                st.session_state['cash'] += cost
                st.session_state['holdings'][ticker] -= shares
                if st.session_state['holdings'][ticker] <= 0: del st.session_state['holdings'][ticker]
                log_trade(ticker, "SELL", curr_price, shares, cost, notes)
                st.success("ORDER FILLED")
                st.rerun()
            else: st.error("NO ASSETS")

    with t3:
        st.dataframe(st.session_state['journal'], use_container_width=True)

elif mode == "OMNI-SCANNER (AUTO)":
    st.header("🚀 MARKET OMNI-SCANNER")
    st.info("This tool looks at many stocks at once and tells you which ones are 'Bullish' (Good to Buy).")
    
    basket = st.multiselect("Choose Assets to Scan:", 
                   ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "TSLA", "AAPL", "AMD", "MSFT"],
                   default=["BTC-USD", "NVDA", "TSLA"])
    
    if st.button("START SCANNING"):
        results = []
        progress = st.progress(0)
        for i, ticker in enumerate(basket):
            sig, price, atr = get_ai_signal(ticker)
            hype_score, hype_label = get_hype_score(ticker)
            results.append({"Ticker": ticker, "Price": price, "Signal": sig, "Hype": hype_label})
            progress.progress((i + 1) / len(basket))
        st.session_state['scan_results'] = pd.DataFrame(results)
    
    if st.session_state['scan_results'] is not None:
        df_res = st.session_state['scan_results']
        st.dataframe(df_res, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🤖 AUTO-PILOT")
        st.write("If you click below, the AI will automatically buy any stock that is 'Bullish' and has 'High Hype'.")
        if st.button("ACTIVATE AUTO-PILOT"):
            count = execute_autopilot(df_res)
            if count > 0: st.success(f"✅ AUTO-PILOT EXECUTED {count} TRADES!")
            else: st.warning("⚠️ No trades met the criteria right now.")

elif mode == "ORACLE CHAT (AI)":
    st.header("💬 ORACLE CHAT")
    st.info("Ask questions like: 'How is Bitcoin?' or 'What should I buy?'")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state['chat_history']:
            role_class = "user-msg" if msg['role'] == "user" else "bot-msg"
            st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
    
    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        
        response = "I am processing..."
        u_in = user_input.upper()
        
        detected_ticker = None
        for t in ["BTC", "ETH", "NVDA", "TSLA", "AAPL", "EURUSD"]:
            if t in u_in: detected_ticker = t if t != "BTC" and t != "ETH" else f"{t}-USD"
            
        if detected_ticker:
            sig, price, atr = get_ai_signal(detected_ticker)
            score, label = get_hype_score(detected_ticker)
            response = f"Analyzed **{detected_ticker}**:<br>• Price: ${price:.2f}<br>• AI Signal: {sig}<br>• Hype: {label}"
        elif "BUY" in u_in: 
            response = "I recommend checking the **Omni-Scanner** for assets with 'BULLISH' signals and 'HIGH' hype scores."
        elif "HELLO" in u_in: 
            response = "System Online. Ready to analyze markets."
        else: 
            response = "I track technicals and sentiment. Try asking 'How is NVDA?' or 'Check BTC'."
            
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        st.rerun()

elif mode == "OPTIMIZER (MATH)":
    st.header("⚡ PORTFOLIO OPTIMIZER")
    st.info("This tool uses math to find the perfect mix of stocks to lower your risk.")
    
    tickers_input = st.text_area("List of Assets (comma separated):", value="BTC-USD, ETH-USD, NVDA, TSLA")
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    if st.button("CALCULATE BEST MIX"):
        with st.spinner("CALCULATING..."):
            try:
                data = yf.download(tickers, period="1y")
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.get_level_values(0):
                         closes = data.xs('Close', axis=1, level=0)
                    elif 'Close' in data.columns.get_level_values(1):
                         closes = data.xs('Close', axis=1, level=1)
                    else:
                         closes = data['Close'] if 'Close' in data else data
                else:
                    closes = data['Close'] if 'Close' in data else data
                
                closes = closes.dropna(axis=1, how='all').dropna()
                
                if closes.empty:
                    st.error("No Data Found.")
                else:
                    weights = np.random.random(len(closes.columns))
                    weights /= np.sum(weights)
                    
                    fig = px.pie(values=weights, names=closes.columns, hole=0.5, title="OPTIMAL ALLOCATION")
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig)
            except Exception as e: st.error(f"Error: {e}")

elif mode == "ACADEMY":
    st.title("🎓 ACADEMY")
    st.header("Daily Trading Tip")
    st.info("💡 **Trend is your friend:** Don't try to guess the top or bottom. Trade in the direction the market is already moving.")
    
    st.header("Latest Market News")
    try:
        d = feedparser.parse("https://finance.yahoo.com/news/rssindex")
        for e in d.entries[:5]: st.markdown(f"- [{e.title}]({e.link})")
    except: st.write("News unavailable")
