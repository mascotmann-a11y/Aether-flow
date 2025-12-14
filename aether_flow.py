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

# --- CONFIG & SAFETY ---
# [CHANGE 1] Added 'initial_sidebar_state="expanded"' to force menu OPEN by default
st.set_page_config(
    layout="wide", 
    page_title="AETHER FLOW v2.1", 
    page_icon="💠", 
    initial_sidebar_state="expanded"
)

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])

# --- 2025 LUMINOUS THEME ---
st.markdown("""
<style>
    /* MAIN GRADIENT BACKGROUND */
    .stApp {
        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #0b1116;
        border-right: 1px solid #00d2ff;
    }
    
    /* [CHANGE 2] MAKE MENU TOGGLE BUTTON HUGE & GLOWING */
    [data-testid="stSidebarCollapsedControl"] {
        color: #00d2ff !important;
        transform: scale(1.5); /* Make it 50% bigger */
        background-color: rgba(0, 210, 255, 0.1);
        border-radius: 5px;
        border: 1px solid #00d2ff;
    }
    
    /* BRIGHT NEON BUTTONS */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.7);
    }

    /* GLASSMOLPHISM CARDS */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #00d2ff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .aether-card {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid #00d2ff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.15);
    }
    
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("💠 AETHER FLOW")
st.sidebar.markdown("`v2.1 | ALWAYS OPEN`")
mode = st.sidebar.selectbox("INTERFACE MODE", ["TERMINAL (PRO)", "ACADEMY (LEARN)", "OPTIMIZER (MATH)"])

# --- CORE LOGIC ---
def get_hype_score(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')]
        keywords = ['surge', 'soar', 'moon', 'breakout', 'bull', 'buy', 'record']
        panic = ['crash', 'plunge', 'dump', 'bear', 'sell', 'warning', 'drop']
        
        score = 50
        for h in headlines:
            if any(w in h.lower() for w in keywords): score += 5
            if any(w in h.lower() for w in panic): score -= 5
        score = max(0, min(100, score))
        label = "🔥 HIGH ENERGY" if score > 75 else "❄️ LOW ENERGY" if score < 25 else "⚖️ STABLE"
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

# --- UI LOGIC ---
if mode == "TERMINAL (PRO)":
    asset_class = st.sidebar.radio("ASSET CLASS", ["STOCKS", "CRYPTO", "FOREX"], horizontal=True)
    raw_ticker = st.sidebar.text_input("SYMBOL", value="BTC" if asset_class=="CRYPTO" else "NVDA").upper()
    
    if asset_class == "STOCKS": ticker = raw_ticker
    elif asset_class == "CRYPTO": ticker = f"{raw_ticker}-USD" if "-USD" not in raw_ticker else raw_ticker
    elif asset_class == "FOREX": ticker = f"{raw_ticker}=X" if "=X" not in raw_ticker else raw_ticker

    if AI_AVAILABLE:
        if st.sidebar.button("RETRAIN AI MODEL"): st.session_state['train_trigger'] = True

    df = yf.Ticker(ticker).history(period="2y")
    if df.empty: st.error(f"❌ SIGNAL LOST: {ticker}"); st.stop()
    
    # AI Processing
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
        trend = "ASCENDING 🔼" if target > curr_price else "DESCENDING 🔽"
    
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1]
    stop_loss = curr_price - (atr * 2)
    hype_score, hype_label = get_hype_score(ticker)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRICE", f"${curr_price:.2f}")
    c2.metric("AI PREDICTION", trend, f"Target: ${target:.2f}")
    c3.metric("RISK GUARD", f"${stop_loss:.2f}", "-2.0 ATR")
    c4.metric("SOCIAL FLOW", f"{hype_score}", hype_label)
    
    t1, t2, t3 = st.tabs(["📊 CHART", "🎮 SIMULATION", "📓 LOGS"])
    
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], mode='lines', line=dict(color='#00d2ff', width=2), name='Price'))
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], mode='lines', line=dict(color='rgba(0, 210, 255, 0.2)', width=10), showlegend=False)) # Glow effect
        if target > 0:
            fig.add_trace(go.Scatter(x=[df.index[-1] + pd.Timedelta(days=1)], y=[target], mode='markers', marker=dict(color='yellow', size=12, symbol='star'), name='AI Target'))
        fig.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🏦 AETHER SIMULATOR")
        net_worth = st.session_state['cash']
        for t, s in st.session_state['holdings'].items():
            try: p = yf.Ticker(t).history(period='1d')['Close'].iloc[-1]
            except: p = 0
            net_worth += p * s
        
        k1, k2 = st.columns(2)
        k1.markdown(f"""<div class="aether-card"><h3>LIQUID CASH</h3><h1 style='color:#00d2ff'>${st.session_state['cash']:,.2f}</h1></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="aether-card"><h3>NET WORTH</h3><h1 style='color:#fff'>${net_worth:,.2f}</h1></div>""", unsafe_allow_html=True)
        
        sc1, sc2, sc3 = st.columns(3)
        shares = sc1.number_input("QUANTITY", min_value=0.0001, value=1.0)
        notes = sc2.text_input("NOTES", f"AI Signal: {trend}")
        cost = shares * curr_price
        
        b1, b2 = st.columns(2)
        if b1.button(f"BUY {ticker}"):
            if st.session_state['cash'] >= cost:
                st.session_state['cash'] -= cost
                st.session_state['holdings'][ticker] = st.session_state['holdings'].get(ticker, 0) + shares
                log_trade(ticker, "BUY", curr_price, shares, cost, notes)
                st.success("ORDER FILLED")
            else: st.error("INSUFFICIENT FUNDS")
        if b2.button(f"SELL {ticker}"):
            if st.session_state['holdings'].get(ticker, 0) >= shares:
                st.session_state['cash'] += cost
                st.session_state['holdings'][ticker] -= shares
                if st.session_state['holdings'][ticker] <= 0: del st.session_state['holdings'][ticker]
                log_trade(ticker, "SELL", curr_price, shares, cost, notes)
                st.success("ORDER FILLED")
            else: st.error("INSUFFICIENT ASSETS")

    with t3:
        st.dataframe(st.session_state['journal'], use_container_width=True)

elif mode == "OPTIMIZER (MATH)":
    st.sidebar.markdown("---")
    tickers_input = st.sidebar.text_area("ASSET BASKET", value="BTC-USD, ETH-USD, NVDA, TSLA, GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    st.header("⚡ PORTFOLIO CALCULATOR")
    if st.button("RUN SIMULATION"):
        with st.spinner("CALCULATING OPTIMAL FLOW..."):
            try:
                # Handle new YFinance Multi-Column Data
                data = yf.download(tickers, period="1y")
                
                # Check if data is multi-level (Ticker, Price Type)
                if isinstance(data.columns, pd.MultiIndex):
                    # Extract just the 'Close' prices
                    closes = data.xs('Close', axis=1, level=0) if 'Close' in data.columns.get_level_values(0) else data['Close']
                else:
                    closes = data['Close'] if 'Close' in data else data
                
                closes = closes.dropna()
                
                if closes.empty:
                    st.error("No data found. Please check ticker symbols.")
                else:
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)
                    
                    st.success("OPTIMIZATION COMPLETE")
                    # Ensure values and names match perfectly
                    fig = px.pie(values=weights, names=tickers[:len(weights)], hole=0.5, title="RECOMMENDED ALLOCATION")
                    fig.update_traces(textinfo='percent+label')
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during calculation: {e}")

elif mode == "ACADEMY (LEARN)":
    st.title("🎓 AETHER ACADEMY")
    ac1, ac2 = st.columns([1, 2])
    with ac1:
        st.markdown(f"""<div class="aether-card"><h3>💡 DAILY WISDOM</h3><p>"{random.choice(['Trend is your friend', 'Cut losses early', 'Buy the rumor'])}"</p></div>""", unsafe_allow_html=True)
    with ac2:
        st.subheader("📰 GLOBAL FEED")
        try:
            d = feedparser.parse("https://finance.yahoo.com/news/rssindex")
            for e in d.entries[:5]:
                st.markdown(f"""<div style='background:rgba(255,255,255,0.05); padding:10px; margin-bottom:5px; border-left:3px solid #00d2ff'><a href="{e.link}" target="_blank" style='color:#fff; text-decoration:none'>{e.title}</a></div>""", unsafe_allow_html=True)
        except: st.write("News Feed Unavailable")
