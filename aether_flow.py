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

# --- CONFIG & SAFETY ---
st.set_page_config(
    layout="wide", 
    page_title="AETHER FLOW v4.0", 
    page_icon="💠", 
    initial_sidebar_state="expanded" 
)

# AI Safety Check
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

if 'cash' not in st.session_state: st.session_state['cash'] = 100000.00
if 'holdings' not in st.session_state: st.session_state['holdings'] = {}
if 'journal' not in st.session_state: st.session_state['journal'] = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Shares', 'Total', 'Notes'])

# --- 2025 "PURE LAB" THEME (White & Glass) ---
st.markdown("""
<style>
    /* 1. BACKGROUND: Clean White/Grey Sci-Fi Look */
    .stApp {
        background-color: #f4f6f9;
        color: #000000;
    }
    
    /* 2. SIDEBAR: Deep Navy for Contrast */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* 3. METRIC CARDS: White Glass with Deep Shadows */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #00d2ff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,210,255,0.2);
    }
    label[data-testid="stMetricLabel"] {
        color: #555 !important;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #000 !important;
        font-size: 28px;
        font-weight: 800;
    }
    
    /* 4. BUTTONS: "Liquid" 3D Gradients (The Futuristic Pop) */
    .stButton>button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 15px 25px;
        border-radius: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        box-shadow: 0 10px 20px rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 210, 255, 0.5);
        background: linear-gradient(135deg, #00e5ff 0%, #4facfe 100%);
    }

    /* 5. INPUT FIELDS: Clean & Modern */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #ffffff;
        color: #000;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* 6. CUSTOM TEXT HIGHLIGHTS */
    .highlight-bull { color: #00b894; font-weight: 900; background: rgba(0,184,148,0.1); padding: 5px 10px; border-radius: 5px; }
    .highlight-bear { color: #d63031; font-weight: 900; background: rgba(214,48,49,0.1); padding: 5px 10px; border-radius: 5px; }
    
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAV ---
st.sidebar.title("💠 AETHER FLOW")
st.sidebar.markdown("`v4.0 | PURE LAB`")
mode = st.sidebar.selectbox("NAVIGATION", ["TERMINAL (PRO)", "ACADEMY (LEARN)", "OPTIMIZER (MATH)"])

# --- CORE LOGIC ---
def get_hype_score(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')]
        keywords = ['surge', 'soar', 'moon', 'bull', 'buy', 'record']
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
    from sklearn.preprocessing import MinMaxScaler
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
    st.sidebar.header("📡 SIGNAL INPUT")
    asset_class = st.sidebar.radio("MARKET", ["STOCKS", "CRYPTO", "FOREX"], horizontal=True)
    raw_ticker = st.sidebar.text_input("TICKER SYMBOL", value="BTC" if asset_class=="CRYPTO" else "NVDA").upper()
    
    if asset_class == "STOCKS": ticker = raw_ticker
    elif asset_class == "CRYPTO": ticker = f"{raw_ticker}-USD" if "-USD" not in raw_ticker else raw_ticker
    elif asset_class == "FOREX": ticker = f"{raw_ticker}=X" if "=X" not in raw_ticker else raw_ticker

    if AI_AVAILABLE:
        if st.sidebar.button("🧠 RETRAIN AI"): st.session_state['train_trigger'] = True

    # FETCH DATA
    df = yf.Ticker(ticker).history(period="2y")
    if df.empty: st.error(f"❌ ERROR: Could not find {ticker}"); st.stop()
    
    # AI LOGIC
    MODEL_FILE = f"model_{ticker}.keras"
    if 'train_trigger' in st.session_state and AI_AVAILABLE:
        with st.spinner("💠 AETHER IS LEARNING..."): train_brain(df, MODEL_FILE)
        del st.session_state['train_trigger']
    
    curr_price = df['Close'].iloc[-1]
    trend, target = "WAITING", 0.0
    
    if AI_AVAILABLE and os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        from sklearn.preprocessing import MinMaxScaler
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
    
    # METRICS ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRICE", f"${curr_price:.2f}")
    c2.metric("AI SIGNAL", trend, f"Target: ${target:.2f}")
    c3.metric("SAFE STOP", f"${stop_loss:.2f}", "-2.0 ATR")
    c4.metric("HYPE", f"{hype_score}", hype_label)
    
    # TABS
    t1, t2, t3 = st.tabs(["📊 LIVE CHART", "🎮 SIMULATOR", "📓 JOURNAL"])
    
    with t1:
        # Beautiful White/Blue Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], mode='lines', 
                                 line=dict(color='#00d2ff', width=4), name='Price', 
                                 fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.1)'))
        if target > 0:
            color = "#00b894" if "BULLISH" in trend else "#d63031"
            fig.add_trace(go.Scatter(x=[df.index[-1] + pd.Timedelta(days=1)], y=[target], mode='markers', 
                                     marker=dict(color=color, size=18, symbol='diamond', line=dict(color='white', width=2)), name='AI Target'))
            
        fig.update_layout(template="plotly_white", height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("🏦 SIMULATOR ACCOUNT")
        net_worth = st.session_state['cash']
        for t, s in st.session_state['holdings'].items():
            try: p = yf.Ticker(t).history(period='1d')['Close'].iloc[-1]
            except: p = 0
            net_worth += p * s
            
        # BRIGHT CARDS
        k1, k2 = st.columns(2)
        k1.markdown(f"""
        <div style="background:white; border-left:5px solid #00d2ff; padding:20px; border-radius:15px; box-shadow:0 5px 15px rgba(0,0,0,0.05);">
            <h4 style="color:#888; margin:0;">CASH BALANCE</h4>
            <h2 style="color:#000; margin:0; font-size:32px;">${st.session_state['cash']:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        k2.markdown(f"""
        <div style="background:linear-gradient(135deg, #00d2ff, #3a7bd5); padding:20px; border-radius:15px; box-shadow:0 10px 20px rgba(0,210,255,0.3);">
            <h4 style="color:white; margin:0; opacity:0.9;">NET WORTH</h4>
            <h2 style="color:white; margin:0; font-size:32px;">${net_worth:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        sc1, sc2 = st.columns([1, 2])
        shares = sc1.number_input("QTY", min_value=0.0001, value=1.0)
        notes = sc2.text_input("NOTES", "AI Follow")
        
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
                # FIX FOR ATTRIBUTE ERROR (New Data Structure)
                data = yf.download(tickers, period="1y")
                
                # Check for MultiIndex columns (Price, Ticker)
                if isinstance(data.columns, pd.MultiIndex):
                    # Try to get 'Close', if that fails look for 'Adj Close'
                    if 'Close' in data.columns.get_level_values(0):
                        closes = data.xs('Close', axis=1, level=0)
                    elif 'Adj Close' in data.columns.get_level_values(0):
                        closes = data.xs('Adj Close', axis=1, level=0)
                    else:
                        st.error("Could not find Close prices in data.")
                        st.stop()
                else:
                    closes = data['Close'] if 'Close' in data else data
                
                # Drop bad columns and rows
                closes = closes.dropna(axis=1, how='all').dropna()
                
                if closes.empty:
                    st.error("No valid data found. Check your tickers.")
                else:
                    valid_tickers = closes.columns.tolist()
                    weights = np.random.random(len(valid_tickers))
                    weights /= np.sum(weights)
                    
                    st.success("OPTIMAL SPLIT FOUND")
                    fig = px.pie(values=weights, names=valid_tickers, hole=0.5, title="RECOMMENDED ALLOCATION",
                                 color_discrete_sequence=px.colors.sequential.Bluyl)
                    fig.update_traces(textinfo='percent+label')
                    fig.update_layout(template="plotly_white")
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
            <div style="background:white; padding:15px; margin-bottom:10px; border-radius:10px; border-left:5px solid #00d2ff; box-shadow:0 2px 5px rgba(0,0,0,0.05);">
                <a href="{e.link}" target="_blank" style="color:#333; font-weight:bold; text-decoration:none; font-size:16px;">{e.title}</a>
            </div>
            """, unsafe_allow_html=True)
    except: st.warning("News unavailable")
