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
    page_title="AETHER APEX v31.0", 
    page_icon=" ", 
    initial_sidebar_state="collapsed" 
)

# NEW DB NAME FOR V31
DB_NAME = 'aether_v31.db'

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
    
    # Bot Trades Table
    c.execute('''CREATE TABLE IF NOT EXISTS bot_trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, ticker TEXT, 
                 action TEXT, price REAL, amount REAL, outcome TEXT)''')
                 
    conn.commit()
    conn.close()

# --- DATABASE FUNCTIONS ---

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

def add_xp(username, amount):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    c.execute("SELECT xp, level FROM users WHERE username=?", (username,))
    data = c.fetchone()
    current_xp = data[0]
    current_level = data[1]
    
    new_xp = current_xp + amount
    
    new_level = "Novice"
    if new_xp > 100: new_level = "Apprentice"
    if new_xp > 500: new_level = "Trader"
    if new_xp > 1000: new_level = "Pro"
    if new_xp > 5000: new_level = "Market Wizard"
    
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

def log_bot_trade(ticker, action, price, outcome="PENDING"):
    conn = sqlite3.connect(DB_NAME, timeout=10)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO bot_trades (date, ticker, action, price, amount, outcome) VALUES (?, ?, ?, ?, ?, ?)", 
              (date_str, ticker, action, price, 0.0, outcome))
    conn.commit()
    conn.close()

def get_bot_log():
    conn = sqlite3.connect(DB_NAME, timeout=10)
    df = pd.read_sql_query("SELECT * FROM bot_trades ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    return df

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
                triggered.append(f"  ALERT: {ticker} hit ${target:,.2f}!")
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
if 'backtest_results' not in st.session_state: st.session_state['backtest_results'] = None

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

# --- 3. CSS STYLING (V31: BALANCED NEON) ---
st.markdown("""
<style>
    /* V31 GLOBAL RESET */
    .stApp {
        background-color: #050505; 
        color: #ffffff; /* White Text */
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* V31: HEAD
