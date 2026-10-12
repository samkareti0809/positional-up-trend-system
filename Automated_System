import os
import datetime
import sqlite3
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
DB_NAME = "master_nse_warehouse.db"
WATCHLIST_DB = "watchlist.db"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN:
        print("\n[Telegram Not Configured] Message Preview:\n")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")

# =====================================================================
# 1. SETUP WATCHLIST DATABASE
# =====================================================================
def init_watchlist_db():
    conn = sqlite3.connect(WATCHLIST_DB)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_watchlist (
            Ticker TEXT PRIMARY KEY,
            Added_Date TEXT,
            Setup_Type TEXT,
            Status TEXT DEFAULT 'Monitoring'
        )
    ''')
    conn.commit()
    conn.close()

# =====================================================================
# 2. PHASE 1: END-OF-DAY SCREENER (Builds the Watchlist)
# Run at 4:00 PM
# =====================================================================
def run_eod_screener():
    print("Running EOD Screener to find Coiled Springs...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Check if we have tables
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stock_%';")
        tables = [row[0] for row in cursor.fetchall()]
    except:
        print("Database not found or empty.")
        return

    coiled_springs = []
    
    for table in tables:
        ticker = table.replace("stock_", "").replace("_NS", ".NS")
        try:
            df = pd.read_sql(f"SELECT * FROM {table} ORDER BY Date ASC", conn)
            if df.empty or len(df) < 50: continue

            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            
            # Calculate metrics
            df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["Dist_EMA_20_Pct"] = ((df["Close"] - df["EMA_20"]) / df["EMA_20"]) * 100
            
            df["BB_Mid"] = df["Close"].rolling(window=20).mean()
            df["BB_Std"] = df["Close"].rolling(window=20).std()
            df["BB_Upper"] = df["BB_Mid"] + (df["BB_Std"] * 2)
            df["BB_Lower"] = df["BB_Mid"] - (df["BB_Std"] * 2)
            df["BB_Width_Pct"] = ((df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]) * 100
            
            df["Vol_50D_Avg"] = df["Volume"].rolling(window=50).mean()
            df["RVOL_50D_Pct"] = (df["Volume"] / df["Vol_50D_Avg"].replace(0, np.nan)) * 100
            
            curr = df.iloc[-1]
            
            # The 3 Rules of the Coiled Spring
            is_tight_bb = curr["BB_Width_Pct"] < 15.0
            is_near_20ema = abs(curr["Dist_EMA_20_Pct"]) < 2.0
            is_volume_dry = curr["RVOL_50D_Pct"] < 80.0
            
            if is_tight_bb and is_near_20ema and is_volume_dry:
                coiled_springs.append({
                    "ticker": ticker,
                    "bb_width": curr["BB_Width_Pct"],
                    "close": curr["Close"]
                })
        except Exception as e:
            continue
            
    conn.close()
    
    # Save to Watchlist DB
    if coiled_springs:
        conn_wl = sqlite3.connect(WATCHLIST_DB)
        cursor_wl = conn_wl.cursor()
        
        # Clear old watchlist
        cursor_wl.execute("DELETE FROM active_watchlist")
        
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        msg_lines = [f"<b>🎯 EOD COILED SPRING WATCHLIST ({today_str})</b>", "<i>These stocks are tightly coiled and resting. Do not buy yet. Wait for the intraday volume ignition alert tomorrow.</i>\n"]
        
        for stock in coiled_springs:
            cursor_wl.execute("INSERT INTO active_watchlist (Ticker, Added_Date, Setup_Type) VALUES (?, ?, ?)", 
                              (stock['ticker'], today_str, 'Coiled Spring'))
            msg_lines.append(f"• <b>{stock['ticker']}</b> (₹{stock['close']:.2f}) | BB Width: {stock['bb_width']:.1f}%")
            
        conn_wl.commit()
        conn_wl.close()
        
        send_telegram_alert("\n".join(msg_lines))
    else:
        send_telegram_alert("<b>🎯 EOD WATCHLIST</b>\nNo coiled springs detected today.")


# =====================================================================
# 3. PHASE 2: INTRADAY IGNITION SCANNER (Elephant Footprint Tracker)
# Run at 10:15 AM
# =====================================================================
def run_intraday_scanner():
    print("Running Intraday Scanner on Watchlist...")
    conn_wl = sqlite3.connect(WATCHLIST_DB)
    cursor_wl = conn_wl.cursor()
    
    try:
        cursor_wl.execute("SELECT Ticker FROM active_watchlist WHERE Status='Monitoring'")
        watchlist = [row[0] for row in cursor_wl.fetchall()]
    except:
        print("Watchlist DB not found.")
        return
        
    conn_wl.close()
    
    if not watchlist:
        print("Watchlist is empty. Nothing to monitor.")
        return
        
    # Connect to the main warehouse to get the 50-day average liquidity to compare against
    conn = sqlite3.connect(DB_NAME)
    
    alerts = []
    
    for ticker in watchlist:
        try:
            table = f"stock_{ticker.replace('.NS', '_NS')}"
            # Get historical 50-day average volume/turnover
            df_hist = pd.read_sql(f"SELECT * FROM {table} ORDER BY Date DESC LIMIT 50", conn)
            
            if df_hist.empty: continue
            
            avg_50d_vol = df_hist['Volume'].mean()
            
            # Fetch Live Intraday Data (Using Yahoo Finance for 1m/5m data)
            # In a real environment, you would use your broker API here (Zerodha, Dhan, etc.)
            live_data = yf.download(ticker, period="1d", interval="5m", progress=False)
            
            if live_data.empty: continue
            
            # Calculate Volume traded so far TODAY
            live_volume = live_data['Volume'].sum()
            current_price = live_data['Close'].iloc[-1].item() if isinstance(live_data['Close'].iloc[-1], pd.Series) else float(live_data['Close'].iloc[-1])
            
            # RULE: If it's 10:15 AM, 75 minutes of the day have passed (1/5th of the trading day).
            # If the volume so far is already > 50% of the normal FULL DAY average volume, 
            # an institution is aggressively buying right out of the gate.
            
            if live_volume > (avg_50d_vol * 0.5):
                rvol_so_far = (live_volume / avg_50d_vol) * 100
                alerts.append({
                    "ticker": ticker,
                    "price": current_price,
                    "rvol": rvol_so_far
                })
                
                # Update status so we don't alert again tomorrow
                conn_wl = sqlite3.connect(WATCHLIST_DB)
                conn_wl.execute(f"UPDATE active_watchlist SET Status='Triggered' WHERE Ticker='{ticker}'")
                conn_wl.commit()
                conn_wl.close()
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    conn.close()
    
    if alerts:
        msg_lines = ["<b>🚨 INTRADAY ELEPHANT IGNITION ALERT 🚨</b>", "<i>10:15 AM Scan: Massive institutional footprint detected on watchlist stocks!</i>\n"]
        for alert in alerts:
            msg_lines.append(f"🔥 <b>{alert['ticker']}</b> @ ₹{alert['price']:.2f}")
            msg_lines.append(f"  └ Volume so far is already <b>{alert['rvol']:.0f}%</b> of its full daily average!")
            
        send_telegram_alert("\n".join(msg_lines))
    else:
        print("No intraday ignition detected on watchlist.")


# =====================================================================
# MAIN EXECUTION ROUTER
# =====================================================================
import sys

if __name__ == "__main__":
    # Ensure Watchlist DB exists
    init_watchlist_db()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--eod":
            run_eod_screener()
        elif sys.argv[1] == "--intraday":
            run_intraday_scanner()
        else:
            print("Invalid argument. Use --eod or --intraday")
    else:
        print("Please run with --eod (for 4:00 PM scan) or --intraday (for 10:15 AM scan).")
        # For demonstration, we'll run both sequentially right now:
        print("\n--- DEMONSTRATION RUN ---")
        run_eod_screener()
        run_intraday_scanner()
