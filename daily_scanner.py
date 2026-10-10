import os
import datetime
import sqlite3
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import warnings
import io

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
DB_NAME = "master_nse_warehouse.db"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MAX_HISTORY_DAYS = 350             
MIN_DAILY_TURNOVER_INR = 30000000  
STOP_LOSS_PCT = 0.08               
BATCH_SIZE = 200 

def send_telegram_alert(message: str):
    """Dispatches formatted message to Telegram."""
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
# 1. MACRO BENCHMARK & REGIME CHECK
# =====================================================================
today_str = datetime.date.today().strftime("%Y-%m-%d")
nifty_raw = yf.download("^NSEI", period="1y", interval="1d", progress=False, auto_adjust=False)

if isinstance(nifty_raw.columns, pd.MultiIndex):
    nifty_raw.columns = nifty_raw.columns.get_level_values(0)
nifty_raw.columns = [str(c).strip() for c in nifty_raw.columns]
nifty_raw.dropna(subset=["Close"], inplace=True)

nifty_raw["EMA_50"] = nifty_raw["Close"].ewm(span=50, adjust=False).mean()
nifty_raw["SMA_200"] = nifty_raw["Close"].rolling(200, min_periods=100).mean()
nifty_raw["ROC_20"] = nifty_raw["Close"].pct_change(20) * 100

latest_nifty = nifty_raw.iloc[-1]
nifty_close = float(latest_nifty["Close"])
nifty_ema50 = float(latest_nifty["EMA_50"])
nifty_sma200 = float(latest_nifty["SMA_200"])
nifty_roc20 = float(latest_nifty["ROC_20"])

is_macro_bull = (nifty_close > nifty_ema50) or (nifty_close >= (nifty_sma200 * 0.98))

# =====================================================================
# 2. DATABASE CONNECTION & GHOST FETCH BOOTSTRAP
# =====================================================================
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stock_%';")
tables = [row[0] for row in cursor.fetchall()]

bootstrap_log = ""

# --- DYNAMIC BOOTSTRAP ---
if len(tables) == 0:
    bootstrap_log += "\n<b>⚙️ BOOTSTRAP PROTOCOL TRIGGERED:</b>\n"
    try:
        # Ghost Fetcher: Spoof browser headers to bypass NSE firewall
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=15) # Acquire NSE cookies
        resp = session.get("https://archives.nseindia.com/content/equities/EQUITY_L.csv", headers=headers, timeout=15)
        
        if resp.status_code == 200:
            df_sym = pd.read_csv(io.StringIO(resp.text))
            tickers = [f"{sym}.NS" for sym in df_sym['SYMBOL']]
            bootstrap_log += f"✅ NSE Ticker list acquired ({len(tickers)} stocks).\n"
            
            # Download 1-year history in batches
            bootstrap_log += f"⏳ Downloading YFinance data in batches of {BATCH_SIZE}...\n"
            for i in range(0, len(tickers), BATCH_SIZE):
                batch_tickers = tickers[i:i+BATCH_SIZE]
                batch_data = yf.download(batch_tickers, period="1y", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
                
                for ticker in batch_tickers:
                    if len(batch_tickers) > 1:
                        if ticker not in batch_data.columns.get_level_values(0): continue
                        df_new = batch_data[ticker].copy()
                    else:
                        df_new = batch_data.copy()
                        
                    df_new.dropna(subset=["Close"], inplace=True)
                    if df_new.empty or len(df_new) < 60: 
                        continue
                        
                    df_new.reset_index(inplace=True)
                    date_col = "Date" if "Date" in df_new.columns else df_new.columns[0]
                    df_new["Date"] = pd.to_datetime(df_new[date_col]).dt.normalize()
                    df_new.columns = [str(c).strip() for c in df_new.columns]
                    
                    table_name = f"stock_{ticker.replace('.NS', '_NS')}"
                    df_new = df_new[["Date", "Open", "High", "Low", "Close", "Volume"]]
                    df_new.to_sql(table_name, conn, if_exists="replace", index=False)
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stock_%';")
            tables = [row[0] for row in cursor.fetchall()]
            bootstrap_log += f"✅ Database successfully built with {len(tables)} tables!\n"
        else:
            bootstrap_log += f"❌ NSE Firewall Blocked Fetch (Status Code: {resp.status_code}).\n"
            
    except Exception as e:
        bootstrap_log += f"❌ Bootstrap Failed Error: {str(e)}\n"

# =====================================================================
# 3. BATCH DATA SYNC & SYSTEM SCANNER
# =====================================================================
actionable_signals = []
radar_watchlist = []
bear_rs_watchlist = []

processed_count = 0
error_count = 0
skipped_liquidity_count = 0

for i in range(0, len(tables), BATCH_SIZE):
    batch_tables = tables[i:i + BATCH_SIZE]
    batch_tickers = [t.replace("stock_", "").replace("_NS", ".NS") for t in batch_tables]
    
    try:
        batch_data = yf.download(batch_tickers, period="7d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
    except Exception:
        error_count += len(batch_tickers)
        continue

    for table, ticker in zip(batch_tables, batch_tickers):
        try:
            if len(batch_tickers) > 1:
                if ticker not in batch_data.columns.get_level_values(0): continue
                df_new = batch_data[ticker].copy()
            else:
                df_new = batch_data.copy()

            df_new.dropna(subset=["Close"], inplace=True)
            
            if not df_new.empty:
                df_new.reset_index(inplace=True)
                date_col = "Date" if "Date" in df_new.columns else df_new.columns[0]
                df_new["Date"] = pd.to_datetime(df_new[date_col]).dt.normalize()
                df_new.columns = [str(c).strip() for c in df_new.columns]
                
                cursor.execute(f"SELECT MAX(Date) FROM {table}")
                max_date_val = cursor.fetchone()[0]
                
                if max_date_val:
                    max_date = pd.to_datetime(max_date_val).normalize()
                    df_new = df_new[df_new["Date"] > max_date]
                
                if not df_new.empty:
                    df_new = df_new[["Date", "Open", "High", "Low", "Close", "Volume"]]
                    df_new.to_sql(table, conn, if_exists="append", index=False)
            
            conn.execute(f"""
                DELETE FROM {table} 
                WHERE rowid NOT IN (
                    SELECT rowid FROM {table} ORDER BY Date DESC LIMIT {MAX_HISTORY_DAYS}
                )
            """)
            conn.commit()

            df = pd.read_sql(f"SELECT * FROM {table} ORDER BY Date ASC", conn)
            if df.empty: continue

            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            df.set_index("Date", inplace=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=["Close", "Volume"], inplace=True)

            if len(df) < 65: continue

            df["Turnover"] = df["Close"] * df["Volume"]
            if df["Turnover"].tail(30).mean() < MIN_DAILY_TURNOVER_INR:
                skipped_liquidity_count += 1
                continue

            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["EMA_50_Slope"] = df["EMA_50"] - df["EMA_50"].shift(20)
            df["SMA_200"] = df["Close"].rolling(window=200, min_periods=60).mean()
            df["ROC_3M"] = df["Close"].pct_change(63).fillna(0) * 100
            df["ROC_20"] = df["Close"].pct_change(20).fillna(0) * 100
            
            df["High_52W"] = df["High"].rolling(window=252, min_periods=60).max()
            df["Low_52W"] = df["Low"].rolling(window=252, min_periods=60).min()
            df["Rolling_Low_90"] = df["Low"].rolling(window=90, min_periods=30).min()
            
            df["Distance_From_High"] = (df["High_52W"] - df["Close"]) / df["High_52W"]
            df["Distance_From_Low_52W"] = (df["Close"] - df["Low_52W"]) / df["Low_52W"]
            df["Gain_From_Low_90"] = (df["Close"] - df["Rolling_Low_90"]) / df["Rolling_Low_90"]
            df["EMA_Distance_Pct"] = (abs(df["Close"] - df["EMA_50"]) / df["EMA_50"]) * 100
            
            high_low_diff = df["High"] - df["Low"]
            df["DCR"] = ((df["Close"] - df["Low"]) / high_low_diff.replace(0, np.nan)).fillna(0.5) * 100
            df["Vol_SMA_20"] = df["Volume"].rolling(window=20, min_periods=10).mean()
            df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA_20"].replace(0, np.nan)
            df["SMA_200_Slope"] = df["SMA_200"] - df["SMA_200"].shift(20)

            df["Bull_Aligned"] = (df["EMA_50"] > df["SMA_200"]).astype(int)
            df["Bull_Trend_Days"] = df.groupby((df["Bull_Aligned"] != df["Bull_Aligned"].shift()).cumsum())["Bull_Aligned"].cumsum()

            curr = df.iloc[-1]
            processed_count += 1

            if is_macro_bull and len(df) >= 200:
                cond_trend = (curr["Close"] > curr["EMA_50"]) and (curr["EMA_50"] > curr["SMA_200"])
                cond_slope = curr["EMA_50_Slope"] > 0
                
                # Applying structural rules from the "Prior Uptrend " file
                cond_prior_uptrend = (curr["SMA_200_Slope"] > 0) and (curr["Distance_From_Low_52W"] >= 0.30) and (curr["Bull_Trend_Days"] >= 20)
                
                cond_rs = curr["ROC_20"] > nifty_roc20
                cond_mom = curr["ROC_3M"] >= 35.0
                cond_90d_bounce = curr["Gain_From_Low_90"] > 0.40

                if not (cond_trend and cond_slope and cond_prior_uptrend and cond_rs and cond_mom and cond_90d_bounce):
                    continue

                cond_highs = curr["Distance_From_High"] <= 0.18
                cond_comp = (curr["EMA_Distance_Pct"] <= 7.0) or (curr["ROC_3M"] >= 80.0)
                cond_dcr = curr["DCR"] >= 75.0
                cond_vol = curr["Vol_Ratio"] >= 1.8

                if cond_highs and cond_comp and cond_dcr and cond_vol:
                    actionable_signals.append({
                        "ticker": ticker,
                        "price": curr["Close"],
                        "stop_loss": curr["Close"] * (1 - STOP_LOSS_PCT),
                        "vol_ratio": curr["Vol_Ratio"],
                        "dcr": curr["DCR"],
                        "roc_3m": curr["ROC_3M"],
                        "rs_spread": curr["ROC_20"] - nifty_roc20,
                        "dist_high": curr["Distance_From_High"] * 100,
                    })
                    continue

                cond_radar_proximity = (curr["Distance_From_High"] <= 0.08) and (curr["EMA_Distance_Pct"] <= 10.0)
                cond_radar_activity = (curr["Vol_Ratio"] >= 1.2) or (curr["DCR"] >= 65.0)

                if cond_radar_proximity and cond_radar_activity:
                    radar_watchlist.append({
                        "ticker": ticker,
                        "price": curr["Close"],
                        "vol_ratio": curr["Vol_Ratio"],
                        "dcr": curr["DCR"],
                        "dist_high": curr["Distance_From_High"] * 100,
                        "roc_3m": curr["ROC_3M"],
                    })
            else:
                if not pd.isna(curr["ROC_20"]):
                    dist_high_val = curr["Distance_From_High"] * 100 if not pd.isna(curr["Distance_From_High"]) else 0.0
                    bear_rs_watchlist.append({
                        "ticker": ticker,
                        "price": curr["Close"],
                        "rs_spread": curr["ROC_20"] - nifty_roc20,
                        "dist_high": dist_high_val,
                    })

        except Exception as e:
            error_count += 1
            continue

conn.close()

# =====================================================================
# 4. CONSTRUCT TELEGRAM DISPATCH
# =====================================================================
msg_lines = [f"<b>🎯 DAILY SYSTEM SCANNER REPORT ({today_str})</b>"]
msg_lines.append(f"<i>Diag: {len(tables)} DB Tables | {processed_count} Evaluated | {skipped_liquidity_count} Low Liq | {error_count} Errors</i>\n")

if bootstrap_log:
    msg_lines.append(bootstrap_log)

if is_macro_bull:
    msg_lines.append(f"<b>Macro Regime:</b> 🟢 BULLISH (Nifty: {nifty_close:.1f} | 20D ROC: {nifty_roc20:.1f}%)")
else:
    msg_lines.append(f"<b>Macro Regime:</b> 🔴 BEARISH / CAUTION (Nifty: {nifty_close:.1f})")
    msg_lines.append("<b>⚠️ MACRO SHIELD DOWN:</b> Standard setups blocked. Hunting Extreme RS Anomalies for study.")

msg_lines.append("=" * 35)

if actionable_signals and is_macro_bull:
    msg_lines.append("\n<b>🚀 ACTIONABLE TRANCHE-1 PROBES:</b>")
    msg_lines.append("<i>Allocation: 33% of 25% Position | Stop-Loss: 8%</i>\n")
    for sig in actionable_signals:
        msg_lines.append(f"• <b>{sig['ticker']}</b> @ ₹{sig['price']:.2f}")
        msg_lines.append(f"  ├ 🛑 <b>Initial Stop-Loss:</b> ₹{sig['stop_loss']:.2f} (-8.0%)")
        msg_lines.append(f"  ├ 📈 <b>3M Momentum:</b> +{sig['roc_3m']:.1f}% | <b>RS vs Nifty:</b> +{sig['rs_spread']:.1f}%")
        msg_lines.append(f"  ├ 📊 <b>Volume Surge:</b> {sig['vol_ratio']:.2f}x | <b>Closing Range (DCR):</b> {sig['dcr']:.1f}%")
        msg_lines.append(f"  └ 🎯 <b>Distance from 52W High:</b> {sig['dist_high']:.1f}%\n")
elif is_macro_bull:
    msg_lines.append("\n<b>🚀 ACTIONABLE TRANCHE-1 PROBES:</b> None today.")

if radar_watchlist and is_macro_bull:
    msg_lines.append("\n<b>👀 RADAR WATCHLIST (NEARING BREAKOUT):</b>")
    for r in radar_watchlist[:8]:
        msg_lines.append(f"• <b>{r['ticker']}</b> (₹{r['price']:.2f}) | High Gap: -{r['dist_high']:.1f}% | Vol: {r['vol_ratio']:.1f}x | DCR: {r['dcr']:.0f}%")

if not is_macro_bull and bear_rs_watchlist:
    msg_lines.append("\n<b>🛡️ PURE RS PULSE CHECK (STUDY ONLY):</b>")
    msg_lines.append("<i>Top 15 stocks showing relative strength vs Nifty:</i>\n")
    
    bear_rs_watchlist = sorted(bear_rs_watchlist, key=lambda x: x["rs_spread"], reverse=True)
    
    for r in bear_rs_watchlist[:15]: 
        msg_lines.append(f"• <b>{r['ticker']}</b> (₹{r['price']:.2f})")
        msg_lines.append(f"  └ <b>RS Outperformance:</b> +{r['rs_spread']:.1f}% | High Gap: -{r['dist_high']:.1f}%")
elif not is_macro_bull:
    msg_lines.append("\n<b>🛡️ PURE RS PULSE CHECK:</b> No data available.")

final_message = "\n".join(msg_lines)
send_telegram_alert(final_message)
