import os
import datetime
import sqlite3
import pandas as pd
import requests
import yfinance as yf
import warnings

# Suppress yfinance warnings for cleaner logs
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
DB_NAME = "master_nse_warehouse.db"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MAX_HISTORY_DAYS = 350             # Rolling window preserved per ticker
MIN_DAILY_TURNOVER_INR = 50000000  # 5 Crore INR liquidity floor
STOP_LOSS_PCT = 0.08               # 8% hard stop loss

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
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("Telegram notification sent successfully.")
        else:
            print(f"Telegram dispatch failed: {resp.text}")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")

# =====================================================================
# 1. MACRO BENCHMARK & REGIME CHECK
# =====================================================================
print("Fetching Nifty 50 for Regime and Relative Strength calculation...")
today_str = datetime.date.today().strftime("%Y-%m-%d")
nifty_raw = yf.download("^NSEI", period="1y", interval="1d", progress=False, auto_adjust=False)

if isinstance(nifty_raw.columns, pd.MultiIndex):
    nifty_raw.columns = nifty_raw.columns.get_level_values(0)
nifty_raw.columns = [str(c).strip() for c in nifty_raw.columns]
nifty_raw.dropna(subset=["Close"], inplace=True)

nifty_raw["EMA_50"] = nifty_raw["Close"].ewm(span=50, adjust=False).mean()
nifty_raw["SMA_200"] = nifty_raw["Close"].rolling(200).mean()
nifty_raw["ROC_20"] = nifty_raw["Close"].pct_change(20) * 100

latest_nifty = nifty_raw.iloc[-1]
nifty_close = float(latest_nifty["Close"])
nifty_ema50 = float(latest_nifty["EMA_50"])
nifty_sma200 = float(latest_nifty["SMA_200"])
nifty_roc20 = float(latest_nifty["ROC_20"])

# Relaxed Macro Shield Logic
is_macro_bull = (nifty_close > nifty_ema50) or (nifty_close >= (nifty_sma200 * 0.98))

print(f"Nifty 50: {nifty_close:.2f} | 50 EMA: {nifty_ema50:.2f} | 200 SMA: {nifty_sma200:.2f}")
print(f"Macro Shield Status: {'BULLISH (ACTIVE)' if is_macro_bull else 'BEARISH (WARNING)'}")

# =====================================================================
# 2. INCREMENTAL DATA SYNC & SYSTEM SCANNER
# =====================================================================
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'stock_%';")
tables = [row[0] for row in cursor.fetchall()]

actionable_signals = []
radar_watchlist = []
bear_rs_watchlist = []

print(f"Syncing daily data and evaluating {len(tables)} stocks...")

for table in tables:
    ticker = table.replace("stock_", "").replace("_NS", ".NS")
    try:
        # Load cached historical data
        df = pd.read_sql(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT {MAX_HISTORY_DAYS}", conn)
        if df.empty:
            continue

        date_col = "Date" if "Date" in df.columns else ("index" if "index" in df.columns else df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        df.sort_values(by=date_col, ascending=True, inplace=True)
        last_cached_date = df[date_col].iloc[-1]

        # Download missing data if DB is behind today's date
        if last_cached_date.date() < datetime.date.today():
            start_date = (last_cached_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            new_data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            if not new_data.empty:
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)
                new_data.columns = [str(c).strip() for c in new_data.columns]
                new_data.reset_index(inplace=True)
                new_data["Date"] = pd.to_datetime(new_data["Date"]).dt.normalize()

                # Append to SQLite DB
                new_data.to_sql(table, conn, if_exists="append", index=False)
                df = pd.concat([df, new_data], ignore_index=True)

                # Truncate rolling database to max history
                conn.execute(f"""
                    DELETE FROM {table} 
                    WHERE rowid NOT IN (
                        SELECT rowid FROM {table} ORDER BY rowid DESC LIMIT {MAX_HISTORY_DAYS}
                    )
                """)
                conn.commit()

        # Format dataframe for calculations
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["Close", "Volume"], inplace=True)

        if len(df) < 250:
            continue

        # Liquidity Filter
        df["Turnover"] = df["Close"] * df["Volume"]
        if df["Turnover"].tail(30).mean() < MIN_DAILY_TURNOVER_INR:
            continue

        # --- INDICATOR CALCULATIONS ---
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA_50_Slope"] = df["EMA_50"] - df["EMA_50"].shift(20)
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["ROC_3M"] = df["Close"].pct_change(63) * 100
        df["ROC_20"] = df["Close"].pct_change(20) * 100
        df["High_52W"] = df["High"].rolling(window=252).max()
        df["Low_52W"] = df["Low"].rolling(window=252).min()
        df["Distance_From_High"] = (df["High_52W"] - df["Close"]) / df["High_52W"]
        df["Distance_From_Low_52W"] = (df["Close"] - df["Low_52W"]) / df["Low_52W"]
        df["Rolling_Low_90"] = df["Low"].rolling(window=90).min()
        df["Gain_From_Low_90"] = (df["Close"] - df["Rolling_Low_90"]) / df["Rolling_Low_90"]
        df["EMA_Distance_Pct"] = (abs(df["Close"] - df["EMA_50"]) / df["EMA_50"]) * 100
        df["DCR"] = ((df["Close"] - df["Low"]) / (df["High"] - df["Low"])) * 100
        df["Vol_SMA_20"] = df["Volume"].rolling(window=20).mean()
        df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA_20"]
        df["SMA_200_Slope"] = df["SMA_200"] - df["SMA_200"].shift(20)

        # Prior Uptrend Maturity Tracker
        df["Bull_Aligned"] = (df["EMA_50"] > df["SMA_200"]).astype(int)
        df["Bull_Trend_Days"] = df.groupby((df["Bull_Aligned"] != df["Bull_Aligned"].shift()).cumsum())["Bull_Aligned"].cumsum()

        curr = df.iloc[-1]

        # =====================================================================
        # SCENARIO A: BULL REGIME (Active Trading)
        # =====================================================================
        if is_macro_bull:
            cond_trend = (curr["Close"] > curr["EMA_50"]) and (curr["EMA_50"] > curr["SMA_200"])
            cond_slope = curr["EMA_50_Slope"] > 0
            # Strict integration of the "Prior Uptrend " file rules
            cond_prior_uptrend = (curr["SMA_200_Slope"] > 0) and (curr["Distance_From_Low_52W"] >= 0.30) and (curr["Bull_Trend_Days"] >= 20)
            cond_rs = curr["ROC_20"] > nifty_roc20
            cond_mom = curr["ROC_3M"] >= 35.0
            cond_90d_bounce = curr["Gain_From_Low_90"] > 0.40

            if not (cond_trend and cond_slope and cond_prior_uptrend and cond_rs and cond_mom and cond_90d_bounce):
                continue

            # --- TIER 1: ACTIONABLE BREAKOUT SIGNAL ---
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

            # --- TIER 2: RADAR WATCHLIST (NEAR THRESHOLD) ---
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

        # =====================================================================
        # SCENARIO B: BEAR REGIME (Study Only)
        # =====================================================================
        else:
            # --- TIER 3: BEAR MARKET RELATIVE STRENGTH ---
            cond_holding_200 = curr["Close"] > curr["SMA_200"]
            cond_strong_rs = curr["ROC_20"] > (nifty_roc20 + 5.0) # Beating Nifty by > 5% over 20 days
            cond_resilient = curr["Distance_From_High"] <= 0.20   # Hasn't crashed > 20%
            
            if cond_holding_200 and cond_strong_rs and cond_resilient:
                bear_rs_watchlist.append({
                    "ticker": ticker,
                    "price": curr["Close"],
                    "rs_spread": curr["ROC_20"] - nifty_roc20,
                    "dist_high": curr["Distance_From_High"] * 100,
                })

    except Exception:
        continue

conn.close()

# =====================================================================
# 3. CONSTRUCT TELEGRAM DISPATCH
# =====================================================================
msg_lines = [f"**🎯 DAILY SYSTEM SCANNER REPORT ({today_str})**", ""]

if is_macro_bull:
    msg_lines.append(f"**Macro Regime:** 🟢 BULLISH (Nifty: {nifty_close:.1f} | 20D ROC: {nifty_roc20:.1f}%)")
else:
    msg_lines.append(f"**Macro Regime:** 🔴 BEARISH / CAUTION (Nifty: {nifty_close:.1f})")
    msg_lines.append("**⚠️ MACRO SHIELD DOWN:** Standard setups blocked. Displaying Relative Strength leaders for study.")

msg_lines.append("=" * 35)

# --- BULL MARKET OUTPUT (TIERS 1 & 2) ---
if actionable_signals and is_macro_bull:
    msg_lines.append("\n**🚀 ACTIONABLE TRANCHE-1 PROBES:**")
    msg_lines.append("*Allocation: 33% of 25% Position | Stop-Loss: 8%*\n")
    for sig in actionable_signals:
        msg_lines.append(f"• **{sig['ticker']}** @ ₹{sig['price']:.2f}")
        msg_lines.append(f"  ├ 🛑 **Initial Stop-Loss:** ₹{sig['stop_loss']:.2f} (-8.0%)")
        msg_lines.append(f"  ├ 📈 **3M Momentum:** +{sig['roc_3m']:.1f}% | **RS vs Nifty:** +{sig['rs_spread']:.1f}%")
        msg_lines.append(f"  ├ 📊 **Volume Surge:** {sig['vol_ratio']:.2f}x | **Closing Range (DCR):** {sig['dcr']:.1f}%")
        msg_lines.append(f"  └ 🎯 **Distance from 52W High:** {sig['dist_high']:.1f}%\n")
elif is_macro_bull:
    msg_lines.append("\n**🚀 ACTIONABLE TRANCHE-1 PROBES:** None today.")

if radar_watchlist and is_macro_bull:
    msg_lines.append("\n**👀 RADAR WATCHLIST (NEARING BREAKOUT):**")
    for r in radar_watchlist[:8]:
        msg_lines.append(f"• **{r['ticker']}** (₹{r['price']:.2f}) | High Gap: -{r['dist_high']:.1f}% | Vol: {r['vol_ratio']:.1f}x | DCR: {r['dcr']:.0f}%")

# --- BEAR MARKET OUTPUT (TIER 3) ---
if not is_macro_bull and bear_rs_watchlist:
    msg_lines.append("\n**🛡️ BEAR MARKET RS LEADERS (STUDY ONLY):**")
    msg_lines.append("*Stocks holding 200 SMA and heavily outperforming the Nifty:*\n")
    
    # Sort by Relative Strength Spread (Highest to Lowest)
    bear_rs_watchlist = sorted(bear_rs_watchlist, key=lambda x: x["rs_spread"], reverse=True)
    
    for r in bear_rs_watchlist[:10]: # Display top 10 strongest stocks
        msg_lines.append(f"• **{r['ticker']}** (₹{r['price']:.2f})")
        msg_lines.append(f"  └ **RS Outperformance:** +{r['rs_spread']:.1f}% | High Gap: -{r['dist_high']:.1f}%")

elif not is_macro_bull:
    msg_lines.append("\n**🛡️ BEAR MARKET RS LEADERS:** None found. Total market washout.")

final_message = "\n".join(msg_lines)
send_telegram_alert(final_message)
