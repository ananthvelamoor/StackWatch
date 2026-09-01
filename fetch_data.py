"""
fetch_data.py
Pulls historical daily stock data via yfinance and loads it into a local SQLite database.
"""

import sqlite3
import yfinance as yf
import pandas as pd

# --- Config ---
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",   # tech
    "JPM", "BAC", "GS",                         # finance
    "KO", "PEP", "WMT",                         # consumer
    "XOM", "CVX",                                # energy
    "JNJ", "PFE",                                # healthcare
    "TSLA", "DIS", "NFLX"                        # misc / growth
]
PERIOD = "2y"  # 2 years of daily data
DB_PATH = "stocks.db"


def create_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, date)
        );
    """)
    conn.commit()


def fetch_and_store(conn, ticker):
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, period=PERIOD, interval="1d", progress=False, auto_adjust=True)

    if df.empty:
        print(f"  No data returned for {ticker}, skipping.")
        return

    # yfinance sometimes returns multi-index columns — flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["ticker"] = ticker
    df["date"] = df["Date"].dt.strftime("%Y-%m-%d")

    rows = df[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]].values.tolist()

    conn.executemany("""
        INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    print(f"  Inserted {len(rows)} rows for {ticker}.")


def main():
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    for ticker in TICKERS:
        fetch_and_store(conn, ticker)

    total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"\nDone. {total} total rows in {DB_PATH}.")
    conn.close()


if __name__ == "__main__":
    main()