"""
analysis.py
Queries stocks.db with SQL to compute moving averages and RSI,
then layers MACD on top using pandas.
"""

import sqlite3
import pandas as pd

DB_PATH = "stocks.db"


def get_moving_averages(conn, ticker):
    """20-day and 50-day simple moving averages, computed in SQL."""
    query = """
        SELECT
            date,
            close,
            AVG(close) OVER (
                ORDER BY date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) AS ma_20,
            AVG(close) OVER (
                ORDER BY date
                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
            ) AS ma_50
        FROM prices
        WHERE ticker = ?
        ORDER BY date;
    """
    return pd.read_sql_query(query, conn, params=(ticker,))


def get_rsi(conn, ticker, period=14):
    """
    RSI computed via SQL window functions.
    Steps: daily price change -> separate gains/losses -> rolling average -> RSI formula.
    """
    query = """
        WITH price_changes AS (
            SELECT
                date,
                close,
                close - LAG(close) OVER (ORDER BY date) AS change
            FROM prices
            WHERE ticker = ?
        ),
        gains_losses AS (
            SELECT
                date,
                close,
                CASE WHEN change > 0 THEN change ELSE 0 END AS gain,
                CASE WHEN change < 0 THEN -change ELSE 0 END AS loss
            FROM price_changes
        ),
        rolling AS (
            SELECT
                date,
                close,
                AVG(gain) OVER (
                    ORDER BY date
                    ROWS BETWEEN ? PRECEDING AND CURRENT ROW
                ) AS avg_gain,
                AVG(loss) OVER (
                    ORDER BY date
                    ROWS BETWEEN ? PRECEDING AND CURRENT ROW
                ) AS avg_loss
            FROM gains_losses
        )
        SELECT
            date,
            close,
            avg_gain,
            avg_loss,
            CASE
                WHEN avg_loss = 0 THEN 100
                ELSE 100 - (100 / (1 + (avg_gain / avg_loss)))
            END AS rsi
        FROM rolling
        ORDER BY date;
    """
    return pd.read_sql_query(query, conn, params=(ticker, period - 1, period - 1))


def add_macd(df):
    """
    MACD needs exponential moving averages, which SQL can't do natively,
    so this part runs in pandas on top of the SQL-pulled data.
    """
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df


def analyze_ticker(conn, ticker):
    ma_df = get_moving_averages(conn, ticker)
    rsi_df = get_rsi(conn, ticker)

    merged = ma_df.merge(rsi_df[["date", "rsi"]], on="date")
    merged = add_macd(merged)

    latest = merged.iloc[-1]
    print(f"\n--- {ticker} ---")
    print(f"Date: {latest['date']}")
    print(f"Close: {latest['close']:.2f}")
    print(f"20-day MA: {latest['ma_20']:.2f} | 50-day MA: {latest['ma_50']:.2f}")
    print(f"RSI (14): {latest['rsi']:.2f}")
    print(f"MACD: {latest['macd']:.2f} | Signal: {latest['signal_line']:.2f}")

    return merged


def main():
    conn = sqlite3.connect(DB_PATH)

    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM prices", conn)["ticker"].tolist()

    for ticker in tickers:
        analyze_ticker(conn, ticker)

    conn.close()


if __name__ == "__main__":
    main()
