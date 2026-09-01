"""
app.py
Flask backend for the Stock Advisor app.
Serves SQL-computed indicators, full price history for charting, and
cached AI agent recommendations (so we're not hitting the API on every page load).
"""

import sqlite3
import json
import os
from flask import Flask, jsonify, render_template
import pandas as pd

from analysis import get_moving_averages, get_rsi, add_macd
from agent import get_latest_snapshot, get_recommendations

app = Flask(__name__)
DB_PATH = "stocks.db"
CACHE_PATH = "recommendations_cache.json"


def get_conn():
    return sqlite3.connect(DB_PATH)


def get_all_tickers():
    conn = get_conn()
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM prices", conn)["ticker"].tolist()
    conn.close()
    return sorted(tickers)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/snapshots")
def api_snapshots():
    """Latest indicator values for every ticker."""
    conn = get_conn()
    snapshots = [get_latest_snapshot(conn, t) for t in get_all_tickers()]
    conn.close()
    return jsonify(snapshots)


@app.route("/api/prices/<ticker>")
def api_prices(ticker):
    """Full price history + moving averages for one ticker, for charting."""
    conn = get_conn()
    df = get_moving_averages(conn, ticker.upper())
    conn.close()
    if df.empty:
        return jsonify({"error": "ticker not found"}), 404
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/recommendations")
def api_recommendations():
    """
    Returns cached AI recommendations if available, otherwise calls the agent.
    Caching avoids burning API calls every time someone loads the page.
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return jsonify(json.load(f))

    conn = get_conn()
    snapshots = [get_latest_snapshot(conn, t) for t in get_all_tickers()]
    conn.close()

    result = get_recommendations(snapshots)

    with open(CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return jsonify(result)


@app.route("/api/recommendations/refresh", methods=["POST"])
def api_refresh_recommendations():
    """Forces a fresh AI call, overwriting the cache."""
    conn = get_conn()
    snapshots = [get_latest_snapshot(conn, t) for t in get_all_tickers()]
    conn.close()

    result = get_recommendations(snapshots)

    with open(CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
