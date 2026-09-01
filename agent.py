"""
agent.py
Pulls the latest technical indicators for all tickers (via analysis.py's SQL + pandas logic),
sends them to Claude in a single call, and gets back ranked recommendations.
"""

import sqlite3
import json
import os
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

from analysis import get_moving_averages, get_rsi, add_macd

load_dotenv()  # reads variables from a .env file in the project root, if present

DB_PATH = "stocks.db"
MODEL = "claude-sonnet-5"  # swap to "claude-haiku-4-5-20251001" if you want cheaper/faster calls


def get_latest_snapshot(conn, ticker):
    """Reuses the SQL + pandas logic from analysis.py, returns just the latest row as a dict."""
    ma_df = get_moving_averages(conn, ticker)
    rsi_df = get_rsi(conn, ticker)
    merged = ma_df.merge(rsi_df[["date", "rsi"]], on="date")
    merged = add_macd(merged)

    latest = merged.iloc[-1]
    return {
        "ticker": ticker,
        "date": latest["date"],
        "close": round(latest["close"], 2),
        "ma_20": round(latest["ma_20"], 2),
        "ma_50": round(latest["ma_50"], 2),
        "rsi": round(latest["rsi"], 2),
        "macd": round(latest["macd"], 2),
        "signal_line": round(latest["signal_line"], 2),
    }


def build_prompt(snapshots):
    return f"""You are a financial analysis assistant. Below is technical indicator data for {len(snapshots)} stocks, computed from historical price data (moving averages, RSI, MACD).

Data:
{json.dumps(snapshots, indent=2)}

For each ticker, assign a recommendation of "Buy", "Watch", or "Avoid" based on the technical indicators. Consider:
- RSI above 70 = potentially overbought, below 30 = potentially oversold
- MACD above signal line = bullish momentum, below = bearish momentum
- Price relative to 20-day and 50-day moving averages = trend direction

Rank the tickers from strongest to weakest opportunity. Respond with ONLY valid JSON, no other text, in this exact format:

{{
  "rankings": [
    {{
      "rank": 1,
      "ticker": "XXX",
      "recommendation": "Buy",
      "reasoning": "one to two sentence explanation referencing the specific indicator values"
    }}
  ]
}}
"""


def get_recommendations(snapshots):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your environment or a .env file.")

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": build_prompt(snapshots)}],
    )

    # Find the text block — response may include other block types (e.g. thinking) first
    text_block = next((block for block in response.content if block.type == "text"), None)
    if text_block is None:
        raise RuntimeError(f"No text block found in response: {response.content}")

    raw_text = text_block.text.strip()

    # Strip markdown code fences if Claude adds them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    return json.loads(raw_text)


def main():
    conn = sqlite3.connect(DB_PATH)
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM prices", conn)["ticker"].tolist()

    print("Gathering latest indicators for all tickers...")
    snapshots = [get_latest_snapshot(conn, t) for t in tickers]
    conn.close()

    print("Asking Claude for ranked recommendations...")
    result = get_recommendations(snapshots)

    print("\n=== RANKED RECOMMENDATIONS ===\n")
    for entry in result["rankings"]:
        print(f"{entry['rank']}. {entry['ticker']} — {entry['recommendation']}")
        print(f"   {entry['reasoning']}\n")

    return result


if __name__ == "__main__":
    main()
