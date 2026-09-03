"""
agentic_agent.py
Agentic version of the stock recommendation engine.

Instead of pre-computing every ticker's indicators and stuffing them into one prompt,
this gives Claude TOOLS to query the SQLite database itself. Claude decides which
tickers to investigate, what data it needs, and when it has enough information to
make a recommendation. This is a real agentic loop: the model calls tools, we execute
them and return results, and the model keeps going until it produces a final answer.
"""

import sqlite3
import json
import os
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

from analysis import get_moving_averages, get_rsi, add_macd

load_dotenv()

DB_PATH = "stocks.db"
MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# Tool implementations — plain Python functions the agent can trigger
# ---------------------------------------------------------------------------

def tool_list_tickers():
    conn = sqlite3.connect(DB_PATH)
    tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM prices", conn)["ticker"].tolist()
    conn.close()
    return sorted(tickers)


def tool_get_indicators(ticker):
    conn = sqlite3.connect(DB_PATH)
    ma_df = get_moving_averages(conn, ticker.upper())
    rsi_df = get_rsi(conn, ticker.upper())
    conn.close()

    if ma_df.empty:
        return {"error": f"No data found for ticker {ticker}"}

    merged = ma_df.merge(rsi_df[["date", "rsi"]], on="date")
    merged = add_macd(merged)
    latest = merged.iloc[-1]

    return {
        "ticker": ticker.upper(),
        "date": latest["date"],
        "close": round(latest["close"], 2),
        "ma_20": round(latest["ma_20"], 2),
        "ma_50": round(latest["ma_50"], 2),
        "rsi": round(latest["rsi"], 2),
        "macd": round(latest["macd"], 2),
        "signal_line": round(latest["signal_line"], 2),
    }


def tool_get_price_history(ticker, days=30):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date, close FROM prices
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(ticker.upper(), days))
    conn.close()

    if df.empty:
        return {"error": f"No data found for ticker {ticker}"}

    df = df.sort_values("date")
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Tool schema — tells Claude what tools exist and how to call them
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_tickers",
        "description": "Returns the list of all stock tickers available in the database.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_indicators",
        "description": "Returns the latest technical indicators (close price, 20-day MA, 50-day MA, RSI, MACD) for a given ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_price_history",
        "description": "Returns recent daily closing prices for a ticker, useful for spotting trends over a specific window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                "days": {"type": "integer", "description": "Number of most recent trading days to return (default 30)"},
            },
            "required": ["ticker"],
        },
    },
]

TOOL_FUNCTIONS = {
    "list_tickers": lambda **kwargs: tool_list_tickers(),
    "get_indicators": lambda **kwargs: tool_get_indicators(kwargs["ticker"]),
    "get_price_history": lambda **kwargs: tool_get_price_history(kwargs["ticker"], kwargs.get("days", 30)),
}


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analysis agent. You have tools to explore a database
of historical stock data. Your job is to investigate the available stocks and produce ranked
buy/watch/avoid recommendations with reasoning.

Process:
1. Call list_tickers to see what's available.
2. Call get_indicators for each ticker to see its current technical picture (RSI, MACD, moving averages).
3. If a ticker's signals are ambiguous or borderline, use get_price_history to look at recent trend
   before deciding — don't just rely on the snapshot indicators for close calls.
4. Once you've investigated all tickers, respond with ONLY valid JSON (no other text) in this format:

{
  "rankings": [
    {"rank": 1, "ticker": "XXX", "recommendation": "Buy", "reasoning": "..."}
  ]
}
"""


def run_agent():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    client = Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": "Investigate the available stocks and give me ranked recommendations."}]

    max_turns = 40  # safety cap so a stuck loop can't run forever
    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Log what the agent is doing, for visibility into the agentic process
        for block in response.content:
            if block.type == "tool_use":
                print(f"[agent] calling {block.name}({block.input})")

        if response.stop_reason != "tool_use":
            # Agent is done — extract the final text/JSON answer
            text_block = next((b for b in response.content if b.type == "text"), None)
            if text_block is None:
                raise RuntimeError("Agent finished without producing a text response.")

            raw_text = text_block.text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            return json.loads(raw_text)

        # Agent wants to use tools — execute them and feed results back
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                func = TOOL_FUNCTIONS[block.name]
                result = func(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent did not finish within {max_turns} turns.")


if __name__ == "__main__":
    result = run_agent()
    print("\n=== AGENTIC RANKED RECOMMENDATIONS ===\n")
    for entry in result["rankings"]:
        print(f"{entry['rank']}. {entry['ticker']} — {entry['recommendation']}")
        print(f"   {entry['reasoning']}\n")
