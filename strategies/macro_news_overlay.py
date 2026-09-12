"""
Strategy: macro_news_overlay
------------------------------
Adds two extra layers on top of a base trend/momentum signal:

  1. MACRO OVERLAY (always on, no API key needed):
     Looks at the recent direction of the S&P 500 (global cue), the Nifty
     50 (domestic index), and the rupee (INR=X) to build a rough "is the
     tide with you or against you" tailwind/headwind score. This is a
     simple heuristic (10-day % change, thresholded), not an economic
     model — treat it as context, not causation.

  2. NEWS SENTIMENT OVERLAY (optional, needs a free API key):
     Pulls aggregate financial-market news sentiment from Alpha Vantage's
     NEWS_SENTIMENT endpoint. Get a free key at
     https://www.alphavantage.co/support/#api-key and set it as an
     environment variable before running:
         export ALPHAVANTAGE_API_KEY=your_key_here      (Mac/Linux)
         setx ALPHAVANTAGE_API_KEY "your_key_here"       (Windows)
     If the key or GitHub Actions secret isn't set, this layer is skipped
     (contributes 0) — the strategy still runs fine without it.

  HONEST LIMITATION: this news layer is a BROAD market-mood signal
  (queried once per run, cached, not once per stock), not stock-specific
  news. Free news APIs have thin-to-no headline coverage of individual
  Indian small/mid-cap tickers, so per-stock news scoring isn't reliable
  from free sources — a broad "is market sentiment positive or negative
  right now" reading is what's actually achievable here, and that's what
  this does. Don't read the news number as "what people are saying about
  this specific stock."

Requires: pip install requests  (in addition to pandas/numpy/yfinance)
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf

STRATEGY_LABEL = "Trend/momentum + global macro + news sentiment overlay"

_macro_cache = None
_news_cache = None


def _get_macro_tailwind():
    global _macro_cache
    if _macro_cache is not None:
        return _macro_cache
    try:
        tickers = {"sp500": "^GSPC", "nifty": "^NSEI", "inr": "INR=X"}
        scores = []
        for key, tk in tickers.items():
            hist = yf.download(tk, period="1mo", interval="1d", progress=False, auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if hist.empty or len(hist) < 10:
                continue
            chg = (hist["Close"].iloc[-1] - hist["Close"].iloc[-10]) / hist["Close"].iloc[-10]
            if key == "inr":
                # a weakening rupee (INR=X rising) is a mild headwind for importers/the broad market
                scores.append(-1 if chg > 0.01 else (1 if chg < -0.01 else 0))
            else:
                scores.append(1 if chg > 0.01 else (-1 if chg < -0.01 else 0))
        _macro_cache = round(sum(scores) / max(len(scores), 1), 2) if scores else 0.0
    except Exception:
        _macro_cache = 0.0
    return _macro_cache


def _get_news_sentiment():
    global _news_cache
    if _news_cache is not None:
        return _news_cache
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        _news_cache = 0.0
        return _news_cache
    try:
        import requests
        url = (
            "https://www.alphavantage.co/query"
            "?function=NEWS_SENTIMENT&topics=financial_markets,economy_macro"
            f"&apikey={api_key}&limit=50"
        )
        resp = requests.get(url, timeout=10).json()
        feed = resp.get("feed", [])
        if not feed:
            _news_cache = 0.0
            return _news_cache
        scores = [float(item.get("overall_sentiment_score", 0)) for item in feed]
        avg = sum(scores) / len(scores)
        _news_cache = round(max(-1.0, min(1.0, avg)), 2)
    except Exception:
        _news_cache = 0.0
    return _news_cache


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def build_signal(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["VolAvg20"] = df["Volume"].rolling(20).mean()

    trend_score = np.where(df["SMA20"] > df["SMA50"], 1, -1)
    momentum_score = np.select([df["RSI"] < 35, df["RSI"] > 65], [1, -1], default=0)
    volume_confirm = (df["Volume"] > 1.3 * df["VolAvg20"]).astype(int)

    macro = _get_macro_tailwind()
    news = _get_news_sentiment()

    df["composite"] = (
        trend_score * 1.0
        + momentum_score * 1.0
        + volume_confirm * 0.5
        + macro * 1.0
        + news * 0.5
    )

    df["rsi_display"] = df["RSI"]
    df["trend_display"] = np.where(df["SMA20"] > df["SMA50"], "up", "down")
    df["volume_confirmed_display"] = volume_confirm.astype(bool)
    df["macro_tailwind_display"] = macro
    df["news_sentiment_display"] = news
    return df


def label_recommendation(composite):
    if composite >= 2.5:
        return "Strong watch (bullish, macro-aligned)"
    if composite >= 1.0:
        return "Watch (bullish lean)"
    if composite <= -2.5:
        return "Strong caution (bearish, macro headwind)"
    if composite <= -1.0:
        return "Caution (bearish lean)"
    return "Neutral / monitor"
