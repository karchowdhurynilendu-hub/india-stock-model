"""
India Equities Signal Model — pluggable strategy runner
----------------------------------------------------------
Fetches daily OHLCV data for a universe of NSE-listed stocks, hands it to
whichever strategy is selected below, backtests that strategy's signal,
and writes today's ranked recommendations to a JSON file for the
dashboard to read.

TO SWITCH STRATEGIES:
    Change the STRATEGY variable below to the filename (without .py) of
    any module in the strategies/ folder. Four are included:
        "trend_momentum_volume"  (default — trend + RSI + volume)
        "macd_crossover"         (MACD line/signal crossover + volume)
        "momentum_breakout"      (aggressive early-momentum / breakout watchlist)
        "macro_news_overlay"     (trend/momentum + global macro + news sentiment)

TO ADD YOUR OWN STRATEGY:
    Create a new file in strategies/, e.g. strategies/my_strategy.py,
    implementing this contract:

        build_signal(df) -> df
            Takes a pandas DataFrame with columns Open, High, Low, Close,
            Volume (daily bars, oldest first). Must return that same
            DataFrame with an added 'composite' column: a float score
            where positive = bullish lean, negative = bearish lean,
            magnitude = conviction. You can add any other columns you
            want (e.g. your own indicator values).

        label_recommendation(composite_value) -> str
            Takes the latest composite score and returns a short label,
            e.g. "Strong watch (bullish)".

    Optional:
        detect_pattern(row) -> (name, bias) or None
            Row is the latest bar. Return a (pattern_name, bias) tuple
            or None. Shown as an informational tag only.

        STRATEGY_LABEL = "Your strategy's display name"

    Then set STRATEGY = "macd_crossover" below. No other code needs to
    change — the backtest and output format are generic and work with
    any strategy that produces a 'composite' column.

Requires: pip install yfinance pandas numpy
"""

import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config — this is what you edit day to day
# ---------------------------------------------------------------------------
STRATEGY = "trend_momentum_volume"   # <-- change this to switch strategies

UNIVERSE = [
    {"ticker": "ZENSARTECH.NS", "name": "Zensar Tech", "cap": "mid"},
    {"ticker": "KEI.NS", "name": "KEI Industries", "cap": "mid"},
    {"ticker": "SONATSOFTW.NS", "name": "Sonata Software", "cap": "small"},
    {"ticker": "APTUS.NS", "name": "Aptus Value Housing", "cap": "small"},
    {"ticker": "REDINGTON.NS", "name": "Redington", "cap": "mid"},
    {"ticker": "ANURAS.NS", "name": "Anupam Rasayan", "cap": "small"},
    {"ticker": "CRAFTSMAN.NS", "name": "Craftsman Automation", "cap": "mid"},
    {"ticker": "GRAVITA.NS", "name": "Gravita India", "cap": "small"},
    {"ticker": "HBLENGINE.NS", "name": "HBL Engineering (formerly HBL Power)", "cap": "small"},
    {"ticker": "COHANCE.NS", "name": "Cohance Lifesciences (formerly Suven Pharma)", "cap": "small"},
    {"ticker": "RAINBOW.NS", "name": "Rainbow Childrens Hosp", "cap": "small"},
    {"ticker": "TCIEXP.NS", "name": "TCI Express", "cap": "mid"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "cap": "large"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "cap": "large"},
]

LOOKBACK_DAYS = "1y"
BACKTEST_HOLD_DAYS = 10
BACKTEST_THRESHOLD = 1.0   # |composite| at or above this counts as a signal in the backtest


# ---------------------------------------------------------------------------
# Strategy loading — generic, doesn't need to change when you add strategies
# ---------------------------------------------------------------------------
def load_strategy(name):
    strategies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies")
    if strategies_dir not in sys.path:
        sys.path.insert(0, strategies_dir)
    module = importlib.import_module(name)
    for required in ("build_signal", "label_recommendation"):
        if not hasattr(module, required):
            raise ImportError(
                f"Strategy '{name}' is missing required function: {required}(). "
                f"See the contract described at the top of india_stock_model.py."
            )
    return module


# ---------------------------------------------------------------------------
# Generic backtest — works against any strategy's 'composite' column
# ---------------------------------------------------------------------------
def backtest(df, threshold=BACKTEST_THRESHOLD):
    signals = df[df["composite"].abs() >= threshold].copy()
    outcomes = []
    for idx in signals.index:
        pos = df.index.get_loc(idx)
        if pos + BACKTEST_HOLD_DAYS >= len(df):
            continue
        entry_price = df["Close"].iloc[pos]
        exit_price = df["Close"].iloc[pos + BACKTEST_HOLD_DAYS]
        fwd_return = (exit_price - entry_price) / entry_price
        direction = 1 if df["composite"].iloc[pos] > 0 else -1
        correct = (fwd_return > 0 and direction > 0) or (fwd_return < 0 and direction < 0)
        outcomes.append(correct)
    if not outcomes:
        return None
    return round(100 * sum(outcomes) / len(outcomes), 1)


def _sanitize_for_json(obj):
    """Recursively replace NaN/Infinity floats with None so the output is
    always valid JSON — Python's json module writes bare NaN/Infinity by
    default, which is NOT valid JSON and breaks browser JSON.parse()."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Run the scan
# ---------------------------------------------------------------------------
def run_scan():
    strategy = load_strategy(STRATEGY)
    strategy_label = getattr(strategy, "STRATEGY_LABEL", STRATEGY)
    results = []

    for stock in UNIVERSE:
        try:
            hist = yf.download(
                stock["ticker"], period=LOOKBACK_DAYS, interval="1d",
                auto_adjust=True, progress=False,
            )
            # Newer yfinance versions return two-level ("MultiIndex") columns
            # even for a single ticker, which breaks comparisons like
            # SMA20 > SMA50 with "Operands are not aligned". Flatten it.
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if hist.empty or len(hist) < 60 or pd.isna(hist["Close"].iloc[-1]):
                continue

            hist = strategy.build_signal(hist)
            if "composite" not in hist.columns:
                raise ValueError("strategy.build_signal() did not add a 'composite' column")

            latest = hist.iloc[-1]
            pattern = strategy.detect_pattern(latest) if hasattr(strategy, "detect_pattern") else None
            win_rate = backtest(hist)

            # Generic pass-through: any column a strategy names with a
            # "_display" suffix is surfaced in the output automatically,
            # under its name with that suffix stripped. This lets new
            # strategies add their own fields (e.g. macro_tailwind,
            # roc_20d_pct) without any changes needed here.
            extra_fields = {}
            for col in hist.columns:
                if col.endswith("_display"):
                    val = latest[col]
                    key = col[: -len("_display")]
                    if isinstance(val, (float, np.floating)) and pd.isna(val):
                        extra_fields[key] = None
                    elif isinstance(val, (np.bool_,)):
                        extra_fields[key] = bool(val)
                    elif isinstance(val, (np.integer, np.floating)):
                        extra_fields[key] = round(float(val), 2)
                    else:
                        extra_fields[key] = val

            result = {
                "ticker": stock["ticker"],
                "name": stock["name"],
                "cap": stock["cap"],
                "price": round(float(latest["Close"]), 2),
                "composite_score": round(float(latest["composite"]), 2),
                "recommendation": strategy.label_recommendation(latest["composite"]),
                "candle_pattern": pattern[0] if pattern else None,
                "candle_bias": pattern[1] if pattern else None,
                "backtest_win_rate_pct": win_rate,
            }
            result.update(extra_fields)
            results.append(result)
            time.sleep(0.3)  # be polite to the data source
        except Exception as e:
            print(f"Skipped {stock['ticker']}: {e}")

    results.sort(key=lambda r: abs(r["composite_score"]), reverse=True)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": strategy_label,
        "strategy_file": STRATEGY,
        "hold_period_days_for_backtest": BACKTEST_HOLD_DAYS,
        "results": results,
    }

    output = _sanitize_for_json(output)

    with open("india_stock_signals.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Strategy: {strategy_label}")
    print(f"Wrote {len(results)} results to india_stock_signals.json")
    return output


if __name__ == "__main__":
    run_scan()
