"""
Strategy: macd_crossover
--------------------------
Example second strategy, provided so you can see the contract in action
before writing your own. Uses MACD line vs signal line crossover, with
volume as a secondary confirmation on the composite score.

To switch to this strategy: open india_stock_model.py and set
    STRATEGY = "macd_crossover"
"""

import numpy as np

STRATEGY_LABEL = "MACD crossover + volume confirmation"


def build_signal(df):
    df = df.copy()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["VolAvg20"] = df["Volume"].rolling(20).mean()

    crossed_up = (df["MACD"] > df["MACD_signal"]) & (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))
    crossed_down = (df["MACD"] < df["MACD_signal"]) & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))

    df["cross_score"] = np.select([crossed_up, crossed_down], [1, -1], default=0)
    # let a crossover's effect linger for a few days rather than firing for one bar only
    df["cross_score"] = df["cross_score"].replace(0, np.nan).ffill(limit=3).fillna(0)
    df["volume_confirm"] = (df["Volume"] > 1.3 * df["VolAvg20"]).astype(int)

    df["composite"] = df["cross_score"] * 1.5 + df["volume_confirm"] * 0.5

    # display columns picked up by the main script if present
    df["rsi_display"] = np.nan  # this strategy doesn't use RSI
    df["trend_display"] = np.where(df["MACD"] > df["MACD_signal"], "up", "down")
    df["volume_confirmed_display"] = df["volume_confirm"].astype(bool)
    return df


def label_recommendation(composite):
    if composite >= 1.5:
        return "Strong watch (bullish)"
    if composite > 0:
        return "Watch (bullish lean)"
    if composite <= -1.5:
        return "Strong caution (bearish)"
    if composite < 0:
        return "Caution (bearish lean)"
    return "Neutral / monitor"


# no detect_pattern() here — it's optional. The main script will just
# show "—" for candle pattern when a strategy doesn't define one.
