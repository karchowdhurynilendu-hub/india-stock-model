"""
Strategy: trend_momentum_volume
--------------------------------
Trend (SMA20 vs SMA50) + Momentum (RSI) + Volume confirmation, composited
into a single score. Candlestick pattern is attached as a secondary,
informational tag only — it does not affect the composite score.

Every strategy file must define:
    build_signal(df)          -> df with an added 'composite' float column
    label_recommendation(x)   -> string label for a given composite value

Optional:
    detect_pattern(row)       -> (pattern_name, bias) or None
    STRATEGY_LABEL            -> friendly name shown on the dashboard
"""

import numpy as np

STRATEGY_LABEL = "Trend + momentum (RSI) + volume confirmation"


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_signal(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["VolAvg20"] = df["Volume"].rolling(20).mean()

    df["trend_score"] = np.where(df["SMA20"] > df["SMA50"], 1, -1)
    df["momentum_score"] = np.select(
        [df["RSI"] < 35, df["RSI"] > 65], [1, -1], default=0,
    )
    df["volume_confirm"] = (df["Volume"] > 1.3 * df["VolAvg20"]).astype(int)

    df["composite"] = (
        df["trend_score"] * 1.0
        + df["momentum_score"] * 1.0
        + df["volume_confirm"] * 0.5
    )
    # extra columns the main script pulls into the output, if present
    df["rsi_display"] = df["RSI"]
    df["trend_display"] = np.where(df["SMA20"] > df["SMA50"], "up", "down")
    df["volume_confirmed_display"] = df["volume_confirm"].astype(bool)
    return df


def label_recommendation(composite):
    if composite >= 2.0:
        return "Strong watch (bullish)"
    if composite >= 1.0:
        return "Watch (bullish lean)"
    if composite <= -2.0:
        return "Strong caution (bearish)"
    if composite <= -1.0:
        return "Caution (bearish lean)"
    return "Neutral / monitor"


def detect_pattern(row):
    body = abs(row["Close"] - row["Open"])
    rng = row["High"] - row["Low"]
    if rng == 0:
        return None
    body_ratio = body / rng
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]

    if body_ratio < 0.08:
        if lower_wick > upper_wick * 2.5:
            return ("Dragonfly doji", "bullish")
        if upper_wick > lower_wick * 2.5:
            return ("Gravestone doji", "bearish")
        return ("Standard doji", "neutral")
    if body_ratio < 0.3 and lower_wick > body * 2 and upper_wick < body * 0.5:
        return ("Hammer", "bullish")
    if body_ratio < 0.3 and upper_wick > body * 2 and lower_wick < body * 0.5:
        return ("Shooting star", "bearish")
    return None
