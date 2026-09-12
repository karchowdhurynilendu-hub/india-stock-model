"""
Strategy: momentum_breakout
------------------------------
An aggressive, early-momentum / breakout-hunting strategy — the kind of
approach some traders use to try to catch a stock early in a large move.
It deliberately ignores traditional overbought caution (fast recent gains
are treated as the point, not a warning) because the goal is catching
acceleration early rather than mean-reversion safety.

IMPORTANT — read before using this one:
No factor model reliably identifies "multibaggers" in advance; that
framing describes an outcome, not a repeatable, testable signal. What
this strategy actually does is surface stocks showing statistically
unusual acceleration (price near/above 52-week highs, fast recent rate
of change, volume surges, bullish moving-average stacking). Most
breakouts fail to continue — that's normal for momentum systems, not a
bug. Treat this as a speculative watchlist generator to research
further, not a buy signal. The backtest win rate reported by the main
script is especially important to check for this one before trusting it.

Signals combined:
  - Proximity to / breakout above the 52-week high
  - Rate of change over 20 and 60 trading days (acceleration)
  - Volume surge vs the 50-day average (aggressive threshold: 2x)
  - Bullish moving-average stack (5 > 20 > 50), an "early trend forming" cue
"""

import numpy as np

STRATEGY_LABEL = "Aggressive momentum breakout (early-signal watchlist)"


def build_signal(df):
    df = df.copy()
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["High252"] = df["Close"].rolling(252, min_periods=60).max()
    df["VolAvg50"] = df["Volume"].rolling(50).mean()

    df["pct_from_high"] = (df["Close"] - df["High252"]) / df["High252"]
    df["roc20"] = df["Close"].pct_change(20)
    df["roc60"] = df["Close"].pct_change(60)

    breakout_score = np.where(df["pct_from_high"] >= -0.03, 1, 0)  # within 3% of, or above, 52w high
    roc_score = np.select(
        [df["roc20"] > 0.15, df["roc20"] > 0.07, df["roc20"] < -0.10],
        [2, 1, -1], default=0,
    )
    accel_score = np.where(df["roc20"] > df["roc60"] / 3, 1, 0)  # 20-day pace outrunning 60-day pace
    volume_surge = (df["Volume"] > 2.0 * df["VolAvg50"]).astype(int)
    bullish_stack = ((df["SMA5"] > df["SMA20"]) & (df["SMA20"] > df["SMA50"])).astype(int)

    df["composite"] = (
        breakout_score * 1.5
        + roc_score * 1.0
        + accel_score * 0.5
        + volume_surge * 1.0
        + bullish_stack * 0.5
    )

    df["rsi_display"] = np.nan
    df["trend_display"] = np.where(bullish_stack.astype(bool), "up", "down")
    df["volume_confirmed_display"] = volume_surge.astype(bool)
    df["pct_from_52w_high_pct_display"] = (df["pct_from_high"] * 100).round(1)
    df["roc_20d_pct_display"] = (df["roc20"] * 100).round(1)
    return df


def label_recommendation(composite):
    if composite >= 4.0:
        return "Explosive momentum (high risk/high reward)"
    if composite >= 2.0:
        return "Early momentum forming"
    if composite <= -1.0:
        return "Momentum fading (exit watch)"
    return "No breakout signal"
