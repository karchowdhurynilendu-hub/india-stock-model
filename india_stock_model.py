"""
India Equities Signal Model — fully dynamic universe, pluggable strategy
--------------------------------------------------------------------------
No fixed stock list. Every run:

  STAGE 1 — SCREENER (cheap, broad, dynamic):
      Fetches the full official NSE equity list fresh (not hardcoded),
      batch-downloads recent closing prices across all of it, and flags
      GAINERS: stocks up >= MOVER_THRESHOLD_PCT over the last
      MOVER_LOOKBACK_DAYS trading days. Separately, it detects RECENT
      IPOs dynamically using each stock's actual listing date (from the
      same NSE list) — anything listed within RECENT_IPO_WINDOW_DAYS
      counts, automatically, with no manual list to maintain.

  STAGE 2 — DEEP ANALYSIS (thorough, narrow):
      Runs the full strategy (trend/momentum/backtest/etc, whichever is
      selected below) only on the dynamic set the screener produced:
      gainers ∪ recent IPOs. Nothing is deep-analyzed unless the
      screener actually flagged it today — the universe changes day to
      day based on real market movement, not a list I typed in.

  A rolling history of flagged movers is kept in movers_history.json so
  you can see whether a mover kept climbing, reversed, or was a one-day
  spike, rather than only ever seeing a single day's snapshot.

WHAT THIS DELIBERATELY DOES NOT DO:
  "Coming" / upcoming IPOs (not yet listed) have no trading history, so
  there is nothing for a technical strategy to compute — a moving
  average or RSI needs price data that doesn't exist yet for a stock
  that hasn't started trading. This script does not attempt to score
  those; it only works with stocks that already have real price data.

HONEST LIMITATIONS:
  - NSE's servers sometimes block automated requests from cloud/CI IP
    ranges (including GitHub Actions). Fetching the full NSE list is a
    best-effort attempt with automatic fallback to FALLBACK_UNIVERSE
    (a curated pool) if it fails — expected to happen on some days.
  - True BSE-only stocks (not cross-listed on NSE) aren't reachable
    through yfinance's .NS tickers at all.
  - On a very quiet market day, few or no stocks may clear the gainer
    threshold. MIN_DEEP_DIVE_SIZE backfills with the next-highest
    gainers so the dashboard isn't empty — see the constant below.

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
            magnitude = conviction.

        label_recommendation(composite_value) -> str
            Takes the latest composite score and returns a short label.

    Optional:
        detect_pattern(row) -> (name, bias) or None
        STRATEGY_LABEL = "Your strategy's display name"

    Then set STRATEGY = "my_strategy" below.

Requires: pip install yfinance pandas numpy requests
"""

import csv
import importlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Config — this is what you edit day to day
# ---------------------------------------------------------------------------
STRATEGY = "momentum_breakout"   # <-- change this to switch strategies

# Screener settings
ENABLE_FULL_NSE_SCREEN = True   # try the full NSE list; auto-falls back if blocked
MOVER_DIRECTION = "gainers"     # "gainers" | "losers" | "both" — which moves qualify a stock for deep analysis
MOVER_LOOKBACK_DAYS = 5         # ~1 trading week
MOVER_THRESHOLD_PCT = 10.0      # flag a stock if its % change over the lookback clears this
RECENT_IPO_WINDOW_DAYS = 180    # ~6 months — dynamically detected via each stock's actual listing date
SCREENER_BATCH_SIZE = 200       # tickers per batch download call
MOVERS_HISTORY_DAYS = 90        # how many days of mover history to keep
MIN_DEEP_DIVE_SIZE = 20         # backfill with next-highest gainers if fewer than this qualify, so quiet days aren't empty

LOOKBACK_DAYS = "1y"
BACKTEST_HOLD_DAYS = 10
BACKTEST_THRESHOLD = 1.0   # |composite| at or above this counts as a signal in the backtest

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equity/EQUITY_L.csv"

# Fallback pool used ONLY if the live NSE list can't be fetched (e.g.
# blocked). Not used at all on a normal day when the full list works.
FALLBACK_UNIVERSE = [
    {"ticker": "ZENSARTECH.NS", "name": "Zensar Tech"}, {"ticker": "KEI.NS", "name": "KEI Industries"},
    {"ticker": "SONATSOFTW.NS", "name": "Sonata Software"}, {"ticker": "APTUS.NS", "name": "Aptus Value Housing"},
    {"ticker": "REDINGTON.NS", "name": "Redington"}, {"ticker": "ANURAS.NS", "name": "Anupam Rasayan"},
    {"ticker": "CRAFTSMAN.NS", "name": "Craftsman Automation"}, {"ticker": "GRAVITA.NS", "name": "Gravita India"},
    {"ticker": "HBLENGINE.NS", "name": "HBL Engineering"}, {"ticker": "COHANCE.NS", "name": "Cohance Lifesciences"},
    {"ticker": "RAINBOW.NS", "name": "Rainbow Childrens Hosp"}, {"ticker": "TCIEXP.NS", "name": "TCI Express"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries"}, {"ticker": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"ticker": "IDBI.NS", "name": "IDBI Bank"}, {"ticker": "WELCORP.NS", "name": "Welspun Corp"},
    {"ticker": "ASTERDM.NS", "name": "Aster DM Healthcare"}, {"ticker": "RBLBANK.NS", "name": "RBL Bank"},
    {"ticker": "SONACOMS.NS", "name": "Sona BLW Precision Forgings"}, {"ticker": "AEGISLOG.NS", "name": "Aegis Logistics"},
    {"ticker": "GLAND.NS", "name": "Gland Pharma"}, {"ticker": "HINDCOPPER.NS", "name": "Hindustan Copper"},
    {"ticker": "NAVINFLUOR.NS", "name": "Navin Fluorine International"}, {"ticker": "POONAWALLA.NS", "name": "Poonawalla Fincorp"},
    {"ticker": "NH.NS", "name": "Narayana Hrudayalaya"}, {"ticker": "ANANDRATHI.NS", "name": "Anand Rathi Wealth"},
    {"ticker": "DELHIVERY.NS", "name": "Delhivery"}, {"ticker": "LALPATHLAB.NS", "name": "Dr Lal PathLabs"},
    {"ticker": "NUVAMA.NS", "name": "Nuvama Wealth Management"}, {"ticker": "MANAPPURAM.NS", "name": "Manappuram Finance"},
    {"ticker": "PNBHOUSING.NS", "name": "PNB Housing Finance"}, {"ticker": "TATATECH.NS", "name": "Tata Technologies"},
    {"ticker": "CDSL.NS", "name": "Central Depository Services"}, {"ticker": "BANDHANBNK.NS", "name": "Bandhan Bank"},
    {"ticker": "ANGELONE.NS", "name": "Angel One"}, {"ticker": "IIFL.NS", "name": "IIFL Finance"},
    {"ticker": "AMBER.NS", "name": "Amber Enterprises India"}, {"ticker": "KAYNES.NS", "name": "Kaynes Technology India"},
    {"ticker": "NBCC.NS", "name": "NBCC (India)"}, {"ticker": "AFFLE.NS", "name": "Affle (India)"},
    {"ticker": "CREDITACC.NS", "name": "CreditAccess Grameen"}, {"ticker": "IGL.NS", "name": "Indraprastha Gas"},
    {"ticker": "BRIGADE.NS", "name": "Brigade Enterprises"}, {"ticker": "CESC.NS", "name": "CESC Ltd"},
    {"ticker": "TATACHEM.NS", "name": "Tata Chemicals"}, {"ticker": "CAMS.NS", "name": "Computer Age Management Services"},
    {"ticker": "OLAELEC.NS", "name": "Ola Electric Mobility"}, {"ticker": "SYNGENE.NS", "name": "Syngene International"},
    {"ticker": "CROMPTON.NS", "name": "Crompton Greaves Consumer Electricals"}, {"ticker": "KEC.NS", "name": "KEC International"},
    {"ticker": "WHIRLPOOL.NS", "name": "Whirlpool of India"}, {"ticker": "FIRSTCRY.NS", "name": "Brainbees Solutions (FirstCry)"},
    {"ticker": "MEESHO.NS", "name": "Meesho"}, {"ticker": "URBANCO.NS", "name": "Urban Company"},
    {"ticker": "PWL.NS", "name": "PhysicsWallah"}, {"ticker": "PINELABS.NS", "name": "Pine Labs"},
    {"ticker": "TENNIND.NS", "name": "Tenneco Clean Air India"}, {"ticker": "JSWCEMENT.NS", "name": "JSW Cement"},
    {"ticker": "GROWW.NS", "name": "Groww (Billionbrains Garage Ventures)"},
]


# ---------------------------------------------------------------------------
# Stage 1: Screener — dynamic universe, gainers + recent IPOs
# ---------------------------------------------------------------------------
def fetch_full_nse_list():
    """Best-effort fetch of the official, current NSE equity list —
    including each stock's actual listing date, which is what makes
    recent-IPO detection dynamic instead of a hand-maintained list.

    Uses a session warm-up (visiting the NSE homepage first to pick up
    cookies, like established NSE-scraping libraries do) since a bare
    request with no prior visit is an easy bot signal. NSE frequently
    blocks automated requests from cloud/CI IPs regardless — this can
    and will fail on some days even with the warm-up. Expected, not a
    bug. Returns a list of {ticker, name, listing_date} dicts, or None."""
    urls_to_try = [
        "https://nsearchives.nseindia.com/content/equity/EQUITY_L.csv",
        NSE_EQUITY_LIST_URL,
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/vnd.ms-excel,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading",
    }

    for url in urls_to_try:
        for attempt in range(2):
            try:
                session = requests.Session()
                session.headers.update(headers)
                # Warm-up: visit the homepage first so NSE issues session
                # cookies, then reuse that session for the actual data
                # request — a bare request with no prior visit is an easy
                # bot signal and gets blocked more often.
                session.get("https://www.nseindia.com/", timeout=15)
                time.sleep(1)
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
                reader = csv.DictReader(io.StringIO(resp.text))
                result = []
                for raw_row in reader:
                    row = {(k or "").strip(): (v or "").strip() for k, v in raw_row.items()}
                    symbol = row.get("SYMBOL", "")
                    name = row.get("NAME OF COMPANY", "")
                    series = row.get("SERIES", "")
                    listing_date_raw = row.get("DATE OF LISTING", "")
                    if not symbol or series != "EQ":
                        continue
                    listing_date = None
                    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
                        try:
                            listing_date = datetime.strptime(listing_date_raw, fmt).date()
                            break
                        except ValueError:
                            continue
                    result.append({
                        "ticker": f"{symbol}.NS",
                        "name": name or symbol,
                        "listing_date": listing_date.isoformat() if listing_date else None,
                    })
                if result:
                    return result
            except Exception as e:
                print(f"NSE list fetch attempt failed ({url}, try {attempt + 1}): {e}")
                time.sleep(2)

    print("Could not fetch the full NSE list after retries; falling back to the curated pool for screening.")
    return None


def get_screening_universe():
    """Returns (universe, source_label). universe is a list of
    {ticker, name, listing_date} — listing_date is None for the
    fallback pool, since that's not dynamically sourced."""
    if ENABLE_FULL_NSE_SCREEN:
        full_list = fetch_full_nse_list()
        if full_list:
            print(f"Screening across {len(full_list)} NSE-listed equities (live full list).")
            return full_list, "full_nse_list"
    print(f"Screening across the fallback pool only ({len(FALLBACK_UNIVERSE)} stocks).")
    fallback = [{"ticker": s["ticker"], "name": s["name"], "listing_date": None} for s in FALLBACK_UNIVERSE]
    return fallback, "fallback_pool"


def find_recent_ipos(screening_universe):
    """Dynamically detects recently-listed stocks using each stock's
    real listing date — no hardcoded IPO list. Returns None entries are
    skipped (the fallback pool has no listing dates, so this returns
    nothing when NSE's list wasn't available)."""
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=RECENT_IPO_WINDOW_DAYS)
    recent = []
    for s in screening_universe:
        if not s.get("listing_date"):
            continue
        try:
            if datetime.fromisoformat(s["listing_date"]).date() >= cutoff:
                recent.append(s)
        except ValueError:
            continue
    return recent


def run_screener(screening_universe):
    """Batch-downloads recent closes across the screening universe and
    flags stocks whose % change over MOVER_LOOKBACK_DAYS trading days
    clears MOVER_THRESHOLD_PCT in the configured MOVER_DIRECTION. Cheap
    by design: no indicators, no backtest, just a % change check."""
    tickers = [s["ticker"] for s in screening_universe]
    lookup = {s["ticker"]: s for s in screening_universe}
    scored = []  # every stock with a computed pct_change, for the backfill step

    for i in range(0, len(tickers), SCREENER_BATCH_SIZE):
        batch = tickers[i : i + SCREENER_BATCH_SIZE]
        try:
            data = yf.download(
                batch, period="1mo", interval="1d",
                group_by="ticker", threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"Screener batch {i}-{i + len(batch)} failed: {e}")
            continue

        for tk in batch:
            try:
                closes = data["Close"] if len(batch) == 1 else data[tk]["Close"]
                closes = closes.dropna()
                if len(closes) < MOVER_LOOKBACK_DAYS + 1:
                    continue
                recent = float(closes.iloc[-1])
                past = float(closes.iloc[-(MOVER_LOOKBACK_DAYS + 1)])
                if past == 0 or np.isnan(past) or np.isnan(recent):
                    continue
                pct_change = round(100 * (recent - past) / past, 2)
                entry = dict(lookup[tk])
                entry["screener_pct_change"] = pct_change
                scored.append(entry)
            except Exception:
                continue

    if MOVER_DIRECTION == "gainers":
        movers = [s for s in scored if s["screener_pct_change"] >= MOVER_THRESHOLD_PCT]
    elif MOVER_DIRECTION == "losers":
        movers = [s for s in scored if s["screener_pct_change"] <= -MOVER_THRESHOLD_PCT]
    else:
        movers = [s for s in scored if abs(s["screener_pct_change"]) >= MOVER_THRESHOLD_PCT]

    # Backfill on quiet days so the dashboard isn't empty: add the next
    # highest-ranked movers (by the same direction) until MIN_DEEP_DIVE_SIZE.
    if len(movers) < MIN_DEEP_DIVE_SIZE:
        flagged_tickers = {m["ticker"] for m in movers}
        remaining = [s for s in scored if s["ticker"] not in flagged_tickers]
        if MOVER_DIRECTION == "losers":
            remaining.sort(key=lambda s: s["screener_pct_change"])
        else:
            remaining.sort(key=lambda s: s["screener_pct_change"], reverse=True)
        for s in remaining:
            if len(movers) >= MIN_DEEP_DIVE_SIZE:
                break
            s = dict(s)
            s["backfilled"] = True
            movers.append(s)

    return movers


def update_movers_history(movers):
    """Append today's flagged movers to a rolling log (kept to the last
    MOVERS_HISTORY_DAYS days) so you can track them over time rather
    than only ever seeing today's snapshot."""
    history_path = "movers_history.json"
    today = datetime.now(timezone.utc).date().isoformat()

    try:
        with open(history_path, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append({
        "date": today,
        "movers": [
            {"ticker": m["ticker"], "name": m["name"], "pct_change": m["screener_pct_change"]}
            for m in movers
        ],
    })

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=MOVERS_HISTORY_DAYS)
    history = [h for h in history if datetime.fromisoformat(h["date"]).date() >= cutoff]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Stage 2: Strategy loading & deep analysis
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


def run_scan():
    strategy = load_strategy(STRATEGY)
    strategy_label = getattr(strategy, "STRATEGY_LABEL", STRATEGY)

    print("Stage 1: fetching screening universe and scanning for gainers...")
    screening_universe, universe_source = get_screening_universe()
    movers = run_screener(screening_universe)
    recent_ipos = find_recent_ipos(screening_universe)
    update_movers_history(movers)
    print(f"Screener flagged {len(movers)} stocks (direction={MOVER_DIRECTION}, threshold={MOVER_THRESHOLD_PCT}%).")
    print(f"Dynamically detected {len(recent_ipos)} recent IPOs (listed within {RECENT_IPO_WINDOW_DAYS} days).")

    # Fully dynamic deep-dive universe: movers ∪ recent IPOs. No fixed list.
    seen = set()
    deep_dive_universe = []
    for entry in movers + recent_ipos:
        if entry["ticker"] not in seen:
            deep_dive_universe.append(entry)
            seen.add(entry["ticker"])

    print(f"Stage 2: running full analysis on {len(deep_dive_universe)} dynamically-selected stocks...")
    results = []

    for stock in deep_dive_universe:
        try:
            hist = yf.download(
                stock["ticker"], period=LOOKBACK_DAYS, interval="1d",
                auto_adjust=True, progress=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            if hist.empty or len(hist) < 20 or pd.isna(hist["Close"].iloc[-1]):
                continue  # note: recent IPOs may have <60 days of history, so this floor is lower than before

            hist = strategy.build_signal(hist)
            if "composite" not in hist.columns:
                raise ValueError("strategy.build_signal() did not add a 'composite' column")

            latest = hist.iloc[-1]
            pattern = strategy.detect_pattern(latest) if hasattr(strategy, "detect_pattern") else None
            win_rate = backtest(hist) if len(hist) >= 60 else None  # backtest needs real history depth

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
                "price": round(float(latest["Close"]), 2),
                "composite_score": round(float(latest["composite"]), 2),
                "recommendation": strategy.label_recommendation(latest["composite"]),
                "candle_pattern": pattern[0] if pattern else None,
                "candle_bias": pattern[1] if pattern else None,
                "backtest_win_rate_pct": win_rate,
                "flagged_by_screener": "screener_pct_change" in stock,
                "screener_pct_change": stock.get("screener_pct_change"),
                "backfilled": stock.get("backfilled", False),
                "is_recent_ipo": stock in recent_ipos,
                "listing_date": stock.get("listing_date"),
            }
            result.update(extra_fields)
            results.append(result)
            time.sleep(0.15)  # be polite to the data source
        except Exception as e:
            print(f"Skipped {stock['ticker']}: {e}")

    results.sort(key=lambda r: abs(r["composite_score"]), reverse=True)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": strategy_label,
        "strategy_file": STRATEGY,
        "hold_period_days_for_backtest": BACKTEST_HOLD_DAYS,
        "universe_source": universe_source,
        "screener_direction": MOVER_DIRECTION,
        "screener_threshold_pct": MOVER_THRESHOLD_PCT,
        "screener_lookback_days": MOVER_LOOKBACK_DAYS,
        "recent_ipo_window_days": RECENT_IPO_WINDOW_DAYS,
        "recent_ipos_detected": len(recent_ipos),
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
