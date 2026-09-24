"""
India Equities Signal Model — fully dynamic universe, pluggable strategy
--------------------------------------------------------------------------
No fixed stock list. Every run:

  STAGE 1 — SCREENER (cheap, broad, dynamic):
      Uses the `pybhav` library to fetch NSE's official daily bhavcopy —
      which lists EVERY security traded that day (typically 1,800-2,100+
      symbols) — and flags GAINERS: stocks up >= MOVER_THRESHOLD_PCT over
      the last ~MOVER_LOOKBACK_DAYS trading days. Separately, it detects
      RECENT IPOs dynamically using each stock's actual listing date
      (from NSE's official equity list) — anything listed within
      RECENT_IPO_WINDOW_DAYS counts, automatically, with no manual list
      to maintain. If bhavcopy is unavailable (pybhav not installed, or
      NSE blocked it), this falls back to a smaller yfinance-based scan
      over the curated FALLBACK_UNIVERSE (~106 stocks).

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
    ranges (including GitHub Actions), and this can affect both the
    bhavcopy fetch and the equity-list fetch independently. Each has its
    own fallback — check "universe_source" and "recent_ipos_detected" in
    the output to see what actually happened on a given run.
  - pybhav is a young, minimally-adopted library (v0.0.2, alpha). It
    handles NSE's session/cookie requirements properly, but hasn't been
    battle-tested at scale — treat "bhavcopy_full_market" as usually
    reliable, not guaranteed.
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

Requires: pip install yfinance pandas numpy requests pybhav
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

try:
    from pybhav import NSEBhavcopy, BhavcopNotAvailable, DownloadError
    HAS_PYBHAV = True
except ImportError:
    HAS_PYBHAV = False

# ---------------------------------------------------------------------------
# Config — this is what you edit day to day
# ---------------------------------------------------------------------------
STRATEGY = "trend_momentum_volume"   # <-- change this to switch strategies

# Screener settings
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
    # --- restored additional Nifty Smallcap 100 names ---
    {"ticker": "HSCL.NS", "name": "Himadri Speciality Chemical"}, {"ticker": "SAILIFE.NS", "name": "Sai Life Sciences"},
    {"ticker": "WOCKPHARMA.NS", "name": "Wockhardt"}, {"ticker": "STARHEALTH.NS", "name": "Star Health and Allied Insurance"},
    {"ticker": "KARURVYSYA.NS", "name": "Karur Vysya Bank"}, {"ticker": "IKS.NS", "name": "Inventurus Knowledge Solutions"},
    {"ticker": "NEULANDLAB.NS", "name": "Neuland Laboratories"}, {"ticker": "MRPL.NS", "name": "Mangalore Refinery & Petrochemicals"},
    {"ticker": "CHOLAHLDNG.NS", "name": "Cholamandalam Financial Holdings"}, {"ticker": "NETWEB.NS", "name": "Netweb Technologies India"},
    {"ticker": "PPLPHARMA.NS", "name": "Piramal Pharma"}, {"ticker": "GRSE.NS", "name": "Garden Reach Shipbuilders & Engineers"},
    {"ticker": "CGCL.NS", "name": "Capri Global Capital"}, {"ticker": "DATAPATTNS.NS", "name": "Data Patterns (India)"},
    {"ticker": "ITI.NS", "name": "ITI Ltd"}, {"ticker": "JYOTICNC.NS", "name": "Jyoti CNC Automation"},
    {"ticker": "CUB.NS", "name": "City Union Bank"}, {"ticker": "FORCEMOT.NS", "name": "Force Motors"},
    {"ticker": "ANANTRAJ.NS", "name": "Anant Raj"}, {"ticker": "SAGILITY.NS", "name": "Sagility India"},
    {"ticker": "IFCI.NS", "name": "IFCI Ltd"}, {"ticker": "RAMCOCEM.NS", "name": "The Ramco Cements"},
    {"ticker": "GESHIP.NS", "name": "The Great Eastern Shipping Company"}, {"ticker": "CASTROLIND.NS", "name": "Castrol India"},
    {"ticker": "FSL.NS", "name": "Firstsource Solutions"}, {"ticker": "TRITURBINE.NS", "name": "Triveni Turbine"},
    {"ticker": "SARDAEN.NS", "name": "Sarda Energy & Minerals"}, {"ticker": "AARTIIND.NS", "name": "Aarti Industries"},
    {"ticker": "BEML.NS", "name": "BEML Ltd"}, {"ticker": "GMDCLTD.NS", "name": "Gujarat Mineral Development Corporation"},
    {"ticker": "DEVYANI.NS", "name": "Devyani International"}, {"ticker": "DEEPAKFERT.NS", "name": "Deepak Fertilisers & Petrochemicals Corp"},
    {"ticker": "CHAMBLFERT.NS", "name": "Chambal Fertilisers & Chemicals"}, {"ticker": "FIVESTAR.NS", "name": "Fivestar Business Finance"},
    {"ticker": "GPIL.NS", "name": "Godawari Power & Ispat"}, {"ticker": "KFINTECH.NS", "name": "KFin Technologies"},
    {"ticker": "PGEL.NS", "name": "PG Electroplast"}, {"ticker": "NATCOPHARM.NS", "name": "Natco Pharma"},
    {"ticker": "JBMA.NS", "name": "JBM Auto"}, {"ticker": "INOXWIND.NS", "name": "Inox Wind"},
    {"ticker": "JMFINANCIL.NS", "name": "JM Financial"}, {"ticker": "SIGNATURE.NS", "name": "Signatureglobal (India)"},
    {"ticker": "IRCON.NS", "name": "Ircon International"}, {"ticker": "AFCONS.NS", "name": "Afcons Infrastructure"},
    {"ticker": "BLS.NS", "name": "BLS International Services"}, {"ticker": "SWANENERGY.NS", "name": "Swan Corp (formerly Swan Energy)"},
    {"ticker": "RPOWER.NS", "name": "Reliance Power"},
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


def find_recent_ipos(nse_list):
    """Dynamically detects recently-listed stocks using each stock's
    real listing date — no hardcoded IPO list. Returns [] if the NSE
    name/listing list wasn't available this run."""
    if not nse_list:
        return []
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=RECENT_IPO_WINDOW_DAYS)
    recent = []
    for s in nse_list:
        if not s.get("listing_date"):
            continue
        try:
            if datetime.fromisoformat(s["listing_date"]).date() >= cutoff:
                recent.append(s)
        except ValueError:
            continue
    return recent


def fetch_bhavcopy_movers(name_lookup, window_calendar_days=12):
    """Uses pybhav to fetch NSE's official daily bhavcopy — which lists
    EVERY security traded that day (typically 1,800-2,100+ symbols) —
    across a short window, and computes % change per symbol between the
    earliest and latest available trading day in that window. This is
    what makes 1,000+ stock coverage practical: one bulk file covering
    the whole market, rather than thousands of individual API calls.
    Returns a scored list, or None if bhavcopy is unavailable this run
    (pybhav not installed, NSE blocked it, etc.)."""
    if not HAS_PYBHAV:
        print("pybhav not installed; skipping bhavcopy screening.")
        return None
    try:
        nse = NSEBhavcopy(cache_dir=None)  # no local caching needed for a one-shot CI run
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=window_calendar_days)
        df = nse.get_range(start.isoformat(), end.isoformat(), segment="CM", skip_errors=True)
    except Exception as e:
        print(f"Bhavcopy fetch failed: {e}")
        return None

    if df is None or df.empty or "_date" not in df.columns:
        print("Bhavcopy returned no usable data this run.")
        return None

    df.columns = df.columns.astype(str).str.strip()
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
    if "SYMBOL" not in df.columns or "CLOSE" not in df.columns:
        print("Bhavcopy data missing expected SYMBOL/CLOSE columns; skipping.")
        return None

    df = df.sort_values("_date")
    scored = []
    for symbol, g in df.groupby("SYMBOL"):
        if len(g) < 2:
            continue
        try:
            past_close = float(g.iloc[0]["CLOSE"])
            recent_close = float(g.iloc[-1]["CLOSE"])
        except (KeyError, ValueError, TypeError):
            continue
        if past_close == 0 or np.isnan(past_close) or np.isnan(recent_close):
            continue
        pct_change = round(100 * (recent_close - past_close) / past_close, 2)
        ticker = f"{symbol}.NS"
        scored.append({
            "ticker": ticker,
            "name": name_lookup.get(ticker, str(symbol)),
            "screener_pct_change": pct_change,
        })
    return scored if scored else None


def _yfinance_batch_scan(universe):
    """Fallback screener: batch-downloads recent closes via yfinance
    across a smaller, fixed universe. Used only when bhavcopy isn't
    available (pybhav missing, or NSE blocked the bhavcopy fetch)."""
    tickers = [s["ticker"] for s in universe]
    lookup = {s["ticker"]: s for s in universe}
    scored = []

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

    return scored


def _finalize_movers(scored):
    """Applies the direction/threshold filter, then backfills with the
    next-highest movers if too few qualify so quiet days aren't empty."""
    if MOVER_DIRECTION == "gainers":
        movers = [s for s in scored if s["screener_pct_change"] >= MOVER_THRESHOLD_PCT]
    elif MOVER_DIRECTION == "losers":
        movers = [s for s in scored if s["screener_pct_change"] <= -MOVER_THRESHOLD_PCT]
    else:
        movers = [s for s in scored if abs(s["screener_pct_change"]) >= MOVER_THRESHOLD_PCT]

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


def run_screener(name_lookup):
    """Tries the full-market bhavcopy screener first (1,000+ symbols).
    Falls back to a smaller yfinance batch scan over the curated pool
    only if bhavcopy is unavailable this run. Returns (movers, source_label)."""
    scored = fetch_bhavcopy_movers(name_lookup)
    if scored is not None:
        print(f"Screened {len(scored)} symbols via NSE bhavcopy (full market).")
        return _finalize_movers(scored), "bhavcopy_full_market"

    print("Bhavcopy screening unavailable this run — falling back to the curated pool via yfinance.")
    fallback_universe = [{"ticker": s["ticker"], "name": s["name"]} for s in FALLBACK_UNIVERSE]
    scored = _yfinance_batch_scan(fallback_universe)
    return _finalize_movers(scored), "fallback_pool"


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

    print("Stage 1: fetching NSE name/listing reference and screening for movers...")
    nse_list = fetch_full_nse_list()
    name_lookup = {e["ticker"]: e["name"] for e in nse_list} if nse_list else {}
    recent_ipos = find_recent_ipos(nse_list)

    movers, universe_source = run_screener(name_lookup)
    update_movers_history(movers)
    print(f"Screener flagged {len(movers)} stocks (direction={MOVER_DIRECTION}, threshold={MOVER_THRESHOLD_PCT}%, source={universe_source}).")
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
