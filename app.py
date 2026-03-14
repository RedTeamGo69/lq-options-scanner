from pandas.io.formats.style import Styler
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf
from scipy.stats import norm

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LQ Quant Options Value Screener v3",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# ENV / CONSTANTS
# ============================================================
TRADIER_BASE_URL = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org/fred").rstrip("/")

DB_PATH = os.getenv("LQ_SCANNER_DB_PATH", "lq_options_scanner.db")

NY_TZ = pytz.timezone("America/New_York")
TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0
T_FLOOR_YEARS = 1.0 / (365.0 * 24.0 * 60.0 * 60.0)  # 1 second


# ============================================================
# CONFIG
# ============================================================
@dataclass
class ScannerConfig:
    min_open_interest: int = 100
    min_volume: int = 10
    min_bid: float = 0.10
    max_spread_pct: float = 15.0
    min_abs_delta: float = 0.10
    max_abs_delta: float = 0.85
    top_n: int = 25

    rv20_weight: float = 0.50
    rv60_weight: float = 0.30
    rv120_weight: float = 0.20
    vol_forecast_multiplier: float = 1.00

    use_executable_pricing: bool = True
    execution_slippage_pct: float = 0.0

    confidence_weight_edge: float = 0.45
    confidence_weight_spread: float = 0.20
    confidence_weight_oi: float = 0.15
    confidence_weight_volume: float = 0.10
    confidence_weight_delta: float = 0.10

    iv_history_lookback_days: int = 252


# ============================================================
# HTTP SESSION
# ============================================================
@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def tradier_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TRADIER_API_KEY:
        raise ValueError("Missing Tradier API key. Set TRADIER_API_KEY environment variable.")

    session = get_http_session()
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json",
    }
    response = session.get(f"{TRADIER_BASE_URL}{path}", headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fred_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not FRED_API_KEY:
        raise ValueError("Missing FRED_API_KEY environment variable.")

    session = get_http_session()
    query = {"api_key": FRED_API_KEY, "file_type": "json"}
    if params:
        query.update(params)

    response = session.get(f"{FRED_BASE_URL}{path}", params=query, timeout=10)
    response.raise_for_status()
    return response.json()


# ============================================================
# NUMERIC HELPERS
# ============================================================
def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ============================================================
# BLACK-SCHOLES / MERTON
# ============================================================
class BlackScholesCalculator:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        self.S = max(float(S), 1e-12)
        self.K = max(float(K), 1e-12)
        self.T = max(float(T), 0.0)
        self.r = float(r)
        self.sigma = max(float(sigma), 1e-6)
        self.q = float(q)

    def _d1_d2(self) -> Tuple[float, float]:
        if self.T <= 0:
            return 0.0, 0.0
        sqrt_T = math.sqrt(self.T)
        d1 = (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2

    def price(self, option_type: str) -> float:
        option_type = option_type.upper()
        if self.T <= 0:
            if option_type == "CALL":
                return max(0.0, self.S - self.K)
            return max(0.0, self.K - self.S)

        d1, d2 = self._d1_d2()
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)

        if option_type == "CALL":
            return self.S * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        return self.K * disc_r * norm.cdf(-d2) - self.S * disc_q * norm.cdf(-d1)

    def greeks(self, option_type: str) -> Dict[str, float]:
        option_type = option_type.upper()

        if self.T <= 0:
            delta = 1.0 if (option_type == "CALL" and self.S > self.K) else 0.0
            if option_type == "PUT":
                delta = -1.0 if self.S < self.K else 0.0
            return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        d1, d2 = self._d1_d2()
        sqrt_T = math.sqrt(self.T)
        pdf_d1 = norm.pdf(d1)
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)

        if option_type == "CALL":
            delta = disc_q * norm.cdf(d1)
            theta = (
                -(self.S * disc_q * pdf_d1 * self.sigma) / (2.0 * sqrt_T)
                - self.r * self.K * disc_r * norm.cdf(d2)
                + self.q * self.S * disc_q * norm.cdf(d1)
            ) / 365.0
        else:
            delta = disc_q * (norm.cdf(d1) - 1.0)
            theta = (
                -(self.S * disc_q * pdf_d1 * self.sigma) / (2.0 * sqrt_T)
                + self.r * self.K * disc_r * norm.cdf(-d2)
                - self.q * self.S * disc_q * norm.cdf(-d1)
            ) / 365.0

        gamma = (disc_q * pdf_d1) / (self.S * self.sigma * sqrt_T)
        vega = (self.S * disc_q * pdf_d1 * sqrt_T) / 100.0  # per 1 vol point

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
        }


# ============================================================
# PARSERS
# ============================================================
def parse_history_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    history_obj = payload.get("history") if isinstance(payload, dict) else None
    day_rows = history_obj.get("day") if isinstance(history_obj, dict) else None
    if day_rows is None:
        return []
    if isinstance(day_rows, dict):
        return [day_rows]
    if isinstance(day_rows, list):
        return day_rows
    return []


def parse_option_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    options_obj = payload.get("options") if isinstance(payload, dict) else None
    option_rows = options_obj.get("option") if isinstance(options_obj, dict) else None
    if option_rows is None:
        return []
    if isinstance(option_rows, dict):
        return [option_rows]
    if isinstance(option_rows, list):
        return option_rows
    return []


# ============================================================
# TIME HELPERS
# ============================================================
def market_close_datetime_ny(date_obj) -> datetime:
    naive_dt = datetime.combine(date_obj, time(16, 0))
    return NY_TZ.localize(naive_dt)


def compute_time_to_expiry_years(expiration_date_str: str) -> Tuple[int, float]:
    today_ny = datetime.now(NY_TZ).date()
    expiry_date = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
    dte = (expiry_date - today_ny).days

    if dte < 0:
        return dte, 0.0

    if dte == 0:
        now_ny = datetime.now(NY_TZ)
        close_ny = market_close_datetime_ny(today_ny)
        seconds_left = max((close_ny - now_ny).total_seconds(), 0.0)
        T = max(seconds_left / (365.0 * 24.0 * 60.0 * 60.0), T_FLOOR_YEARS)
        return dte, T

    return dte, max(dte / CALENDAR_DAYS_PER_YEAR, T_FLOOR_YEARS)


def format_date_dropdown(date_str: str) -> str:
    today_ny = datetime.now(NY_TZ).date()
    days_to_exp = (datetime.strptime(date_str, "%Y-%m-%d").date() - today_ny).days
    return f"{date_str} ({days_to_exp} DTE)"


# ============================================================
# SQLITE IV HISTORY
# ============================================================
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@st.cache_resource
def init_db() -> bool:
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS iv_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            dte INTEGER NOT NULL,
            atm_call_iv REAL,
            atm_put_iv REAL,
            atm_avg_iv REAL,
            spot REAL,
            UNIQUE(snapshot_date, ticker, expiration)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_iv_snapshots_ticker_exp
        ON iv_snapshots (ticker, expiration, snapshot_date)
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_iv_snapshots_ticker_date
        ON iv_snapshots (ticker, snapshot_date)
    """)

    conn.commit()
    conn.close()
    return True


def save_iv_snapshot(
    ticker: str,
    expiration: str,
    dte: int,
    atm_call_iv: Optional[float],
    atm_put_iv: Optional[float],
    atm_avg_iv: Optional[float],
    spot: float,
) -> None:
    snapshot_date = datetime.now(NY_TZ).date().isoformat()
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO iv_snapshots
        (snapshot_date, ticker, expiration, dte, atm_call_iv, atm_put_iv, atm_avg_iv, spot)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        snapshot_date,
        ticker.upper(),
        expiration,
        int(dte),
        atm_call_iv,
        atm_put_iv,
        atm_avg_iv,
        float(spot),
    ))

    conn.commit()
    conn.close()


def get_local_iv_history(
    ticker: str,
    lookback_days: int = 252,
    expiration: Optional[str] = None,
) -> pd.DataFrame:
    cutoff = (datetime.now(NY_TZ).date() - timedelta(days=lookback_days + 30)).isoformat()
    conn = get_db_connection()

    if expiration:
        query = """
            SELECT snapshot_date, ticker, expiration, dte, atm_call_iv, atm_put_iv, atm_avg_iv, spot
            FROM iv_snapshots
            WHERE ticker = ? AND expiration = ? AND snapshot_date >= ?
            ORDER BY snapshot_date
        """
        df = pd.read_sql_query(query, conn, params=(ticker.upper(), expiration, cutoff))
    else:
        query = """
            SELECT snapshot_date, ticker, expiration, dte, atm_call_iv, atm_put_iv, atm_avg_iv, spot
            FROM iv_snapshots
            WHERE ticker = ? AND snapshot_date >= ?
            ORDER BY snapshot_date
        """
        df = pd.read_sql_query(query, conn, params=(ticker.upper(), cutoff))

    conn.close()

    if not df.empty:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def compute_local_iv_rank_and_percentile(
    ticker: str,
    current_iv: float,
    lookback_days: int = 252,
) -> Dict[str, Optional[float]]:
    hist = get_local_iv_history(ticker, lookback_days=lookback_days, expiration=None)
    if hist.empty or "atm_avg_iv" not in hist.columns:
        return {
            "iv_rank": None,
            "iv_percentile": None,
            "hist_count": 0,
            "iv_min": None,
            "iv_max": None,
        }

    series = pd.to_numeric(hist["atm_avg_iv"], errors="coerce").dropna()
    if series.empty:
        return {
            "iv_rank": None,
            "iv_percentile": None,
            "hist_count": 0,
            "iv_min": None,
            "iv_max": None,
        }

    iv_min = float(series.min())
    iv_max = float(series.max())
    iv_rank = None
    if not np.isclose(iv_max, iv_min):
        iv_rank = max(0.0, min(100.0, ((current_iv - iv_min) / (iv_max - iv_min)) * 100.0))

    iv_percentile = float((series <= current_iv).mean() * 100.0)

    return {
        "iv_rank": iv_rank,
        "iv_percentile": iv_percentile,
        "hist_count": int(len(series)),
        "iv_min": iv_min,
        "iv_max": iv_max,
    }


# ============================================================
# DATA FETCHERS - TRADIER
# ============================================================
@st.cache_data(ttl=86400)
def get_company_name(ticker_symbol: str) -> str:
    try:
        data = tradier_get("/markets/quotes", params={"symbols": ticker_symbol})
        quote = data.get("quotes", {}).get("quote")
        if isinstance(quote, list):
            quote = quote[0] if quote else {}
        if isinstance(quote, dict):
            desc = quote.get("description")
            if desc:
                return desc
    except Exception:
        pass
    return ticker_symbol


@st.cache_data(ttl=3600)
def get_risk_free_rate() -> float:
    fallback_rate = 0.045

    if not FRED_API_KEY:
        return fallback_rate

    try:
        payload = fred_get(
            "/series/observations",
            params={
                "series_id": "DTB3",
                "sort_order": "desc",
                "limit": 30,
            },
        )
        observations = payload.get("observations", [])
        for obs in observations:
            value = obs.get("value")
            if value and value != ".":
                return float(value) / 100.0
    except Exception:
        pass

    return fallback_rate


@st.cache_data(ttl=60)
def get_quote_and_history(ticker_symbol: str, history_days: int = 420) -> Dict[str, Any]:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=history_days * 2)

    quote_payload = tradier_get("/markets/quotes", params={"symbols": ticker_symbol})
    quote = quote_payload.get("quotes", {}).get("quote")
    if isinstance(quote, list):
        quote = quote[0] if quote else None
    if not isinstance(quote, dict):
        raise ValueError(f"No quote returned for {ticker_symbol}")

    current_price = safe_float(quote.get("last"))
    if np.isnan(current_price):
        current_price = safe_float(quote.get("close"))
    if np.isnan(current_price) or current_price <= 0:
        raise ValueError(f"Invalid current price for {ticker_symbol}")

    div_yield = safe_float(quote.get("div_yield"), 0.0)
    if div_yield > 1.0:
        div_yield /= 100.0
    div_yield = max(div_yield, 0.0)

    history_payload = tradier_get(
        "/markets/history",
        params={
            "symbol": ticker_symbol,
            "interval": "daily",
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        },
    )
    rows = parse_history_rows(history_payload)
    if not rows:
        raise ValueError(f"No history returned for {ticker_symbol}")

    hist = pd.DataFrame(rows)
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
    hist = hist.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if len(hist) < 130:
        raise ValueError(f"Not enough history for {ticker_symbol}")

    hist["log_return"] = np.log(hist["close"] / hist["close"].shift(1))

    return {
        "price": float(current_price),
        "div_yield": float(div_yield),
        "quote": quote,
        "history": hist,
    }


@st.cache_data(ttl=600)
def get_expiration_dates(ticker_symbol: str) -> List[str]:
    payload = tradier_get(
        "/markets/options/expirations",
        params={
            "symbol": ticker_symbol,
            "includeAllRoots": "true",
            "strikes": "false",
        },
    )
    expiration = payload.get("expirations", {}).get("date")
    if isinstance(expiration, str):
        return [expiration]
    if isinstance(expiration, list):
        return expiration
    return []


@st.cache_data(ttl=30)
def get_option_chain(ticker_symbol: str, expiration: str) -> pd.DataFrame:
    payload = tradier_get(
        "/markets/options/chains",
        params={
            "symbol": ticker_symbol,
            "expiration": expiration,
            "greeks": "true",
        },
    )
    rows = parse_option_rows(payload)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["strike", "bid", "ask", "last", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.upper()
    else:
        df["option_type"] = ""

    def extract_greek(obj, key):
        if isinstance(obj, dict):
            return safe_float(obj.get(key))
        return np.nan

    if "greeks" in df.columns:
        df["mid_iv"] = df["greeks"].apply(lambda x: extract_greek(x, "mid_iv"))
        df["delta_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "delta"))
        df["gamma_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "gamma"))
        df["theta_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "theta"))
        df["vega_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "vega"))
    else:
        df["mid_iv"] = np.nan
        df["delta_mkt"] = np.nan
        df["gamma_mkt"] = np.nan
        df["theta_mkt"] = np.nan
        df["vega_mkt"] = np.nan

    return df


# ============================================================
# YAHOO ENRICHMENT
# ============================================================
@st.cache_data(ttl=3600)
def get_yahoo_events(ticker_symbol: str) -> Dict[str, Optional[str]]:
    result = {
        "next_earnings_date": None,
        "ex_dividend_date": None,
    }

    try:
        tkr = yf.Ticker(ticker_symbol)

        earnings_dt = None
        try:
            edf = tkr.get_earnings_dates(limit=8)
            if edf is not None and not edf.empty:
                edf = edf.reset_index()
                first_col = edf.columns[0]
                edf[first_col] = pd.to_datetime(edf[first_col], errors="coerce")
                edf = edf.dropna(subset=[first_col]).sort_values(first_col)
                today = pd.Timestamp(datetime.now(NY_TZ).date())
                future_rows = edf[edf[first_col].dt.date >= today.date()]
                if not future_rows.empty:
                    earnings_dt = future_rows.iloc[0][first_col]
                else:
                    earnings_dt = edf.iloc[-1][first_col]
        except Exception:
            pass

        if earnings_dt is None:
            try:
                cal = tkr.calendar
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    for col in cal.columns:
                        for val in cal[col].tolist():
                            parsed = pd.to_datetime(val, errors="coerce")
                            if pd.notna(parsed):
                                earnings_dt = parsed
                                break
                        if earnings_dt is not None:
                            break
            except Exception:
                pass

        if earnings_dt is not None and pd.notna(earnings_dt):
            result["next_earnings_date"] = pd.Timestamp(earnings_dt).date().isoformat()

        try:
            fast_info = getattr(tkr, "fast_info", None)
            if fast_info:
                ex_div = None
                if isinstance(fast_info, dict):
                    ex_div = fast_info.get("exDividendDate")
                else:
                    ex_div = getattr(fast_info, "exDividendDate", None)

                if ex_div is not None:
                    parsed = pd.to_datetime(ex_div, unit="s", errors="coerce")
                    if pd.isna(parsed):
                        parsed = pd.to_datetime(ex_div, errors="coerce")
                    if pd.notna(parsed):
                        result["ex_dividend_date"] = parsed.date().isoformat()
        except Exception:
            pass

        if result["ex_dividend_date"] is None:
            try:
                actions = tkr.actions
                if actions is not None and not actions.empty and "Dividends" in actions.columns:
                    divs = actions[actions["Dividends"] > 0].copy()
                    if not divs.empty:
                        divs = divs.reset_index()
                        date_col = divs.columns[0]
                        divs[date_col] = pd.to_datetime(divs[date_col], errors="coerce")
                        divs = divs.dropna(subset=[date_col]).sort_values(date_col)
                        today = pd.Timestamp(datetime.now(NY_TZ).date())
                        future_divs = divs[divs[date_col].dt.date >= today.date()]
                        if not future_divs.empty:
                            result["ex_dividend_date"] = future_divs.iloc[0][date_col].date().isoformat()
            except Exception:
                pass

    except Exception:
        pass

    return result


# ============================================================
# VOL FORECASTS
# ============================================================
def realized_vol_from_history(hist: pd.DataFrame, lookback: int) -> Optional[float]:
    if len(hist) < lookback + 1:
        return None
    rv = hist["log_return"].tail(lookback).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    if pd.isna(rv) or rv <= 0:
        return None
    return float(rv)


def build_forward_vol_forecast(hist: pd.DataFrame, cfg: ScannerConfig) -> Dict[str, Optional[float]]:
    rv20 = realized_vol_from_history(hist, 20)
    rv60 = realized_vol_from_history(hist, 60)
    rv120 = realized_vol_from_history(hist, 120)

    values = []
    weights = []

    if rv20 is not None:
        values.append(rv20)
        weights.append(cfg.rv20_weight)
    if rv60 is not None:
        values.append(rv60)
        weights.append(cfg.rv60_weight)
    if rv120 is not None:
        values.append(rv120)
        weights.append(cfg.rv120_weight)

    if not values or sum(weights) == 0:
        forecast = None
    else:
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        forecast = float(np.dot(np.array(values, dtype=float), w))
        forecast *= cfg.vol_forecast_multiplier

    return {
        "rv20": rv20,
        "rv60": rv60,
        "rv120": rv120,
        "forecast_vol": forecast,
    }


# ============================================================
# SCREENING HELPERS
# ============================================================
def label_moneyness(S: float, K: float, option_type: str, strikes: pd.Series) -> str:
    if strikes.empty:
        return "N/A"
    min_dist = (strikes - S).abs().min()
    if np.isclose(abs(K - S), min_dist):
        return "ATM"

    option_type = option_type.upper()
    if option_type == "CALL":
        return "ITM" if K < S else "OTM"
    return "ITM" if K > S else "OTM"


def compute_execution_price(row: pd.Series, action: str, use_executable_pricing: bool, slippage_pct: float) -> float:
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))

    if np.isnan(bid) or np.isnan(ask):
        return np.nan

    mid = (bid + ask) / 2.0

    if not use_executable_pricing:
        return mid

    slip = slippage_pct / 100.0
    if action == "BUY":
        return ask * (1.0 + slip)
    return bid * (1.0 - slip)


def get_market_greeks(
    row: pd.Series,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_iv: float,
    option_type: str,
) -> Dict[str, float]:
    delta = safe_float(row.get("delta_mkt"))
    gamma = safe_float(row.get("gamma_mkt"))
    theta = safe_float(row.get("theta_mkt"))
    vega = safe_float(row.get("vega_mkt"))

    if all(not np.isnan(x) for x in [delta, gamma, theta, vega]):
        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
        }

    calc = BlackScholesCalculator(S=S, K=K, T=T, r=r, sigma=market_iv, q=q)
    return calc.greeks(option_type)


def normalize_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if series.empty:
        return series
    s = pd.to_numeric(series, errors="coerce").astype(float)
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or np.isclose(min_v, max_v):
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    scaled = 100.0 * (s - min_v) / (max_v - min_v)
    if not higher_is_better:
        scaled = 100.0 - scaled
    return scaled.clip(0, 100)


def compute_confidence_score(df: pd.DataFrame, cfg: ScannerConfig) -> pd.Series:
    edge_score = normalize_score(df["Value Edge (%)"], higher_is_better=True)
    spread_score = normalize_score(df["Spread (%)"], higher_is_better=False)
    oi_score = normalize_score(df["OI"], higher_is_better=True)
    volume_score = normalize_score(df["Vol"], higher_is_better=True)
    delta_centered = 1.0 - (df["Delta"].abs() - 0.45).abs()
    delta_score = normalize_score(delta_centered, higher_is_better=True)

    total = (
        cfg.confidence_weight_edge * edge_score
        + cfg.confidence_weight_spread * spread_score
        + cfg.confidence_weight_oi * oi_score
        + cfg.confidence_weight_volume * volume_score
        + cfg.confidence_weight_delta * delta_score
    )
    return total.clip(0, 100)


def short_option_yield_metrics(action: str, option_type: str, S: float, K: float, credit: float, dte: int) -> Dict[str, float]:
    if action != "SELL":
        return {"Simple Yield (%)": np.nan, "Ann Yield (%)": np.nan}

    effective_dte = max(dte, 1)

    if option_type == "PUT":
        capital_base = K
    else:
        capital_base = S

    if capital_base <= 0:
        return {"Simple Yield (%)": np.nan, "Ann Yield (%)": np.nan}

    simple_yield = (credit / capital_base) * 100.0
    ann_yield = simple_yield * (365.0 / effective_dte)
    return {
        "Simple Yield (%)": simple_yield,
        "Ann Yield (%)": ann_yield,
    }


# ============================================================
# ATM IV SNAPSHOTS
# ============================================================
def build_term_structure_snapshot(
    ticker: str,
    expirations: List[str],
    S: float,
    max_expirations: int = 8,
    save_to_db: bool = True,
) -> pd.DataFrame:
    rows = []
    picked = expirations[:max_expirations]

    for exp in picked:
        try:
            dte, _ = compute_time_to_expiry_years(exp)
            if dte < 0:
                continue

                # no-op

            chain = get_option_chain(ticker, exp)
            if chain.empty:
                continue

            chain = chain.copy()
            chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
            chain["mid_iv"] = pd.to_numeric(chain["mid_iv"], errors="coerce")
            chain["option_type"] = chain["option_type"].astype(str).str.upper()

            calls = chain[(chain["option_type"] == "CALL") & chain["strike"].notna() & chain["mid_iv"].notna()].copy()
            puts = chain[(chain["option_type"] == "PUT") & chain["strike"].notna() & chain["mid_iv"].notna()].copy()

            if calls.empty and puts.empty:
                continue

            def nearest_atm_iv(df_side: pd.DataFrame) -> float:
                idx = (df_side["strike"] - S).abs().idxmin()
                return float(df_side.loc[idx, "mid_iv"])

            call_iv = nearest_atm_iv(calls) if not calls.empty else np.nan
            put_iv = nearest_atm_iv(puts) if not puts.empty else np.nan
            avg_iv = np.nanmean([call_iv, put_iv])

            rows.append(
                {
                    "Expiration": exp,
                    "DTE": dte,
                    "ATM Call IV (%)": call_iv * 100.0 if not np.isnan(call_iv) else np.nan,
                    "ATM Put IV (%)": put_iv * 100.0 if not np.isnan(put_iv) else np.nan,
                    "ATM Avg IV (%)": avg_iv * 100.0 if not np.isnan(avg_iv) else np.nan,
                }
            )

            if save_to_db and not np.isnan(avg_iv):
                save_iv_snapshot(
                    ticker=ticker,
                    expiration=exp,
                    dte=dte,
                    atm_call_iv=None if np.isnan(call_iv) else float(call_iv),
                    atm_put_iv=None if np.isnan(put_iv) else float(put_iv),
                    atm_avg_iv=float(avg_iv),
                    spot=S,
                )

        except Exception:
            continue

    return pd.DataFrame(rows).sort_values("DTE").reset_index(drop=True) if rows else pd.DataFrame()


def build_skew_snapshot(chain_df: pd.DataFrame, S: float, option_type: str = "PUT") -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame()

    df = chain_df.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["mid_iv"] = pd.to_numeric(df["mid_iv"], errors="coerce")
    df["option_type"] = df["option_type"].astype(str).str.upper()

    side = df[(df["option_type"] == option_type.upper()) & df["strike"].notna() & df["mid_iv"].notna()].copy()
    if side.empty:
        return pd.DataFrame()

    side["Pct From Spot"] = ((side["strike"] / S) - 1.0) * 100.0
    side["IV (%)"] = side["mid_iv"] * 100.0
    side = side.sort_values("strike").reset_index(drop=True)

    return side[["strike", "Pct From Spot", "IV (%)"]].rename(columns={"strike": "Strike"})


# ============================================================
# MAIN SCREENING ENGINE
# ============================================================
def screen_chain(
    chain_df: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    T: float,
    dte: int,
    action: str,
    option_family: str,
    forecast_vol: float,
    rv20: Optional[float],
    rv60: Optional[float],
    rv120: Optional[float],
    cfg: ScannerConfig,
) -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame()

    option_type = "PUT" if option_family == "PUTS" else "CALL"

    df = chain_df.copy()
    df = df[df["option_type"] == option_type].copy()
    if df.empty:
        return pd.DataFrame()

    for col in ["strike", "bid", "ask", "volume", "open_interest", "mid_iv"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mid"] = np.where(
        df["bid"].notna() & df["ask"].notna() & (df["bid"] >= 0) & (df["ask"] >= 0),
        (df["bid"] + df["ask"]) / 2.0,
        np.nan,
    )

    df["spread_pct"] = np.where(
        df["mid"].notna() & (df["mid"] > 0),
        ((df["ask"] - df["bid"]) / df["mid"]) * 100.0,
        np.nan,
    )

    df["exec_px"] = df.apply(
        lambda row: compute_execution_price(
            row,
            action=action,
            use_executable_pricing=cfg.use_executable_pricing,
            slippage_pct=cfg.execution_slippage_pct,
        ),
        axis=1,
    )

    df = df[
        df["strike"].notna()
        & df["bid"].notna()
        & df["ask"].notna()
        & df["mid_iv"].notna()
        & (df["mid_iv"] > 0)
        & (df["open_interest"].fillna(0) >= cfg.min_open_interest)
        & (df["volume"].fillna(0) >= cfg.min_volume)
        & (df["bid"] >= cfg.min_bid)
        & df["spread_pct"].notna()
        & (df["spread_pct"] <= cfg.max_spread_pct)
        & df["exec_px"].notna()
        & (df["exec_px"] > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    strikes_series = df["strike"].astype(float)
    rows = []

    for _, row in df.iterrows():
        K = float(row["strike"])
        bid = float(row["bid"])
        ask = float(row["ask"])
        mid = float(row["mid"])
        exec_px = float(row["exec_px"])
        market_iv = float(row["mid_iv"])
        oi = safe_int(row.get("open_interest"))
        vol = safe_int(row.get("volume"))

        market_calc = BlackScholesCalculator(S=S, K=K, T=T, r=r, sigma=market_iv, q=q)
        market_theo = market_calc.price(option_type)

        market_greeks = get_market_greeks(
            row=row,
            S=S,
            K=K,
            T=T,
            r=r,
            q=q,
            market_iv=market_iv,
            option_type=option_type,
        )

        if abs(market_greeks["delta"]) < cfg.min_abs_delta or abs(market_greeks["delta"]) > cfg.max_abs_delta:
            continue

        forecast_calc = BlackScholesCalculator(S=S, K=K, T=T, r=r, sigma=forecast_vol, q=q)
        forecast_theo = forecast_calc.price(option_type)
        forecast_greeks = forecast_calc.greeks(option_type)

        if action == "BUY":
            abs_edge = forecast_theo - exec_px
        else:
            abs_edge = exec_px - forecast_theo

        value_edge_pct = (abs_edge / exec_px) * 100.0 if exec_px > 0 else np.nan

        iv_minus_forecast = (market_iv - forecast_vol) * 100.0
        iv_minus_rv20 = ((market_iv - rv20) * 100.0) if rv20 is not None else np.nan
        iv_minus_rv60 = ((market_iv - rv60) * 100.0) if rv60 is not None else np.nan
        iv_minus_rv120 = ((market_iv - rv120) * 100.0) if rv120 is not None else np.nan

        yld = short_option_yield_metrics(action, option_type, S, K, exec_px, dte)

        rows.append(
            {
                "Moneyness": label_moneyness(S, K, option_type, strikes_series),
                "Strike": K,
                "Bid": bid,
                "Ask": ask,
                "Mid": mid,
                "Exec Px": exec_px,
                "Market Theo": market_theo,
                "Forecast Theo": forecast_theo,
                "Abs Edge ($)": abs_edge,
                "Value Edge (%)": value_edge_pct,
                "Spread (%)": float(row["spread_pct"]),
                "Mkt IV (%)": market_iv * 100.0,
                "Forecast Vol (%)": forecast_vol * 100.0,
                "RV20 (%)": rv20 * 100.0 if rv20 is not None else np.nan,
                "RV60 (%)": rv60 * 100.0 if rv60 is not None else np.nan,
                "RV120 (%)": rv120 * 100.0 if rv120 is not None else np.nan,
                "IV - Forecast (pts)": iv_minus_forecast,
                "IV - RV20 (pts)": iv_minus_rv20,
                "IV - RV60 (pts)": iv_minus_rv60,
                "IV - RV120 (pts)": iv_minus_rv120,
                "Delta": market_greeks["delta"],
                "Gamma": market_greeks["gamma"],
                "Theta": market_greeks["theta"],
                "Vega": market_greeks["vega"],
                "Model Delta": forecast_greeks["delta"],
                "Model Gamma": forecast_greeks["gamma"],
                "Model Theta": forecast_greeks["theta"],
                "Model Vega": forecast_greeks["vega"],
                "Vol": vol,
                "OI": oi,
                "DTE": dte,
                "Simple Yield (%)": yld["Simple Yield (%)"],
                "Ann Yield (%)": yld["Ann Yield (%)"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Confidence"] = compute_confidence_score(out, cfg)
    out = out.sort_values(
        by=["Confidence", "Value Edge (%)", "Spread (%)", "OI"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    return out.head(cfg.top_n)


# ============================================================
# DISPLAY HELPERS
# ============================================================
def style_results(df: pd.DataFrame) -> Styler:
    def color_edge(val):
        if pd.isna(val):
            return ""
        if val >= 5:
            return "color: #00FF88; font-weight: bold"
        if val <= -5:
            return "color: #FF5A5A; font-weight: bold"
        return ""

    def color_spread(val):
        if pd.isna(val):
            return ""
        if val > 15:
            return "color: #FF5A5A; font-weight: bold"
        if val > 8:
            return "color: #FFD166"
        return "color: #00FF88"

    def color_conf(val):
        if pd.isna(val):
            return ""
        if val >= 75:
            return "color: #00FF88; font-weight: bold"
        if val >= 50:
            return "color: #FFD166; font-weight: bold"
        return "color: #FF5A5A"

    def color_moneyness(val):
        if val == "ATM":
            return "color: #FFD166; font-weight: bold"
        if val == "OTM":
            return "color: #00FFFF; font-weight: bold"
        if val == "ITM":
            return "color: #FF8A8A; font-weight: bold"
        return ""

    styler = df.style
    for col, func in {
        "Value Edge (%)": color_edge,
        "Spread (%)": color_spread,
        "Confidence": color_conf,
        "Moneyness": color_moneyness,
    }.items():
        if col in df.columns:
            styler = styler.applymap(func, subset=[col])

    return styler.format(
        {
            "Strike": "{:,.2f}",
            "Bid": "{:,.2f}",
            "Ask": "{:,.2f}",
            "Mid": "{:,.2f}",
            "Exec Px": "{:,.2f}",
            "Market Theo": "{:,.2f}",
            "Forecast Theo": "{:,.2f}",
            "Abs Edge ($)": "{:,.2f}",
            "Value Edge (%)": "{:,.1f}",
            "Spread (%)": "{:,.1f}",
            "Mkt IV (%)": "{:,.1f}",
            "Forecast Vol (%)": "{:,.1f}",
            "RV20 (%)": "{:,.1f}",
            "RV60 (%)": "{:,.1f}",
            "RV120 (%)": "{:,.1f}",
            "IV - Forecast (pts)": "{:,.1f}",
            "IV - RV20 (pts)": "{:,.1f}",
            "IV - RV60 (pts)": "{:,.1f}",
            "IV - RV120 (pts)": "{:,.1f}",
            "Delta": "{:,.3f}",
            "Gamma": "{:,.4f}",
            "Theta": "{:,.4f}",
            "Vega": "{:,.4f}",
            "Model Delta": "{:,.3f}",
            "Model Gamma": "{:,.4f}",
            "Model Theta": "{:,.4f}",
            "Model Vega": "{:,.4f}",
            "Simple Yield (%)": "{:,.2f}",
            "Ann Yield (%)": "{:,.1f}",
            "Confidence": "{:,.0f}",
        }
    )


def display_summary(
    ticker: str,
    company_name: str,
    S: float,
    q: float,
    r: float,
    rv20: Optional[float],
    rv60: Optional[float],
    rv120: Optional[float],
    forecast_vol: Optional[float],
    yahoo_events: Dict[str, Optional[str]],
    iv_stats: Dict[str, Optional[float]],
) -> None:
    if company_name.upper() == ticker.upper():
        st.subheader(ticker)
    else:
        st.subheader(f"{company_name} ({ticker})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"${S:,.2f}")
    c2.metric("Dividend Yield", f"{q*100:.2f}%")
    c3.metric("Risk-Free", f"{r*100:.2f}%")
    c4.metric("Forecast Vol", f"{forecast_vol*100:.1f}%" if forecast_vol is not None else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("RV20", f"{rv20*100:.1f}%" if rv20 is not None else "N/A")
    c6.metric("RV60", f"{rv60*100:.1f}%" if rv60 is not None else "N/A")
    c7.metric("RV120", f"{rv120*100:.1f}%" if rv120 is not None else "N/A")
    c8.metric("Local IV Obs", str(iv_stats["hist_count"]))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Local IV Rank", f"{iv_stats['iv_rank']:.0f}%" if iv_stats["iv_rank"] is not None else "N/A")
    c10.metric("Local IV Percentile", f"{iv_stats['iv_percentile']:.0f}%" if iv_stats["iv_percentile"] is not None else "N/A")
    c11.metric("Next Earnings", yahoo_events.get("next_earnings_date") or "N/A")
    c12.metric("Ex-Div Date", yahoo_events.get("ex_dividend_date") or "N/A")


def display_event_warnings(yahoo_events: Dict[str, Optional[str]], target_date: str) -> None:
    today_ny = datetime.now(NY_TZ).date()
    expiry_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

    earn_str = yahoo_events.get("next_earnings_date")
    if earn_str:
        try:
            earn_dt = datetime.strptime(earn_str, "%Y-%m-%d").date()
            if today_ny <= earn_dt <= expiry_dt:
                st.warning(f"Earnings risk: earnings date appears to fall before expiration ({earn_str}).")
        except Exception:
            pass

    ex_div_str = yahoo_events.get("ex_dividend_date")
    if ex_div_str:
        try:
            ex_dt = datetime.strptime(ex_div_str, "%Y-%m-%d").date()
            if today_ny <= ex_dt <= expiry_dt:
                st.warning(f"Dividend event risk: ex-dividend date appears before expiration ({ex_div_str}).")
        except Exception:
            pass


def display_expected_moves(S: float, T: float, forecast_vol: float, best_df: pd.DataFrame) -> None:
    if T <= 0 or best_df.empty:
        return

    atm_rows = best_df[best_df["Moneyness"] == "ATM"].copy()
    if not atm_rows.empty:
        atm_iv = atm_rows.iloc[0]["Mkt IV (%)"] / 100.0
    else:
        temp = best_df.copy()
        temp["dist_to_50"] = (temp["Delta"].abs() - 0.50).abs()
        atm_iv = temp.sort_values("dist_to_50").iloc[0]["Mkt IV (%)"] / 100.0

    market_em = S * atm_iv * math.sqrt(T)
    model_em = S * forecast_vol * math.sqrt(T)

    c1, c2 = st.columns(2)
    c1.metric("Market-Implied Move (1σ)", f"± ${market_em:.2f}")
    c2.metric("Forecast-Vol Move (1σ)", f"± ${model_em:.2f}")

    st.caption(
        f"Market range: {S - market_em:.2f} to {S + market_em:.2f} | "
        f"Forecast range: {S - model_em:.2f} to {S + model_em:.2f}"
    )


def display_interpretation(best_df: pd.DataFrame, action: str) -> None:
    if best_df.empty:
        return

    top = best_df.iloc[0]
    msg = (
        f"Top contract confidence {top['Confidence']:.0f}/100. "
        f"Value edge {top['Value Edge (%)']:.1f}%. "
        f"Spread {top['Spread (%)']:.1f}%. "
        f"Market IV minus forecast vol {top['IV - Forecast (pts)']:.1f} vol points."
    )

    if action == "BUY":
        msg = "Buy-side interpretation: positive edge means the contract screens cheap versus the forecast-vol model. " + msg
    else:
        msg = "Sell-side interpretation: positive edge means the contract screens rich versus the forecast-vol model. " + msg

    st.info(msg)


# ============================================================
# PROCESS TICKER
# ============================================================
def process_ticker(ticker: str, action: str, option_family: str, cfg: ScannerConfig, key_suffix: str = "") -> None:
    with st.spinner(f"Loading {ticker}..."):
        try:
            market_data = get_quote_and_history(ticker)
            expirations = get_expiration_dates(ticker)
            yahoo_events = get_yahoo_events(ticker)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                st.warning(f"{ticker}: Tradier rate limit hit.")
            else:
                st.error(f"{ticker}: HTTP error ({status}).")
            return
        except Exception as e:
            st.error(f"{ticker}: {e}")
            return

    if not expirations:
        st.warning(f"{ticker}: no option expirations found.")
        return

    S = market_data["price"]
    q = market_data["div_yield"]
    hist = market_data["history"]
    r = get_risk_free_rate()
    company_name = get_company_name(ticker)

    vol_pack = build_forward_vol_forecast(hist, cfg)
    rv20 = vol_pack["rv20"]
    rv60 = vol_pack["rv60"]
    rv120 = vol_pack["rv120"]
    forecast_vol = vol_pack["forecast_vol"]

    if forecast_vol is None or forecast_vol <= 0:
        st.error(f"{ticker}: could not build a valid forecast vol.")
        return

    exp_col, ctrl_col = st.columns([3, 1])

    with exp_col:
        target_date = st.selectbox(
            "Select Expiration",
            expirations,
            format_func=format_date_dropdown,
            key=f"exp_{ticker}_{key_suffix}",
        )

    with ctrl_col:
        run_scan = st.button(
            "Scan Chain",
            use_container_width=True,
            type="primary",
            key=f"scan_{ticker}_{key_suffix}",
        )

    dte, T = compute_time_to_expiry_years(target_date)
    if dte < 0:
        st.warning("Selected expiration is in the past.")
        return

    results_key = f"results_{ticker}_{action}_{option_family}_{target_date}"

    if run_scan:
        with st.spinner("Scanning chain, saving IV snapshots, and building analytics..."):
            try:
                chain_df = get_option_chain(ticker, target_date)
                term_df = build_term_structure_snapshot(ticker, expirations, S, save_to_db=True)

                current_atm_iv = None
                if not term_df.empty:
                    row = term_df[term_df["Expiration"] == target_date]
                    if not row.empty and pd.notna(row.iloc[0]["ATM Avg IV (%)"]):
                        current_atm_iv = float(row.iloc[0]["ATM Avg IV (%)"]) / 100.0

                iv_stats = {
                    "iv_rank": None,
                    "iv_percentile": None,
                    "hist_count": 0,
                    "iv_min": None,
                    "iv_max": None,
                }
                if current_atm_iv is not None:
                    iv_stats = compute_local_iv_rank_and_percentile(
                        ticker=ticker,
                        current_iv=current_atm_iv,
                        lookback_days=cfg.iv_history_lookback_days,
                    )

                best_df = screen_chain(
                    chain_df=chain_df,
                    S=S,
                    r=r,
                    q=q,
                    T=T,
                    dte=dte,
                    action=action,
                    option_family=option_family,
                    forecast_vol=forecast_vol,
                    rv20=rv20,
                    rv60=rv60,
                    rv120=rv120,
                    cfg=cfg,
                )

                put_skew_df = build_skew_snapshot(chain_df, S, option_type="PUT")
                call_skew_df = build_skew_snapshot(chain_df, S, option_type="CALL")
                iv_hist_df = get_local_iv_history(ticker, lookback_days=cfg.iv_history_lookback_days)

                st.session_state[results_key] = {
                    "best_df": best_df,
                    "term_df": term_df,
                    "put_skew_df": put_skew_df,
                    "call_skew_df": call_skew_df,
                    "iv_hist_df": iv_hist_df,
                    "S": S,
                    "T": T,
                    "forecast_vol": forecast_vol,
                    "expiration": target_date,
                    "yahoo_events": yahoo_events,
                    "iv_stats": iv_stats,
                    "rv20": rv20,
                    "rv60": rv60,
                    "rv120": rv120,
                    "r": r,
                    "q": q,
                    "company_name": company_name,
                }

            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    st.warning("Tradier rate limit hit while scanning.")
                else:
                    st.error(f"HTTP error while scanning ({status}).")
                return
            except Exception as e:
                st.error(f"Scan error: {e}")
                return

    if results_key not in st.session_state:
        display_summary(
            ticker=ticker,
            company_name=company_name,
            S=S,
            q=q,
            r=r,
            rv20=rv20,
            rv60=rv60,
            rv120=rv120,
            forecast_vol=forecast_vol,
            yahoo_events=yahoo_events,
            iv_stats={"iv_rank": None, "iv_percentile": None, "hist_count": 0, "iv_min": None, "iv_max": None},
        )
        st.caption("Pick an expiration and click Scan Chain.")
        return

    cached = st.session_state[results_key]

    display_summary(
        ticker=ticker,
        company_name=cached["company_name"],
        S=cached["S"],
        q=cached["q"],
        r=cached["r"],
        rv20=cached["rv20"],
        rv60=cached["rv60"],
        rv120=cached["rv120"],
        forecast_vol=cached["forecast_vol"],
        yahoo_events=cached["yahoo_events"],
        iv_stats=cached["iv_stats"],
    )

    display_event_warnings(cached["yahoo_events"], cached["expiration"])

    best_df = cached["best_df"]
    term_df = cached["term_df"]
    put_skew_df = cached["put_skew_df"]
    call_skew_df = cached["call_skew_df"]
    iv_hist_df = cached["iv_hist_df"]

    if best_df.empty:
        st.warning("No contracts passed the filters.")
        return

    display_expected_moves(cached["S"], cached["T"], cached["forecast_vol"], best_df)
    display_interpretation(best_df, action)

    tabs = st.tabs(["Top Contracts", "Term Structure", "Put Skew", "Call Skew", "Local IV History"])

    with tabs[0]:
        st.subheader(f"Top Contracts | {ticker} | {action} {option_family} | {cached['expiration']}")
        styled = style_results(best_df)

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn("Strike", format="$ %.2f"),
                "Bid": st.column_config.NumberColumn("Bid", format="$ %.2f"),
                "Ask": st.column_config.NumberColumn("Ask", format="$ %.2f"),
                "Mid": st.column_config.NumberColumn("Mid", format="$ %.2f"),
                "Exec Px": st.column_config.NumberColumn("Exec Px", format="$ %.2f"),
                "Market Theo": st.column_config.NumberColumn("Market Theo", format="$ %.2f"),
                "Forecast Theo": st.column_config.NumberColumn("Forecast Theo", format="$ %.2f"),
                "Abs Edge ($)": st.column_config.NumberColumn("Abs Edge ($)", format="$ %.2f"),
                "Value Edge (%)": st.column_config.NumberColumn("Value Edge (%)", format="%.1f%%"),
                "Spread (%)": st.column_config.NumberColumn("Spread (%)", format="%.1f%%"),
                "Mkt IV (%)": st.column_config.NumberColumn("Mkt IV (%)", format="%.1f%%"),
                "Forecast Vol (%)": st.column_config.NumberColumn("Forecast Vol (%)", format="%.1f%%"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f"),
            },
        )

        csv = best_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{ticker}_{action}_{option_family}_{cached['expiration']}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_{ticker}_{action}_{option_family}_{cached['expiration']}",
        )

    with tabs[1]:
        st.subheader("ATM IV Term Structure")
        if term_df.empty:
            st.warning("No term-structure data available.")
        else:
            st.dataframe(term_df, use_container_width=True, hide_index=True)
            chart_df = term_df.set_index("DTE")[["ATM Avg IV (%)"]]
            st.line_chart(chart_df)

    with tabs[2]:
        st.subheader("Put Skew Snapshot")
        if put_skew_df.empty:
            st.warning("No put skew data available.")
        else:
            st.dataframe(put_skew_df, use_container_width=True, hide_index=True)
            st.line_chart(put_skew_df.set_index("Pct From Spot")[["IV (%)"]])

    with tabs[3]:
        st.subheader("Call Skew Snapshot")
        if call_skew_df.empty:
            st.warning("No call skew data available.")
        else:
            st.dataframe(call_skew_df, use_container_width=True, hide_index=True)
            st.line_chart(call_skew_df.set_index("Pct From Spot")[["IV (%)"]])

    with tabs[4]:
        st.subheader("Local IV History")
        if iv_hist_df.empty:
            st.warning("No local IV history saved yet. Each scan saves ATM IV snapshots to SQLite.")
        else:
            iv_hist_df = iv_hist_df.copy()
            iv_hist_df["snapshot_date"] = pd.to_datetime(iv_hist_df["snapshot_date"], errors="coerce")
            iv_hist_df["atm_avg_iv_pct"] = pd.to_numeric(iv_hist_df["atm_avg_iv"], errors="coerce") * 100.0

            st.dataframe(iv_hist_df, use_container_width=True, hide_index=True)

            daily = (
                iv_hist_df.groupby("snapshot_date", as_index=False)["atm_avg_iv_pct"]
                .mean()
                .sort_values("snapshot_date")
            )
            if not daily.empty:
                st.line_chart(daily.set_index("snapshot_date")[["atm_avg_iv_pct"]])


# ============================================================
# MAIN APP
# ============================================================
_ = init_db()

st.title("📈 LQ Quant Options Value Screener v3")
st.markdown(
    "Tradier powers live chain pricing and Greeks. Yahoo enriches earnings and ex-dividend dates. "
    "SQLite stores daily ATM IV snapshots so the app can build its own real IV history over time."
)

if not TRADIER_API_KEY:
    st.error("Missing Tradier API key. Set TRADIER_API_KEY before using the app.")
    st.stop()

with st.sidebar:
    st.header("Scanner Settings")

    action = st.radio("Action", ["SELL", "BUY"], horizontal=True)
    option_family = st.radio("Type", ["PUTS", "CALLS"], horizontal=True)

    st.subheader("Liquidity Filters")
    min_open_interest = st.number_input("Min OI", min_value=0, value=100, step=10)
    min_volume = st.number_input("Min Volume", min_value=0, value=10, step=1)
    min_bid = st.number_input("Min Bid", min_value=0.0, value=0.10, step=0.05, format="%.2f")
    max_spread_pct = st.number_input("Max Spread %", min_value=1.0, value=15.0, step=1.0)
    min_abs_delta = st.number_input("Min |Delta|", min_value=0.01, max_value=0.99, value=0.10, step=0.01, format="%.2f")
    max_abs_delta = st.number_input("Max |Delta|", min_value=0.01, max_value=0.99, value=0.85, step=0.01, format="%.2f")
    top_n = st.number_input("Top N", min_value=5, max_value=100, value=25, step=5)

    st.subheader("Forecast Vol Model")
    rv20_weight = st.slider("RV20 Weight", 0.0, 1.0, 0.50, 0.05)
    rv60_weight = st.slider("RV60 Weight", 0.0, 1.0, 0.30, 0.05)
    rv120_weight = st.slider("RV120 Weight", 0.0, 1.0, 0.20, 0.05)
    vol_forecast_multiplier = st.slider("Forecast Vol Multiplier", 0.50, 1.50, 1.00, 0.01)

    st.subheader("Execution Model")
    use_executable_pricing = st.toggle("Use bid/ask execution pricing", value=True)
    execution_slippage_pct = st.slider("Slippage %", 0.0, 5.0, 0.0, 0.1)

    st.subheader("Local IV History")
    iv_history_lookback_days = st.number_input("IV History Lookback Days", min_value=30, max_value=1000, value=252, step=10)

cfg = ScannerConfig(
    min_open_interest=int(min_open_interest),
    min_volume=int(min_volume),
    min_bid=float(min_bid),
    max_spread_pct=float(max_spread_pct),
    min_abs_delta=float(min_abs_delta),
    max_abs_delta=float(max_abs_delta),
    top_n=int(top_n),
    rv20_weight=float(rv20_weight),
    rv60_weight=float(rv60_weight),
    rv120_weight=float(rv120_weight),
    vol_forecast_multiplier=float(vol_forecast_multiplier),
    use_executable_pricing=bool(use_executable_pricing),
    execution_slippage_pct=float(execution_slippage_pct),
    iv_history_lookback_days=int(iv_history_lookback_days),
)

with st.form("search_form"):
    ticker_input = st.text_input(
        "Enter ticker(s), comma-separated",
        value="",
        placeholder="AAPL, TSLA, SPY",
    ).strip().upper()

    submit_search = st.form_submit_button("Fetch Options Data", type="primary", use_container_width=True)

if submit_search:
    if ticker_input:
        tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        tickers = dedupe_preserve_order(tickers)
        st.session_state["active_tickers"] = tickers
    else:
        st.warning("Please enter at least one ticker.")

if "active_tickers" in st.session_state:
    tickers = st.session_state["active_tickers"]

    if len(tickers) == 1:
        process_ticker(tickers[0], action, option_family, cfg)
    else:
        tabs = st.tabs(tickers)
        for tab, tkr in zip(tabs, tickers):
            with tab:
                process_ticker(tkr, action, option_family, cfg, key_suffix=tkr)
