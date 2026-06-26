import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config import (
    TRADIER_BASE_URL,
    TRADIER_FUNDAMENTALS_BASE_URL,
    TRADIER_API_KEY,
    FRED_API_KEY,
    FRED_BASE_URL,
    NY_TZ,
)
from utils import safe_float

logger = logging.getLogger(__name__)


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


def tradier_fundamentals_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET against Tradier's beta fundamentals API (Morningstar-sourced).

    Same auth/session as tradier_get, but a separate base URL since fundamentals
    live under /beta rather than /v1. Returns the parsed JSON (typically a list).
    """
    if not TRADIER_API_KEY:
        raise ValueError("Missing Tradier API key. Set TRADIER_API_KEY environment variable.")

    session = get_http_session()
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json",
    }
    response = session.get(
        f"{TRADIER_FUNDAMENTALS_BASE_URL}{path}", headers=headers, params=params, timeout=10
    )
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
    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise requests.HTTPError(
            f"FRED API error: {response.status_code} for {FRED_BASE_URL}{path}"
        ) from None
    return response.json()


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
        logger.debug("Failed to fetch company name for %s", ticker_symbol, exc_info=True)
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
        logger.warning("FRED risk-free rate fetch failed, using fallback %.2f%%", fallback_rate * 100, exc_info=True)

    return fallback_rate


@st.cache_data(ttl=60)
def get_quote_and_history(ticker_symbol: str, history_days: int = 420) -> Dict[str, Any]:
    end_date = datetime.now(NY_TZ).date()
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
    # Keep OHLC when present so range-based estimators (Yang-Zhang) can use it.
    # These are optional: close-only history still works, just falls back to
    # the close-to-close estimator downstream.
    for col in ["open", "high", "low"]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce") if col in hist.columns else np.nan
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
        # bid/ask IV let the edge compute in IV space with an honest executable
        # haircut (pay ask_iv, receive bid_iv) instead of only the mid.
        df["bid_iv"] = df["greeks"].apply(lambda x: extract_greek(x, "bid_iv"))
        df["ask_iv"] = df["greeks"].apply(lambda x: extract_greek(x, "ask_iv"))
        df["delta_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "delta"))
        df["gamma_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "gamma"))
        df["theta_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "theta"))
        df["vega_mkt"] = df["greeks"].apply(lambda x: extract_greek(x, "vega"))
    else:
        df["mid_iv"] = np.nan
        df["bid_iv"] = np.nan
        df["ask_iv"] = np.nan
        df["delta_mkt"] = np.nan
        df["gamma_mkt"] = np.nan
        df["theta_mkt"] = np.nan
        df["vega_mkt"] = np.nan

    return df


# ============================================================
# TRADIER FUNDAMENTALS (dividends + corporate calendar)
# ============================================================
def _iso_date(value: Any) -> Optional[str]:
    """Coerce a date-like value to an ISO yyyy-mm-dd string, or None."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    try:
        return pd.Timestamp(parsed).date().isoformat()
    except (AttributeError, ValueError):
        return None


def _collect_fundamentals_rows(payload: Any, table_key: str) -> List[Dict[str, Any]]:
    """Recursively gather every dict row stored under ``table_key`` anywhere in
    the deeply-nested, Morningstar-shaped fundamentals payload. Resilient to the
    exact nesting depth and to Tradier's list/dict/null collection quirks."""
    rows: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == table_key and isinstance(value, list):
                    rows.extend(item for item in value if isinstance(item, dict))
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    return rows


def _row_value(row: Dict[str, Any], *keys: str) -> Any:
    """First non-null value among the given keys (tolerates field-name variants)."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _pick_event_date(dated_rows: List[tuple], today) -> Optional[str]:
    """From (date, iso) tuples, return the soonest date >= today, else the latest."""
    valid = [(d, iso) for d, iso in dated_rows if d is not None]
    if not valid:
        return None
    future = sorted((t for t in valid if t[0] >= today), key=lambda t: t[0])
    if future:
        return future[0][1]
    return max(valid, key=lambda t: t[0])[1]


@st.cache_data(ttl=3600)
def get_tradier_fundamentals(ticker_symbol: str) -> Dict[str, Any]:
    """Next earnings date, ex-dividend date, and trailing-twelve-month dividend
    total from Tradier's beta fundamentals API (Morningstar-sourced).

    Returns the same keys the rest of the app consumes. Every field degrades to
    None on any error or missing coverage, so the UI never breaks on this call.
    """
    result: Dict[str, Any] = {
        "next_earnings_date": None,
        "ex_dividend_date": None,
        "trailing_annual_dividend": None,
    }

    today = datetime.now(NY_TZ).date()

    # --- Dividends: ex-dividend date + trailing-12-month total ---
    try:
        payload = tradier_fundamentals_get(
            "/markets/fundamentals/dividends", params={"symbols": ticker_symbol}
        )
        div_rows = _collect_fundamentals_rows(payload, "cash_dividends")

        # Dedupe by ex-date (the same dividend can appear under multiple result
        # blocks, e.g. Company vs Stock share-class entries).
        by_ex_date: Dict[str, Dict[str, Any]] = {}
        for row in div_rows:
            iso = _iso_date(_row_value(row, "ex_date", "ex_dividend_date"))
            if iso is not None and iso not in by_ex_date:
                by_ex_date[iso] = row

        if by_ex_date:
            dated = [(datetime.strptime(iso, "%Y-%m-%d").date(), iso) for iso in by_ex_date]
            result["ex_dividend_date"] = _pick_event_date(dated, today)

            cutoff = today - timedelta(days=365)
            ttm = 0.0
            for iso, row in by_ex_date.items():
                ex_d = datetime.strptime(iso, "%Y-%m-%d").date()
                if cutoff <= ex_d <= today:
                    amt = safe_float(_row_value(row, "cash_amount", "amount"))
                    if not np.isnan(amt) and amt > 0:
                        ttm += amt
            if ttm > 0:
                result["trailing_annual_dividend"] = ttm
    except Exception:
        logger.warning("Tradier dividends fetch failed for %s", ticker_symbol, exc_info=True)

    # --- Corporate calendar: next earnings date ---
    try:
        payload = tradier_fundamentals_get(
            "/markets/fundamentals/calendars", params={"symbols": ticker_symbol}
        )
        cal_rows = _collect_fundamentals_rows(payload, "corporate_calendars")

        earnings_dates = []
        for row in cal_rows:
            event_text = _row_value(row, "event")
            event_type = _row_value(row, "event_type")
            is_earnings = (
                isinstance(event_text, str) and "earnings" in event_text.lower()
            ) or str(event_type) == "14"  # Morningstar earnings event_type
            if not is_earnings:
                continue
            iso = _iso_date(_row_value(row, "begin_date_time", "begin_date"))
            if iso is not None:
                earnings_dates.append((datetime.strptime(iso, "%Y-%m-%d").date(), iso))

        result["next_earnings_date"] = _pick_event_date(earnings_dates, today)
    except Exception:
        logger.warning("Tradier calendar fetch failed for %s", ticker_symbol, exc_info=True)

    return result
