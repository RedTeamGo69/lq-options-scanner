import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

import public_api
from config import FRED_API_KEY, FRED_BASE_URL, NY_TZ
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
# YFINANCE-DERIVED FIELDS
# ============================================================
# Public's API exposes neither a company name nor a dividend yield, so these
# two fields come from yfinance (already used below for earnings / ex-div dates).
@st.cache_data(ttl=86400)
def get_company_name(ticker_symbol: str) -> str:
    try:
        info = getattr(yf.Ticker(ticker_symbol), "info", None) or {}
        for key in ("longName", "shortName"):
            name = info.get(key)
            if name:
                return str(name)
    except Exception:
        logger.debug("Failed to fetch company name for %s", ticker_symbol, exc_info=True)
    return ticker_symbol


@st.cache_data(ttl=86400)
def get_dividend_yield(ticker_symbol: str, spot: float) -> float:
    """Trailing-12-month cash dividends / spot, from yfinance."""
    if spot is None or spot <= 0:
        return 0.0
    try:
        divs = yf.Ticker(ticker_symbol).dividends
        if divs is not None and not divs.empty:
            idx = divs.index
            now = pd.Timestamp.now(tz=idx.tz) if getattr(idx, "tz", None) is not None else pd.Timestamp.now()
            ttm = float(divs[idx >= (now - pd.Timedelta(days=365))].sum())
            if ttm > 0:
                return max(min(ttm / float(spot), 1.0), 0.0)
    except Exception:
        logger.debug("Failed to fetch dividend yield for %s", ticker_symbol, exc_info=True)
    return 0.0


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


def _bars_to_history(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if df.empty or "close" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame()

    dt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if dt.isna().all():
        epoch = pd.to_numeric(df["timestamp"], errors="coerce")
        dt = pd.to_datetime(epoch, unit="ms", errors="coerce", utc=True)
        if dt.isna().all():
            dt = pd.to_datetime(epoch, unit="s", errors="coerce", utc=True)

    df["date"] = dt.dt.tz_convert(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "close"]]


@st.cache_data(ttl=60)
def get_quote_and_history(ticker_symbol: str, history_days: int = 420) -> Dict[str, Any]:
    quote = public_api.get_equity_quote(ticker_symbol)
    if not isinstance(quote, dict):
        raise ValueError(f"No quote returned for {ticker_symbol}")

    current_price = safe_float(quote.get("last"))
    if np.isnan(current_price):
        current_price = safe_float(quote.get("previousClose"))
    if np.isnan(current_price) or current_price <= 0:
        raise ValueError(f"Invalid current price for {ticker_symbol}")

    div_yield = get_dividend_yield(ticker_symbol, float(current_price))

    hist = _bars_to_history(public_api.get_daily_bars(ticker_symbol))
    if hist.empty:
        raise ValueError(f"No history returned for {ticker_symbol}")
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
    return public_api.get_option_expirations(ticker_symbol)


@st.cache_data(ttl=30)
def get_option_chain(ticker_symbol: str, expiration: str) -> pd.DataFrame:
    rows = public_api.get_option_chain_rows(ticker_symbol, expiration)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["strike", "bid", "ask", "last", "volume", "open_interest",
                "mid_iv", "delta", "gamma", "theta", "vega"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.upper()
    else:
        df["option_type"] = ""

    df["delta_mkt"] = df["delta"]
    df["gamma_mkt"] = df["gamma"]
    df["theta_mkt"] = df["theta"]
    df["vega_mkt"] = df["vega"]

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
            logger.debug("yfinance earnings date fetch failed for %s", ticker_symbol, exc_info=True)

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
                logger.debug("yfinance calendar fallback failed for %s", ticker_symbol, exc_info=True)

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
            logger.debug("yfinance fast_info ex-div fetch failed for %s", ticker_symbol, exc_info=True)

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
                logger.debug("yfinance dividend actions fallback failed for %s", ticker_symbol, exc_info=True)

    except Exception:
        logger.warning("Yahoo event enrichment failed entirely for %s", ticker_symbol, exc_info=True)

    return result
