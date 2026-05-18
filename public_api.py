"""
Public.com market-data client.

Pure `requests` (no Streamlit / pandas / numpy) so it can be shared by the
Streamlit app and the standalone GitHub Actions IV scanner.

Auth flow:
  1. A long-lived *secret* is generated once in Public's web UI
     (Settings -> Security -> API) and provided via PUBLIC_API_SECRET.
  2. The secret is exchanged for a short-lived bearer access token at
     POST /userapiauthservice/personal/access-tokens. The token is cached
     and transparently re-minted before it expires.
  3. Market-data endpoints live under /userapigateway/marketdata/{accountId}/...
     so the account id is discovered once via /userapigateway/trading/account
     (or taken from PUBLIC_ACCOUNT_ID).
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from config import (
    PUBLIC_API_BASE_URL,
    PUBLIC_API_SECRET,
    PUBLIC_ACCOUNT_ID,
    PUBLIC_TOKEN_VALIDITY_MINUTES,
)

logger = logging.getLogger(__name__)

# Cash-settled index underlyings that Public types as INDEX rather than EQUITY.
_INDEX_SYMBOLS = {"SPX", "XSP", "NDX", "RUT", "VIX", "DJX", "OEX"}

_TOKEN_PATH = "/userapiauthservice/personal/access-tokens"
_ACCOUNT_PATH = "/userapigateway/trading/account"
_MD_PREFIX = "/userapigateway/marketdata"
_HIST_PREFIX = "/userapigateway/historicdata"

# Refresh the token this many seconds before it actually expires.
_TOKEN_SLACK_SECONDS = 120

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=2)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# RLock (not Lock): _get_account_id may call _auth_headers -> _get_token,
# both of which take this lock.
_lock = threading.RLock()
_token_state: Dict[str, Any] = {"value": None, "expires_at": 0.0}
_account_state: Dict[str, Optional[str]] = {"value": PUBLIC_ACCOUNT_ID or None}


def _to_float(value: Any) -> Optional[float]:
    """Public returns most numeric fields as strings; coerce safely."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def instrument_type(symbol: str) -> str:
    return "INDEX" if symbol.upper() in _INDEX_SYMBOLS else "EQUITY"


# ============================================================
# AUTH
# ============================================================
def _mint_token() -> str:
    if not PUBLIC_API_SECRET:
        raise ValueError(
            "Missing Public API secret. Set PUBLIC_API_SECRET (generate it in "
            "Public: Settings -> Security -> API)."
        )
    resp = _session.post(
        f"{PUBLIC_API_BASE_URL}{_TOKEN_PATH}",
        json={
            "validityInMinutes": PUBLIC_TOKEN_VALIDITY_MINUTES,
            "secret": PUBLIC_API_SECRET,
        },
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    token = (resp.json() or {}).get("accessToken")
    if not token:
        raise ValueError("Public API did not return an access token.")
    _token_state["value"] = token
    _token_state["expires_at"] = time.time() + PUBLIC_TOKEN_VALIDITY_MINUTES * 60
    return token


def _get_token(force: bool = False) -> str:
    with _lock:
        if (
            not force
            and _token_state["value"]
            and time.time() < _token_state["expires_at"] - _TOKEN_SLACK_SECONDS
        ):
            return _token_state["value"]
        return _mint_token()


def _auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Accept": "application/json",
    }


def _get_account_id() -> str:
    with _lock:
        if _account_state["value"]:
            return _account_state["value"]
        resp = _session.get(
            f"{PUBLIC_API_BASE_URL}{_ACCOUNT_PATH}",
            headers=_auth_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        accounts = (resp.json() or {}).get("accounts") or []
        if not accounts:
            raise ValueError("No Public brokerage accounts found for this API secret.")
        account_id = accounts[0].get("accountId")
        if not account_id:
            raise ValueError("Public account response did not include an accountId.")
        _account_state["value"] = account_id
        return account_id


# ============================================================
# LOW-LEVEL REQUEST (with one 401 retry)
# ============================================================
def _request(
    method: str,
    path: str,
    *,
    json: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{PUBLIC_API_BASE_URL}{path}"
    for attempt in (1, 2):
        headers = _auth_headers()
        resp = _session.request(
            method, url, headers=headers, json=json, params=params, timeout=15
        )
        if resp.status_code == 401 and attempt == 1:
            # Token rejected — force a fresh mint and retry once.
            _get_token(force=True)
            continue
        resp.raise_for_status()
        return resp.json() or {}
    raise requests.HTTPError("Public API authentication failed after token refresh.")


# ============================================================
# MARKET DATA
# ============================================================
def get_equity_quote(symbol: str) -> Dict[str, Any]:
    account_id = _get_account_id()
    data = _request(
        "POST",
        f"{_MD_PREFIX}/{account_id}/quotes",
        json={"instruments": [{"symbol": symbol, "type": instrument_type(symbol)}]},
    )
    quotes = data.get("quotes") or []
    if not quotes:
        raise ValueError(f"No quote returned for {symbol}.")
    return quotes[0]


def get_spot(symbol: str) -> float:
    quote = get_equity_quote(symbol)
    price = _to_float(quote.get("last"))
    if price is None:
        price = _to_float(quote.get("previousClose"))
    if price is None or price <= 0:
        raise ValueError(f"Invalid price for {symbol}.")
    return float(price)


def get_daily_bars(symbol: str) -> List[Dict[str, Any]]:
    itype = instrument_type(symbol)
    data = _request(
        "GET",
        f"{_HIST_PREFIX}/{itype}/{symbol}/YEAR",
        params={"aggregation": "ONE_DAY"},
    )
    regular = data.get("regularMarket") or {}
    return regular.get("bars") or []


def get_option_expirations(symbol: str) -> List[str]:
    account_id = _get_account_id()
    data = _request(
        "POST",
        f"{_MD_PREFIX}/{account_id}/option-expirations",
        json={"instrument": {"symbol": symbol, "type": instrument_type(symbol)}},
    )
    expirations = data.get("expirations") or []
    return [str(e) for e in expirations if e]


def _normalize_contract(contract: Dict[str, Any], option_type: str) -> Dict[str, Any]:
    details = contract.get("optionDetails") or {}
    greeks = details.get("greeks") or {}
    return {
        "option_type": option_type,
        "strike": _to_float(details.get("strikePrice")),
        "bid": _to_float(contract.get("bid")),
        "ask": _to_float(contract.get("ask")),
        "last": _to_float(contract.get("last")),
        "volume": _to_float(contract.get("volume")),
        "open_interest": _to_float(contract.get("openInterest")),
        "mid_iv": _to_float(greeks.get("impliedVolatility")),
        "delta": _to_float(greeks.get("delta")),
        "gamma": _to_float(greeks.get("gamma")),
        "theta": _to_float(greeks.get("theta")),
        "vega": _to_float(greeks.get("vega")),
    }


def get_option_chain_rows(symbol: str, expiration: str) -> List[Dict[str, Any]]:
    account_id = _get_account_id()
    data = _request(
        "POST",
        f"{_MD_PREFIX}/{account_id}/option-chain",
        json={
            "instrument": {"symbol": symbol, "type": instrument_type(symbol)},
            "expirationDate": expiration,
        },
    )
    rows: List[Dict[str, Any]] = []
    for side, label in (("calls", "CALL"), ("puts", "PUT")):
        for contract in data.get(side) or []:
            rows.append(_normalize_contract(contract, label))
    return rows
