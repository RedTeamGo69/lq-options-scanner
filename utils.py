from datetime import datetime, time
from typing import Any, List, Tuple

import numpy as np

from config import NY_TZ, CALENDAR_DAYS_PER_YEAR, T_FLOOR_YEARS


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
