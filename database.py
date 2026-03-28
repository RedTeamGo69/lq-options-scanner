import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import DB_PATH, NY_TZ

logger = logging.getLogger(__name__)


# ============================================================
# SQLITE IV HISTORY
# ============================================================
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def db_connection():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


@st.cache_resource
def init_db() -> bool:
    with db_connection() as conn:
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
    with db_connection() as conn:
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


def get_local_iv_history(
    ticker: str,
    lookback_days: int = 252,
    expiration: Optional[str] = None,
) -> pd.DataFrame:
    cutoff = (datetime.now(NY_TZ).date() - timedelta(days=lookback_days + 30)).isoformat()
    with db_connection() as conn:
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
