import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import DB_PATH, NY_TZ
from tickers_default import DEFAULT_TICKERS
from utils import dedupe_preserve_order

logger = logging.getLogger(__name__)

# ============================================================
# DATABASE BACKEND SELECTION
# ============================================================
# If DATABASE_URL is set, use PostgreSQL (Supabase, Neon, etc.)
# Otherwise, fall back to local SQLite.
DATABASE_URL = os.getenv("DATABASE_URL", "")


def _use_postgres() -> bool:
    return bool(DATABASE_URL)


# ============================================================
# CONNECTION HELPERS
# ============================================================
def _get_pg_connection():
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def _get_sqlite_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def db_connection():
    if _use_postgres():
        conn = _get_pg_connection()
    else:
        conn = _get_sqlite_connection()
    try:
        yield conn
    finally:
        conn.close()


# ============================================================
# INIT
# ============================================================
@st.cache_resource
def init_db() -> bool:
    with db_connection() as conn:
        cur = conn.cursor()

        if _use_postgres():
            cur.execute("""
                CREATE TABLE IF NOT EXISTS iv_snapshots (
                    id SERIAL PRIMARY KEY,
                    snapshot_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    expiration TEXT NOT NULL,
                    dte INTEGER NOT NULL,
                    atm_call_iv DOUBLE PRECISION,
                    atm_put_iv DOUBLE PRECISION,
                    atm_avg_iv DOUBLE PRECISION,
                    spot DOUBLE PRECISION,
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

            cur.execute("""
                CREATE TABLE IF NOT EXISTS tracked_tickers (
                    ticker TEXT PRIMARY KEY,
                    added_at TEXT NOT NULL
                )
            """)
        else:
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

            cur.execute("""
                CREATE TABLE IF NOT EXISTS tracked_tickers (
                    ticker TEXT PRIMARY KEY,
                    added_at TEXT NOT NULL
                )
            """)

        # Seed the tracked-ticker universe on first run so the daily scan and
        # the sidebar both start from the known default list.
        cur.execute("SELECT COUNT(*) FROM tracked_tickers")
        count = cur.fetchone()[0]
        if count == 0:
            now = datetime.now(NY_TZ).isoformat()
            ph = "%s" if _use_postgres() else "?"
            for tkr in DEFAULT_TICKERS:
                cur.execute(
                    f"INSERT INTO tracked_tickers (ticker, added_at) VALUES ({ph}, {ph}) "
                    f"ON CONFLICT (ticker) DO NOTHING",
                    (tkr.upper(), now),
                )

        conn.commit()
    return True


# ============================================================
# SAVE
# ============================================================
def save_iv_snapshot(
    ticker: str,
    expiration: str,
    dte: int,
    atm_call_iv: Optional[float],
    atm_put_iv: Optional[float],
    atm_avg_iv: Optional[float],
    spot: float,
) -> None:
    today = datetime.now(NY_TZ).date()
    # Skip weekends to keep IV history clean
    if today.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return
    snapshot_date = today.isoformat()
    with db_connection() as conn:
        cur = conn.cursor()

        if _use_postgres():
            cur.execute("""
                INSERT INTO iv_snapshots
                (snapshot_date, ticker, expiration, dte, atm_call_iv, atm_put_iv, atm_avg_iv, spot)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_date, ticker, expiration)
                DO UPDATE SET
                    dte = EXCLUDED.dte,
                    atm_call_iv = EXCLUDED.atm_call_iv,
                    atm_put_iv = EXCLUDED.atm_put_iv,
                    atm_avg_iv = EXCLUDED.atm_avg_iv,
                    spot = EXCLUDED.spot
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
        else:
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


# ============================================================
# QUERY
# ============================================================
def get_local_iv_history(
    ticker: str,
    lookback_days: int = 252,
    expiration: Optional[str] = None,
) -> pd.DataFrame:
    cutoff = (datetime.now(NY_TZ).date() - timedelta(days=lookback_days + 30)).isoformat()

    with db_connection() as conn:
        ph = "%s" if _use_postgres() else "?"

        if expiration:
            query = f"""
                SELECT snapshot_date, ticker, expiration, dte, atm_call_iv, atm_put_iv, atm_avg_iv, spot
                FROM iv_snapshots
                WHERE ticker = {ph} AND expiration = {ph} AND snapshot_date >= {ph}
                ORDER BY snapshot_date
            """
            df = pd.read_sql_query(query, conn, params=(ticker.upper(), expiration, cutoff))
        else:
            query = f"""
                SELECT snapshot_date, ticker, expiration, dte, atm_call_iv, atm_put_iv, atm_avg_iv, spot
                FROM iv_snapshots
                WHERE ticker = {ph} AND snapshot_date >= {ph}
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
    target_dte: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    empty = {
        "iv_rank": None,
        "iv_percentile": None,
        "hist_count": 0,
        "iv_min": None,
        "iv_max": None,
    }

    hist = get_local_iv_history(ticker, lookback_days=lookback_days, expiration=None)
    if hist.empty or "atm_avg_iv" not in hist.columns:
        return dict(empty)

    hist = hist.copy()
    hist["atm_avg_iv"] = pd.to_numeric(hist["atm_avg_iv"], errors="coerce")
    hist = hist.dropna(subset=["atm_avg_iv"])
    if hist.empty:
        return dict(empty)

    # Constant-maturity comparison: for each snapshot day, keep the expiration
    # whose DTE is closest to the one being scanned, so the rank compares
    # like-for-like tenors instead of pooling every expiration together.
    if target_dte is not None and "dte" in hist.columns and "snapshot_date" in hist.columns:
        h = hist.copy()
        h["dte"] = pd.to_numeric(h["dte"], errors="coerce")
        h = h.dropna(subset=["dte"])
        if not h.empty:
            h["_dte_dist"] = (h["dte"] - target_dte).abs()
            h = h.sort_values("_dte_dist").groupby("snapshot_date", as_index=False).first()
            hist = h

    series = hist["atm_avg_iv"].dropna()
    if series.empty:
        return dict(empty)

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
# TRACKED TICKERS (daily IV scan universe)
# ============================================================
def get_tracked_tickers() -> list:
    """Return the alphabetically-sorted list of tickers the daily scan tracks."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT ticker FROM tracked_tickers ORDER BY ticker")
        return [row[0] for row in cur.fetchall()]


def add_tracked_tickers(tickers) -> list:
    """Add one or more tickers. Returns the list actually inserted (new ones)."""
    cleaned = dedupe_preserve_order(
        [t.strip().upper() for t in tickers if t and t.strip()]
    )
    if not cleaned:
        return []

    existing = set(get_tracked_tickers())
    to_add = [t for t in cleaned if t not in existing]
    if not to_add:
        return []

    now = datetime.now(NY_TZ).isoformat()
    with db_connection() as conn:
        cur = conn.cursor()
        ph = "%s" if _use_postgres() else "?"
        for tkr in to_add:
            cur.execute(
                f"INSERT INTO tracked_tickers (ticker, added_at) VALUES ({ph}, {ph}) "
                f"ON CONFLICT (ticker) DO NOTHING",
                (tkr, now),
            )
        conn.commit()
    return to_add


def remove_tracked_ticker(ticker: str) -> None:
    """Remove a single ticker from the tracked universe."""
    with db_connection() as conn:
        cur = conn.cursor()
        ph = "%s" if _use_postgres() else "?"
        cur.execute(f"DELETE FROM tracked_tickers WHERE ticker = {ph}", (ticker.strip().upper(),))
        conn.commit()
