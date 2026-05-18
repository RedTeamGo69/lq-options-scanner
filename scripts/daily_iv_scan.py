"""
Daily IV Scanner — runs via GitHub Actions to collect ATM IV snapshots.

Fetches option chains from Public for a list of tickers, computes the
ATM implied vol for each expiration, and saves to PostgreSQL (Neon).

Required env vars:
    PUBLIC_API_SECRET  — Public API secret (Settings -> Security -> API)
    DATABASE_URL       — PostgreSQL connection string (Neon)
"""

import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import psycopg2
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import public_api  # noqa: E402  (repo root added to sys.path above)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
# PUBLIC_API_SECRET is read by public_api via config; surface a clear error
# here if it is missing rather than failing on the first request.
if not os.getenv("PUBLIC_API_SECRET"):
    raise SystemExit("PUBLIC_API_SECRET environment variable is required.")
DATABASE_URL = os.environ["DATABASE_URL"]

TICKERS = [
    # Mega-cap / tech
    "NVDA", "AAPL", "GOOG", "GOOGL", "MSFT", "AMZN", "TSM", "AVGO",
    "TSLA", "META", "NFLX", "ORCL", "PLTR", "ARM", "INTC", "MU",
    # Index / ETF
    "SPX", "XSP", "SPY", "QQQ", "IBIT", "SCHD",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "KEY", "COIN", "HOOD", "AFRM", "SOFI",
    # Healthcare / consumer / industrial
    "LLY", "NVO", "JNJ", "WMT", "COST", "EBAY", "HIMS", "UNH",
    # Blue chips
    "BRK.B", "V", "MA", "XOM", "CVX", "VZ", "F",
    # Other
    "NU", "CRWV", "SOLS",
]

# ── Public helpers ───────────────────────────────────────────
def get_spot(ticker: str) -> float:
    return public_api.get_spot(ticker)


def get_expirations(ticker: str) -> list[str]:
    return public_api.get_option_expirations(ticker)


def get_chain(ticker: str, expiration: str) -> list[dict]:
    return public_api.get_option_chain_rows(ticker, expiration)


# ── ATM IV extraction ────────────────────────────────────────
def compute_atm_iv(chain: list[dict], spot: float) -> dict:
    """Find ATM call/put IV from chain rows."""
    calls, puts = [], []
    for row in chain:
        strike = row.get("strike")
        mid_iv = row.get("mid_iv")
        if strike is None or mid_iv is None:
            continue
        strike = float(strike)
        iv = float(mid_iv)
        if iv <= 0:
            continue
        otype = (row.get("option_type") or "").upper()
        if otype == "CALL":
            calls.append((strike, iv))
        elif otype == "PUT":
            puts.append((strike, iv))

    def nearest_iv(pairs):
        if not pairs:
            return None
        best = min(pairs, key=lambda p: abs(p[0] - spot))
        return best[1]

    call_iv = nearest_iv(calls)
    put_iv = nearest_iv(puts)

    vals = [v for v in [call_iv, put_iv] if v is not None]
    avg_iv = float(np.mean(vals)) if vals else None

    return {"atm_call_iv": call_iv, "atm_put_iv": put_iv, "atm_avg_iv": avg_iv}


# ── Database ─────────────────────────────────────────────────
def ensure_table(conn):
    with conn.cursor() as cur:
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
    conn.commit()


def save_row(conn, snapshot_date: str, ticker: str, expiration: str,
             dte: int, spot: float, ivs: dict):
    with conn.cursor() as cur:
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
            snapshot_date, ticker.upper(), expiration, dte,
            ivs["atm_call_iv"], ivs["atm_put_iv"], ivs["atm_avg_iv"],
            spot,
        ))


# ── Main ─────────────────────────────────────────────────────
def main():
    today = datetime.now(pytz.timezone("America/New_York")).date()
    snapshot_date = today.isoformat()

    conn = psycopg2.connect(DATABASE_URL)
    try:
        ensure_table(conn)

        total_rows = 0
        errors = 0

        for ticker in TICKERS:
            try:
                spot = get_spot(ticker)
                expirations = get_expirations(ticker)
                if not expirations:
                    logger.warning(f"{ticker}: no expirations found, skipping")
                    continue

                ticker_rows = 0
                for exp in expirations:
                    try:
                        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                        dte = (exp_date - today).days
                        if dte < 0:
                            continue

                        chain = get_chain(ticker, exp)
                        if not chain:
                            continue

                        ivs = compute_atm_iv(chain, spot)
                        if ivs["atm_avg_iv"] is None:
                            continue

                        save_row(conn, snapshot_date, ticker, exp, dte, spot, ivs)
                        ticker_rows += 1

                    except Exception:
                        logger.debug(f"{ticker} {exp}: chain error", exc_info=True)
                        continue

                conn.commit()
                total_rows += ticker_rows
                logger.info(f"{ticker}: spot=${spot:.2f}, saved {ticker_rows} expirations")

                # Small delay to be kind to Public API rate limits
                time.sleep(0.3)

            except Exception as e:
                errors += 1
                logger.error(f"{ticker}: FAILED — {e}")
                continue
    finally:
        conn.close()

    logger.info(f"Done. {total_rows} rows saved, {errors} ticker errors.")

    if errors > len(TICKERS) // 2:
        logger.error("More than half of tickers failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
