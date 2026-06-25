"""
Default ticker universe for the daily IV scan.

This is the *seed* list: it is written into the ``tracked_tickers`` table the
first time the table is empty. After that, the database is the source of truth
and you add/remove tickers from the app's sidebar — editing this file is no
longer required.

Kept dependency-free on purpose so both the Streamlit app (``database.py``) and
the standalone GitHub Actions scan script (``scripts/daily_iv_scan.py``) can
import it without pulling in heavy packages.
"""

DEFAULT_TICKERS = [
    # Mega-cap / tech
    "NVDA", "AAPL", "GOOG", "GOOGL", "MSFT", "AMZN", "TSM", "AVGO",
    "TSLA", "META", "NFLX", "ORCL", "PLTR", "ARM", "INTC", "MU", "AMD",
    # Index / ETF
    "SPX", "XSP", "NDX", "SPY", "QQQ", "IBIT", "SCHD",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "KEY", "COIN", "HOOD", "AFRM", "SOFI", "OWL",
    # Healthcare / consumer / industrial
    "LLY", "NVO", "JNJ", "WMT", "COST", "EBAY", "HIMS", "UNH",
    # Blue chips
    "BRK.B", "V", "MA", "XOM", "CVX", "VZ", "F",
    # Other
    "NU", "CRWV", "SOLS",
]
