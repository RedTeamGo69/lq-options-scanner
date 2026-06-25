import streamlit as st

from config import ScannerConfig, TRADIER_API_KEY, TICKER_PATTERN
from utils import dedupe_preserve_order
from database import init_db
from ui import process_ticker

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LQ Quant Options Value Screener v3",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# INIT
# ============================================================
_ = init_db()

st.title("📈 LQ Quant Options Value Screener v3")
st.markdown(
    "Scan any ticker for the single contracts that screen cheapest to buy or richest to sell "
    "versus a realized-vol model. Tradier powers live chain pricing and Greeks; Yahoo adds earnings "
    "and ex-dividend dates; SQLite stores daily ATM IV snapshots to build its own IV history over time."
)

if not TRADIER_API_KEY:
    st.error("Missing Tradier API key. Set TRADIER_API_KEY before using the app.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Scanner Settings")

    action = st.radio("Action", ["SELL", "BUY"], horizontal=True)
    option_family = st.radio("Type", ["PUTS", "CALLS"], horizontal=True)

# All scan parameters (liquidity filters, vol model, vol adjustments, execution)
# run on the tuned ScannerConfig defaults. They are no longer exposed as sidebar
# knobs to keep the scanner focused on its one job: ranking single contracts by value edge.
cfg = ScannerConfig()

# ============================================================
# SEARCH & MAIN LOOP
# ============================================================
with st.form("search_form"):
    ticker_input = st.text_input(
        "Enter ticker(s), comma-separated",
        value="",
        placeholder="AAPL, TSLA, SPY",
    ).strip().upper()

    submit_search = st.form_submit_button("Fetch Options Data", type="primary", use_container_width=True)

if submit_search:
    if ticker_input:
        raw_tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
        invalid = [t for t in raw_tickers if not TICKER_PATTERN.match(t)]
        tickers = [t for t in raw_tickers if TICKER_PATTERN.match(t)]
        if invalid:
            st.warning(f"Skipped invalid ticker(s): {', '.join(invalid)}. Use 1-5 letter symbols (e.g. AAPL, SPY).")
        if tickers:
            tickers = dedupe_preserve_order(tickers)
            st.session_state["active_tickers"] = tickers
        else:
            st.warning("No valid ticker symbols entered.")
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
