import streamlit as st

from config import ScannerConfig, TRADIER_API_KEY, TICKER_PATTERN
from utils import dedupe_preserve_order
from database import (
    init_db,
    get_tracked_tickers,
    add_tracked_tickers,
    remove_tracked_ticker,
)
from ui import process_ticker

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LQ Quant Options Value Screener v3",
    page_icon="📈",
    layout="wide",
)

# Custom styling for less prominent scrollbars
st.markdown(
    """
    <style>
    /* Firefox scrollbar */
    * {
        scrollbar-width: thin;
        scrollbar-color: #444 #1a1a1a;
    }

    /* Chrome/Safari/Edge scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }

    ::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

    # --- Daily IV tracking universe -------------------------------------
    # These tickers are what the nightly GitHub Actions scan logs ATM IV for.
    # The list lives in the database, so edits here change what the scheduled
    # job collects — no code changes needed.
    st.divider()
    tracked = get_tracked_tickers()
    with st.expander(f"🗓️ Daily IV Tracking ({len(tracked)})", expanded=False):
        st.caption(
            "Tickers the nightly scan logs ATM IV for. Stored in the database, "
            "so changes here take effect on the next scheduled run."
        )

        with st.form("add_ticker_form", clear_on_submit=True):
            new_ticker_input = st.text_input(
                "Add ticker(s)",
                placeholder="e.g. AAPL or AAPL, TSLA",
            ).strip().upper()
            add_clicked = st.form_submit_button("➕ Add", use_container_width=True)

        if add_clicked:
            raw = [t.strip() for t in new_ticker_input.split(",") if t.strip()]
            invalid = [t for t in raw if not TICKER_PATTERN.match(t)]
            valid = [t for t in raw if TICKER_PATTERN.match(t)]
            if invalid:
                st.warning(f"Skipped invalid: {', '.join(invalid)}")
            if valid:
                added = add_tracked_tickers(valid)
                if added:
                    st.success(f"Added: {', '.join(added)}")
                    st.rerun()
                else:
                    st.info("Already tracked — nothing to add.")
            elif not invalid:
                st.warning("Enter a ticker symbol to add.")

        if tracked:
            to_remove = st.selectbox(
                "Remove a ticker", ["—"] + tracked, index=0
            )
            if st.button("🗑️ Remove", use_container_width=True, disabled=(to_remove == "—")):
                remove_tracked_ticker(to_remove)
                st.success(f"Removed: {to_remove}")
                st.rerun()

            st.caption("Currently tracked:")
            st.write(", ".join(tracked))
        else:
            st.info("No tickers tracked yet.")

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
