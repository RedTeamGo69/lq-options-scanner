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
    "Tradier powers live chain pricing and Greeks. Yahoo enriches earnings and ex-dividend dates. "
    "SQLite stores daily ATM IV snapshots so the app can build its own real IV history over time."
)

if not TRADIER_API_KEY:
    st.error("Missing Tradier API key. Set TRADIER_API_KEY before using the app.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Scanner Settings")

    action = st.radio("Action", ["SELL", "BUY", "SELL SPREAD"], horizontal=True)
    option_family = st.radio("Type", ["PUTS", "CALLS"], horizontal=True, disabled=(action == "SELL SPREAD"))

    st.subheader("Liquidity Filters")
    min_open_interest = st.number_input("Min OI", min_value=0, value=100, step=10)
    min_volume = st.number_input("Min Volume", min_value=0, value=10, step=1)
    min_bid = st.number_input("Min Bid", min_value=0.0, value=0.10, step=0.05, format="%.2f")
    max_spread_pct = st.number_input("Max Spread %", min_value=1.0, value=15.0, step=1.0)
    min_abs_delta = st.number_input("Min |Delta|", min_value=0.01, max_value=0.99, value=0.10, step=0.01, format="%.2f")
    max_abs_delta = st.number_input("Max |Delta|", min_value=0.01, max_value=0.99, value=0.85, step=0.01, format="%.2f")
    top_n = st.number_input("Top N", min_value=5, max_value=100, value=25, step=5)

    st.subheader("Forecast Vol Model")
    rv20_weight = st.slider("RV20 Weight", 0.0, 1.0, 0.50, 0.05)
    rv60_weight = st.slider("RV60 Weight", 0.0, 1.0, 0.30, 0.05)
    rv120_weight = st.slider("RV120 Weight", 0.0, 1.0, 0.20, 0.05)
    vol_forecast_multiplier = st.slider("Forecast Vol Multiplier", 0.50, 1.50, 1.00, 0.01)

    st.subheader("Vol Adjustments")
    enable_earnings_vol_adj = st.toggle("Adjust forecast for earnings", value=True)
    expected_earnings_move = st.slider("Expected Earnings Move %", 1.0, 20.0, 5.0, 0.5) / 100.0
    enable_term_structure_scaling = st.toggle("Scale forecast by term structure", value=True)

    st.subheader("Execution Model")
    use_executable_pricing = st.toggle("Use bid/ask execution pricing", value=True)
    execution_slippage_pct = st.slider("Slippage %", 0.0, 5.0, 0.0, 0.1)

    st.subheader("Spread Filters")
    spread_max_width = st.number_input("Max Spread Width ($)", min_value=1, max_value=50, value=10, step=1)
    spread_min_credit = st.number_input("Min Net Credit ($)", min_value=0.01, value=0.10, step=0.05, format="%.2f")
    spread_top_n = st.number_input("Top N Spreads", min_value=5, max_value=50, value=15, step=5)

    st.subheader("Position Sizing")
    account_size = st.number_input("Account Size ($)", min_value=1000.0, value=10000.0, step=1000.0, format="%.0f")
    risk_per_trade_pct = st.slider("Max Risk Per Trade %", 0.5, 10.0, 2.0, 0.5)
    sizing_method = st.radio("Sizing Method", ["fixed_risk", "half_kelly"], format_func=lambda x: "Fixed Risk %" if x == "fixed_risk" else "Half-Kelly", horizontal=True)

    st.subheader("Local IV History")
    iv_history_lookback_days = st.number_input("IV History Lookback Days", min_value=30, max_value=1000, value=252, step=10)

cfg = ScannerConfig(
    min_open_interest=int(min_open_interest),
    min_volume=int(min_volume),
    min_bid=float(min_bid),
    max_spread_pct=float(max_spread_pct),
    min_abs_delta=float(min_abs_delta),
    max_abs_delta=float(max_abs_delta),
    top_n=int(top_n),
    rv20_weight=float(rv20_weight),
    rv60_weight=float(rv60_weight),
    rv120_weight=float(rv120_weight),
    vol_forecast_multiplier=float(vol_forecast_multiplier),
    enable_earnings_vol_adj=bool(enable_earnings_vol_adj),
    expected_earnings_move=float(expected_earnings_move),
    enable_term_structure_scaling=bool(enable_term_structure_scaling),
    use_executable_pricing=bool(use_executable_pricing),
    execution_slippage_pct=float(execution_slippage_pct),
    iv_history_lookback_days=int(iv_history_lookback_days),
    spread_max_width=int(spread_max_width),
    spread_min_credit=float(spread_min_credit),
    spread_top_n=int(spread_top_n),
    account_size=float(account_size),
    risk_per_trade_pct=float(risk_per_trade_pct),
    sizing_method=str(sizing_method),
)

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
