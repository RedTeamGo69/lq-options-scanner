import math
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from pandas.io.formats.style import Styler

from config import ScannerConfig, NY_TZ
from utils import safe_float, format_date_dropdown, compute_time_to_expiry_years
from pricing import (
    build_forward_vol_forecast,
    adjust_forecast_vol_for_earnings,
    compute_payoff_curve,
    compute_scenario_table,
    compute_position_size,
)
from data import (
    get_quote_and_history,
    get_expiration_dates,
    get_yahoo_events,
    get_risk_free_rate,
    get_company_name,
    get_option_chain,
)
from database import compute_local_iv_rank_and_percentile, get_local_iv_history
from screening import (
    screen_chain,
    build_term_structure_snapshot,
    compute_term_structure_scaling_factor,
    build_skew_snapshot,
)
from strategies import build_vertical_spreads

logger = logging.getLogger(__name__)


# ============================================================
# DISPLAY HELPERS
# ============================================================
def style_results(df: pd.DataFrame) -> Styler:
    def color_edge(val):
        if pd.isna(val):
            return ""
        if val >= 5:
            return "color: #00FF88; font-weight: bold"
        if val <= -5:
            return "color: #FF5A5A; font-weight: bold"
        return ""

    def color_spread(val):
        if pd.isna(val):
            return ""
        if val > 15:
            return "color: #FF5A5A; font-weight: bold"
        if val > 8:
            return "color: #FFD166"
        return "color: #00FF88"

    def color_conf(val):
        if pd.isna(val):
            return ""
        if val >= 75:
            return "color: #00FF88; font-weight: bold"
        if val >= 50:
            return "color: #FFD166; font-weight: bold"
        return "color: #FF5A5A"

    def color_moneyness(val):
        if val == "ATM":
            return "color: #FFD166; font-weight: bold"
        if val == "OTM":
            return "color: #00FFFF; font-weight: bold"
        if val == "ITM":
            return "color: #FF8A8A; font-weight: bold"
        return ""

    styler = df.style
    for col, func in {
        "Value Edge (%)": color_edge,
        "Spread (%)": color_spread,
        "Confidence": color_conf,
        "Moneyness": color_moneyness,
    }.items():
        if col in df.columns:
            styler = styler.map(func, subset=[col])

    return styler.format(
        {
            "Strike": "{:,.2f}",
            "Bid": "{:,.2f}",
            "Ask": "{:,.2f}",
            "Mid": "{:,.2f}",
            "Exec Px": "{:,.2f}",
            "Market Theo": "{:,.2f}",
            "Forecast Theo": "{:,.2f}",
            "Abs Edge ($)": "{:,.2f}",
            "Value Edge (%)": "{:,.1f}",
            "Spread (%)": "{:,.1f}",
            "Mkt IV (%)": "{:,.1f}",
            "Forecast Vol (%)": "{:,.1f}",
            "RV20 (%)": "{:,.1f}",
            "RV60 (%)": "{:,.1f}",
            "RV120 (%)": "{:,.1f}",
            "IV - Forecast (pts)": "{:,.1f}",
            "IV - RV20 (pts)": "{:,.1f}",
            "IV - RV60 (pts)": "{:,.1f}",
            "IV - RV120 (pts)": "{:,.1f}",
            "Delta": "{:,.3f}",
            "Gamma": "{:,.4f}",
            "Theta": "{:,.4f}",
            "Vega": "{:,.4f}",
            "Model Delta": "{:,.3f}",
            "Model Gamma": "{:,.4f}",
            "Model Theta": "{:,.4f}",
            "Model Vega": "{:,.4f}",
            "Simple Yield (%)": "{:,.2f}",
            "Ann Yield (%)": "{:,.1f}",
            "Confidence": "{:,.0f}",
        }
    )


def display_summary(
    ticker: str,
    company_name: str,
    S: float,
    q: float,
    r: float,
    rv20: Optional[float],
    rv60: Optional[float],
    rv120: Optional[float],
    forecast_vol: Optional[float],
    yahoo_events: Dict[str, Optional[str]],
    iv_stats: Dict[str, Optional[float]],
    base_forecast_vol: Optional[float] = None,
    ts_factor: Optional[float] = None,
    earnings_adj_applied: bool = False,
) -> None:
    if company_name.upper() == ticker.upper():
        st.subheader(ticker)
    else:
        st.subheader(f"{company_name} ({ticker})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"${S:,.2f}")
    c2.metric("Dividend Yield", f"{q*100:.2f}%")
    c3.metric("Risk-Free", f"{r*100:.2f}%")

    vol_delta = None
    if forecast_vol is not None and base_forecast_vol is not None and forecast_vol != base_forecast_vol:
        vol_delta = f"+{(forecast_vol - base_forecast_vol)*100:.1f} pts"
    c4.metric("Forecast Vol", f"{forecast_vol*100:.1f}%" if forecast_vol is not None else "N/A", delta=vol_delta)

    adj_parts = []
    if ts_factor is not None and ts_factor != 1.0:
        adj_parts.append(f"Term structure: {ts_factor:.2f}x")
    if earnings_adj_applied:
        adj_parts.append("Earnings-adjusted")
    if adj_parts:
        c4.caption(" | ".join(adj_parts))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("RV20", f"{rv20*100:.1f}%" if rv20 is not None else "N/A")
    c6.metric("RV60", f"{rv60*100:.1f}%" if rv60 is not None else "N/A")
    c7.metric("RV120", f"{rv120*100:.1f}%" if rv120 is not None else "N/A")
    c8.metric("Local IV Obs", str(iv_stats["hist_count"]))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Local IV Rank", f"{iv_stats['iv_rank']:.0f}%" if iv_stats["iv_rank"] is not None else "N/A")
    c10.metric("Local IV Percentile", f"{iv_stats['iv_percentile']:.0f}%" if iv_stats["iv_percentile"] is not None else "N/A")
    c11.metric("Next Earnings", yahoo_events.get("next_earnings_date") or "N/A")
    c12.metric("Ex-Div Date", yahoo_events.get("ex_dividend_date") or "N/A")


def display_event_warnings(yahoo_events: Dict[str, Optional[str]], target_date: str) -> None:
    today_ny = datetime.now(NY_TZ).date()
    expiry_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

    earn_str = yahoo_events.get("next_earnings_date")
    if earn_str:
        try:
            earn_dt = datetime.strptime(earn_str, "%Y-%m-%d").date()
            if today_ny <= earn_dt <= expiry_dt:
                st.warning(f"Earnings risk: earnings date appears to fall before expiration ({earn_str}).")
        except Exception:
            logger.debug("Failed to parse earnings date: %s", earn_str, exc_info=True)

    ex_div_str = yahoo_events.get("ex_dividend_date")
    if ex_div_str:
        try:
            ex_dt = datetime.strptime(ex_div_str, "%Y-%m-%d").date()
            if today_ny <= ex_dt <= expiry_dt:
                st.warning(f"Dividend event risk: ex-dividend date appears before expiration ({ex_div_str}).")
        except Exception:
            logger.debug("Failed to parse ex-dividend date: %s", ex_div_str, exc_info=True)


def display_expected_moves(S: float, T: float, forecast_vol: float, best_df: pd.DataFrame) -> None:
    if T <= 0 or best_df.empty:
        return

    atm_rows = best_df[best_df["Moneyness"] == "ATM"].copy()
    if not atm_rows.empty:
        atm_iv = atm_rows.iloc[0]["Mkt IV (%)"] / 100.0
    else:
        temp = best_df.copy()
        temp["dist_to_50"] = (temp["Delta"].abs() - 0.50).abs()
        temp = temp.dropna(subset=["dist_to_50"])
        if temp.empty:
            return
        atm_iv = temp.sort_values("dist_to_50").iloc[0]["Mkt IV (%)"] / 100.0

    market_em = S * atm_iv * math.sqrt(T)
    model_em = S * forecast_vol * math.sqrt(T)

    c1, c2 = st.columns(2)
    c1.metric("Market-Implied Move (1σ)", f"± ${market_em:.2f}")
    c2.metric("Forecast-Vol Move (1σ)", f"± ${model_em:.2f}")

    st.caption(
        f"Market range: {S - market_em:.2f} to {S + market_em:.2f} | "
        f"Forecast range: {S - model_em:.2f} to {S + model_em:.2f}"
    )


def display_interpretation(best_df: pd.DataFrame, action: str) -> None:
    if best_df.empty:
        return

    top = best_df.iloc[0]
    msg = (
        f"Top contract confidence {top['Confidence']:.0f}/100. "
        f"Value edge {top['Value Edge (%)']:.1f}%. "
        f"Spread {top['Spread (%)']:.1f}%. "
        f"Market IV minus forecast vol {top['IV - Forecast (pts)']:.1f} vol points."
    )

    if action == "BUY":
        msg = "Buy-side interpretation: positive edge means the contract screens cheap versus the forecast-vol model. " + msg
    else:
        msg = "Sell-side interpretation: positive edge means the contract screens rich versus the forecast-vol model. " + msg

    st.info(msg)


# ============================================================
# SPREAD DISPLAY
# ============================================================
def style_spreads(df: pd.DataFrame) -> Styler:
    def color_pop(val):
        if pd.isna(val):
            return ""
        if val >= 70:
            return "color: #00FF88; font-weight: bold"
        if val >= 50:
            return "color: #FFD166; font-weight: bold"
        return "color: #FF5A5A"

    def color_score(val):
        if pd.isna(val):
            return ""
        if val >= 70:
            return "color: #00FF88; font-weight: bold"
        if val >= 50:
            return "color: #FFD166; font-weight: bold"
        return "color: #FF5A5A"

    def color_edge(val):
        if pd.isna(val):
            return ""
        if val >= 5:
            return "color: #00FF88; font-weight: bold"
        if val <= -5:
            return "color: #FF5A5A; font-weight: bold"
        return ""

    styler = df.style
    for col, func in {
        "PoP (%)": color_pop,
        "Score": color_score,
        "Model Edge (%)": color_edge,
    }.items():
        if col in df.columns:
            styler = styler.map(func, subset=[col])

    return styler.format(
        {
            "Short Strike": "{:,.2f}",
            "Long Strike": "{:,.2f}",
            "Width ($)": "{:,.2f}",
            "Net Credit ($)": "{:,.2f}",
            "Max Profit ($)": "{:,.2f}",
            "Max Loss ($)": "{:,.2f}",
            "Breakeven": "{:,.2f}",
            "Risk/Reward": "{:,.2f}",
            "PoP (%)": "{:,.1f}",
            "Model Edge (%)": "{:,.1f}",
            "Ann Return (%)": "{:,.1f}",
            "Net Delta": "{:,.3f}",
            "Net Theta": "{:,.4f}",
            "Net Gamma": "{:,.4f}",
            "Net Vega": "{:,.4f}",
            "Score": "{:,.0f}",
        }
    )


def display_spread_pnl(spread_row: pd.Series, S: float, T: float, r: float, q: float) -> None:
    """Show P&L diagram and scenario table for a selected spread."""
    strategy = spread_row["Strategy"]
    short_K = spread_row["Short Strike"]
    long_K = spread_row["Long Strike"]
    net_credit = spread_row["Net Credit ($)"]

    if strategy == "Bull Put":
        legs = [
            {"strike": short_K, "option_type": "PUT", "action": "SELL",
             "premium": net_credit + (short_K - long_K - net_credit) * 0.5,
             "iv": 0.30, "quantity": 1},
            {"strike": long_K, "option_type": "PUT", "action": "BUY",
             "premium": (short_K - long_K - net_credit) * 0.5,
             "iv": 0.30, "quantity": 1},
        ]
        # Simpler: use net credit directly for the spread payoff
        legs = [
            {"strike": short_K, "option_type": "PUT", "action": "SELL",
             "premium": 0.0, "iv": 0.30, "quantity": 1},
            {"strike": long_K, "option_type": "PUT", "action": "BUY",
             "premium": 0.0, "iv": 0.30, "quantity": 1},
        ]
    else:
        legs = [
            {"strike": short_K, "option_type": "CALL", "action": "SELL",
             "premium": 0.0, "iv": 0.30, "quantity": 1},
            {"strike": long_K, "option_type": "CALL", "action": "BUY",
             "premium": 0.0, "iv": 0.30, "quantity": 1},
        ]

    # For expiration payoff, we need to account for net credit received
    # Build proper legs with estimated premiums from BS
    opt_type = "PUT" if strategy == "Bull Put" else "CALL"
    from pricing import BlackScholesCalculator
    short_calc = BlackScholesCalculator(S=S, K=short_K, T=T, r=r, sigma=0.30, q=q)
    long_calc = BlackScholesCalculator(S=S, K=long_K, T=T, r=r, sigma=0.30, q=q)
    short_price = short_calc.price(opt_type)
    long_price = long_calc.price(opt_type)

    legs = [
        {"strike": short_K, "option_type": opt_type, "action": "SELL",
         "premium": short_price, "iv": 0.30, "quantity": 1},
        {"strike": long_K, "option_type": opt_type, "action": "BUY",
         "premium": long_price, "iv": 0.30, "quantity": 1},
    ]

    payoff_df = compute_payoff_curve(legs, S, T, r, q)
    scenario_df = compute_scenario_table(legs, S, T, r, q)

    if not payoff_df.empty:
        st.subheader(f"P&L Diagram: {strategy} {short_K}/{long_K}")

        chart_df = payoff_df.set_index("Spot")[["Expiration P&L", "Mid-Life P&L"]]
        st.line_chart(chart_df)

        # Zero line reference
        st.caption(
            f"Net credit: ${net_credit:.2f} | "
            f"Max profit: ${spread_row['Max Profit ($)']:.2f} | "
            f"Max loss: ${spread_row['Max Loss ($)']:.2f} | "
            f"Breakeven: ${spread_row['Breakeven']:.2f}"
        )

    if not scenario_df.empty:
        st.subheader("Scenario Analysis")
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)


def display_position_sizing(
    max_loss: float,
    cfg: ScannerConfig,
    edge_pct: float = 0.0,
    win_prob: float = 0.5,
) -> None:
    """Show position sizing recommendation."""
    sizing = compute_position_size(
        max_loss_per_contract=max_loss * 100.0,  # convert per-share to per-contract
        account_size=cfg.account_size,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        method=cfg.sizing_method,
        edge_pct=edge_pct,
        win_prob=win_prob,
    )

    contracts = sizing["contracts"]
    total_risk = sizing["total_risk"]
    pct = sizing["pct_of_account"]
    method_label = "Fixed Risk %" if cfg.sizing_method == "fixed_risk" else "Half-Kelly"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suggested Contracts", str(contracts))
    c2.metric("Total Risk", f"${total_risk:,.0f}")
    c3.metric("% of Account", f"{pct:.1f}%")
    c4.metric("Method", method_label)

    if contracts == 0 and max_loss > 0:
        st.caption(
            f"Max loss per contract (${max_loss * 100:.0f}) exceeds risk budget "
            f"(${cfg.account_size * cfg.risk_per_trade_pct / 100:.0f}). "
            f"Consider a narrower spread or larger account allocation."
        )


# ============================================================
# PROCESS TICKER
# ============================================================
def process_ticker(ticker: str, action: str, option_family: str, cfg: ScannerConfig, key_suffix: str = "") -> None:
    with st.spinner(f"Loading {ticker}..."):
        try:
            market_data = get_quote_and_history(ticker)
            expirations = get_expiration_dates(ticker)
            yahoo_events = get_yahoo_events(ticker)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                st.warning(f"{ticker}: Tradier rate limit hit.")
            else:
                st.error(f"{ticker}: HTTP error ({status}).")
            return
        except Exception as e:
            st.error(f"{ticker}: {e}")
            return

    if not expirations:
        st.warning(f"{ticker}: no option expirations found.")
        return

    S = market_data["price"]
    q = market_data["div_yield"]
    hist = market_data["history"]
    r = get_risk_free_rate()
    company_name = get_company_name(ticker)

    vol_pack = build_forward_vol_forecast(hist, cfg)
    rv20 = vol_pack["rv20"]
    rv60 = vol_pack["rv60"]
    rv120 = vol_pack["rv120"]
    forecast_vol = vol_pack["forecast_vol"]

    if forecast_vol is None or forecast_vol <= 0:
        st.error(f"{ticker}: could not build a valid forecast vol.")
        return

    exp_col, ctrl_col = st.columns([3, 1])

    with exp_col:
        target_date = st.selectbox(
            "Select Expiration",
            expirations,
            format_func=format_date_dropdown,
            key=f"exp_{ticker}_{key_suffix}",
        )

    with ctrl_col:
        run_scan = st.button(
            "Scan Chain",
            use_container_width=True,
            type="primary",
            key=f"scan_{ticker}_{key_suffix}",
        )

    dte, T = compute_time_to_expiry_years(target_date)
    if dte < 0:
        st.warning("Selected expiration is in the past.")
        return

    results_key = f"results_{ticker}_{action}_{option_family}_{target_date}"

    if run_scan:
        with st.spinner("Scanning chain, saving IV snapshots, and building analytics..."):
            try:
                chain_df = get_option_chain(ticker, target_date)
                term_df = build_term_structure_snapshot(ticker, expirations, S, save_to_db=True)

                current_atm_iv = None
                if not term_df.empty:
                    row = term_df[term_df["Expiration"] == target_date]
                    if not row.empty and pd.notna(row.iloc[0]["ATM Avg IV (%)"]):
                        current_atm_iv = float(row.iloc[0]["ATM Avg IV (%)"]) / 100.0

                iv_stats = {
                    "iv_rank": None,
                    "iv_percentile": None,
                    "hist_count": 0,
                    "iv_min": None,
                    "iv_max": None,
                }
                if current_atm_iv is not None:
                    iv_stats = compute_local_iv_rank_and_percentile(
                        ticker=ticker,
                        current_iv=current_atm_iv,
                        lookback_days=cfg.iv_history_lookback_days,
                    )

                effective_forecast_vol = forecast_vol
                ts_factor = None
                earnings_adj_applied = False

                if forecast_vol is not None:
                    if cfg.enable_term_structure_scaling and not term_df.empty:
                        ts_factor = compute_term_structure_scaling_factor(term_df, target_date)
                        if ts_factor is not None:
                            effective_forecast_vol = forecast_vol * ts_factor

                    if cfg.enable_earnings_vol_adj:
                        effective_forecast_vol, earnings_adj_applied = adjust_forecast_vol_for_earnings(
                            forecast_vol=effective_forecast_vol,
                            T=T,
                            earnings_date_str=yahoo_events.get("next_earnings_date"),
                            expiration_date_str=target_date,
                            expected_earnings_move=cfg.expected_earnings_move,
                        )

                best_df = screen_chain(
                    chain_df=chain_df,
                    S=S,
                    r=r,
                    q=q,
                    T=T,
                    dte=dte,
                    action=action,
                    option_family=option_family,
                    forecast_vol=effective_forecast_vol if effective_forecast_vol is not None else forecast_vol,
                    rv20=rv20,
                    rv60=rv60,
                    rv120=rv120,
                    cfg=cfg,
                )

                put_skew_df = build_skew_snapshot(chain_df, S, option_type="PUT")
                call_skew_df = build_skew_snapshot(chain_df, S, option_type="CALL")
                iv_hist_df = get_local_iv_history(ticker, lookback_days=cfg.iv_history_lookback_days)

                # Build vertical spreads only when user selects SELL SPREAD
                spreads_df = pd.DataFrame()
                if action == "SELL SPREAD":
                    final_fv = effective_forecast_vol if effective_forecast_vol is not None else forecast_vol
                    spreads_df = build_vertical_spreads(
                        chain_df=chain_df,
                        S=S, r=r, q=q, T=T, dte=dte,
                        forecast_vol=final_fv,
                        cfg=cfg,
                    )

                st.session_state[results_key] = {
                    "best_df": best_df,
                    "term_df": term_df,
                    "put_skew_df": put_skew_df,
                    "call_skew_df": call_skew_df,
                    "iv_hist_df": iv_hist_df,
                    "spreads_df": spreads_df,
                    "chain_df": chain_df,
                    "S": S,
                    "T": T,
                    "base_forecast_vol": forecast_vol,
                    "forecast_vol": effective_forecast_vol if effective_forecast_vol is not None else forecast_vol,
                    "ts_factor": ts_factor,
                    "earnings_adj_applied": earnings_adj_applied,
                    "expiration": target_date,
                    "yahoo_events": yahoo_events,
                    "iv_stats": iv_stats,
                    "rv20": rv20,
                    "rv60": rv60,
                    "rv120": rv120,
                    "r": r,
                    "q": q,
                    "company_name": company_name,
                }

            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    st.warning("Tradier rate limit hit while scanning.")
                else:
                    st.error(f"HTTP error while scanning ({status}).")
                return
            except Exception as e:
                st.error(f"Scan error: {e}")
                return

    if results_key not in st.session_state:
        display_summary(
            ticker=ticker,
            company_name=company_name,
            S=S,
            q=q,
            r=r,
            rv20=rv20,
            rv60=rv60,
            rv120=rv120,
            forecast_vol=forecast_vol,
            yahoo_events=yahoo_events,
            iv_stats={"iv_rank": None, "iv_percentile": None, "hist_count": 0, "iv_min": None, "iv_max": None},
        )
        st.caption("Pick an expiration and click Scan Chain.")
        return

    cached = st.session_state[results_key]

    display_summary(
        ticker=ticker,
        company_name=cached["company_name"],
        S=cached["S"],
        q=cached["q"],
        r=cached["r"],
        rv20=cached["rv20"],
        rv60=cached["rv60"],
        rv120=cached["rv120"],
        forecast_vol=cached["forecast_vol"],
        yahoo_events=cached["yahoo_events"],
        iv_stats=cached["iv_stats"],
        base_forecast_vol=cached.get("base_forecast_vol"),
        ts_factor=cached.get("ts_factor"),
        earnings_adj_applied=cached.get("earnings_adj_applied", False),
    )

    display_event_warnings(cached["yahoo_events"], cached["expiration"])

    best_df = cached["best_df"]
    term_df = cached["term_df"]
    put_skew_df = cached["put_skew_df"]
    call_skew_df = cached["call_skew_df"]
    iv_hist_df = cached["iv_hist_df"]
    spreads_df = cached.get("spreads_df", pd.DataFrame())

    if best_df.empty and spreads_df.empty:
        st.warning("No contracts passed the filters.")
        return

    is_spread_mode = (action == "SELL SPREAD")

    if not best_df.empty and not is_spread_mode:
        display_expected_moves(cached["S"], cached["T"], cached["forecast_vol"], best_df)
        display_interpretation(best_df, action)

    if is_spread_mode:
        tab_names = ["Vertical Spreads", "P&L Analysis", "Term Structure", "Put Skew", "Call Skew", "Local IV History"]
    else:
        tab_names = ["Top Contracts", "P&L Analysis", "Term Structure", "Put Skew", "Call Skew", "Local IV History"]
    tabs = st.tabs(tab_names)

    tab_map = {name: tab for name, tab in zip(tab_names, tabs)}

    # --- First tab: Top Contracts or Vertical Spreads ---
    first_tab = tab_map.get("Top Contracts") or tab_map.get("Vertical Spreads")
    with first_tab:
        if is_spread_mode:
            st.subheader(f"Vertical Spreads | {ticker} | {cached['expiration']}")
            if spreads_df.empty:
                st.warning("No vertical spreads found. Try adjusting spread filters (max width, min credit) in the sidebar.")
            else:
                styled_spreads = style_spreads(spreads_df)
                st.dataframe(
                    styled_spreads,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Short Strike": st.column_config.NumberColumn("Short Strike", format="$ %.2f"),
                        "Long Strike": st.column_config.NumberColumn("Long Strike", format="$ %.2f"),
                        "Net Credit ($)": st.column_config.NumberColumn("Net Credit ($)", format="$ %.2f"),
                        "Max Profit ($)": st.column_config.NumberColumn("Max Profit ($)", format="$ %.2f"),
                        "Max Loss ($)": st.column_config.NumberColumn("Max Loss ($)", format="$ %.2f"),
                        "Breakeven": st.column_config.NumberColumn("Breakeven", format="$ %.2f"),
                        "PoP (%)": st.column_config.NumberColumn("PoP (%)", format="%.1f%%"),
                        "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                    },
                )

                top_spread = spreads_df.iloc[0]
                st.markdown("---")
                st.subheader("Position Sizing (Top Spread)")
                display_position_sizing(
                    max_loss=top_spread["Max Loss ($)"],
                    cfg=cfg,
                    edge_pct=safe_float(top_spread.get("Model Edge (%)"), 0.0),
                    win_prob=safe_float(top_spread.get("PoP (%)"), 50.0) / 100.0,
                )

                spread_csv = spreads_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Spreads CSV",
                    data=spread_csv,
                    file_name=f"{ticker}_spreads_{cached['expiration']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_spreads_{ticker}_{cached['expiration']}",
                )
        else:
            if best_df.empty:
                st.warning("No single-leg contracts passed the filters.")
            else:
                st.subheader(f"Top Contracts | {ticker} | {action} {option_family} | {cached['expiration']}")
                styled = style_results(best_df)

                st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Strike": st.column_config.NumberColumn("Strike", format="$ %.2f"),
                        "Bid": st.column_config.NumberColumn("Bid", format="$ %.2f"),
                        "Ask": st.column_config.NumberColumn("Ask", format="$ %.2f"),
                        "Mid": st.column_config.NumberColumn("Mid", format="$ %.2f"),
                        "Exec Px": st.column_config.NumberColumn("Exec Px", format="$ %.2f"),
                        "Market Theo": st.column_config.NumberColumn("Market Theo", format="$ %.2f"),
                        "Forecast Theo": st.column_config.NumberColumn("Forecast Theo", format="$ %.2f"),
                        "Abs Edge ($)": st.column_config.NumberColumn("Abs Edge ($)", format="$ %.2f"),
                        "Value Edge (%)": st.column_config.NumberColumn("Value Edge (%)", format="%.1f%%"),
                        "Spread (%)": st.column_config.NumberColumn("Spread (%)", format="%.1f%%"),
                        "Mkt IV (%)": st.column_config.NumberColumn("Mkt IV (%)", format="%.1f%%"),
                        "Forecast Vol (%)": st.column_config.NumberColumn("Forecast Vol (%)", format="%.1f%%"),
                        "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f"),
                    },
                )

                csv = best_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{ticker}_{action}_{option_family}_{cached['expiration']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_{ticker}_{action}_{option_family}_{cached['expiration']}",
                )

    # --- P&L Analysis tab ---
    with tab_map["P&L Analysis"]:
        st.subheader(f"P&L Analysis | {ticker} | {cached['expiration']}")

        S_cached = cached["S"]
        T_cached = cached["T"]
        r_cached = cached["r"]
        q_cached = cached["q"]

        if is_spread_mode and not spreads_df.empty:
            display_spread_pnl(spreads_df.iloc[0], S_cached, T_cached, r_cached, q_cached)

        elif not best_df.empty:
            top = best_df.iloc[0]
            opt_type = "PUT" if option_family == "PUTS" else "CALL"
            strike = top["Strike"]
            exec_px = top["Exec Px"]
            iv = top["Mkt IV (%)"] / 100.0

            legs = [{"strike": strike, "option_type": opt_type, "action": action,
                     "premium": exec_px, "iv": iv, "quantity": 1}]

            payoff_df = compute_payoff_curve(legs, S_cached, T_cached, r_cached, q_cached)
            scenario_df = compute_scenario_table(legs, S_cached, T_cached, r_cached, q_cached)

            if not payoff_df.empty:
                st.subheader(f"P&L: {action} {strike} {opt_type}")
                chart_df = payoff_df.set_index("Spot")[["Expiration P&L", "Mid-Life P&L"]]
                st.line_chart(chart_df)
                st.caption(
                    f"Entry price: ${exec_px:.2f} | "
                    f"IV: {iv*100:.1f}% | "
                    f"Delta: {top['Delta']:.3f}"
                )

            if not scenario_df.empty:
                st.subheader("Scenario Analysis")
                st.dataframe(scenario_df, use_container_width=True, hide_index=True)

            if action == "SELL":
                if opt_type == "PUT":
                    max_loss_single = strike - exec_px
                else:
                    max_loss_single = 10.0 * exec_px
            else:
                max_loss_single = exec_px

            st.markdown("---")
            st.subheader("Position Sizing")
            display_position_sizing(
                max_loss=max_loss_single,
                cfg=cfg,
                edge_pct=safe_float(top.get("Value Edge (%)"), 0.0),
                win_prob=0.5,
            )
        else:
            st.warning("No contracts available for P&L analysis.")

    # --- Term Structure tab ---
    with tab_map["Term Structure"]:
        st.subheader("ATM IV Term Structure")
        if term_df.empty:
            st.warning("No term-structure data available.")
        else:
            st.dataframe(term_df, use_container_width=True, hide_index=True)
            chart_df = term_df.set_index("DTE")[["ATM Avg IV (%)"]]
            st.line_chart(chart_df)

    # --- Put Skew tab ---
    with tab_map["Put Skew"]:
        st.subheader("Put Skew Snapshot")
        if put_skew_df.empty:
            st.warning("No put skew data available.")
        else:
            st.dataframe(put_skew_df, use_container_width=True, hide_index=True)
            st.line_chart(put_skew_df.set_index("Pct From Spot")[["IV (%)"]])

    # --- Call Skew tab ---
    with tab_map["Call Skew"]:
        st.subheader("Call Skew Snapshot")
        if call_skew_df.empty:
            st.warning("No call skew data available.")
        else:
            st.dataframe(call_skew_df, use_container_width=True, hide_index=True)
            st.line_chart(call_skew_df.set_index("Pct From Spot")[["IV (%)"]])

    # --- Local IV History tab ---
    with tab_map["Local IV History"]:
        st.subheader("Local IV History")
        if iv_hist_df.empty:
            st.warning("No local IV history saved yet. Each scan saves ATM IV snapshots to SQLite.")
        else:
            iv_hist_df = iv_hist_df.copy()
            iv_hist_df["snapshot_date"] = pd.to_datetime(iv_hist_df["snapshot_date"], errors="coerce")
            iv_hist_df["atm_avg_iv_pct"] = pd.to_numeric(iv_hist_df["atm_avg_iv"], errors="coerce") * 100.0

            st.dataframe(iv_hist_df, use_container_width=True, hide_index=True)

            daily = (
                iv_hist_df.groupby("snapshot_date", as_index=False)["atm_avg_iv_pct"]
                .mean()
                .sort_values("snapshot_date")
            )
            if not daily.empty:
                st.line_chart(daily.set_index("snapshot_date")[["atm_avg_iv_pct"]])
