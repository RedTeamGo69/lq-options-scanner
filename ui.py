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
from utils import format_date_dropdown, compute_time_to_expiry_years
from pricing import (
    build_forward_vol_forecast,
    adjust_forecast_vol_for_earnings,
    compute_payoff_curve,
    compute_scenario_table,
)
from data import (
    get_quote_and_history,
    get_expiration_dates,
    get_tradier_fundamentals,
    get_risk_free_rate,
    get_company_name,
    get_option_chain,
)
from database import compute_local_iv_rank_and_percentile, get_local_iv_history
from screening import (
    screen_chain,
    build_term_structure_snapshot,
    compute_term_structure_scaling_factor,
    implied_earnings_move_from_term_structure,
    build_skew_snapshot,
)

logger = logging.getLogger(__name__)


# ============================================================
# DISPLAY HELPERS
# ============================================================
# Columns shown in the simplified table view (full data is in CSV download)
_DISPLAY_COLS_SELL = [
    "Moneyness", "Strike", "Bid", "Ask", "IV Edge (vol pts)", "$ Edge",
    "Spread (%)", "Delta", "Ann Yield (%)", "Confidence",
]
_DISPLAY_COLS_BUY = [
    "Moneyness", "Strike", "Bid", "Ask", "IV Edge (vol pts)", "$ Edge",
    "Spread (%)", "Delta", "Confidence",
]


def _get_display_columns(df: pd.DataFrame, action: str) -> list:
    """Return the subset of columns to display based on action."""
    template = _DISPLAY_COLS_SELL if action == "SELL" else _DISPLAY_COLS_BUY
    return [c for c in template if c in df.columns]


def style_results(
    df: pd.DataFrame,
    action: str = "SELL",
    edge_green: float = 2.0,
    edge_red: float = -2.0,
) -> Styler:
    display_cols = _get_display_columns(df, action)
    view = df[display_cols]

    def color_edge(val):
        if pd.isna(val):
            return ""
        if val >= edge_green:
            return "color: #00FF88; font-weight: bold"
        if val <= edge_red:
            return "color: #FF5A5A; font-weight: bold"
        return ""

    def color_dollar(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #00FF88"
        if val < 0:
            return "color: #FF5A5A"
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

    styler = view.style
    for col, func in {
        "IV Edge (vol pts)": color_edge,
        "RV Edge (vol pts)": color_edge,
        "$ Edge": color_dollar,
        "Spread (%)": color_spread,
        "Confidence": color_conf,
        "Moneyness": color_moneyness,
    }.items():
        if col in view.columns:
            styler = styler.map(func, subset=[col])

    fmt = {}
    fmt_map = {
        "Strike": "{:,.2f}", "Bid": "{:,.2f}", "Ask": "{:,.2f}",
        "IV Edge (vol pts)": "{:,.1f}", "RV Edge (vol pts)": "{:,.1f}", "$ Edge": "{:,.0f}",
        "Spread (%)": "{:,.1f}",
        "Mkt IV (%)": "{:,.1f}", "Fair IV (%)": "{:,.1f}", "Delta": "{:,.3f}", "Theta": "{:,.4f}",
        "Ann Yield (%)": "{:,.1f}", "Confidence": "{:,.0f}",
    }
    for col, f in fmt_map.items():
        if col in view.columns:
            fmt[col] = f

    return styler.format(fmt)


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
    fundamentals: Dict[str, Optional[str]],
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
    c11.metric("Next Earnings", fundamentals.get("next_earnings_date") or "N/A")
    c12.metric("Ex-Div Date", fundamentals.get("ex_dividend_date") or "N/A")


def display_event_warnings(fundamentals: Dict[str, Optional[str]], target_date: str) -> None:
    today_ny = datetime.now(NY_TZ).date()
    expiry_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

    earn_str = fundamentals.get("next_earnings_date")
    if earn_str:
        try:
            earn_dt = datetime.strptime(earn_str, "%Y-%m-%d").date()
            if today_ny <= earn_dt <= expiry_dt:
                st.warning(f"Earnings risk: earnings date appears to fall before expiration ({earn_str}).")
        except Exception:
            logger.debug("Failed to parse earnings date: %s", earn_str, exc_info=True)

    ex_div_str = fundamentals.get("ex_dividend_date")
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


def display_level_signal(scan_meta: Dict, action: str) -> None:
    """Surface-level read: is the whole vol surface rich/cheap vs realized, and
    is that gap statistically meaningful (noise-gated z-score)?"""
    if not scan_meta:
        return
    level = scan_meta.get("level_edge_volpts")
    z = scan_meta.get("level_z")
    gate = scan_meta.get("noise_gate_z", 1.0)
    if level is None or z is None:
        return

    # level is favorable-signed for the chosen action; translate to an absolute
    # rich/cheap read of the surface.
    surface = "rich" if ((level > 0) == (action == "SELL")) else "cheap"
    mkt_iv = scan_meta.get("market_atm_iv")
    fv = scan_meta.get("fv_atm")
    detail = ""
    if mkt_iv is not None and fv is not None:
        detail = f" Market ATM IV {mkt_iv*100:.1f}% vs forecast {fv*100:.1f}%."

    if z >= gate:
        st.caption(
            f"Surface level: ATM IV screens **{surface}** vs the realized-vol forecast by "
            f"{abs(level):.1f} vol pts (z={z:.1f} ≥ {gate:.1f}, significant).{detail}"
        )
    else:
        st.caption(
            f"Surface level: ATM IV is within noise of the realized-vol forecast "
            f"({abs(level):.1f} vol pts, z={z:.1f} < {gate:.1f}) — treat the level signal as weak.{detail}"
        )


def display_interpretation(best_df: pd.DataFrame, action: str, forecast_vol: float) -> None:
    if best_df.empty:
        return

    top = best_df.iloc[0]
    msg = (
        f"Top contract confidence {top['Confidence']:.0f}/100. "
        f"IV edge {top['IV Edge (vol pts)']:+.1f} vol pts (${top['$ Edge']:+,.0f}/contract). "
        f"Relative-value edge {top['RV Edge (vol pts)']:+.1f} vol pts vs the fitted smile. "
        f"Spread {top['Spread (%)']:.1f}%."
    )

    if action == "BUY":
        msg = "Buy-side: a positive IV edge means you pay less vol than fair (screens cheap). " + msg
    else:
        msg = "Sell-side: a positive IV edge means you receive more vol than fair (screens rich). " + msg

    st.info(msg)


def display_headline_pick(ticker: str, action: str, option_family: str, best_df: pd.DataFrame, expiration: str) -> None:
    """Surface the single top-ranked contract as a prominent one-line recommendation."""
    if best_df.empty:
        return

    top = best_df.iloc[0]
    opt_label = "CALL" if option_family == "CALLS" else "PUT"
    verb = "Best buy" if action == "BUY" else "Best sell"

    strike = float(top["Strike"])
    strike_label = f"{strike:,.0f}" if strike.is_integer() else f"{strike:,.2f}"
    exec_px = float(top["Exec Px"])

    metrics = []
    edge = top["IV Edge (vol pts)"]
    if pd.notna(edge):
        metrics.append(f"IV edge {edge:+.1f} vol pts")
    dollar_edge = top.get("$ Edge", np.nan)
    if pd.notna(dollar_edge):
        metrics.append(f"${dollar_edge:+,.0f}/contract")
    conf = top["Confidence"]
    if pd.notna(conf):
        metrics.append(f"confidence {conf:.0f}/100")
    delta = top["Delta"]
    if pd.notna(delta):
        metrics.append(f"Δ {delta:+.2f}")
    # Annualized yield is only meaningful on the sell side (premium collection)
    ann_yld = top.get("Ann Yield (%)", np.nan)
    if action == "SELL" and pd.notna(ann_yld):
        metrics.append(f"ann. yield {ann_yld:.1f}%")

    detail = " — " + ", ".join(metrics) if metrics else ""
    st.success(
        f"\U0001F3AF **{verb}: {ticker} ${strike_label} {opt_label} @ ${exec_px:.2f}** "
        f"(exp {expiration}){detail}"
    )


# ============================================================
# PROCESS TICKER
# ============================================================
def process_ticker(ticker: str, action: str, option_family: str, cfg: ScannerConfig, key_suffix: str = "") -> None:
    with st.spinner(f"Loading {ticker}..."):
        try:
            market_data = get_quote_and_history(ticker)
            expirations = get_expiration_dates(ticker)
            fundamentals = get_tradier_fundamentals(ticker)
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
    # Tradier quotes don't carry a dividend yield, so derive one from the
    # fundamentals trailing-twelve-month dividend total when the quote reports none.
    if (q is None or q <= 0) and S > 0:
        ttm_div = fundamentals.get("trailing_annual_dividend")
        if ttm_div:
            q = min(float(ttm_div) / S, 0.25)
    hist = market_data["history"]
    r = get_risk_free_rate()
    company_name = get_company_name(ticker)

    vol_pack = build_forward_vol_forecast(hist, cfg)
    rv20 = vol_pack["rv20"]
    rv60 = vol_pack["rv60"]
    rv120 = vol_pack["rv120"]
    forecast_vol = vol_pack["forecast_vol"]
    forecast_vol_uncertainty = vol_pack.get("forecast_vol_uncertainty")

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
                        target_dte=dte,
                    )

                effective_forecast_vol = forecast_vol
                ts_factor = None
                earnings_adj_applied = False
                earnings_move = None

                if forecast_vol is not None:
                    today_ny = datetime.now(NY_TZ).date()
                    earnings_date_str = fundamentals.get("next_earnings_date")

                    # Earnings jump: prefer the move implied by the IV term
                    # structure; fall back to the configured default.
                    if cfg.enable_earnings_vol_adj:
                        earnings_move = implied_earnings_move_from_term_structure(
                            term_df, earnings_date_str, today=today_ny
                        )
                        if earnings_move is None:
                            earnings_move = cfg.expected_earnings_move

                    # Diffusive term-structure scaling, with the earnings jump
                    # stripped from the curve so it isn't double-counted when the
                    # jump is added back per-expiry below.
                    if cfg.enable_term_structure_scaling and not term_df.empty:
                        earnings_dte = None
                        if earnings_date_str:
                            try:
                                e_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
                                earnings_dte = (e_dt - today_ny).days
                            except (ValueError, TypeError):
                                earnings_dte = None
                        jump_var = (earnings_move ** 2) if earnings_move else 0.0
                        ts_factor = compute_term_structure_scaling_factor(
                            term_df,
                            target_dte=dte,
                            earnings_dte=earnings_dte,
                            jump_var=jump_var,
                        )
                        if ts_factor is not None:
                            effective_forecast_vol = forecast_vol * ts_factor

                    # Add the earnings jump back, but only for an expiry that
                    # actually spans the earnings date.
                    if cfg.enable_earnings_vol_adj and earnings_move:
                        effective_forecast_vol, earnings_adj_applied = adjust_forecast_vol_for_earnings(
                            forecast_vol=effective_forecast_vol,
                            T=T,
                            earnings_date_str=earnings_date_str,
                            expiration_date_str=target_date,
                            expected_earnings_move=earnings_move,
                        )

                # Scale the forecast uncertainty by the same level adjustment so
                # the noise-gate z-score stays consistent with the adjusted vol.
                eff_vol = effective_forecast_vol if effective_forecast_vol is not None else forecast_vol
                eff_unc = forecast_vol_uncertainty
                if forecast_vol_uncertainty is not None and forecast_vol not in (None, 0) and eff_vol is not None:
                    eff_unc = forecast_vol_uncertainty * (eff_vol / forecast_vol)

                best_df = screen_chain(
                    chain_df=chain_df,
                    S=S,
                    r=r,
                    q=q,
                    T=T,
                    dte=dte,
                    action=action,
                    option_family=option_family,
                    forecast_vol=eff_vol,
                    cfg=cfg,
                    forecast_vol_uncertainty=eff_unc,
                )

                put_skew_df = build_skew_snapshot(chain_df, S, option_type="PUT")
                call_skew_df = build_skew_snapshot(chain_df, S, option_type="CALL")
                iv_hist_df = get_local_iv_history(ticker, lookback_days=cfg.iv_history_lookback_days)

                st.session_state[results_key] = {
                    "best_df": best_df,
                    "scan_meta": best_df.attrs.get("scan_meta", {}),
                    "term_df": term_df,
                    "put_skew_df": put_skew_df,
                    "call_skew_df": call_skew_df,
                    "iv_hist_df": iv_hist_df,
                    "chain_df": chain_df,
                    "S": S,
                    "T": T,
                    "base_forecast_vol": forecast_vol,
                    "forecast_vol": effective_forecast_vol if effective_forecast_vol is not None else forecast_vol,
                    "ts_factor": ts_factor,
                    "earnings_adj_applied": earnings_adj_applied,
                    "expiration": target_date,
                    "fundamentals": fundamentals,
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
            fundamentals=fundamentals,
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
        fundamentals=cached["fundamentals"],
        iv_stats=cached["iv_stats"],
        base_forecast_vol=cached.get("base_forecast_vol"),
        ts_factor=cached.get("ts_factor"),
        earnings_adj_applied=cached.get("earnings_adj_applied", False),
    )

    display_event_warnings(cached["fundamentals"], cached["expiration"])

    best_df = cached["best_df"]
    put_skew_df = cached["put_skew_df"]
    call_skew_df = cached["call_skew_df"]
    iv_hist_df = cached["iv_hist_df"]

    if best_df.empty:
        st.warning("No contracts passed the filters.")
        return

    display_headline_pick(ticker, action, option_family, best_df, cached["expiration"])
    display_level_signal(cached.get("scan_meta", {}), action)
    display_expected_moves(cached["S"], cached["T"], cached["forecast_vol"], best_df)
    display_interpretation(best_df, action, forecast_vol=cached["forecast_vol"])

    tab_names = ["Top Contracts", "P&L Analysis", "Put Skew", "Call Skew", "Local IV History"]
    tabs = st.tabs(tab_names)

    tab_map = {name: tab for name, tab in zip(tab_names, tabs)}

    # --- First tab: Top Contracts ---
    with tab_map["Top Contracts"]:
        st.subheader(f"Top Contracts | {ticker} | {action} {option_family} | {cached['expiration']}")
        styled = style_results(
            best_df,
            action=action,
            edge_green=cfg.iv_edge_green_volpts,
            edge_red=cfg.iv_edge_red_volpts,
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Strike": st.column_config.NumberColumn("Strike", format="$ %.2f"),
                "Bid": st.column_config.NumberColumn("Bid", format="$ %.2f"),
                "Ask": st.column_config.NumberColumn("Ask", format="$ %.2f"),
                "IV Edge (vol pts)": st.column_config.NumberColumn("IV Edge (vol pts)", format="%.1f"),
                "$ Edge": st.column_config.NumberColumn("$ Edge", format="$ %.0f"),
                "Spread (%)": st.column_config.NumberColumn("Spread (%)", format="%.1f%%"),
                "Ann Yield (%)": st.column_config.NumberColumn("Ann Yield (%)", format="%.1f%%"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f"),
            },
        )
        with st.expander("ℹ️ What do these columns mean?"):
            st.markdown(
                "- **Moneyness** — where the strike sits vs. the current price: "
                "**ITM** (in the money), **ATM** (at the money), **OTM** (out of the money).\n"
                "- **Strike** — the contract's strike price.\n"
                "- **Bid / Ask** — the price you can sell at (bid) and buy at (ask). "
                "The gap between them is the bid–ask spread.\n"
                "- **IV Edge (vol pts)** — how mispriced the option is, in volatility points, "
                "**in your favor**. Higher / green is better; it's the core signal this scanner ranks on.\n"
                "- **$ Edge** — that same edge translated into dollars per contract (1 contract = 100 shares).\n"
                "- **Spread (%)** — the bid–ask spread as a % of price, i.e. how cheap and easy the "
                "contract is to trade. Green = tight (good), red = wide (costly).\n"
                "- **Delta** — how much the option's value moves for a $1 move in the stock "
                "(roughly its odds of finishing in the money). Negative for puts.\n"
                "- **Confidence** — a 0–100 composite rank combining edge, liquidity and spread. "
                "Green ≥ 75 (strong), yellow 50–74 (moderate), red < 50 (weak)."
            )
        st.caption("Full data with all Greeks and vol metrics available in CSV download below.")

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

        with st.expander("ℹ️ How to read the P&L chart"):
            st.markdown(
                "This is the profit/loss of the **top-ranked contract** (1 contract = 100 shares), "
                "in dollars, across a range of stock prices.\n\n"
                "- **Expiration P&L** — what the trade is worth if you hold it all the way to expiration.\n"
                "- **Mid-Life P&L** — what it's worth roughly halfway to expiration; the gap between "
                "the two lines shows how time decay erodes value.\n"
                "- Where a line crosses **$0** is your **break-even** price.\n\n"
                "The **Scenario Analysis** table below is the same thing as a quick grid: P&L at fixed "
                "stock moves (−10% … +10%), plus **Net Delta** (how much you make/lose per $1 move) and "
                "**Net Theta** (how much value you lose to time decay each day)."
            )

    # --- Put Skew tab ---
    with tab_map["Put Skew"]:
        st.subheader("Put Skew Snapshot")
        if put_skew_df.empty:
            st.warning("No put skew data available.")
        else:
            st.dataframe(put_skew_df, use_container_width=True, hide_index=True)
            st.line_chart(put_skew_df.set_index("Pct From Spot")[["IV (%)"]])
            with st.expander("ℹ️ How to read the skew"):
                st.markdown(
                    "This is the **volatility smile/skew**: implied volatility (IV, the y-axis) for "
                    "each put strike, plotted by its distance from the current price (% from spot, x-axis).\n\n"
                    "- A **higher** line means the market is charging **more** volatility (a richer price) "
                    "for options at that strike.\n"
                    "- For stocks the curve usually **rises toward lower strikes** — downside protection "
                    "is in demand, so out-of-the-money puts carry higher IV.\n"
                    "- Use it as a sanity check: a strike poking **above** the smooth curve is relatively "
                    "**expensive** vol, one sitting **below** it is relatively **cheap**."
                )

    # --- Call Skew tab ---
    with tab_map["Call Skew"]:
        st.subheader("Call Skew Snapshot")
        if call_skew_df.empty:
            st.warning("No call skew data available.")
        else:
            st.dataframe(call_skew_df, use_container_width=True, hide_index=True)
            st.line_chart(call_skew_df.set_index("Pct From Spot")[["IV (%)"]])
            with st.expander("ℹ️ How to read the skew"):
                st.markdown(
                    "This is the **volatility smile/skew**: implied volatility (IV, the y-axis) for "
                    "each call strike, plotted by its distance from the current price (% from spot, x-axis).\n\n"
                    "- A **higher** line means the market is charging **more** volatility (a richer price) "
                    "for options at that strike.\n"
                    "- Use it as a sanity check: a strike poking **above** the smooth curve is relatively "
                    "**expensive** vol, one sitting **below** it is relatively **cheap**."
                )

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
