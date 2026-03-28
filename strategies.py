import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import ScannerConfig
from utils import safe_float, safe_int
from pricing import BlackScholesCalculator

logger = logging.getLogger(__name__)


# ============================================================
# VERTICAL SPREAD BUILDER
# ============================================================
def _leg_liquidity_ok(row: pd.Series, cfg: ScannerConfig) -> bool:
    """Check if a single leg passes minimum liquidity filters."""
    oi = safe_int(row.get("open_interest"))
    vol = safe_int(row.get("volume"))
    bid = safe_float(row.get("bid"))
    return (
        oi >= cfg.min_open_interest
        and vol >= cfg.min_volume
        and not np.isnan(bid)
        and bid >= cfg.min_bid
    )


def build_vertical_spreads(
    chain_df: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    T: float,
    dte: int,
    forecast_vol: float,
    cfg: ScannerConfig,
) -> pd.DataFrame:
    """
    Generate and score vertical credit spreads from the option chain.

    For SELL action (credit spreads):
      - Bull put spreads: sell higher-strike put, buy lower-strike put
      - Bear call spreads: sell lower-strike call, buy higher-strike call

    Returns a DataFrame of spreads sorted by a composite score.
    """
    if chain_df.empty or T <= 0:
        return pd.DataFrame()

    df = chain_df.copy()
    for col in ["strike", "bid", "ask", "mid_iv", "open_interest", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["strike"].notna()
        & df["bid"].notna()
        & df["ask"].notna()
        & df["mid_iv"].notna()
        & (df["mid_iv"] > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    puts = df[df["option_type"] == "PUT"].sort_values("strike").reset_index(drop=True)
    calls = df[df["option_type"] == "CALL"].sort_values("strike").reset_index(drop=True)

    spreads = []

    # Bull put spreads (credit): sell high put, buy low put
    if len(puts) >= 2:
        spreads.extend(_build_put_spreads(puts, S, r, q, T, dte, forecast_vol, cfg))

    # Bear call spreads (credit): sell low call, buy high call
    if len(calls) >= 2:
        spreads.extend(_build_call_spreads(calls, S, r, q, T, dte, forecast_vol, cfg))

    if not spreads:
        return pd.DataFrame()

    result = pd.DataFrame(spreads)

    # Score spreads
    result["Score"] = _score_spreads(result)
    result = result.sort_values("Score", ascending=False).reset_index(drop=True)

    return result.head(cfg.spread_top_n)


def _build_put_spreads(
    puts: pd.DataFrame,
    S: float, r: float, q: float, T: float, dte: int,
    forecast_vol: float, cfg: ScannerConfig,
) -> List[Dict]:
    """Generate bull put credit spreads."""
    spreads = []
    strikes = puts["strike"].values

    for i in range(len(puts)):
        short_row = puts.iloc[i]
        short_K = float(short_row["strike"])

        if not _leg_liquidity_ok(short_row, cfg):
            continue

        for j in range(i):
            long_row = puts.iloc[j]
            long_K = float(long_row["strike"])
            width = short_K - long_K

            if width <= 0 or width > cfg.spread_max_width:
                continue

            if not _leg_liquidity_ok(long_row, cfg):
                continue

            short_bid = safe_float(short_row.get("bid"))
            long_ask = safe_float(long_row.get("ask"))

            if np.isnan(short_bid) or np.isnan(long_ask):
                continue

            net_credit = short_bid - long_ask
            if net_credit < cfg.spread_min_credit:
                continue

            max_loss = width - net_credit
            if max_loss <= 0:
                continue

            spread = _compute_spread_metrics(
                strategy="Bull Put",
                short_row=short_row,
                long_row=long_row,
                short_K=short_K,
                long_K=long_K,
                width=width,
                net_credit=net_credit,
                max_loss=max_loss,
                S=S, r=r, q=q, T=T, dte=dte,
                forecast_vol=forecast_vol,
                option_type="PUT",
            )
            spreads.append(spread)

    return spreads


def _build_call_spreads(
    calls: pd.DataFrame,
    S: float, r: float, q: float, T: float, dte: int,
    forecast_vol: float, cfg: ScannerConfig,
) -> List[Dict]:
    """Generate bear call credit spreads."""
    spreads = []

    for i in range(len(calls)):
        short_row = calls.iloc[i]
        short_K = float(short_row["strike"])

        if not _leg_liquidity_ok(short_row, cfg):
            continue

        for j in range(i + 1, len(calls)):
            long_row = calls.iloc[j]
            long_K = float(long_row["strike"])
            width = long_K - short_K

            if width <= 0 or width > cfg.spread_max_width:
                continue

            if not _leg_liquidity_ok(long_row, cfg):
                continue

            short_bid = safe_float(short_row.get("bid"))
            long_ask = safe_float(long_row.get("ask"))

            if np.isnan(short_bid) or np.isnan(long_ask):
                continue

            net_credit = short_bid - long_ask
            if net_credit < cfg.spread_min_credit:
                continue

            max_loss = width - net_credit
            if max_loss <= 0:
                continue

            spread = _compute_spread_metrics(
                strategy="Bear Call",
                short_row=short_row,
                long_row=long_row,
                short_K=short_K,
                long_K=long_K,
                width=width,
                net_credit=net_credit,
                max_loss=max_loss,
                S=S, r=r, q=q, T=T, dte=dte,
                forecast_vol=forecast_vol,
                option_type="CALL",
            )
            spreads.append(spread)

    return spreads


def build_debit_spreads(
    chain_df: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    T: float,
    dte: int,
    forecast_vol: float,
    cfg: ScannerConfig,
) -> pd.DataFrame:
    """
    Generate and score vertical debit spreads from the option chain.

    For BUY action (debit spreads):
      - Bull call spreads: buy lower-strike call, sell higher-strike call
      - Bear put spreads: buy higher-strike put, sell lower-strike put

    Returns a DataFrame of spreads sorted by a composite score.
    """
    if chain_df.empty or T <= 0:
        return pd.DataFrame()

    df = chain_df.copy()
    for col in ["strike", "bid", "ask", "mid_iv", "open_interest", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["strike"].notna()
        & df["bid"].notna()
        & df["ask"].notna()
        & df["mid_iv"].notna()
        & (df["mid_iv"] > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    puts = df[df["option_type"] == "PUT"].sort_values("strike").reset_index(drop=True)
    calls = df[df["option_type"] == "CALL"].sort_values("strike").reset_index(drop=True)

    spreads = []

    # Bull call spreads (debit): buy low call, sell high call
    if len(calls) >= 2:
        spreads.extend(_build_bull_call_spreads(calls, S, r, q, T, dte, forecast_vol, cfg))

    # Bear put spreads (debit): buy high put, sell low put
    if len(puts) >= 2:
        spreads.extend(_build_bear_put_spreads(puts, S, r, q, T, dte, forecast_vol, cfg))

    if not spreads:
        return pd.DataFrame()

    result = pd.DataFrame(spreads)

    result["Score"] = _score_debit_spreads(result)
    result = result.sort_values("Score", ascending=False).reset_index(drop=True)

    return result.head(cfg.spread_top_n)


def _build_bull_call_spreads(
    calls: pd.DataFrame,
    S: float, r: float, q: float, T: float, dte: int,
    forecast_vol: float, cfg: ScannerConfig,
) -> List[Dict]:
    """Generate bull call debit spreads: buy lower call, sell higher call."""
    spreads = []

    for i in range(len(calls)):
        long_row = calls.iloc[i]
        long_K = float(long_row["strike"])

        if not _leg_liquidity_ok(long_row, cfg):
            continue

        for j in range(i + 1, len(calls)):
            short_row = calls.iloc[j]
            short_K = float(short_row["strike"])
            width = short_K - long_K

            if width <= 0 or width > cfg.spread_max_width:
                continue

            if not _leg_liquidity_ok(short_row, cfg):
                continue

            long_ask = safe_float(long_row.get("ask"))
            short_bid = safe_float(short_row.get("bid"))

            if np.isnan(long_ask) or np.isnan(short_bid):
                continue

            net_debit = long_ask - short_bid
            if net_debit <= 0:
                continue

            max_profit = width - net_debit
            if max_profit <= 0:
                continue

            spread = _compute_debit_spread_metrics(
                strategy="Bull Call",
                long_row=long_row,
                short_row=short_row,
                long_K=long_K,
                short_K=short_K,
                width=width,
                net_debit=net_debit,
                max_profit=max_profit,
                S=S, r=r, q=q, T=T, dte=dte,
                forecast_vol=forecast_vol,
                option_type="CALL",
            )
            spreads.append(spread)

    return spreads


def _build_bear_put_spreads(
    puts: pd.DataFrame,
    S: float, r: float, q: float, T: float, dte: int,
    forecast_vol: float, cfg: ScannerConfig,
) -> List[Dict]:
    """Generate bear put debit spreads: buy higher put, sell lower put."""
    spreads = []

    for i in range(len(puts)):
        long_row = puts.iloc[i]
        long_K = float(long_row["strike"])

        if not _leg_liquidity_ok(long_row, cfg):
            continue

        for j in range(i):
            short_row = puts.iloc[j]
            short_K = float(short_row["strike"])
            width = long_K - short_K

            if width <= 0 or width > cfg.spread_max_width:
                continue

            if not _leg_liquidity_ok(short_row, cfg):
                continue

            long_ask = safe_float(long_row.get("ask"))
            short_bid = safe_float(short_row.get("bid"))

            if np.isnan(long_ask) or np.isnan(short_bid):
                continue

            net_debit = long_ask - short_bid
            if net_debit <= 0:
                continue

            max_profit = width - net_debit
            if max_profit <= 0:
                continue

            spread = _compute_debit_spread_metrics(
                strategy="Bear Put",
                long_row=long_row,
                short_row=short_row,
                long_K=long_K,
                short_K=short_K,
                width=width,
                net_debit=net_debit,
                max_profit=max_profit,
                S=S, r=r, q=q, T=T, dte=dte,
                forecast_vol=forecast_vol,
                option_type="PUT",
            )
            spreads.append(spread)

    return spreads


def _compute_debit_spread_metrics(
    strategy: str,
    long_row: pd.Series,
    short_row: pd.Series,
    long_K: float,
    short_K: float,
    width: float,
    net_debit: float,
    max_profit: float,
    S: float, r: float, q: float, T: float, dte: int,
    forecast_vol: float,
    option_type: str,
) -> Dict:
    """Compute all metrics for a single vertical debit spread."""
    max_loss = net_debit

    # Breakeven
    if strategy == "Bull Call":
        breakeven = long_K + net_debit
    else:  # Bear Put
        breakeven = long_K - net_debit

    # Risk/reward ratio (profit per dollar risked)
    risk_reward = max_profit / max_loss if max_loss > 0 else np.nan

    # Net Greeks from BS model using market IV
    long_iv = safe_float(long_row.get("mid_iv"))
    short_iv = safe_float(short_row.get("mid_iv"))

    net_delta = net_gamma = net_theta = net_vega = np.nan

    if not np.isnan(long_iv) and not np.isnan(short_iv):
        long_calc = BlackScholesCalculator(S=S, K=long_K, T=T, r=r, sigma=long_iv, q=q)
        short_calc = BlackScholesCalculator(S=S, K=short_K, T=T, r=r, sigma=short_iv, q=q)
        long_greeks = long_calc.greeks(option_type)
        short_greeks = short_calc.greeks(option_type)

        # Long leg bought, short leg sold
        net_delta = long_greeks["delta"] - short_greeks["delta"]
        net_gamma = long_greeks["gamma"] - short_greeks["gamma"]
        net_theta = long_greeks["theta"] - short_greeks["theta"]
        net_vega = long_greeks["vega"] - short_greeks["vega"]

    # Probability of profit
    pop = _probability_of_profit(S, breakeven, T, r, q, forecast_vol, strategy)

    # Model edge: BS model debit vs market debit (negative edge = cheaper than model)
    model_edge = np.nan
    if not np.isnan(long_iv) and not np.isnan(short_iv):
        long_model = BlackScholesCalculator(S=S, K=long_K, T=T, r=r, sigma=forecast_vol, q=q)
        short_model = BlackScholesCalculator(S=S, K=short_K, T=T, r=r, sigma=forecast_vol, q=q)
        model_long_price = long_model.price(option_type)
        model_short_price = short_model.price(option_type)
        model_debit = model_long_price - model_short_price
        # Positive edge means we're paying less than model says it's worth
        model_edge = (model_debit - net_debit) / net_debit * 100.0 if net_debit > 0 else np.nan

    # Annualized return on risk
    effective_dte = max(dte, 1)
    ann_return = (max_profit / max_loss) * (365.0 / effective_dte) * 100.0 if max_loss > 0 else np.nan

    # Liquidity: worst leg OI
    min_oi = min(
        safe_int(long_row.get("open_interest")),
        safe_int(short_row.get("open_interest")),
    )

    return {
        "Strategy": strategy,
        "Long Strike": long_K,
        "Short Strike": short_K,
        "Width ($)": width,
        "Net Debit ($)": net_debit,
        "Max Profit ($)": max_profit,
        "Max Loss ($)": max_loss,
        "Breakeven": breakeven,
        "Risk/Reward": risk_reward,
        "PoP (%)": pop,
        "Model Edge (%)": model_edge,
        "Ann Return (%)": ann_return,
        "Net Delta": net_delta,
        "Net Theta": net_theta,
        "Net Gamma": net_gamma,
        "Net Vega": net_vega,
        "Min OI": min_oi,
        "DTE": dte,
    }


def _compute_spread_metrics(
    strategy: str,
    short_row: pd.Series,
    long_row: pd.Series,
    short_K: float,
    long_K: float,
    width: float,
    net_credit: float,
    max_loss: float,
    S: float, r: float, q: float, T: float, dte: int,
    forecast_vol: float,
    option_type: str,
) -> Dict:
    """Compute all metrics for a single vertical spread."""
    max_profit = net_credit

    # Breakeven
    if strategy == "Bull Put":
        breakeven = short_K - net_credit
    else:
        breakeven = short_K + net_credit

    # Risk/reward ratio
    risk_reward = max_profit / max_loss if max_loss > 0 else np.nan

    # Net Greeks from BS model using market IV
    short_iv = safe_float(short_row.get("mid_iv"))
    long_iv = safe_float(long_row.get("mid_iv"))

    net_delta = net_gamma = net_theta = net_vega = np.nan

    if not np.isnan(short_iv) and not np.isnan(long_iv):
        short_calc = BlackScholesCalculator(S=S, K=short_K, T=T, r=r, sigma=short_iv, q=q)
        long_calc = BlackScholesCalculator(S=S, K=long_K, T=T, r=r, sigma=long_iv, q=q)
        short_greeks = short_calc.greeks(option_type)
        long_greeks = long_calc.greeks(option_type)

        # Short leg sold, long leg bought
        net_delta = -short_greeks["delta"] + long_greeks["delta"]
        net_gamma = -short_greeks["gamma"] + long_greeks["gamma"]
        net_theta = -short_greeks["theta"] + long_greeks["theta"]
        net_vega = -short_greeks["vega"] + long_greeks["vega"]

    # Probability of profit (prob spot stays above breakeven for bull put,
    # or below breakeven for bear call) using forecast vol
    pop = _probability_of_profit(S, breakeven, T, r, q, forecast_vol, strategy)

    # Model edge: BS theo spread value vs market credit
    model_edge = np.nan
    if not np.isnan(short_iv) and not np.isnan(long_iv):
        short_model = BlackScholesCalculator(S=S, K=short_K, T=T, r=r, sigma=forecast_vol, q=q)
        long_model = BlackScholesCalculator(S=S, K=long_K, T=T, r=r, sigma=forecast_vol, q=q)
        model_short_price = short_model.price(option_type)
        model_long_price = long_model.price(option_type)
        model_credit = model_short_price - model_long_price
        model_edge = (net_credit - model_credit) / net_credit * 100.0 if net_credit > 0 else np.nan

    # Annualized return on risk
    effective_dte = max(dte, 1)
    ann_return = (max_profit / max_loss) * (365.0 / effective_dte) * 100.0 if max_loss > 0 else np.nan

    # Liquidity: worst leg OI
    min_oi = min(
        safe_int(short_row.get("open_interest")),
        safe_int(long_row.get("open_interest")),
    )

    return {
        "Strategy": strategy,
        "Short Strike": short_K,
        "Long Strike": long_K,
        "Width ($)": width,
        "Net Credit ($)": net_credit,
        "Max Profit ($)": max_profit,
        "Max Loss ($)": max_loss,
        "Breakeven": breakeven,
        "Risk/Reward": risk_reward,
        "PoP (%)": pop,
        "Model Edge (%)": model_edge,
        "Ann Return (%)": ann_return,
        "Net Delta": net_delta,
        "Net Theta": net_theta,
        "Net Gamma": net_gamma,
        "Net Vega": net_vega,
        "Min OI": min_oi,
        "DTE": dte,
    }


def _probability_of_profit(
    S: float, breakeven: float, T: float, r: float, q: float,
    sigma: float, strategy: str,
) -> float:
    """
    Probability that spot finishes on the profitable side of breakeven
    under log-normal dynamics with forecast vol.
    """
    if T <= 0 or sigma <= 0:
        return np.nan

    d2 = (np.log(S / breakeven) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if strategy in ("Bull Put", "Bull Call"):
        # Profit if spot > breakeven
        return float(norm.cdf(d2) * 100.0)
    else:
        # Bear Call / Bear Put: profit if spot < breakeven
        return float(norm.cdf(-d2) * 100.0)


def _score_spreads(df: pd.DataFrame) -> pd.Series:
    """
    Composite score for ranking spreads.
    Blend of: PoP, risk/reward, model edge, liquidity.
    """
    def _norm(s, higher_is_better=True):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or np.isclose(mn, mx):
            return pd.Series(np.full(len(s), 50.0), index=s.index)
        scaled = 100.0 * (s - mn) / (mx - mn)
        if not higher_is_better:
            scaled = 100.0 - scaled
        return scaled.clip(0, 100)

    pop_score = _norm(df["PoP (%)"], higher_is_better=True)
    rr_score = _norm(df["Risk/Reward"], higher_is_better=True)
    edge_score = _norm(df["Model Edge (%)"], higher_is_better=True)
    oi_score = _norm(df["Min OI"], higher_is_better=True)

    return (0.35 * pop_score + 0.25 * rr_score + 0.25 * edge_score + 0.15 * oi_score).clip(0, 100)


def _score_debit_spreads(df: pd.DataFrame) -> pd.Series:
    """
    Composite score for ranking debit spreads.
    Same blend as credit spreads: PoP, risk/reward, model edge, liquidity.
    """
    def _norm(s, higher_is_better=True):
        s = pd.to_numeric(s, errors="coerce").astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or np.isclose(mn, mx):
            return pd.Series(np.full(len(s), 50.0), index=s.index)
        scaled = 100.0 * (s - mn) / (mx - mn)
        if not higher_is_better:
            scaled = 100.0 - scaled
        return scaled.clip(0, 100)

    pop_score = _norm(df["PoP (%)"], higher_is_better=True)
    rr_score = _norm(df["Risk/Reward"], higher_is_better=True)
    edge_score = _norm(df["Model Edge (%)"], higher_is_better=True)
    oi_score = _norm(df["Min OI"], higher_is_better=True)

    return (0.35 * pop_score + 0.25 * rr_score + 0.25 * edge_score + 0.15 * oi_score).clip(0, 100)
