import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import ScannerConfig
from utils import safe_int, compute_time_to_expiry_years
from pricing import (
    BlackScholesCalculator,
    label_moneyness,
    compute_execution_price,
    get_market_greeks,
    compute_confidence_score,
    short_option_yield_metrics,
    label_atm_strike,
)
from data import get_option_chain
from database import save_iv_snapshot

logger = logging.getLogger(__name__)


# ============================================================
# ATM IV SNAPSHOTS
# ============================================================
def build_term_structure_snapshot(
    ticker: str,
    expirations: List[str],
    S: float,
    max_expirations: int = 8,
    save_to_db: bool = True,
) -> pd.DataFrame:
    rows = []
    picked = expirations[:max_expirations]

    for exp in picked:
        try:
            dte, _ = compute_time_to_expiry_years(exp)
            if dte < 0:
                continue

            chain = get_option_chain(ticker, exp)
            if chain.empty:
                continue

            chain = chain.copy()
            chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
            chain["mid_iv"] = pd.to_numeric(chain["mid_iv"], errors="coerce")
            chain["option_type"] = chain["option_type"].astype(str).str.upper()

            calls = chain[(chain["option_type"] == "CALL") & chain["strike"].notna() & chain["mid_iv"].notna()].copy()
            puts = chain[(chain["option_type"] == "PUT") & chain["strike"].notna() & chain["mid_iv"].notna()].copy()

            if calls.empty and puts.empty:
                continue

            def nearest_atm_iv(df_side: pd.DataFrame) -> float:
                idx = (df_side["strike"] - S).abs().idxmin()
                return float(df_side.loc[idx, "mid_iv"])

            call_iv = nearest_atm_iv(calls) if not calls.empty else np.nan
            put_iv = nearest_atm_iv(puts) if not puts.empty else np.nan
            avg_iv = np.nanmean([call_iv, put_iv])

            rows.append(
                {
                    "Expiration": exp,
                    "DTE": dte,
                    "ATM Call IV (%)": call_iv * 100.0 if not np.isnan(call_iv) else np.nan,
                    "ATM Put IV (%)": put_iv * 100.0 if not np.isnan(put_iv) else np.nan,
                    "ATM Avg IV (%)": avg_iv * 100.0 if not np.isnan(avg_iv) else np.nan,
                }
            )

            if save_to_db and not np.isnan(avg_iv):
                save_iv_snapshot(
                    ticker=ticker,
                    expiration=exp,
                    dte=dte,
                    atm_call_iv=None if np.isnan(call_iv) else float(call_iv),
                    atm_put_iv=None if np.isnan(put_iv) else float(put_iv),
                    atm_avg_iv=float(avg_iv),
                    spot=S,
                )

        except Exception:
            logger.debug("Term structure snapshot failed for expiration", exc_info=True)
            continue

    return pd.DataFrame(rows).sort_values("DTE").reset_index(drop=True) if rows else pd.DataFrame()


def compute_term_structure_scaling_factor(
    term_df: pd.DataFrame,
    target_expiration: str,
) -> Optional[float]:
    if term_df.empty or "ATM Avg IV (%)" not in term_df.columns:
        return None

    avg_col = term_df["ATM Avg IV (%)"].dropna()
    if avg_col.empty:
        return None

    mean_iv = avg_col.mean()
    if mean_iv <= 0:
        return None

    row = term_df[term_df["Expiration"] == target_expiration]
    if row.empty or pd.isna(row.iloc[0]["ATM Avg IV (%)"]):
        return None

    this_iv = float(row.iloc[0]["ATM Avg IV (%)"])
    factor = this_iv / mean_iv

    return float(np.clip(factor, 0.5, 2.0))


def build_skew_snapshot(chain_df: pd.DataFrame, S: float, option_type: str = "PUT") -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame()

    df = chain_df.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["mid_iv"] = pd.to_numeric(df["mid_iv"], errors="coerce")
    df["option_type"] = df["option_type"].astype(str).str.upper()

    side = df[(df["option_type"] == option_type.upper()) & df["strike"].notna() & df["mid_iv"].notna()].copy()
    if side.empty:
        return pd.DataFrame()

    side["Pct From Spot"] = ((side["strike"] / S) - 1.0) * 100.0
    side["IV (%)"] = side["mid_iv"] * 100.0
    side = side.sort_values("strike").reset_index(drop=True)

    return side[["strike", "Pct From Spot", "IV (%)"]].rename(columns={"strike": "Strike"})


# ============================================================
# MAIN SCREENING ENGINE
# ============================================================
def screen_chain(
    chain_df: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    T: float,
    dte: int,
    action: str,
    option_family: str,
    forecast_vol: float,
    cfg: ScannerConfig,
) -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame()

    option_type = "PUT" if option_family == "PUTS" else "CALL"

    df = chain_df.copy()
    df = df[df["option_type"] == option_type].copy()
    if df.empty:
        return pd.DataFrame()

    for col in ["strike", "bid", "ask", "volume", "open_interest", "mid_iv"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mid"] = np.where(
        df["bid"].notna() & df["ask"].notna() & (df["bid"] >= 0) & (df["ask"] >= 0),
        (df["bid"] + df["ask"]) / 2.0,
        np.nan,
    )

    df["spread_pct"] = np.where(
        df["mid"].notna() & (df["mid"] > 0),
        ((df["ask"] - df["bid"]) / df["mid"]) * 100.0,
        np.nan,
    )

    df["exec_px"] = df.apply(
        lambda row: compute_execution_price(
            row,
            action=action,
            use_executable_pricing=cfg.use_executable_pricing,
            slippage_pct=cfg.execution_slippage_pct,
        ),
        axis=1,
    )

    df = df[
        df["strike"].notna()
        & df["bid"].notna()
        & df["ask"].notna()
        & df["mid_iv"].notna()
        & (df["mid_iv"] > 0)
        & (df["open_interest"].fillna(0) >= cfg.min_open_interest)
        & (df["volume"].fillna(0) >= cfg.min_volume)
        & (df["bid"] >= cfg.min_bid)
        & df["spread_pct"].notna()
        & (df["spread_pct"] <= cfg.max_spread_pct)
        & df["exec_px"].notna()
        & (df["exec_px"] > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    rows = []

    for row in df.itertuples(index=False):
        K = float(row.strike)
        bid = float(row.bid)
        ask = float(row.ask)
        exec_px = float(row.exec_px)
        market_iv = float(row.mid_iv)
        oi = safe_int(getattr(row, "open_interest", None))
        vol = safe_int(getattr(row, "volume", None))

        row_series = pd.Series({
            "delta_mkt": getattr(row, "delta_mkt", np.nan),
            "gamma_mkt": getattr(row, "gamma_mkt", np.nan),
            "theta_mkt": getattr(row, "theta_mkt", np.nan),
            "vega_mkt": getattr(row, "vega_mkt", np.nan),
        })
        market_greeks = get_market_greeks(
            row=row_series,
            S=S,
            K=K,
            T=T,
            r=r,
            q=q,
            market_iv=market_iv,
            option_type=option_type,
        )

        if abs(market_greeks["delta"]) < cfg.min_abs_delta or abs(market_greeks["delta"]) > cfg.max_abs_delta:
            continue

        forecast_calc = BlackScholesCalculator(S=S, K=K, T=T, r=r, sigma=forecast_vol, q=q)
        forecast_theo = forecast_calc.price(option_type)

        if action == "BUY":
            abs_edge = forecast_theo - exec_px
        else:
            abs_edge = exec_px - forecast_theo

        value_edge_pct = (abs_edge / exec_px) * 100.0 if exec_px > 0 else np.nan

        yld = short_option_yield_metrics(action, option_type, S, K, exec_px, dte)

        rows.append(
            {
                "Moneyness": label_moneyness(S, K, option_type),
                "Strike": K,
                "Bid": bid,
                "Ask": ask,
                "Exec Px": exec_px,
                "Value Edge (%)": value_edge_pct,
                "Spread (%)": float(row.spread_pct),
                "Mkt IV (%)": market_iv * 100.0,
                "Delta": market_greeks["delta"],
                "Theta": market_greeks["theta"],
                "Vol": vol,
                "OI": oi,
                "Ann Yield (%)": yld["Ann Yield (%)"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = label_atm_strike(out, S)
    out["Confidence"] = compute_confidence_score(out, cfg)
    out = out.sort_values(
        by=["Confidence", "Value Edge (%)", "Spread (%)", "OI"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    return out.head(cfg.top_n)
