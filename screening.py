import logging
import math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import ScannerConfig, NY_TZ, CALENDAR_DAYS_PER_YEAR
from utils import safe_int, compute_time_to_expiry_years
from pricing import (
    BlackScholesCalculator,
    label_moneyness,
    compute_execution_price,
    get_market_greeks,
    compute_confidence_score,
    short_option_yield_metrics,
    label_atm_strike,
    fit_smile,
    iv_edge_components,
    dollar_edge_from_vol,
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


def implied_earnings_move_from_term_structure(
    term_df: pd.DataFrame,
    earnings_date_str: Optional[str],
    today=None,
) -> Optional[float]:
    """Estimate the earnings jump (fraction of spot) from the ATM IV term
    structure: the first post-earnings expiry carries extra total variance vs
    the last pre-earnings expiry, and that difference is the jump variance.
    Returns None when the curve can't isolate it (e.g. no pre-earnings expiry),
    so the caller can fall back to a configured default.
    """
    if term_df.empty or not earnings_date_str or "Expiration" not in term_df.columns:
        return None
    try:
        earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    if today is None:
        today = datetime.now(NY_TZ).date()
    if earnings_dt < today:
        return None

    pre, post = [], []
    for _, row in term_df.iterrows():
        iv_pct = row.get("ATM Avg IV (%)")
        exp = row.get("Expiration")
        dte = row.get("DTE")
        if pd.isna(iv_pct) or pd.isna(dte) or not exp:
            continue
        try:
            exp_dt = datetime.strptime(str(exp), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        iv = float(iv_pct) / 100.0
        if iv <= 0:
            continue
        t = max(float(dte) / CALENDAR_DAYS_PER_YEAR, 1e-6)
        var_total = iv * iv * t
        (pre if exp_dt < earnings_dt else post).append((t, var_total))

    if not pre or not post:
        return None

    _, v_pre = max(pre, key=lambda p: p[0])   # last expiry before earnings
    _, v_post = min(post, key=lambda p: p[0])  # first expiry spanning earnings
    jump_var = v_post - v_pre
    if jump_var <= 0:
        return None
    return float(np.clip(math.sqrt(jump_var), 0.0, 0.5))


def _term_structure_points(
    term_df: pd.DataFrame,
    earnings_dte: Optional[float] = None,
    jump_var: float = 0.0,
):
    """Sorted (t_years, total_variance) points from the ATM IV term structure.

    When an earnings jump is supplied, strip its variance from every expiry that
    spans the earnings date, so the resulting curve reflects *diffusive* vol only
    and won't double-count earnings once the jump is added back per-expiry.
    """
    if term_df.empty or "ATM Avg IV (%)" not in term_df.columns or "DTE" not in term_df.columns:
        return None
    pts = []
    for _, row in term_df.iterrows():
        iv_pct = row.get("ATM Avg IV (%)")
        dte = row.get("DTE")
        if pd.isna(iv_pct) or pd.isna(dte):
            continue
        iv = float(iv_pct) / 100.0
        dte = float(dte)
        if iv <= 0 or dte < 0:
            continue
        t = max(dte / CALENDAR_DAYS_PER_YEAR, 1e-6)
        v = iv * iv * t
        if jump_var > 0 and earnings_dte is not None and earnings_dte >= 0 and dte >= earnings_dte:
            v = max(v - jump_var, 1e-8)
        pts.append((t, v))
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def compute_term_structure_scaling_factor(
    term_df: pd.DataFrame,
    target_dte: float,
    earnings_dte: Optional[float] = None,
    jump_var: float = 0.0,
    reference_dte: float = 45.0,
) -> Optional[float]:
    """Diffusive term-structure scaling factor for the realized-vol forecast.

    Interpolates the ATM IV curve to the exact target DTE, linear in total
    variance (sigma^2 * t) vs time — the standard, arbitrage-aware interpolation —
    and divides by the curve's value at a ~1-month reference tenor. Captures the
    *shape* of the term structure (contango/backwardation) projected onto the
    target tenor; earnings are stripped first so this stays purely diffusive.
    """
    pts = _term_structure_points(term_df, earnings_dte, jump_var)
    if pts is None:
        return None
    t_arr, v_arr = pts

    t_target = max(target_dte / CALENDAR_DAYS_PER_YEAR, 1e-6)
    dte_min = float(t_arr.min() * CALENDAR_DAYS_PER_YEAR)
    dte_max = float(t_arr.max() * CALENDAR_DAYS_PER_YEAR)
    ref_dte = float(np.clip(reference_dte, dte_min, dte_max))
    t_ref = max(ref_dte / CALENDAR_DAYS_PER_YEAR, 1e-6)

    w_target = float(np.interp(t_target, t_arr, v_arr))
    w_ref = float(np.interp(t_ref, t_arr, v_arr))
    if w_target <= 0 or w_ref <= 0:
        return None

    iv_target = math.sqrt(w_target / t_target)
    iv_ref = math.sqrt(w_ref / t_ref)
    if iv_ref <= 0:
        return None

    return float(np.clip(iv_target / iv_ref, 0.5, 2.0))


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
    forecast_vol_uncertainty: Optional[float] = None,
) -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame()

    option_type = "PUT" if option_family == "PUTS" else "CALL"

    df = chain_df.copy()
    df = df[df["option_type"] == option_type].copy()
    if df.empty:
        return pd.DataFrame()

    for col in ["strike", "bid", "ask", "volume", "open_interest", "mid_iv", "bid_iv", "ask_iv"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fit one smooth smile for this side from the broad set of valid strikes
    # (before liquidity filtering), so fair value is skew-aware. Liquidity-weight
    # by open interest so thin wing quotes don't distort the curve.
    smile_src = df[df["strike"].notna() & df["mid_iv"].notna() & (df["mid_iv"] > 0)]
    smile = fit_smile(
        strikes=smile_src["strike"].to_numpy(),
        mid_ivs=smile_src["mid_iv"].to_numpy(),
        S=S,
        weights=(smile_src["open_interest"].fillna(0.0).to_numpy() + 1.0),
        degree=cfg.smile_fit_degree,
        min_points=cfg.min_smile_points,
    )

    fv_atm = float(forecast_vol)
    fv_unc = float(forecast_vol_uncertainty) if forecast_vol_uncertainty and forecast_vol_uncertainty > 0 else 0.005

    # Scan-level surface signal: market ATM IV vs the realized-vol forecast.
    if smile is not None:
        market_atm_iv = float(smile.atm_value)
        residual_std = float(smile.residual_std)
        sign = 1.0 if action == "SELL" else -1.0
        level_edge_decimal = sign * (market_atm_iv - fv_atm)
        level_z = abs(market_atm_iv - fv_atm) / fv_unc
    else:
        market_atm_iv = np.nan
        residual_std = 0.0
        level_edge_decimal = np.nan
        level_z = np.nan

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

    # Per-strike noise floor: the level (forecast) uncertainty and the smile-fit
    # residual dispersion added in quadrature.
    row_noise = math.sqrt(fv_unc ** 2 + residual_std ** 2)
    row_noise = max(row_noise, 1e-4)

    rows = []

    for row in df.itertuples(index=False):
        K = float(row.strike)
        bid = float(row.bid)
        ask = float(row.ask)
        exec_px = float(row.exec_px)
        market_iv = float(row.mid_iv)
        bid_iv = float(getattr(row, "bid_iv", np.nan))
        ask_iv = float(getattr(row, "ask_iv", np.nan))
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

        if smile is None:
            continue

        edge = iv_edge_components(
            market_iv=market_iv,
            bid_iv=bid_iv,
            ask_iv=ask_iv,
            K=K,
            smile=smile,
            fv_atm=fv_atm,
            action=action,
        )
        iv_edge_decimal = edge["iv_edge"]
        vega = float(market_greeks.get("vega", np.nan))
        dollar_edge = dollar_edge_from_vol(vega, iv_edge_decimal)
        signal_z = iv_edge_decimal / row_noise

        yld = short_option_yield_metrics(action, option_type, S, K, exec_px, dte)

        rows.append(
            {
                "Moneyness": label_moneyness(S, K, option_type),
                "Strike": K,
                "Bid": bid,
                "Ask": ask,
                "Exec Px": exec_px,
                "IV Edge (vol pts)": iv_edge_decimal * 100.0,
                "RV Edge (vol pts)": edge["rv_edge"] * 100.0,
                "Level Edge (vol pts)": level_edge_decimal * 100.0,
                "Signal Z": signal_z,
                "$ Edge": dollar_edge,
                "Spread (%)": float(row.spread_pct),
                "Mkt IV (%)": market_iv * 100.0,
                "Fair IV (%)": edge["fair_iv"] * 100.0,
                "Delta": market_greeks["delta"],
                "Theta": market_greeks["theta"],
                "Vega": vega,
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
        by=["Confidence", "IV Edge (vol pts)", "Spread (%)", "OI"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    out = out.head(cfg.top_n)

    # Scan-level surface summary for the UI (constant across rows).
    out.attrs["scan_meta"] = {
        "level_edge_volpts": level_edge_decimal * 100.0 if np.isfinite(level_edge_decimal) else None,
        "level_z": level_z if np.isfinite(level_z) else None,
        "market_atm_iv": market_atm_iv if np.isfinite(market_atm_iv) else None,
        "fv_atm": fv_atm,
        "noise_gate_z": cfg.noise_gate_z,
        "smile_is_flat": bool(smile.is_flat) if smile is not None else True,
    }
    return out
