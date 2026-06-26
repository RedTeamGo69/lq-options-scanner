import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import ScannerConfig, NY_TZ, TRADING_DAYS_PER_YEAR, T_FLOOR_YEARS
from utils import safe_float


# ============================================================
# BLACK-SCHOLES / MERTON
# ============================================================
class BlackScholesCalculator:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        self.S = max(float(S), 1e-12)
        self.K = max(float(K), 1e-12)
        self.T = max(float(T), 0.0)
        self.r = float(r)
        self.sigma = max(float(sigma), 1e-6)
        self.q = float(q)

    def _d1_d2(self) -> Tuple[float, float]:
        if self.T <= 0:
            return 0.0, 0.0
        sqrt_T = math.sqrt(self.T)
        d1 = (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2

    def price(self, option_type: str) -> float:
        option_type = option_type.upper()
        if self.T <= 0:
            if option_type == "CALL":
                return max(0.0, self.S - self.K)
            return max(0.0, self.K - self.S)

        d1, d2 = self._d1_d2()
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)

        if option_type == "CALL":
            return self.S * disc_q * norm.cdf(d1) - self.K * disc_r * norm.cdf(d2)
        return self.K * disc_r * norm.cdf(-d2) - self.S * disc_q * norm.cdf(-d1)

    def greeks(self, option_type: str) -> Dict[str, float]:
        option_type = option_type.upper()

        if self.T <= 0:
            delta = 1.0 if (option_type == "CALL" and self.S > self.K) else 0.0
            if option_type == "PUT":
                delta = -1.0 if self.S < self.K else 0.0
            return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        d1, d2 = self._d1_d2()
        sqrt_T = math.sqrt(self.T)
        pdf_d1 = norm.pdf(d1)
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)

        if option_type == "CALL":
            delta = disc_q * norm.cdf(d1)
            theta = (
                -(self.S * disc_q * pdf_d1 * self.sigma) / (2.0 * sqrt_T)
                - self.r * self.K * disc_r * norm.cdf(d2)
                + self.q * self.S * disc_q * norm.cdf(d1)
            ) / 365.0
        else:
            delta = disc_q * (norm.cdf(d1) - 1.0)
            theta = (
                -(self.S * disc_q * pdf_d1 * self.sigma) / (2.0 * sqrt_T)
                + self.r * self.K * disc_r * norm.cdf(-d2)
                - self.q * self.S * disc_q * norm.cdf(-d1)
            ) / 365.0

        gamma = (disc_q * pdf_d1) / (self.S * self.sigma * sqrt_T)
        vega = (self.S * disc_q * pdf_d1 * sqrt_T) / 100.0  # per 1 vol point

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
        }


# ============================================================
# VOL FORECASTS
# ============================================================
def realized_vol_from_history(hist: pd.DataFrame, lookback: int) -> Optional[float]:
    """Close-to-close annualized realized vol. The simple, noisy baseline; kept
    as the fallback when OHLC is unavailable for the range-based estimator."""
    if len(hist) < lookback + 1:
        return None
    rv = hist["log_return"].tail(lookback).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    if pd.isna(rv) or rv <= 0:
        return None
    return float(rv)


def yang_zhang_vol(hist: pd.DataFrame, lookback: int) -> Optional[float]:
    """Yang-Zhang annualized realized vol from OHLC.

    Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
    intraday variance. It is drift-independent and far more statistically
    efficient than close-to-close (lower variance for the same window), so the
    forecast is steadier. Returns None when OHLC is missing/insufficient, so the
    caller can fall back to the close-to-close estimator.
    """
    needed = ["open", "high", "low", "close"]
    if not all(col in hist.columns for col in needed) or lookback < 2:
        return None

    df = hist[needed].apply(pd.to_numeric, errors="coerce").copy()
    df["prev_close"] = df["close"].shift(1)
    df = df.dropna(subset=needed + ["prev_close"])
    # All inputs must be strictly positive for the logs to be defined.
    df = df[(df[needed] > 0).all(axis=1) & (df["prev_close"] > 0)]
    if len(df) < lookback:
        return None

    window = df.tail(lookback)
    n = len(window)
    if n < 2:
        return None

    o = np.log(window["open"] / window["prev_close"])   # overnight
    u = np.log(window["high"] / window["open"])          # high vs open
    d = np.log(window["low"] / window["open"])           # low vs open
    c = np.log(window["close"] / window["open"])         # intraday close vs open

    var_o = float(o.var(ddof=1))
    var_c = float(c.var(ddof=1))
    var_rs = float((u * (u - c) + d * (d - c)).mean())   # Rogers-Satchell (1/n)

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_yz = var_o + k * var_c + (1.0 - k) * var_rs
    if not np.isfinite(var_yz) or var_yz <= 0:
        return None

    return float(math.sqrt(var_yz * TRADING_DAYS_PER_YEAR))


def realized_vol(hist: pd.DataFrame, lookback: int, estimator: str = "yang_zhang") -> Optional[float]:
    """Dispatch to the configured realized-vol estimator, with a graceful
    fallback to close-to-close when the range-based estimator can't run."""
    if estimator == "yang_zhang":
        yz = yang_zhang_vol(hist, lookback)
        if yz is not None:
            return yz
    return realized_vol_from_history(hist, lookback)


def forecast_vol_uncertainty(forecast: float, rv_values: list) -> float:
    """Estimate the standard error (in decimal vol) of the realized-vol forecast.

    Two contributions, take the larger:
      - Sampling error of an annualized vol estimate ~ sigma / sqrt(2 * n_eff),
        using the shortest (most responsive, noisiest) window as n_eff.
      - Dispersion across the RV20/60/120 windows, which captures regime and
        horizon uncertainty the sampling error alone misses.
    A small floor keeps the downstream z-score well-defined.
    """
    n_eff = 20.0
    std_error = abs(forecast) / math.sqrt(2.0 * n_eff)
    dispersion = float(np.std(rv_values, ddof=0)) if len(rv_values) > 1 else 0.0
    floor = 0.005  # 0.5 vol points
    return max(std_error, dispersion, floor)


def build_forward_vol_forecast(hist: pd.DataFrame, cfg: ScannerConfig) -> Dict[str, Optional[float]]:
    estimator = getattr(cfg, "vol_estimator", "yang_zhang")
    rv20 = realized_vol(hist, 20, estimator)
    rv60 = realized_vol(hist, 60, estimator)
    rv120 = realized_vol(hist, 120, estimator)

    values = []
    weights = []

    if rv20 is not None:
        values.append(rv20)
        weights.append(cfg.rv20_weight)
    if rv60 is not None:
        values.append(rv60)
        weights.append(cfg.rv60_weight)
    if rv120 is not None:
        values.append(rv120)
        weights.append(cfg.rv120_weight)

    if not values:
        forecast = None
        uncertainty = None
    else:
        w = np.array(weights, dtype=float)
        w_sum = w.sum()
        if w_sum == 0:
            forecast = None
            uncertainty = None
        else:
            w = w / w_sum
            forecast = float(np.dot(np.array(values, dtype=float), w))
            forecast *= cfg.vol_forecast_multiplier
            uncertainty = forecast_vol_uncertainty(forecast, values)

    return {
        "rv20": rv20,
        "rv60": rv60,
        "rv120": rv120,
        "forecast_vol": forecast,
        "forecast_vol_uncertainty": uncertainty,
    }


def adjust_forecast_vol_for_earnings(
    forecast_vol: float,
    T: float,
    earnings_date_str: Optional[str],
    expiration_date_str: str,
    expected_earnings_move: float,
) -> Tuple[float, bool]:
    if not earnings_date_str or T < T_FLOOR_YEARS or expected_earnings_move <= 0:
        return forecast_vol, False

    try:
        earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
        expiry_dt = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
        today = datetime.now(NY_TZ).date()
    except (ValueError, TypeError):
        return forecast_vol, False

    if not (today <= earnings_dt <= expiry_dt):
        return forecast_vol, False

    diffusion_var = forecast_vol ** 2 * T
    jump_var = expected_earnings_move ** 2
    total_var = diffusion_var + jump_var
    adjusted_vol = math.sqrt(total_var / T)

    return adjusted_vol, True


# ============================================================
# SCREENING HELPERS
# ============================================================
def label_moneyness(S: float, K: float, option_type: str) -> str:
    """Label a single strike as ITM or OTM. ATM is assigned separately
    by label_atm_strike() to the single closest strike."""
    option_type = option_type.upper()
    if option_type == "CALL":
        return "ITM" if K < S else "OTM"
    return "ITM" if K > S else "OTM"


def label_atm_strike(df: pd.DataFrame, S: float) -> pd.DataFrame:
    """Mark the single strike closest to spot as ATM in the Moneyness column."""
    if df.empty or "Strike" not in df.columns:
        return df
    df = df.copy()
    closest_idx = (df["Strike"] - S).abs().idxmin()
    df.loc[closest_idx, "Moneyness"] = "ATM"
    return df


def compute_execution_price(row: pd.Series, action: str, use_executable_pricing: bool, slippage_pct: float) -> float:
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))

    if np.isnan(bid) or np.isnan(ask):
        return np.nan

    mid = (bid + ask) / 2.0

    if not use_executable_pricing:
        return mid

    slip = slippage_pct / 100.0
    if action == "BUY":
        return ask * (1.0 + slip)
    return bid * (1.0 - slip)


def get_market_greeks(
    row: pd.Series,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_iv: float,
    option_type: str,
) -> Dict[str, float]:
    delta = safe_float(row.get("delta_mkt"))
    gamma = safe_float(row.get("gamma_mkt"))
    theta = safe_float(row.get("theta_mkt"))
    vega = safe_float(row.get("vega_mkt"))

    if all(not np.isnan(x) for x in [delta, gamma, theta, vega]):
        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
        }

    calc = BlackScholesCalculator(S=S, K=K, T=T, r=r, sigma=market_iv, q=q)
    return calc.greeks(option_type)


def normalize_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if series.empty:
        return series
    s = pd.to_numeric(series, errors="coerce").astype(float)
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or np.isclose(min_v, max_v):
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    scaled = 100.0 * (s - min_v) / (max_v - min_v)
    if not higher_is_better:
        scaled = 100.0 - scaled
    return scaled.clip(0, 100)


# ============================================================
# VOL SMILE + IV-SPACE EDGE
# ============================================================
@dataclass
class SmileFit:
    """A smooth fit of market IV vs log-moneyness for one option side.

    ``iv_at(K)`` gives the fitted IV at a strike; ``atm_value`` is the fitted IV
    at spot (log-moneyness 0); ``residual_std`` is the dispersion of market IVs
    around the fit (the noise floor for the relative-value signal).
    """
    S: float
    coeffs: np.ndarray  # np.polyval coefficients (highest power first), in log-moneyness
    atm_value: float
    residual_std: float
    is_flat: bool

    def iv_at(self, K):
        x = np.log(np.asarray(K, dtype=float) / self.S)
        return np.maximum(np.polyval(self.coeffs, x), 1e-4)


def fit_smile(
    strikes,
    mid_ivs,
    S: float,
    weights=None,
    degree: int = 2,
    min_points: int = 5,
) -> Optional[SmileFit]:
    """Fit a smooth smile (quadratic in log-moneyness by default) to a side's
    market IVs, optionally liquidity-weighted. Falls back to a flat level (the
    ATM market IV) when there are too few strikes to fit a shape, in which case
    the relative-value edge is identically zero and only the level edge acts.
    """
    K = np.asarray(strikes, dtype=float)
    iv = np.asarray(mid_ivs, dtype=float)
    mask = np.isfinite(K) & np.isfinite(iv) & (K > 0) & (iv > 0)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        mask &= np.isfinite(w) & (w >= 0)
        w = w[mask]

    K, iv = K[mask], iv[mask]
    if len(K) == 0:
        return None

    if len(K) < min_points:
        atm_iv = float(iv[np.argmin(np.abs(K - S))])
        return SmileFit(
            S=S,
            coeffs=np.array([atm_iv], dtype=float),
            atm_value=max(atm_iv, 1e-4),
            residual_std=float(np.std(iv, ddof=1)) if len(iv) > 1 else 0.0,
            is_flat=True,
        )

    x = np.log(K / S)
    deg = int(min(degree, len(K) - 1))
    if w is not None and not np.any(w > 0):
        w = None  # degenerate weights -> unweighted
    coeffs = np.polyfit(x, iv, deg, w=w)
    fitted = np.polyval(coeffs, x)
    resid = iv - fitted
    atm_value = float(np.polyval(coeffs, 0.0))

    return SmileFit(
        S=S,
        coeffs=coeffs,
        atm_value=max(atm_value, 1e-4),
        residual_std=float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0,
        is_flat=False,
    )


def _executable_iv(action: str, mid_iv: float, bid_iv: float, ask_iv: float) -> float:
    """The IV you actually transact at: pay the ask vol to buy, receive the bid
    vol to sell. Falls back to mid when bid/ask IVs are unavailable."""
    has_bid = np.isfinite(bid_iv) and bid_iv > 0
    has_ask = np.isfinite(ask_iv) and ask_iv > 0
    if action == "BUY":
        if has_ask:
            return float(ask_iv)
        if has_bid and np.isfinite(mid_iv):
            return float(mid_iv + (mid_iv - bid_iv))  # mirror the half-spread up
        return float(mid_iv)
    # SELL
    if has_bid:
        return float(bid_iv)
    if has_ask and np.isfinite(mid_iv):
        return float(mid_iv - (ask_iv - mid_iv))
    return float(mid_iv)


def iv_edge_components(
    market_iv: float,
    bid_iv: float,
    ask_iv: float,
    K: float,
    smile: SmileFit,
    fv_atm: float,
    action: str,
) -> Dict[str, float]:
    """Decompose the cheap/expensive signal in IV space (decimal vol).

    fair_iv(K) = market smile re-leveled to the realized-vol forecast at ATM, so
    the edge separates a *level* component (whole surface vs realized) from a
    *relative-value* component (this strike vs its smooth smile). All values are
    signed so that positive = favorable for the chosen action.
    """
    action = action.upper()
    fair_iv = float(smile.iv_at(K)) - smile.atm_value + fv_atm
    exec_iv = _executable_iv(action, market_iv, bid_iv, ask_iv)

    # Surface level: market ATM IV vs the realized-vol forecast (same for every strike).
    level_raw = smile.atm_value - fv_atm           # >0 => surface rich vs realized
    # Strike dislocation: this strike's IV vs the smooth smile.
    rv_raw = float(market_iv) - float(smile.iv_at(K))  # >0 => strike rich vs its smile
    # Total executable edge in IV space.
    total_raw = exec_iv - fair_iv                  # >0 => you transact richer than fair

    sign = 1.0 if action == "SELL" else -1.0
    return {
        "fair_iv": fair_iv,
        "exec_iv": exec_iv,
        "level_edge": sign * level_raw,
        "rv_edge": sign * rv_raw,
        "iv_edge": sign * total_raw,
    }


def dollar_edge_from_vol(vega: float, iv_edge_decimal: float) -> float:
    """Convert a vol-point edge to dollars per contract via vega.

    ``vega`` is per 1 vol point (Tradier's convention and pricing.greeks'),
    ``iv_edge_decimal`` is in decimal vol (0.03 = 3 vol points). $/share =
    vega * (edge in vol points); * 100 for a standard 100-share contract.
    """
    if not np.isfinite(vega) or not np.isfinite(iv_edge_decimal):
        return np.nan
    return float(vega * (iv_edge_decimal * 100.0) * 100.0)


def compute_confidence_score(df: pd.DataFrame, cfg: ScannerConfig) -> pd.Series:
    edge_score = normalize_score(df["IV Edge (vol pts)"], higher_is_better=True)
    if "Signal Z" in df.columns:
        sig_score = normalize_score(df["Signal Z"], higher_is_better=True)
    else:
        sig_score = pd.Series(np.full(len(df), 50.0), index=df.index)
    spread_score = normalize_score(df["Spread (%)"], higher_is_better=False)
    oi_score = normalize_score(df["OI"], higher_is_better=True)
    volume_score = normalize_score(df["Vol"], higher_is_better=True)
    delta_centered = 1.0 - (df["Delta"].abs() - 0.45).abs()
    delta_score = normalize_score(delta_centered, higher_is_better=True)

    total = (
        cfg.confidence_weight_edge * edge_score
        + cfg.confidence_weight_significance * sig_score
        + cfg.confidence_weight_spread * spread_score
        + cfg.confidence_weight_oi * oi_score
        + cfg.confidence_weight_volume * volume_score
        + cfg.confidence_weight_delta * delta_score
    )
    return total.clip(0, 100)


def short_option_yield_metrics(action: str, option_type: str, S: float, K: float, credit: float, dte: int) -> Dict[str, float]:
    if action != "SELL":
        return {"Simple Yield (%)": np.nan, "Ann Yield (%)": np.nan}

    effective_dte = max(dte, 1)

    if option_type == "PUT":
        capital_base = K
    else:
        capital_base = S

    if capital_base <= 0:
        return {"Simple Yield (%)": np.nan, "Ann Yield (%)": np.nan}

    simple_yield = (credit / capital_base) * 100.0
    ann_yield = simple_yield * (365.0 / effective_dte)
    return {
        "Simple Yield (%)": simple_yield,
        "Ann Yield (%)": ann_yield,
    }


# ============================================================
# P&L PAYOFF COMPUTATION
# ============================================================
def compute_payoff_curve(
    legs: list,
    S: float,
    T: float,
    r: float,
    q: float,
    n_points: int = 80,
    sigma_range: float = 2.5,
) -> pd.DataFrame:
    """
    Compute P&L at expiration and mid-life across a range of spot prices.

    Each leg is a dict:
        {
            "strike": float,
            "option_type": "CALL" or "PUT",
            "action": "BUY" or "SELL",
            "premium": float,        # price paid/received per share
            "iv": float,             # implied vol for mid-life repricing
            "quantity": int,         # number of contracts (default 1)
        }

    Returns DataFrame with columns: Spot, Expiration P&L, Mid-Life P&L
    """
    if not legs:
        return pd.DataFrame()

    # Estimate vol for spot range from first leg
    avg_iv = np.mean([leg.get("iv", 0.30) for leg in legs])
    move = S * avg_iv * np.sqrt(max(T, 1 / 365)) * sigma_range
    spot_low = max(S - move, S * 0.5)
    spot_high = S + move
    spots = np.linspace(spot_low, spot_high, n_points)

    # Mid-life: reprice at T/2
    T_mid = max(T / 2.0, 1.0 / (365.0 * 24.0 * 60.0))

    expiry_pnl = np.zeros(n_points)
    midlife_pnl = np.zeros(n_points)

    for leg in legs:
        K = leg["strike"]
        opt_type = leg["option_type"].upper()
        action = leg["action"].upper()
        premium = leg["premium"]
        iv = leg.get("iv", 0.30)
        qty = leg.get("quantity", 1)
        sign = 1.0 if action == "BUY" else -1.0

        # Expiration payoff (intrinsic)
        if opt_type == "CALL":
            intrinsic = np.maximum(spots - K, 0.0)
        else:
            intrinsic = np.maximum(K - spots, 0.0)

        # P&L = sign * (intrinsic - premium)
        expiry_pnl += sign * (intrinsic - premium) * qty

        # Mid-life payoff (BS reprice)
        for idx, spot_i in enumerate(spots):
            calc = BlackScholesCalculator(S=spot_i, K=K, T=T_mid, r=r, sigma=iv, q=q)
            midlife_price = calc.price(opt_type)
            midlife_pnl[idx] += sign * (midlife_price - premium) * qty

    # Scale to per-contract (100 shares)
    return pd.DataFrame({
        "Spot": spots,
        "Expiration P&L": expiry_pnl * 100.0,
        "Mid-Life P&L": midlife_pnl * 100.0,
    })


def compute_scenario_table(
    legs: list,
    S: float,
    T: float,
    r: float,
    q: float,
    pct_moves: tuple = (-10, -5, -2, 0, 2, 5, 10),
) -> pd.DataFrame:
    """
    Show P&L at specific percentage moves from current spot.
    Returns a small summary table.
    """
    if not legs:
        return pd.DataFrame()

    T_mid = max(T / 2.0, 1.0 / (365.0 * 24.0 * 60.0))
    rows = []

    for pct in pct_moves:
        spot_i = S * (1.0 + pct / 100.0)

        expiry_pnl = 0.0
        midlife_pnl = 0.0
        net_delta = 0.0
        net_theta = 0.0

        for leg in legs:
            K = leg["strike"]
            opt_type = leg["option_type"].upper()
            action = leg["action"].upper()
            premium = leg["premium"]
            iv = leg.get("iv", 0.30)
            qty = leg.get("quantity", 1)
            sign = 1.0 if action == "BUY" else -1.0

            # Expiration
            if opt_type == "CALL":
                intrinsic = max(spot_i - K, 0.0)
            else:
                intrinsic = max(K - spot_i, 0.0)
            expiry_pnl += sign * (intrinsic - premium) * qty

            # Mid-life
            calc = BlackScholesCalculator(S=spot_i, K=K, T=T_mid, r=r, sigma=iv, q=q)
            midlife_pnl += sign * (calc.price(opt_type) - premium) * qty
            greeks = calc.greeks(opt_type)
            net_delta += sign * greeks["delta"] * qty
            net_theta += sign * greeks["theta"] * qty

        rows.append({
            "Spot Move": f"{pct:+d}%",
            "Spot": round(spot_i, 2),
            "Expiry P&L": round(expiry_pnl * 100.0, 2),
            "Mid-Life P&L": round(midlife_pnl * 100.0, 2),
            "Net Delta": round(net_delta, 3),
            "Net Theta": round(net_theta, 4),
        })

    return pd.DataFrame(rows)
