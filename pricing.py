import math
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
    if len(hist) < lookback + 1:
        return None
    rv = hist["log_return"].tail(lookback).std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    if pd.isna(rv) or rv <= 0:
        return None
    return float(rv)


def build_forward_vol_forecast(hist: pd.DataFrame, cfg: ScannerConfig) -> Dict[str, Optional[float]]:
    rv20 = realized_vol_from_history(hist, 20)
    rv60 = realized_vol_from_history(hist, 60)
    rv120 = realized_vol_from_history(hist, 120)

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
    else:
        w = np.array(weights, dtype=float)
        w_sum = w.sum()
        if w_sum == 0:
            forecast = None
        else:
            w = w / w_sum
            forecast = float(np.dot(np.array(values, dtype=float), w))
            forecast *= cfg.vol_forecast_multiplier

    return {
        "rv20": rv20,
        "rv60": rv60,
        "rv120": rv120,
        "forecast_vol": forecast,
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
    if int(K) == int(S):
        return "ATM"

    option_type = option_type.upper()
    if option_type == "CALL":
        return "ITM" if K < S else "OTM"
    return "ITM" if K > S else "OTM"


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


def compute_confidence_score(df: pd.DataFrame, cfg: ScannerConfig) -> pd.Series:
    edge_score = normalize_score(df["Value Edge (%)"], higher_is_better=True)
    spread_score = normalize_score(df["Spread (%)"], higher_is_better=False)
    oi_score = normalize_score(df["OI"], higher_is_better=True)
    volume_score = normalize_score(df["Vol"], higher_is_better=True)
    delta_centered = 1.0 - (df["Delta"].abs() - 0.45).abs()
    delta_score = normalize_score(delta_centered, higher_is_better=True)

    total = (
        cfg.confidence_weight_edge * edge_score
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
