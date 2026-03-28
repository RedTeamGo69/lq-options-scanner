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


# ============================================================
# POSITION SIZING
# ============================================================
def compute_position_size(
    max_loss_per_contract: float,
    account_size: float,
    risk_per_trade_pct: float,
    method: str = "fixed_risk",
    edge_pct: float = 0.0,
    win_prob: float = 0.5,
) -> Dict[str, float]:
    """
    Compute suggested number of contracts.

    Methods:
      - fixed_risk: contracts = (account * risk_pct) / max_loss
      - half_kelly: contracts = (kelly_fraction / 2) * account / max_loss
        where kelly = (win_prob * payoff_ratio - lose_prob) / payoff_ratio

    Returns dict with suggested contracts, dollar risk, % of account.
    """
    if max_loss_per_contract <= 0 or account_size <= 0 or risk_per_trade_pct <= 0:
        return {
            "contracts": 0,
            "total_risk": 0.0,
            "pct_of_account": 0.0,
            "method": method,
        }

    # max_loss_per_contract is already in dollars (e.g. $300 for a $3 wide spread)
    max_dollar_risk = account_size * (risk_per_trade_pct / 100.0)

    if method == "half_kelly":
        # Kelly criterion: f* = (p * b - q) / b
        # where p = win_prob, q = 1 - p, b = payoff ratio (reward / risk)
        # We use edge_pct as a proxy: if edge > 0, favorable
        lose_prob = 1.0 - win_prob
        if lose_prob <= 0 or win_prob <= 0:
            kelly_frac = 0.0
        else:
            # payoff_ratio from edge: approximate as (credit / max_loss)
            # but we use win_prob directly for a cleaner formula
            # Kelly = (p * (1 + b) - 1) / b where b = reward/risk
            # Simplified: kelly = win_prob - lose_prob / payoff_ratio
            # With payoff_ratio estimated from edge_pct
            payoff_ratio = max(edge_pct / 100.0 + 1.0, 0.01)
            kelly_frac = max(0.0, (win_prob * payoff_ratio - lose_prob) / payoff_ratio)

        half_kelly = kelly_frac / 2.0
        kelly_dollar_risk = account_size * half_kelly
        # Cap at fixed risk limit
        dollar_risk = min(kelly_dollar_risk, max_dollar_risk)
    else:
        dollar_risk = max_dollar_risk

    contracts = int(dollar_risk / max_loss_per_contract)
    contracts = max(contracts, 0)

    actual_risk = contracts * max_loss_per_contract
    pct_of_account = (actual_risk / account_size) * 100.0 if account_size > 0 else 0.0

    return {
        "contracts": contracts,
        "total_risk": actual_risk,
        "pct_of_account": pct_of_account,
        "method": method,
    }
