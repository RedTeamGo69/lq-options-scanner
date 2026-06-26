import os
import re
from dataclasses import dataclass

import pytz

# ============================================================
# ENV / CONSTANTS
# ============================================================
TRADIER_BASE_URL = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")
TRADIER_FUNDAMENTALS_BASE_URL = os.getenv(
    "TRADIER_FUNDAMENTALS_BASE_URL", "https://api.tradier.com/beta"
).rstrip("/")
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org/fred").rstrip("/")

DB_PATH = os.getenv("LQ_SCANNER_DB_PATH", "lq_options_scanner.db")

NY_TZ = pytz.timezone("America/New_York")
TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0
T_FLOOR_YEARS = 1.0 / (365.0 * 24.0 * 60.0 * 60.0)  # 1 second
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')


# ============================================================
# CONFIG
# ============================================================
@dataclass
class ScannerConfig:
    min_open_interest: int = 100
    min_volume: int = 10
    min_bid: float = 0.10
    max_spread_pct: float = 15.0
    min_abs_delta: float = 0.10
    max_abs_delta: float = 0.85
    top_n: int = 25

    rv20_weight: float = 0.50
    rv60_weight: float = 0.30
    rv120_weight: float = 0.20
    vol_forecast_multiplier: float = 1.00

    # Realized-vol estimator: "yang_zhang" (OHLC range-based, lower variance) or
    # "close_close" (the original close-to-close fallback).
    vol_estimator: str = "yang_zhang"

    enable_earnings_vol_adj: bool = True
    expected_earnings_move: float = 0.05  # fallback when the term structure can't supply one

    enable_term_structure_scaling: bool = True

    use_executable_pricing: bool = True
    execution_slippage_pct: float = 0.0

    # --- Skew-aware fair value + vol-point edge ---
    smile_fit_degree: int = 2        # quadratic smile in log-moneyness (level/slope/curvature)
    min_smile_points: int = 5        # below this, fall back to a flat (skew-blind) fair vol
    # Noise gating: how many forecast-vol std errors the level edge must clear to
    # be treated as a real surface-level signal rather than estimation noise.
    noise_gate_z: float = 1.0
    # Vol-point thresholds for coloring the IV Edge column (green/red).
    iv_edge_green_volpts: float = 2.0
    iv_edge_red_volpts: float = -2.0

    confidence_weight_edge: float = 0.40
    confidence_weight_significance: float = 0.10
    confidence_weight_spread: float = 0.20
    confidence_weight_oi: float = 0.10
    confidence_weight_volume: float = 0.10
    confidence_weight_delta: float = 0.10

    iv_history_lookback_days: int = 252
