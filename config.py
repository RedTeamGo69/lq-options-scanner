import os
import re
from dataclasses import dataclass

import pytz

# ============================================================
# ENV / CONSTANTS
# ============================================================
TRADIER_BASE_URL = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")
TRADIER_API_KEY = os.getenv("TRADIER_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE_URL = os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org/fred").rstrip("/")

DB_PATH = os.getenv("LQ_SCANNER_DB_PATH", "lq_options_scanner.db")

NY_TZ = pytz.timezone("America/New_York")
TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0
T_FLOOR_YEARS = 1.0 / (365.0 * 24.0 * 60.0 * 60.0)  # 1 second
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}$')


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

    enable_earnings_vol_adj: bool = True
    expected_earnings_move: float = 0.05

    enable_term_structure_scaling: bool = True

    use_executable_pricing: bool = True
    execution_slippage_pct: float = 0.0

    confidence_weight_edge: float = 0.45
    confidence_weight_spread: float = 0.20
    confidence_weight_oi: float = 0.15
    confidence_weight_volume: float = 0.10
    confidence_weight_delta: float = 0.10

    iv_history_lookback_days: int = 252

    # Vertical Spreads
    enable_spread_scanner: bool = True
    spread_max_width: int = 10         # max strike width in dollars
    spread_min_credit: float = 0.10    # min net credit for credit spreads
    spread_top_n: int = 15             # top N spreads to display

    # Position Sizing
    account_size: float = 10000.0
    risk_per_trade_pct: float = 2.0    # max % of account to risk
    sizing_method: str = "fixed_risk"  # "fixed_risk" or "half_kelly"
