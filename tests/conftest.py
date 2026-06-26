"""Shared test helpers.

The screening module pulls in data/database, which decorate functions with
Streamlit caching; running outside a Streamlit runtime just logs harmless
warnings. Filter them so test output stays readable.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger("streamlit").setLevel(logging.ERROR)


def simulate_ohlc(n_days, daily_vol, overnight_frac=0.3, steps=64, seed=0, start=100.0):
    """Simulate OHLC bars from an intraday geometric Brownian path so that
    range-based estimators have a well-defined ground-truth daily vol."""
    rng = np.random.default_rng(seed)
    on_vol = daily_vol * np.sqrt(overnight_frac)
    id_vol = daily_vol * np.sqrt(1.0 - overnight_frac)
    rows = []
    prev_close = start
    for _ in range(n_days):
        o = prev_close * np.exp(rng.normal(0.0, on_vol))
        path = o * np.exp(np.cumsum(rng.normal(0.0, id_vol / np.sqrt(steps), steps)))
        rows.append((o, max(o, path.max()), min(o, path.min()), path[-1]))
        prev_close = path[-1]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


@pytest.fixture
def ohlc_factory():
    return simulate_ohlc
