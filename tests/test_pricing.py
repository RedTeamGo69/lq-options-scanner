import math

import numpy as np
import pandas as pd
import pytest

from config import ScannerConfig
from pricing import (
    yang_zhang_vol,
    realized_vol_from_history,
    realized_vol,
    forecast_vol_uncertainty,
    build_forward_vol_forecast,
    fit_smile,
    iv_edge_components,
    dollar_edge_from_vol,
    compute_confidence_score,
)

TRADING_DAYS = 252.0


# ============================================================
# YANG-ZHANG REALIZED VOL
# ============================================================
def test_yang_zhang_recovers_true_vol(ohlc_factory):
    true_daily = 0.0189  # ~30% annualized
    true_ann = true_daily * math.sqrt(TRADING_DAYS)
    estimates = [yang_zhang_vol(ohlc_factory(260, true_daily, seed=s), 250) for s in range(15)]
    mean_est = float(np.mean(estimates))
    # Discrete intraday monitoring biases the range slightly low, but the
    # estimate should track the true vol within a loose tolerance.
    assert 0.80 * true_ann <= mean_est <= 1.12 * true_ann


def test_yang_zhang_is_steadier_than_close_to_close(ohlc_factory):
    true_daily = 0.0189
    yz, cc = [], []
    for s in range(15):
        df = ohlc_factory(260, true_daily, seed=s)
        yz.append(yang_zhang_vol(df, 250))
        cc.append(realized_vol_from_history(df, 250))
    assert np.std(yz) < np.std(cc)


def test_yang_zhang_returns_none_without_ohlc(ohlc_factory):
    df = ohlc_factory(60, 0.02, seed=1)[["close", "log_return"]]
    assert yang_zhang_vol(df, 40) is None


def test_realized_vol_dispatcher_falls_back_to_close_close(ohlc_factory):
    df = ohlc_factory(60, 0.02, seed=2)[["close", "log_return"]]
    rv = realized_vol(df, 40, estimator="yang_zhang")
    assert rv is not None
    assert rv == pytest.approx(realized_vol_from_history(df, 40))


def test_forecast_uncertainty_positive_and_floored():
    assert forecast_vol_uncertainty(0.30, [0.30]) >= 0.005
    # Dispersion across windows dominates when they disagree.
    disp = forecast_vol_uncertainty(0.30, [0.20, 0.30, 0.40])
    assert disp >= float(np.std([0.20, 0.30, 0.40], ddof=0)) - 1e-12


def test_build_forecast_returns_uncertainty(ohlc_factory):
    df = ohlc_factory(300, 0.0189, seed=0)
    pack = build_forward_vol_forecast(df, ScannerConfig())
    assert pack["forecast_vol"] is not None
    assert pack["forecast_vol_uncertainty"] is not None
    assert pack["forecast_vol_uncertainty"] > 0


# ============================================================
# SMILE FIT
# ============================================================
def _quadratic_smile(S, strikes, atm=0.30, slope=-0.15, curv=0.5):
    x = np.log(np.asarray(strikes, float) / S)
    return atm + curv * x ** 2 + slope * x


def test_fit_smile_recovers_quadratic_and_atm():
    S = 100.0
    strikes = np.arange(80, 121, 5.0)
    iv = _quadratic_smile(S, strikes)
    sm = fit_smile(strikes, iv, S, degree=2, min_points=5)
    assert not sm.is_flat
    assert sm.atm_value == pytest.approx(0.30, abs=1e-6)
    assert sm.residual_std == pytest.approx(0.0, abs=1e-6)
    # Fitted value at a wing strike matches the generating function.
    assert float(sm.iv_at(120)) == pytest.approx(_quadratic_smile(S, [120])[0], abs=1e-6)


def test_fit_smile_flat_fallback_when_too_few_points():
    S = 100.0
    strikes = np.array([95.0, 100.0, 105.0])
    iv = np.array([0.33, 0.31, 0.32])
    sm = fit_smile(strikes, iv, S, degree=2, min_points=5)
    assert sm.is_flat
    # Flat level is the ATM (nearest-spot) market IV.
    assert sm.atm_value == pytest.approx(0.31, abs=1e-9)


# ============================================================
# IV-SPACE EDGE
# ============================================================
def test_releveling_identity_fair_atm_equals_forecast():
    S = 100.0
    strikes = np.arange(80, 121, 5.0)
    sm = fit_smile(strikes, _quadratic_smile(S, strikes), S)
    fv = 0.25
    ed = iv_edge_components(float(sm.iv_at(S)), np.nan, np.nan, S, sm, fv, "SELL")
    assert ed["fair_iv"] == pytest.approx(fv, abs=1e-9)


def test_normal_skew_strike_has_zero_rv_edge_but_bump_does_not():
    S = 100.0
    strikes = np.arange(80, 121, 5.0)
    sm = fit_smile(strikes, _quadratic_smile(S, strikes), S)
    fv = 0.25
    on = iv_edge_components(float(sm.iv_at(120)), np.nan, np.nan, 120, sm, fv, "SELL")
    assert on["rv_edge"] == pytest.approx(0.0, abs=1e-6)  # wing not auto-flagged
    bumped = iv_edge_components(float(sm.iv_at(120)) + 0.05, np.nan, np.nan, 120, sm, fv, "SELL")
    assert bumped["rv_edge"] == pytest.approx(0.05, abs=1e-6)  # genuine dislocation


def test_level_edge_sign_by_action():
    S = 100.0
    strikes = np.arange(80, 121, 5.0)
    sm = fit_smile(strikes, _quadratic_smile(S, strikes), S)  # atm IV 0.30
    fv = 0.25  # market richer than forecast
    sell = iv_edge_components(float(sm.iv_at(S)), np.nan, np.nan, S, sm, fv, "SELL")
    buy = iv_edge_components(float(sm.iv_at(S)), np.nan, np.nan, S, sm, fv, "BUY")
    assert sell["level_edge"] > 0      # rich surface favors selling
    assert buy["level_edge"] < 0       # rich surface is bad for buying
    assert sell["level_edge"] == pytest.approx(-buy["level_edge"], abs=1e-9)


def test_executable_haircut_reduces_edge():
    S = 100.0
    strikes = np.arange(80, 121, 5.0)
    sm = fit_smile(strikes, _quadratic_smile(S, strikes), S)
    fv = 0.25
    mid = iv_edge_components(0.32, np.nan, np.nan, S, sm, fv, "SELL")["iv_edge"]
    with_haircut = iv_edge_components(0.32, 0.31, 0.33, S, sm, fv, "SELL")["iv_edge"]
    assert with_haircut < mid  # selling at the bid vol yields less edge than mid


def test_dollar_edge_sign_consistent_with_vol_edge():
    assert dollar_edge_from_vol(0.12, 0.045) == pytest.approx(0.12 * 4.5 * 100.0)
    assert np.sign(dollar_edge_from_vol(0.12, -0.03)) == -1
    assert math.isnan(dollar_edge_from_vol(np.nan, 0.03))


# ============================================================
# CONFIDENCE
# ============================================================
def test_confidence_uses_iv_edge_and_significance():
    df = pd.DataFrame({
        "IV Edge (vol pts)": [5.0, 0.0],
        "Signal Z": [2.0, 0.1],
        "Spread (%)": [3.0, 12.0],
        "OI": [5000, 50],
        "Vol": [800, 5],
        "Delta": [0.45, 0.12],
    })
    conf = compute_confidence_score(df, ScannerConfig())
    assert conf.iloc[0] > conf.iloc[1]  # high edge + significance + liquidity ranks first
    assert (conf >= 0).all() and (conf <= 100).all()
