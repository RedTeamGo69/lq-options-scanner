import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from config import ScannerConfig
from screening import (
    implied_earnings_move_from_term_structure,
    compute_term_structure_scaling_factor,
    screen_chain,
)

TODAY = date(2026, 6, 26)


def _exp(dte):
    return (TODAY + timedelta(days=dte)).isoformat()


def _term_df():
    # Contango curve with an earnings spike in the ~25 DTE expiry.
    return pd.DataFrame({
        "Expiration": [_exp(7), _exp(14), _exp(25), _exp(45), _exp(80)],
        "DTE": [7, 14, 25, 45, 80],
        "ATM Avg IV (%)": [28.0, 29.0, 42.0, 33.0, 34.0],
    })


# ============================================================
# EARNINGS MOVE FROM TERM STRUCTURE
# ============================================================
def test_implied_earnings_move_matches_variance_gap():
    term = _term_df()
    move = implied_earnings_move_from_term_structure(term, _exp(20), today=TODAY)
    # sqrt(0.42^2 * 25/365 - 0.29^2 * 14/365)
    expected = math.sqrt(0.42 ** 2 * 25 / 365.0 - 0.29 ** 2 * 14 / 365.0)
    assert move == pytest.approx(expected, abs=1e-4)


def test_implied_earnings_move_none_without_pre_expiry():
    term = _term_df()
    term = term[term["DTE"] >= 25]  # no expiry before earnings
    assert implied_earnings_move_from_term_structure(term, _exp(5), today=TODAY) is None


def test_implied_earnings_move_none_for_past_earnings():
    assert implied_earnings_move_from_term_structure(_term_df(), _exp(-3), today=TODAY) is None


# ============================================================
# TERM-STRUCTURE FACTOR (no double counting)
# ============================================================
def test_term_structure_factor_interpolates_and_clips():
    term = _term_df()
    f = compute_term_structure_scaling_factor(term, target_dte=45, earnings_dte=20, jump_var=0.0)
    assert 0.5 <= f <= 2.0


def test_stripping_earnings_lowers_factor_at_earnings_expiry():
    term = _term_df()
    move = implied_earnings_move_from_term_structure(term, _exp(20), today=TODAY)
    raw = compute_term_structure_scaling_factor(term, target_dte=25, earnings_dte=20, jump_var=0.0)
    stripped = compute_term_structure_scaling_factor(term, target_dte=25, earnings_dte=20, jump_var=move ** 2)
    # Removing the jump variance from the spiked expiry must not double-count it.
    assert stripped < raw


def test_zero_jump_var_ignores_earnings_dte():
    # With no jump to strip, the earnings_dte argument must not change the curve:
    # the factor equals the plain (earnings-unaware) interpolation.
    term = _term_df()
    plain = compute_term_structure_scaling_factor(term, target_dte=45)
    with_dte = compute_term_structure_scaling_factor(term, target_dte=45, earnings_dte=20, jump_var=0.0)
    assert with_dte == pytest.approx(plain, abs=1e-12)


def test_stripping_earnings_affects_all_post_earnings_expiries():
    # Every expiry dated after earnings carries the jump, so stripping changes
    # the whole post-earnings curve (not just the spiked tenor).
    term = _term_df()
    move = implied_earnings_move_from_term_structure(term, _exp(20), today=TODAY)
    raw = compute_term_structure_scaling_factor(term, target_dte=80, earnings_dte=20, jump_var=0.0)
    stripped = compute_term_structure_scaling_factor(term, target_dte=80, earnings_dte=20, jump_var=move ** 2)
    assert stripped != pytest.approx(raw, abs=1e-6)
    assert 0.5 <= stripped <= 2.0


def test_term_structure_factor_none_with_insufficient_points():
    term = _term_df().iloc[:1]
    assert compute_term_structure_scaling_factor(term, target_dte=30) is None


# ============================================================
# END-TO-END screen_chain
# ============================================================
def _synthetic_chain(S=100.0):
    strikes = np.arange(80, 121, 5.0)
    x = np.log(strikes / S)
    iv = 0.30 + 0.5 * x ** 2 - 0.15 * x  # skewed smile

    def side(otype, delta_sign):
        rows = []
        for k, v in zip(strikes, iv):
            rows.append(dict(
                option_type=otype, strike=k, bid=2.0, ask=2.1, volume=500,
                open_interest=2000, mid_iv=v, bid_iv=v - 0.005, ask_iv=v + 0.005,
                delta_mkt=delta_sign * 0.4, gamma_mkt=0.02, theta_mkt=-0.03, vega_mkt=0.12,
            ))
        return rows

    return pd.DataFrame(side("PUT", -1) + side("CALL", 1))


def test_screen_chain_emits_iv_space_columns_and_meta():
    out = screen_chain(
        _synthetic_chain(), S=100.0, r=0.045, q=0.0, T=30 / 365, dte=30,
        action="SELL", option_family="PUTS", forecast_vol=0.25, cfg=ScannerConfig(),
        forecast_vol_uncertainty=0.05,
    )
    for col in ["IV Edge (vol pts)", "RV Edge (vol pts)", "Level Edge (vol pts)", "$ Edge", "Signal Z"]:
        assert col in out.columns
    meta = out.attrs["scan_meta"]
    assert meta["market_atm_iv"] == pytest.approx(0.30, abs=1e-6)
    assert meta["fv_atm"] == pytest.approx(0.25, abs=1e-9)
    # Surface is rich vs the 25% forecast -> +5 vol points (favorable for SELL).
    assert meta["level_edge_volpts"] == pytest.approx(5.0, abs=1e-6)


def test_screen_chain_smooth_smile_has_negligible_rv_edge():
    out = screen_chain(
        _synthetic_chain(), S=100.0, r=0.045, q=0.0, T=30 / 365, dte=30,
        action="SELL", option_family="PUTS", forecast_vol=0.25, cfg=ScannerConfig(),
        forecast_vol_uncertainty=0.05,
    )
    # On a smooth smile, no strike (incl. wings) should be flagged as dislocated.
    assert out["RV Edge (vol pts)"].abs().max() < 0.5


def test_screen_chain_dollar_edge_matches_vega_times_vol_edge():
    out = screen_chain(
        _synthetic_chain(), S=100.0, r=0.045, q=0.0, T=30 / 365, dte=30,
        action="SELL", option_family="PUTS", forecast_vol=0.25, cfg=ScannerConfig(),
        forecast_vol_uncertainty=0.05,
    )
    row = out.iloc[0]
    assert row["$ Edge"] == pytest.approx(0.12 * row["IV Edge (vol pts)"] * 100.0, rel=1e-6)


def test_screen_chain_injected_rich_strike_is_flagged():
    chain = _synthetic_chain()
    # Bump one OTM put's IV well above its smile neighbors.
    mask = (chain["option_type"] == "PUT") & (chain["strike"] == 90.0)
    chain.loc[mask, ["mid_iv", "bid_iv", "ask_iv"]] += 0.06
    out = screen_chain(
        chain, S=100.0, r=0.045, q=0.0, T=30 / 365, dte=30,
        action="SELL", option_family="PUTS", forecast_vol=0.25, cfg=ScannerConfig(),
        forecast_vol_uncertainty=0.05,
    )
    bumped = out[out["Strike"] == 90.0].iloc[0]
    assert bumped["RV Edge (vol pts)"] > 2.0  # genuine dislocation surfaces


def test_screen_chain_noise_gate_z_scales_with_uncertainty():
    args = dict(
        chain_df=_synthetic_chain(), S=100.0, r=0.045, q=0.0, T=30 / 365, dte=30,
        action="SELL", option_family="PUTS", forecast_vol=0.25, cfg=ScannerConfig(),
    )
    tight = screen_chain(**args, forecast_vol_uncertainty=0.01).attrs["scan_meta"]["level_z"]
    loose = screen_chain(**args, forecast_vol_uncertainty=0.10).attrs["scan_meta"]["level_z"]
    # Same edge, larger uncertainty -> lower significance.
    assert tight > loose
