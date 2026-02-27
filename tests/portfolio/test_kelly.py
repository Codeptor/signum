"""Tests for Kelly criterion position sizing."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.kelly import (
    fractional_kelly,
    full_kelly,
    kelly_edge_sizing,
    kelly_from_predictions,
    kelly_growth_rate,
)


@pytest.fixture
def tickers():
    return ["AAPL", "MSFT", "GOOG"]


@pytest.fixture
def expected_returns(tickers):
    return pd.Series([0.10, 0.08, 0.12], index=tickers)


@pytest.fixture
def covariance(tickers):
    """Simple covariance matrix with moderate correlation."""
    cov = np.array([
        [0.04, 0.01, 0.015],
        [0.01, 0.03, 0.01],
        [0.015, 0.01, 0.05],
    ])
    return pd.DataFrame(cov, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Full Kelly
# ---------------------------------------------------------------------------


class TestFullKelly:
    def test_returns_series(self, expected_returns, covariance):
        w = full_kelly(expected_returns, covariance)
        assert isinstance(w, pd.Series)
        assert len(w) == 3

    def test_positive_edge_positive_weights(self, expected_returns, covariance):
        """With positive expected returns, full Kelly should produce positive weights."""
        w = full_kelly(expected_returns, covariance)
        assert (w > 0).all()

    def test_zero_returns_zero_weights(self, covariance):
        mu = pd.Series([0.0, 0.0, 0.0], index=covariance.index)
        w = full_kelly(mu, covariance)
        np.testing.assert_array_almost_equal(w.values, 0.0)

    def test_risk_free_rate(self, expected_returns, covariance):
        """Higher risk-free rate should reduce weights."""
        w0 = full_kelly(expected_returns, covariance, risk_free_rate=0.0)
        w5 = full_kelly(expected_returns, covariance, risk_free_rate=0.05)
        assert w0.sum() > w5.sum()


# ---------------------------------------------------------------------------
# Fractional Kelly
# ---------------------------------------------------------------------------


class TestFractionalKelly:
    def test_fraction_scales_down(self, expected_returns, covariance):
        w_full = full_kelly(expected_returns, covariance)
        w_quarter = fractional_kelly(expected_returns, covariance, fraction=0.25, long_only=False)
        # Quarter kelly should be ~0.25x the full (before normalization)
        # Can't test exactly due to normalization, but should be smaller
        assert w_quarter.sum() <= w_full.sum() + 0.01

    def test_sums_to_at_most_one(self, expected_returns, covariance):
        w = fractional_kelly(expected_returns, covariance, fraction=0.5)
        assert w.sum() <= 1.0 + 1e-6

    def test_long_only(self, covariance):
        """Long-only constraint should eliminate negative weights."""
        mu = pd.Series([0.10, -0.05, 0.08], index=covariance.index)
        w = fractional_kelly(mu, covariance, fraction=0.5, long_only=True)
        assert (w >= 0).all()

    def test_max_weight(self, expected_returns, covariance):
        w = fractional_kelly(expected_returns, covariance, fraction=1.0, max_weight=0.20)
        assert w.max() <= 0.20 + 1e-6

    def test_zero_returns_equal_weight(self, covariance):
        """Zero returns should produce equal weight fallback."""
        mu = pd.Series([0.0, 0.0, 0.0], index=covariance.index)
        w = fractional_kelly(mu, covariance)
        expected = 1.0 / 3
        np.testing.assert_array_almost_equal(w.values, expected)

    def test_half_kelly(self, expected_returns, covariance):
        w_quarter = fractional_kelly(expected_returns, covariance, fraction=0.25)
        w_half = fractional_kelly(expected_returns, covariance, fraction=0.50)
        # Half Kelly should invest more than quarter Kelly
        assert w_half.sum() >= w_quarter.sum() - 0.01


# ---------------------------------------------------------------------------
# Kelly from predictions
# ---------------------------------------------------------------------------


class TestKellyFromPredictions:
    def test_basic(self, covariance):
        preds = pd.Series([0.05, 0.03, 0.08], index=covariance.index)
        pred_std = pd.Series([0.02, 0.02, 0.02], index=covariance.index)
        w = kelly_from_predictions(preds, pred_std, covariance)
        assert isinstance(w, pd.Series)
        assert (w >= 0).all()
        assert w.sum() <= 1.0 + 1e-6

    def test_confidence_scaling(self, covariance):
        """Higher confidence (lower std) should increase allocation."""
        preds = pd.Series([0.05, 0.05, 0.05], index=covariance.index)
        high_conf = pd.Series([0.01, 0.05, 0.05], index=covariance.index)
        w = kelly_from_predictions(preds, high_conf, covariance, confidence_scaling=True)
        # AAPL has lowest std → highest confidence → should get largest weight
        assert w.iloc[0] > w.iloc[1]

    def test_max_weight_respected(self, covariance):
        preds = pd.Series([0.10, 0.02, 0.02], index=covariance.index)
        pred_std = pd.Series([0.01, 0.05, 0.05], index=covariance.index)
        w = kelly_from_predictions(preds, pred_std, covariance, max_weight=0.25)
        assert w.max() <= 0.25 + 1e-6


# ---------------------------------------------------------------------------
# Kelly edge sizing
# ---------------------------------------------------------------------------


class TestKellyEdgeSizing:
    def test_basic(self):
        edges = pd.Series({"A": 0.02, "B": 0.01, "C": 0.03})
        variances = pd.Series({"A": 0.04, "B": 0.03, "C": 0.05})
        w = kelly_edge_sizing(edges, variances)
        assert isinstance(w, pd.Series)
        assert (w >= 0).all()
        assert w.sum() <= 1.0 + 1e-6

    def test_higher_edge_higher_weight(self):
        edges = pd.Series({"A": 0.10, "B": 0.01})
        variances = pd.Series({"A": 0.04, "B": 0.04})
        w = kelly_edge_sizing(edges, variances, fraction=0.5)
        assert w["A"] > w["B"]

    def test_max_weight(self):
        edges = pd.Series({"A": 1.0, "B": 1.0})
        variances = pd.Series({"A": 0.01, "B": 0.01})
        w = kelly_edge_sizing(edges, variances, max_weight=0.30)
        assert w.max() <= 0.30 + 1e-6

    def test_negative_edge_clipped(self):
        edges = pd.Series({"A": 0.05, "B": -0.02})
        variances = pd.Series({"A": 0.04, "B": 0.04})
        w = kelly_edge_sizing(edges, variances)
        assert w["B"] == 0.0


# ---------------------------------------------------------------------------
# Growth rate
# ---------------------------------------------------------------------------


class TestGrowthRate:
    def test_positive_for_good_allocation(self, expected_returns, covariance):
        w = fractional_kelly(expected_returns, covariance, fraction=0.25)
        g = kelly_growth_rate(w, expected_returns, covariance)
        assert g > 0

    def test_zero_weights_zero_growth(self, expected_returns, covariance):
        w = pd.Series([0.0, 0.0, 0.0], index=expected_returns.index)
        g = kelly_growth_rate(w, expected_returns, covariance)
        assert g == pytest.approx(0.0)

    def test_full_kelly_maximizes(self, expected_returns, covariance):
        """Full Kelly should have higher growth rate than random allocation."""
        w_kelly = full_kelly(expected_returns, covariance)
        w_random = pd.Series([0.3, 0.3, 0.4], index=expected_returns.index)
        g_kelly = kelly_growth_rate(w_kelly, expected_returns, covariance)
        g_random = kelly_growth_rate(w_random, expected_returns, covariance)
        assert g_kelly >= g_random - 1e-6
