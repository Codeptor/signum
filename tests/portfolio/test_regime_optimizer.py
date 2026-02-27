"""Tests for regime-conditional portfolio optimizer."""

import numpy as np
import pandas as pd
import pytest

from python.monitoring.hmm_regime import HMMRegimeDetector, HMMRegimeState
from python.portfolio.regime_optimizer import (
    DEFAULT_REGIME_METHODS,
    RegimeConditionalOptimizer,
)


@pytest.fixture
def price_data():
    """Synthetic price data for 5 assets."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.randn(252, 5) * 0.01, axis=0)) * 100,
        index=dates,
        columns=tickers,
    )
    return prices


@pytest.fixture
def market_returns():
    """Realistic market returns for HMM fitting (regime-switching process)."""
    np.random.seed(42)
    n = 500
    returns = np.empty(n)
    # Simulate regime switching: calm → volatile → calm
    for i in range(n):
        if i < 200:
            returns[i] = np.random.normal(0.0005, 0.008)
        elif i < 350:
            returns[i] = np.random.normal(-0.0002, 0.025)
        else:
            returns[i] = np.random.normal(0.0003, 0.010)
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(returns, index=dates)


def _mock_detector(regime="normal", exposure=0.7):
    """Create a mock detector that always returns the given regime."""

    class MockDetector:
        _fitted = True

        def predict_regime(self_, returns):
            probs = {"low_vol": 0.0, "normal": 0.0, "high_vol": 0.0}
            probs[regime] = 0.85
            remaining = sorted(set(probs.keys()) - {regime})
            probs[remaining[0]] = 0.10
            probs[remaining[1]] = 0.05
            regime_ids = {"low_vol": 0, "normal": 1, "high_vol": 2}
            return HMMRegimeState(
                regime=regime,
                regime_id=regime_ids[regime],
                probabilities=probs,
                exposure_multiplier=exposure,
                message=f"mock {regime}",
            )

    return MockDetector()


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestRegimeOptimizerInit:
    def test_default_regime_methods(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector(),
        )
        assert opt.regime_methods == DEFAULT_REGIME_METHODS

    def test_custom_regime_methods(self, price_data, market_returns):
        custom = {"high_vol": "risk_parity", "normal": "nco", "low_vol": "hrp"}
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            regime_methods=custom,
            hmm_detector=_mock_detector(),
        )
        assert opt.regime_methods == custom

    def test_auto_fits_hmm(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data, market_returns=market_returns
        )
        assert opt.hmm_detector is not None
        assert opt.hmm_detector._fitted

    def test_accepts_pre_fitted_detector(self, price_data, market_returns):
        det = _mock_detector()
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=det,
        )
        assert opt.hmm_detector is det


# ---------------------------------------------------------------------------
# Tests: Regime Detection
# ---------------------------------------------------------------------------


class TestRegimeDetection:
    def test_detect_regime_returns_state(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("high_vol", 0.3),
        )
        state = opt.detect_regime()
        assert isinstance(state, HMMRegimeState)
        assert state.regime == "high_vol"

    def test_regime_has_probabilities(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector(),
        )
        state = opt.detect_regime()
        assert sum(state.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_real_hmm_detects_regime(self, price_data, market_returns):
        """Integration: real HMM detector produces a valid regime."""
        opt = RegimeConditionalOptimizer(
            prices=price_data, market_returns=market_returns
        )
        state = opt.detect_regime()
        assert state.regime in ("low_vol", "normal", "high_vol")


# ---------------------------------------------------------------------------
# Tests: Optimization
# ---------------------------------------------------------------------------


class TestOptimize:
    def test_optimize_returns_weights_and_state(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("low_vol", 1.0),
        )
        weights, state = opt.optimize()
        assert isinstance(weights, pd.Series)
        assert isinstance(state, HMMRegimeState)

    def test_weights_non_negative(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("low_vol", 1.0),
        )
        weights, _ = opt.optimize()
        assert (weights >= -0.01).all()

    def test_weights_sum_le_one(self, price_data, market_returns):
        """Weights sum <= 1.0 (exposure multiplier may reduce total)."""
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("normal", 0.7),
        )
        weights, _ = opt.optimize()
        assert weights.sum() <= 1.0 + 1e-6

    def test_high_vol_uses_hrp_with_reduced_exposure(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("high_vol", 0.3),
        )
        weights, state = opt.optimize()
        assert state.regime == "high_vol"
        # Weights should be scaled by 0.3 exposure
        assert weights.sum() < 0.35

    def test_normal_uses_min_cvar(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("normal", 0.7),
        )
        weights, state = opt.optimize()
        assert state.regime == "normal"
        assert weights.sum() == pytest.approx(0.7, abs=0.01)

    def test_low_vol_uses_herc(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("low_vol", 1.0),
        )
        weights, state = opt.optimize()
        assert state.regime == "low_vol"
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_max_weight_respected(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            max_weight=0.25,
            hmm_detector=_mock_detector("low_vol", 1.0),
        )
        weights, _ = opt.optimize()
        assert weights.max() <= 0.25 + 1e-6


# ---------------------------------------------------------------------------
# Tests: Turnover
# ---------------------------------------------------------------------------


class TestTurnover:
    def test_low_turnover_keeps_current(self, price_data, market_returns):
        current = pd.Series(np.ones(5) / 5, index=price_data.columns)
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            current_weights=current,
            turnover_threshold=0.99,
            hmm_detector=_mock_detector("low_vol", 1.0),
        )
        weights, _ = opt.optimize_with_turnover()
        pd.testing.assert_series_equal(weights, current)

    def test_high_turnover_rebalances(self, price_data, market_returns):
        current = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=price_data.columns)
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            current_weights=current,
            turnover_threshold=0.01,
            hmm_detector=_mock_detector("low_vol", 1.0),
        )
        weights, _ = opt.optimize_with_turnover()
        assert weights.max() < 0.99

    def test_no_current_weights_returns_new(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector(),
        )
        weights, state = opt.optimize_with_turnover()
        assert isinstance(weights, pd.Series)


# ---------------------------------------------------------------------------
# Tests: Compare & Serialize
# ---------------------------------------------------------------------------


class TestCompareAndSerialize:
    def test_compare_methods_returns_dataframe(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector(),
        )
        comparison = opt.compare_methods()
        assert isinstance(comparison, pd.DataFrame)
        assert "hrp" in comparison.columns
        assert "herc" in comparison.columns

    def test_compare_methods_has_regime_attrs(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector(),
        )
        comparison = opt.compare_methods()
        assert "regime" in comparison.attrs
        assert "selected_method" in comparison.attrs

    def test_to_json_structure(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector(),
        )
        result = opt.to_json()
        assert "regime" in result
        assert "regime_probabilities" in result
        assert "exposure_multiplier" in result
        assert "selected_method" in result
        assert "regime_methods" in result

    def test_to_json_regime_valid(self, price_data, market_returns):
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            hmm_detector=_mock_detector("high_vol", 0.3),
        )
        result = opt.to_json()
        assert result["regime"] == "high_vol"
        assert result["selected_method"] == "hrp"

    def test_bl_fallback_without_views(self, price_data, market_returns):
        """BL method without views should fall back to HRP."""
        opt = RegimeConditionalOptimizer(
            prices=price_data,
            market_returns=market_returns,
            regime_methods={
                "high_vol": "hrp",
                "normal": "min_cvar",
                "low_vol": "black_litterman",
            },
            hmm_detector=_mock_detector("low_vol", 1.0),
        )
        weights, _ = opt.optimize()
        assert abs(weights.sum() - 1.0) < 1e-6
