"""Tests for market regime detection."""

import numpy as np
import pytest

from python.alpha.regime_detection import (
    BreakpointResult,
    CUSUMDetector,
    CorrelationRegimeDetector,
    CorrelationRegimeResult,
    GaussianHMM,
    HMMResult,
    MarketRegime,
    RegimeDetector,
    RegimeState,
    VolatilityRegimeClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regime_returns(n=1000, seed=42):
    """Generate returns with clear regime changes."""
    rng = np.random.default_rng(seed)
    # Low vol regime (first third)
    r1 = rng.normal(0.0005, 0.005, n // 3)
    # High vol regime (middle third)
    r2 = rng.normal(-0.001, 0.025, n // 3)
    # Normal regime (last third)
    r3 = rng.normal(0.0003, 0.01, n - 2 * (n // 3))
    return np.concatenate([r1, r2, r3])


def _make_break_series(n=500, seed=42):
    """Series with a clear structural break in the mean."""
    rng = np.random.default_rng(seed)
    first_half = rng.normal(0.0, 1.0, n // 2)
    second_half = rng.normal(3.0, 1.0, n - n // 2)  # Mean shift of 3 sigma
    return np.concatenate([first_half, second_half])


def _make_correlated_returns(n=500, n_assets=10, seed=42):
    """Returns matrix where correlation increases over time."""
    rng = np.random.default_rng(seed)

    # First half: low correlation (independent returns)
    r1 = rng.normal(0, 0.01, (n // 2, n_assets))

    # Second half: high correlation (crisis-like)
    common_factor = rng.normal(0, 0.02, n - n // 2)
    r2 = np.column_stack([
        common_factor + rng.normal(0, 0.003, n - n // 2)
        for _ in range(n_assets)
    ])

    return np.vstack([r1, r2])


# ---------------------------------------------------------------------------
# Gaussian HMM
# ---------------------------------------------------------------------------


class TestGaussianHMM:
    def test_fit_returns_result(self):
        returns = _make_regime_returns()
        hmm = GaussianHMM(n_regimes=2)
        result = hmm.fit(returns)
        assert isinstance(result, HMMResult)
        assert result.n_regimes == 2

    def test_regime_sequence_length(self):
        returns = _make_regime_returns(n=500)
        hmm = GaussianHMM(n_regimes=2)
        result = hmm.fit(returns)
        assert len(result.regime_sequence) == 500

    def test_regime_probs_shape(self):
        n = 300
        returns = _make_regime_returns(n=n)
        hmm = GaussianHMM(n_regimes=3)
        result = hmm.fit(returns)
        assert result.regime_probs.shape == (n, 3)

    def test_regime_probs_sum_to_one(self):
        returns = _make_regime_returns()
        hmm = GaussianHMM(n_regimes=2)
        result = hmm.fit(returns)
        row_sums = result.regime_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_transition_matrix_row_sums(self):
        returns = _make_regime_returns()
        hmm = GaussianHMM(n_regimes=3)
        result = hmm.fit(returns)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_stationary_probs_sum_to_one(self):
        returns = _make_regime_returns()
        hmm = GaussianHMM(n_regimes=2)
        result = hmm.fit(returns)
        assert result.stationary_probs.sum() == pytest.approx(1.0, abs=0.01)

    def test_means_sorted_ascending(self):
        """Regimes should be sorted by volatility (std)."""
        returns = _make_regime_returns()
        hmm = GaussianHMM(n_regimes=3)
        result = hmm.fit(returns)
        # Stds should be sorted ascending after reordering
        for i in range(len(result.stds) - 1):
            assert result.stds[i] <= result.stds[i + 1]

    def test_detects_two_volatility_clusters(self):
        """With clear regime data, HMM should find distinct vol clusters."""
        returns = _make_regime_returns(n=1000)
        hmm = GaussianHMM(n_regimes=2)
        result = hmm.fit(returns)
        # The two regimes should have different stds
        assert result.stds[1] > result.stds[0] * 1.5

    def test_log_likelihood_finite(self):
        returns = _make_regime_returns()
        hmm = GaussianHMM(n_regimes=2)
        result = hmm.fit(returns)
        assert np.isfinite(result.log_likelihood)

    def test_viterbi_valid_states(self):
        returns = _make_regime_returns()
        hmm = GaussianHMM(n_regimes=3)
        result = hmm.fit(returns)
        assert set(result.regime_sequence).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Volatility Regime Classifier
# ---------------------------------------------------------------------------


class TestVolatilityRegimeClassifier:
    def test_classify_returns_array(self):
        returns = _make_regime_returns(n=500)
        vc = VolatilityRegimeClassifier(lookback=21, history_window=100)
        regimes = vc.classify(returns)
        assert len(regimes) == 500

    def test_valid_regime_values(self):
        returns = _make_regime_returns(n=500)
        vc = VolatilityRegimeClassifier(lookback=21, history_window=100)
        regimes = vc.classify(returns)
        assert all(r in (0, 1, 2, 3) for r in regimes)

    def test_crisis_during_high_vol(self):
        """High-vol period should have HIGH_VOL or CRISIS regimes."""
        returns = _make_regime_returns(n=600)
        vc = VolatilityRegimeClassifier(
            lookback=15, history_window=100, thresholds=(25, 50, 85)
        )
        regimes = vc.classify(returns)
        # Middle third is high vol
        high_vol_regimes = regimes[200:400]
        # At least some should be HIGH_VOL or CRISIS (after warmup)
        valid_idx = high_vol_regimes[high_vol_regimes >= 0]
        assert any(r >= MarketRegime.HIGH_VOL for r in valid_idx[-50:])

    def test_current_state(self):
        returns = _make_regime_returns(n=500)
        vc = VolatilityRegimeClassifier(lookback=21, history_window=100)
        state = vc.current_state(returns)
        assert isinstance(state, RegimeState)
        assert 0 <= state.confidence <= 1.0
        assert state.duration >= 1

    def test_short_series(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 10)
        vc = VolatilityRegimeClassifier()
        regimes = vc.classify(returns)
        assert len(regimes) == 10


# ---------------------------------------------------------------------------
# CUSUM Detector
# ---------------------------------------------------------------------------


class TestCUSUMDetector:
    def test_detects_mean_shift(self):
        y = _make_break_series()
        detector = CUSUMDetector(threshold=4.0, drift=0.5)
        result = detector.detect(y)
        assert isinstance(result, BreakpointResult)
        assert result.n_breaks >= 1

    def test_breakpoint_near_true_break(self):
        y = _make_break_series(n=500)
        detector = CUSUMDetector(threshold=4.0, drift=0.5)
        result = detector.detect(y)
        assert len(result.breakpoints) >= 1
        # Break should be near the midpoint
        assert any(200 < bp < 350 for bp in result.breakpoints)

    def test_no_break_in_stationary(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 500)
        detector = CUSUMDetector(threshold=5.0)
        result = detector.detect(y)
        assert result.n_breaks <= 1  # May detect at most 1 spurious break

    def test_cusum_values_shape(self):
        y = _make_break_series()
        detector = CUSUMDetector()
        result = detector.detect(y)
        assert len(result.cusum_values) == len(y)

    def test_short_series(self):
        detector = CUSUMDetector()
        result = detector.detect(np.array([1.0, 2.0, 3.0]))
        assert result.n_breaks == 0

    def test_min_spacing_respected(self):
        y = _make_break_series(n=500)
        detector = CUSUMDetector(threshold=3.0, min_spacing=50)
        result = detector.detect(y)
        for i in range(len(result.breakpoints) - 1):
            gap = result.breakpoints[i + 1] - result.breakpoints[i]
            assert gap >= 50


# ---------------------------------------------------------------------------
# Correlation Regime Detector
# ---------------------------------------------------------------------------


class TestCorrelationRegimeDetector:
    def test_analyze_returns_result(self):
        returns = _make_correlated_returns()
        detector = CorrelationRegimeDetector(lookback=30)
        result = detector.analyze(returns)
        assert isinstance(result, CorrelationRegimeResult)

    def test_absorption_ratio_bounded(self):
        returns = _make_correlated_returns()
        detector = CorrelationRegimeDetector(lookback=30)
        result = detector.analyze(returns)
        valid = result.absorption_ratio[~np.isnan(result.absorption_ratio)]
        assert all(0 <= a <= 1 for a in valid)

    def test_crisis_during_correlated_period(self):
        """High correlation period should show higher absorption ratio."""
        returns = _make_correlated_returns(n=400, n_assets=10)
        detector = CorrelationRegimeDetector(lookback=30, crisis_threshold=0.40)
        result = detector.analyze(returns)
        # Second half has high correlation
        first_half_abs = np.nanmean(result.absorption_ratio[:150])
        second_half_abs = np.nanmean(result.absorption_ratio[250:])
        assert second_half_abs > first_half_abs

    def test_single_asset_returns_empty(self):
        returns = np.random.default_rng(42).normal(0, 0.01, (100, 1))
        detector = CorrelationRegimeDetector()
        result = detector.analyze(returns)
        assert not result.is_crisis

    def test_regime_indicator_binary(self):
        returns = _make_correlated_returns()
        detector = CorrelationRegimeDetector(lookback=30)
        result = detector.analyze(returns)
        assert set(np.unique(result.regime_indicator)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Unified RegimeDetector
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    def test_fit_predict(self):
        returns = _make_regime_returns()
        detector = RegimeDetector(n_regimes=2)
        result = detector.fit_predict(returns)
        assert isinstance(result, HMMResult)

    def test_detect_breaks(self):
        y = _make_break_series()
        detector = RegimeDetector()
        result = detector.detect_breaks(y)
        assert isinstance(result, BreakpointResult)

    def test_classify_volatility(self):
        returns = _make_regime_returns(n=500)
        detector = RegimeDetector()
        regimes = detector.classify_volatility(returns)
        assert len(regimes) == 500

    def test_current_regime(self):
        returns = _make_regime_returns(n=500)
        detector = RegimeDetector()
        state = detector.current_regime(returns)
        assert isinstance(state, RegimeState)
        assert state.regime_name in (
            "Low Volatility", "Normal", "High Volatility", "Crisis"
        )


# ---------------------------------------------------------------------------
# MarketRegime enum
# ---------------------------------------------------------------------------


class TestMarketRegime:
    def test_ordering(self):
        assert MarketRegime.LOW_VOL < MarketRegime.NORMAL
        assert MarketRegime.NORMAL < MarketRegime.HIGH_VOL
        assert MarketRegime.HIGH_VOL < MarketRegime.CRISIS

    def test_values(self):
        assert MarketRegime.LOW_VOL == 0
        assert MarketRegime.CRISIS == 3
