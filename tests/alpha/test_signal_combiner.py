"""Tests for signal combination and alpha blending."""

import numpy as np
import pandas as pd
import pytest

from python.alpha.signal_combiner import (
    CombineMethod,
    CrowdingReport,
    DecayProfile,
    SignalCombiner,
    SignalCrowdingDetector,
    SignalDecayAnalyzer,
    SignalMetadata,
    SignalPerformanceTracker,
    SignalSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_STOCKS = 50


def _make_signal(seed=42, n=N_STOCKS, ic=0.05):
    """Create a signal with controlled correlation to returns."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.02, n)
    noise = rng.normal(0, 0.02, n)
    signal = ic * returns + (1 - abs(ic)) * noise
    return signal, returns


def _make_signal_matrix(n_days=100, n_signals=3, seed=42):
    """Create a DataFrame of daily signal scores."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = rng.normal(0, 1, (n_days, n_signals))
    return pd.DataFrame(data, index=dates, columns=[f"sig_{i}" for i in range(n_signals)])


# ---------------------------------------------------------------------------
# SignalSnapshot
# ---------------------------------------------------------------------------


class TestSignalSnapshot:
    def test_ic_variance_floor(self):
        snap = SignalSnapshot(name="test", timestamp=pd.Timestamp("2024-01-01"), rolling_ic_std=0.0)
        assert snap.ic_variance >= 1e-8

    def test_ic_variance_normal(self):
        snap = SignalSnapshot(name="test", timestamp=pd.Timestamp("2024-01-01"), rolling_ic_std=0.05)
        assert snap.ic_variance == pytest.approx(0.0025)


# ---------------------------------------------------------------------------
# SignalPerformanceTracker
# ---------------------------------------------------------------------------


class TestPerformanceTracker:
    def test_update_and_snapshot(self):
        tracker = SignalPerformanceTracker(halflife=10)
        sig, ret = _make_signal()
        for i in range(30):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            s, r = _make_signal(seed=42 + i)
            tracker.update("momentum", date, s, r)

        snap = tracker.get_snapshot("momentum", pd.Timestamp("2024-02-01"))
        assert isinstance(snap, SignalSnapshot)
        assert snap.name == "momentum"

    def test_empty_snapshot(self):
        tracker = SignalPerformanceTracker()
        snap = tracker.get_snapshot("unknown", pd.Timestamp("2024-01-01"))
        assert snap.rolling_ic == 0.0
        assert snap.rolling_ic_std == 1.0

    def test_signal_names(self):
        tracker = SignalPerformanceTracker()
        sig, ret = _make_signal()
        tracker.update("alpha", pd.Timestamp("2024-01-01"), sig, ret)
        tracker.update("beta", pd.Timestamp("2024-01-01"), sig, ret)
        assert set(tracker.signal_names) == {"alpha", "beta"}

    def test_turnover_computed(self):
        tracker = SignalPerformanceTracker(halflife=10)
        for i in range(5):
            s, r = _make_signal(seed=i)
            tracker.update("sig", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), s, r)
        snap = tracker.get_snapshot("sig", pd.Timestamp("2024-01-06"))
        # Turnover should be between 0 and 2
        assert 0 <= snap.rolling_turnover <= 2.0


# ---------------------------------------------------------------------------
# SignalCrowdingDetector
# ---------------------------------------------------------------------------


class TestCrowdingDetector:
    def test_no_crowding_independent(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=100)
        mat = pd.DataFrame(
            rng.normal(0, 1, (100, 3)),
            index=dates,
            columns=["a", "b", "c"],
        )
        detector = SignalCrowdingDetector(correlation_threshold=0.7)
        report = detector.analyze(mat)
        assert isinstance(report, CrowdingReport)
        assert not report.is_crowded

    def test_crowding_detected(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=100)
        base = rng.normal(0, 1, 100)
        # All signals are nearly identical
        mat = pd.DataFrame(
            {
                "a": base + rng.normal(0, 0.01, 100),
                "b": base + rng.normal(0, 0.01, 100),
                "c": rng.normal(0, 1, 100),
            },
            index=dates,
        )
        detector = SignalCrowdingDetector(correlation_threshold=0.7)
        report = detector.analyze(mat)
        assert report.is_crowded
        assert len(report.crowded_pairs) >= 1

    def test_eigenvalue_ratio(self):
        mat = _make_signal_matrix(n_days=100, n_signals=3)
        detector = SignalCrowdingDetector()
        report = detector.analyze(mat)
        assert report.eigenvalue_ratio >= 1.0

    def test_single_signal(self):
        dates = pd.bdate_range("2024-01-01", periods=50)
        mat = pd.DataFrame({"only": np.random.randn(50)}, index=dates)
        detector = SignalCrowdingDetector()
        report = detector.analyze(mat)
        assert not report.is_crowded


# ---------------------------------------------------------------------------
# SignalDecayAnalyzer
# ---------------------------------------------------------------------------


class TestDecayAnalyzer:
    def test_basic_decay(self):
        rng = np.random.default_rng(42)
        n_days, n_stocks = 100, 30
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        tickers = [f"T{i}" for i in range(n_stocks)]

        signals = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=tickers)
        returns = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_stocks)), index=dates, columns=tickers)

        analyzer = SignalDecayAnalyzer(max_lag=10, min_samples=20)
        profile = analyzer.analyze(signals, returns, "test_signal")
        assert isinstance(profile, DecayProfile)
        assert len(profile.lag_ics) == 10

    def test_half_life_positive(self):
        rng = np.random.default_rng(42)
        n_days, n_stocks = 100, 30
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        tickers = [f"T{i}" for i in range(n_stocks)]

        # Signal with some predictive power
        returns = pd.DataFrame(rng.normal(0.001, 0.01, (n_days, n_stocks)), index=dates, columns=tickers)
        signals = returns.shift(1).fillna(0) + pd.DataFrame(
            rng.normal(0, 0.005, (n_days, n_stocks)), index=dates, columns=tickers
        )

        analyzer = SignalDecayAnalyzer(max_lag=10, min_samples=10)
        profile = analyzer.analyze(signals, returns, "predictive")
        assert profile.half_life_days > 0


# ---------------------------------------------------------------------------
# SignalCombiner — Equal weight
# ---------------------------------------------------------------------------


class TestCombinerEqualWeight:
    @pytest.fixture
    def combiner(self):
        c = SignalCombiner(method=CombineMethod.EQUAL_WEIGHT)
        c.register_signal(SignalMetadata(name="mom", category="momentum"))
        c.register_signal(SignalMetadata(name="val", category="value"))
        return c

    def test_equal_weights(self, combiner):
        signals = {
            "mom": np.random.randn(N_STOCKS),
            "val": np.random.randn(N_STOCKS),
        }
        composite, weights = combiner.combine(signals, pd.Timestamp("2024-01-01"))
        assert weights["mom"] == pytest.approx(0.5, abs=0.01)
        assert weights["val"] == pytest.approx(0.5, abs=0.01)

    def test_composite_shape(self, combiner):
        signals = {
            "mom": np.random.randn(N_STOCKS),
            "val": np.random.randn(N_STOCKS),
        }
        composite, _ = combiner.combine(signals, pd.Timestamp("2024-01-01"))
        assert composite.shape == (N_STOCKS,)

    def test_unregistered_signal_ignored(self, combiner):
        signals = {
            "mom": np.random.randn(N_STOCKS),
            "val": np.random.randn(N_STOCKS),
            "unknown": np.random.randn(N_STOCKS),
        }
        _, weights = combiner.combine(signals, pd.Timestamp("2024-01-01"))
        assert "unknown" not in weights


# ---------------------------------------------------------------------------
# SignalCombiner — Inverse variance
# ---------------------------------------------------------------------------


class TestCombinerInverseVariance:
    def test_stable_signal_gets_more_weight(self):
        combiner = SignalCombiner(method=CombineMethod.INVERSE_VARIANCE)
        combiner.register_signal(SignalMetadata(name="stable"))
        combiner.register_signal(SignalMetadata(name="noisy"))

        rng = np.random.default_rng(42)
        for i in range(50):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            ret = rng.normal(0, 0.02, N_STOCKS)
            # Stable signal has consistent IC
            stable_sig = 0.05 * ret + rng.normal(0, 0.005, N_STOCKS)
            # Noisy signal has erratic IC
            noisy_sig = rng.choice([-1, 1]) * 0.05 * ret + rng.normal(0, 0.02, N_STOCKS)

            combiner.update_performance(date, {"stable": stable_sig, "noisy": noisy_sig}, ret)

        signals = {
            "stable": rng.normal(0, 1, N_STOCKS),
            "noisy": rng.normal(0, 1, N_STOCKS),
        }
        _, weights = combiner.combine(signals, pd.Timestamp("2024-03-01"))
        # Stable signal should get more weight due to lower IC variance
        assert weights["stable"] > weights["noisy"]


# ---------------------------------------------------------------------------
# SignalCombiner — Bayesian model averaging
# ---------------------------------------------------------------------------


class TestCombinerBMA:
    def test_higher_ic_signal_gets_more_weight(self):
        combiner = SignalCombiner(method=CombineMethod.BAYESIAN_MODEL_AVG)
        combiner.register_signal(SignalMetadata(name="good"))
        combiner.register_signal(SignalMetadata(name="bad"))

        rng = np.random.default_rng(42)
        for i in range(50):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            ret = rng.normal(0, 0.02, N_STOCKS)
            good_sig = 0.1 * ret + rng.normal(0, 0.005, N_STOCKS)
            bad_sig = 0.01 * ret + rng.normal(0, 0.02, N_STOCKS)
            combiner.update_performance(date, {"good": good_sig, "bad": bad_sig}, ret)

        signals = {"good": rng.standard_normal(N_STOCKS), "bad": rng.standard_normal(N_STOCKS)}
        _, weights = combiner.combine(signals, pd.Timestamp("2024-03-01"))
        assert weights["good"] > weights["bad"]


# ---------------------------------------------------------------------------
# SignalCombiner — Mean-variance optimization
# ---------------------------------------------------------------------------


class TestCombinerMVO:
    def test_runs_without_error(self):
        combiner = SignalCombiner(method=CombineMethod.MEAN_VARIANCE_OPT)
        combiner.register_signal(SignalMetadata(name="a"))
        combiner.register_signal(SignalMetadata(name="b"))

        rng = np.random.default_rng(42)
        for i in range(50):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            ret = rng.normal(0, 0.02, N_STOCKS)
            combiner.update_performance(
                date,
                {"a": rng.standard_normal(N_STOCKS), "b": rng.standard_normal(N_STOCKS)},
                ret,
            )

        signals = {"a": rng.standard_normal(N_STOCKS), "b": rng.standard_normal(N_STOCKS)}
        composite, weights = combiner.combine(signals, pd.Timestamp("2024-03-01"))
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
        assert composite.shape == (N_STOCKS,)


# ---------------------------------------------------------------------------
# Regime-conditional weighting
# ---------------------------------------------------------------------------


class TestRegimeConditional:
    def test_regime_changes_method(self):
        combiner = SignalCombiner(method=CombineMethod.INVERSE_VARIANCE)
        combiner.register_signal(SignalMetadata(name="a"))
        combiner.register_signal(SignalMetadata(name="b"))

        signals = {"a": np.random.randn(N_STOCKS), "b": np.random.randn(N_STOCKS)}

        _, w_normal = combiner.combine(signals, pd.Timestamp("2024-01-01"), regime="normal")
        _, w_highvol = combiner.combine(signals, pd.Timestamp("2024-01-01"), regime="high_vol")

        # In high_vol regime, should use equal weight → both ~0.5
        assert w_highvol["a"] == pytest.approx(w_highvol["b"], abs=0.05)


# ---------------------------------------------------------------------------
# Crowding penalty
# ---------------------------------------------------------------------------


class TestCrowdingPenalty:
    def test_crowding_reduces_weight(self):
        combiner = SignalCombiner(method=CombineMethod.EQUAL_WEIGHT, crowding_threshold=0.5)
        combiner.register_signal(SignalMetadata(name="a"))
        combiner.register_signal(SignalMetadata(name="b"))
        combiner.register_signal(SignalMetadata(name="c"))

        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=100)
        base = rng.normal(0, 1, 100)
        signal_matrix = pd.DataFrame(
            {
                "a": base + rng.normal(0, 0.01, 100),
                "b": base + rng.normal(0, 0.01, 100),
                "c": rng.normal(0, 1, 100),
            },
            index=dates,
        )

        signals = {
            "a": rng.standard_normal(N_STOCKS),
            "b": rng.standard_normal(N_STOCKS),
            "c": rng.standard_normal(N_STOCKS),
        }

        _, weights = combiner.combine(
            signals, pd.Timestamp("2024-05-01"), signal_matrix=signal_matrix
        )
        # c should have higher weight since a and b are crowded
        assert weights["c"] > weights["a"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_registered_signals_raises(self):
        combiner = SignalCombiner()
        with pytest.raises(ValueError, match="No registered signals"):
            combiner.combine({"x": np.ones(10)}, pd.Timestamp("2024-01-01"))

    def test_weight_constraints(self):
        combiner = SignalCombiner(
            method=CombineMethod.EQUAL_WEIGHT, min_weight=0.1, max_weight=0.5
        )
        for i in range(5):
            combiner.register_signal(SignalMetadata(name=f"s{i}"))

        signals = {f"s{i}": np.random.randn(N_STOCKS) for i in range(5)}
        _, weights = combiner.combine(signals, pd.Timestamp("2024-01-01"))

        for w in weights.values():
            assert w >= 0.1 - 0.01  # Allow small numerical error
            assert w <= 0.5 + 0.01

    def test_weights_sum_to_one(self):
        combiner = SignalCombiner(method=CombineMethod.EQUAL_WEIGHT)
        combiner.register_signal(SignalMetadata(name="a"))
        combiner.register_signal(SignalMetadata(name="b"))
        combiner.register_signal(SignalMetadata(name="c"))

        signals = {
            "a": np.random.randn(N_STOCKS),
            "b": np.random.randn(N_STOCKS),
            "c": np.random.randn(N_STOCKS),
        }
        _, weights = combiner.combine(signals, pd.Timestamp("2024-01-01"))
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
