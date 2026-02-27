"""Tests for cointegration-based pairs trading."""

import numpy as np
import pandas as pd
import pytest

from python.alpha.pairs import (
    PairResult,
    PairScanner,
    PairSignal,
    _adf_test,
    _estimate_half_life,
    _hurst_exponent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cointegrated_prices(n=500, seed=42):
    """Generate two cointegrated price series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)

    # Common stochastic trend
    trend = np.cumsum(rng.normal(0, 0.01, n))

    # A = trend + mean-reverting component
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = 0.95 * ou[t - 1] + rng.normal(0, 0.005)

    price_a = 100 * np.exp(trend + ou)
    price_b = 100 * np.exp(1.2 * trend + rng.normal(0, 0.005, n))

    return pd.DataFrame(
        {"STOCK_A": price_a, "STOCK_B": price_b},
        index=dates,
    )


def _make_independent_prices(n=500, seed=42):
    """Generate two independent (non-cointegrated) price series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)

    price_a = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    price_b = 100 * np.exp(np.cumsum(rng.normal(-0.0001, 0.015, n)))

    return pd.DataFrame(
        {"IND_A": price_a, "IND_B": price_b},
        index=dates,
    )


# ---------------------------------------------------------------------------
# ADF test
# ---------------------------------------------------------------------------


class TestADFTest:
    def test_stationary_series(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 200)
        t_stat, p = _adf_test(y)
        assert t_stat < -2.0
        assert p < 0.1

    def test_random_walk(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.normal(0, 1, 200))
        t_stat, p = _adf_test(y)
        assert p > 0.1

    def test_short_series(self):
        y = np.array([1.0, 2.0, 3.0])
        _, p = _adf_test(y)
        assert p == 1.0


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_mean_reverting_finite(self):
        rng = np.random.default_rng(42)
        ou = np.zeros(500)
        for t in range(1, 500):
            ou[t] = 0.95 * ou[t - 1] + rng.normal(0, 0.1)
        hl = _estimate_half_life(ou)
        assert 1 < hl < 100

    def test_random_walk_long(self):
        rng = np.random.default_rng(42)
        rw = np.cumsum(rng.normal(0, 1, 200))
        hl = _estimate_half_life(rw)
        # Random walk: half-life should be much longer than OU process
        assert hl > 10


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------


class TestHurstExponent:
    def test_returns_value(self):
        rng = np.random.default_rng(42)
        ou = np.zeros(500)
        for t in range(1, 500):
            ou[t] = 0.9 * ou[t - 1] + rng.normal(0, 0.1)
        h = _hurst_exponent(ou)
        # Hurst should be between 0 and 1
        assert 0.0 <= h <= 1.0

    def test_bounded(self):
        rng = np.random.default_rng(42)
        ts = rng.normal(0, 1, 200)
        h = _hurst_exponent(ts)
        assert 0.0 <= h <= 1.0

    def test_short_series(self):
        h = _hurst_exponent(np.array([1.0, 2.0, 3.0]))
        assert h == 0.5


# ---------------------------------------------------------------------------
# PairScanner - analyze_pair
# ---------------------------------------------------------------------------


class TestAnalyzePair:
    def test_cointegrated_pair(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        result = scanner.analyze_pair("STOCK_A", "STOCK_B")
        assert isinstance(result, PairResult)
        assert result.is_cointegrated
        assert result.adf_pvalue < 0.1

    def test_independent_pair(self):
        prices = _make_independent_prices()
        scanner = PairScanner(prices)
        result = scanner.analyze_pair("IND_A", "IND_B")
        assert not result.is_cointegrated

    def test_hedge_ratio_reasonable(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        result = scanner.analyze_pair("STOCK_A", "STOCK_B")
        assert 0.1 < abs(result.hedge_ratio) < 10

    def test_half_life_positive(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        result = scanner.analyze_pair("STOCK_A", "STOCK_B")
        assert result.half_life > 0

    def test_correlation_bounded(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        result = scanner.analyze_pair("STOCK_A", "STOCK_B")
        assert -1 <= result.correlation <= 1

    def test_mean_reversion_speed(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        result = scanner.analyze_pair("STOCK_A", "STOCK_B")
        assert result.mean_reversion_speed > 0

    def test_insufficient_history(self):
        prices = _make_cointegrated_prices(n=20)
        scanner = PairScanner(prices, min_history=100)
        result = scanner.analyze_pair("STOCK_A", "STOCK_B")
        assert not result.is_cointegrated


# ---------------------------------------------------------------------------
# PairScanner - find_cointegrated_pairs
# ---------------------------------------------------------------------------


class TestFindPairs:
    def test_finds_cointegrated(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        pairs = scanner.find_cointegrated_pairs(pvalue_threshold=0.1)
        assert len(pairs) >= 1
        assert pairs[0].is_cointegrated

    def test_sorted_by_pvalue(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        pairs = scanner.find_cointegrated_pairs(pvalue_threshold=0.5)
        for i in range(len(pairs) - 1):
            assert pairs[i].adf_pvalue <= pairs[i + 1].adf_pvalue

    def test_max_pairs_respected(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        pairs = scanner.find_cointegrated_pairs(max_pairs=1)
        assert len(pairs) <= 1

    def test_empty_for_strict_threshold(self):
        prices = _make_independent_prices()
        scanner = PairScanner(prices)
        pairs = scanner.find_cointegrated_pairs(pvalue_threshold=0.001)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


class TestGenerateSignal:
    def test_signal_structure(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        pair = scanner.analyze_pair("STOCK_A", "STOCK_B")
        signal = scanner.generate_signal(pair)
        assert isinstance(signal, PairSignal)
        assert signal.signal in (-1, 0, 1)

    def test_z_score_series(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        pair = scanner.analyze_pair("STOCK_A", "STOCK_B")
        signal = scanner.generate_signal(pair, lookback=30)
        assert isinstance(signal.z_score, pd.Series)
        assert len(signal.z_score) > 0

    def test_entry_exit_thresholds(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        pair = scanner.analyze_pair("STOCK_A", "STOCK_B")
        signal = scanner.generate_signal(pair, entry_z=1.5, exit_z=0.3)
        assert signal.entry_z == 1.5
        assert signal.exit_z == 0.3

    def test_current_z_finite(self):
        prices = _make_cointegrated_prices()
        scanner = PairScanner(prices)
        pair = scanner.analyze_pair("STOCK_A", "STOCK_B")
        signal = scanner.generate_signal(pair)
        assert np.isfinite(signal.current_z)
