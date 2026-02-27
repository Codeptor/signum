"""Tests for market microstructure analysis."""

import numpy as np
import pytest

from python.execution.microstructure import (
    IntradayPattern,
    KyleLambdaEstimator,
    KyleLambdaResult,
    OrderBookImbalance,
    VPINEstimator,
    VPINResult,
    WeightScheme,
    microprice,
    roll_effective_spread,
    roll_spread_rolling,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_book_data(n=100, depth=5, seed=42):
    """Generate synthetic order book snapshots."""
    rng = np.random.default_rng(seed)
    bid_vol = rng.integers(100, 1000, size=(n, depth)).astype(float)
    ask_vol = rng.integers(100, 1000, size=(n, depth)).astype(float)
    return bid_vol, ask_vol


def _make_trade_data(n=10000, seed=42):
    """Generate synthetic trade data with directional flow."""
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 0.01, n))
    volumes = rng.integers(10, 200, size=n).astype(float)
    return prices, volumes


# ---------------------------------------------------------------------------
# Order Book Imbalance
# ---------------------------------------------------------------------------


class TestOrderBookImbalance:
    def test_output_shape(self):
        bid, ask = _make_book_data(n=50, depth=5)
        obi = OrderBookImbalance(depth=5)
        result = obi.compute(bid, ask)
        assert result.shape == (50,)

    def test_bounded(self):
        bid, ask = _make_book_data()
        obi = OrderBookImbalance()
        result = obi.compute(bid, ask)
        assert all(-1.0 <= v <= 1.0 for v in result)

    def test_all_bid_positive(self):
        """When bid >> ask, OBI should be close to +1."""
        bid = np.full((10, 3), 1000.0)
        ask = np.full((10, 3), 1.0)
        obi = OrderBookImbalance(depth=3)
        result = obi.compute(bid, ask)
        assert all(v > 0.9 for v in result)

    def test_all_ask_negative(self):
        """When ask >> bid, OBI should be close to -1."""
        bid = np.full((10, 3), 1.0)
        ask = np.full((10, 3), 1000.0)
        obi = OrderBookImbalance(depth=3)
        result = obi.compute(bid, ask)
        assert all(v < -0.9 for v in result)

    def test_balanced_zero(self):
        vol = np.full((10, 3), 500.0)
        obi = OrderBookImbalance(depth=3)
        result = obi.compute(vol, vol)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_uniform_scheme(self):
        bid, ask = _make_book_data()
        obi = OrderBookImbalance(scheme=WeightScheme.UNIFORM)
        result = obi.compute(bid, ask)
        assert result.shape == (100,)

    def test_inverse_distance_scheme(self):
        bid, ask = _make_book_data()
        obi = OrderBookImbalance(scheme=WeightScheme.INVERSE_DISTANCE)
        result = obi.compute(bid, ask)
        assert result.shape == (100,)

    def test_zero_volume_returns_zero(self):
        bid = np.zeros((5, 3))
        ask = np.zeros((5, 3))
        obi = OrderBookImbalance(depth=3)
        result = obi.compute(bid, ask)
        np.testing.assert_allclose(result, 0.0)


# ---------------------------------------------------------------------------
# Microprice
# ---------------------------------------------------------------------------


class TestMicroprice:
    def test_between_bid_ask(self):
        bid_p = np.array([99.0, 99.5])
        ask_p = np.array([101.0, 100.5])
        bid_v = np.array([500.0, 500.0])
        ask_v = np.array([500.0, 500.0])
        mp = microprice(bid_p, ask_p, bid_v, ask_v)
        assert all(bid_p[i] <= mp[i] <= ask_p[i] for i in range(len(mp)))

    def test_bid_heavy_near_ask(self):
        """When bid volume >> ask, microprice closer to ask."""
        mp = microprice(
            np.array([99.0]), np.array([101.0]),
            np.array([10000.0]), np.array([1.0]),
        )
        assert mp[0] > 100.5

    def test_ask_heavy_near_bid(self):
        """When ask volume >> bid, microprice closer to bid."""
        mp = microprice(
            np.array([99.0]), np.array([101.0]),
            np.array([1.0]), np.array([10000.0]),
        )
        assert mp[0] < 99.5

    def test_equal_volumes_is_midpoint(self):
        mp = microprice(
            np.array([99.0]), np.array([101.0]),
            np.array([500.0]), np.array([500.0]),
        )
        assert mp[0] == pytest.approx(100.0, abs=0.01)

    def test_alpha_parameter(self):
        mp1 = microprice(
            np.array([99.0]), np.array([101.0]),
            np.array([100.0]), np.array([400.0]),
            alpha=1.0,
        )
        mp2 = microprice(
            np.array([99.0]), np.array([101.0]),
            np.array([100.0]), np.array([400.0]),
            alpha=0.5,
        )
        # Both should be between bid and ask but different
        assert 99.0 <= mp1[0] <= 101.0
        assert 99.0 <= mp2[0] <= 101.0


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------


class TestVPIN:
    def test_returns_result(self):
        prices, volumes = _make_trade_data(n=5000)
        vpin = VPINEstimator(bucket_volume=500, n_buckets=20, sigma_window=20)
        result = vpin.compute(prices, volumes)
        assert isinstance(result, VPINResult)

    def test_bounded(self):
        prices, volumes = _make_trade_data(n=5000)
        vpin = VPINEstimator(bucket_volume=500, n_buckets=20, sigma_window=20)
        result = vpin.compute(prices, volumes)
        valid = result.vpin[~np.isnan(result.vpin)]
        assert all(0 <= v <= 1 for v in valid)

    def test_directional_flow_high_vpin(self):
        """Strongly directional price moves should produce higher VPIN."""
        rng = np.random.default_rng(42)
        # Trending prices (lots of informed flow)
        prices = 100 + np.cumsum(np.full(5000, 0.01))  # monotonic up
        volumes = rng.integers(10, 100, size=5000).astype(float)
        vpin_est = VPINEstimator(bucket_volume=500, n_buckets=20, sigma_window=20)
        result = vpin_est.compute(prices, volumes)
        valid = result.vpin[~np.isnan(result.vpin)]
        assert np.mean(valid) > 0.5

    def test_small_data(self):
        """Very small dataset should not crash."""
        prices = np.array([100.0, 100.5, 101.0])
        volumes = np.array([50.0, 50.0, 50.0])
        vpin = VPINEstimator(bucket_volume=1000)
        result = vpin.compute(prices, volumes)
        assert len(result.vpin) >= 1

    def test_buy_fractions_bounded(self):
        prices, volumes = _make_trade_data(n=5000)
        vpin = VPINEstimator(bucket_volume=500, n_buckets=20, sigma_window=20)
        result = vpin.compute(prices, volumes)
        assert all(0 <= f <= 1 for f in result.buy_fractions)


# ---------------------------------------------------------------------------
# Kyle's Lambda
# ---------------------------------------------------------------------------


class TestKyleLambda:
    def test_returns_result(self):
        rng = np.random.default_rng(42)
        dp = rng.normal(0, 0.01, 500)
        of = rng.normal(0, 100, 500)
        est = KyleLambdaEstimator(window=50, min_obs=20)
        result = est.estimate(dp, of)
        assert isinstance(result, KyleLambdaResult)

    def test_positive_lambda_with_impact(self):
        """When flow causes price changes, lambda should be positive."""
        rng = np.random.default_rng(42)
        of = rng.normal(0, 100, 500)
        # Price changes correlated with order flow
        dp = 0.001 * of + rng.normal(0, 0.005, 500)
        est = KyleLambdaEstimator(window=100, min_obs=30)
        result = est.estimate(dp, of)
        assert result.mean_lambda > 0

    def test_lambdas_shape(self):
        rng = np.random.default_rng(42)
        n = 300
        est = KyleLambdaEstimator()
        result = est.estimate(rng.normal(0, 0.01, n), rng.normal(0, 100, n))
        assert len(result.lambdas) == n

    def test_short_series(self):
        est = KyleLambdaEstimator(min_obs=30)
        result = est.estimate(np.array([0.01, 0.02]), np.array([100, 200]))
        assert result.mean_lambda == 0.0


# ---------------------------------------------------------------------------
# Roll Spread
# ---------------------------------------------------------------------------


class TestRollSpread:
    def test_returns_dict(self):
        rng = np.random.default_rng(42)
        dp = rng.normal(0, 0.01, 500)
        result = roll_effective_spread(dp)
        assert "effective_spread" in result
        assert "half_spread" in result

    def test_spread_non_negative(self):
        rng = np.random.default_rng(42)
        dp = rng.normal(0, 0.01, 500)
        result = roll_effective_spread(dp)
        assert result["effective_spread"] >= 0

    def test_known_spread(self):
        """Simulate Roll's model with known spread and verify recovery."""
        rng = np.random.default_rng(42)
        n = 5000
        c = 0.05  # Known half-spread
        q = rng.choice([-1, 1], size=n)
        efficient_price = np.cumsum(rng.normal(0, 0.01, n))
        observed_price = efficient_price + c * q
        dp = np.diff(observed_price)
        result = roll_effective_spread(dp)
        # Should recover approximately 2 * c = 0.10
        assert abs(result["effective_spread"] - 2 * c) < 0.05

    def test_short_series_nan(self):
        result = roll_effective_spread(np.array([0.01, 0.02, 0.03]))
        assert np.isnan(result["effective_spread"])

    def test_rolling(self):
        rng = np.random.default_rng(42)
        dp = rng.normal(0, 0.01, 1000)
        result = roll_spread_rolling(dp, window=200)
        assert len(result) == 1000
        assert np.isnan(result[0])
        assert not np.isnan(result[500])


# ---------------------------------------------------------------------------
# Intraday Pattern
# ---------------------------------------------------------------------------


class TestIntradayPattern:
    def test_volatility_pattern_shape(self):
        rng = np.random.default_rng(42)
        n = 5000
        timestamps = rng.uniform(0, 23400, n)
        sq_ret = rng.exponential(0.0001, n)
        pat = IntradayPattern(n_bins=78)
        result = pat.estimate_volatility_pattern(timestamps, sq_ret)
        assert len(result) == 78

    def test_volatility_pattern_normalized(self):
        """Mean should be approximately 1.0."""
        rng = np.random.default_rng(42)
        n = 10000
        timestamps = rng.uniform(0, 23400, n)
        sq_ret = rng.exponential(0.0001, n)
        pat = IntradayPattern(n_bins=78)
        result = pat.estimate_volatility_pattern(timestamps, sq_ret)
        assert np.mean(result) == pytest.approx(1.0, abs=0.1)

    def test_volume_pattern_sums_to_one(self):
        rng = np.random.default_rng(42)
        n = 5000
        timestamps = rng.uniform(0, 23400, n)
        volumes = rng.integers(100, 1000, n).astype(float)
        pat = IntradayPattern(n_bins=78)
        result = pat.estimate_volume_pattern(timestamps, volumes)
        assert result.sum() == pytest.approx(1.0, abs=0.01)

    def test_volume_pattern_positive(self):
        rng = np.random.default_rng(42)
        n = 5000
        timestamps = rng.uniform(0, 23400, n)
        volumes = rng.integers(100, 1000, n).astype(float)
        pat = IntradayPattern(n_bins=78)
        result = pat.estimate_volume_pattern(timestamps, volumes)
        assert all(v >= 0 for v in result)

    def test_u_shaped_volume(self):
        """Simulate U-shaped volume and verify pattern detection."""
        rng = np.random.default_rng(42)
        n = 20000
        timestamps = rng.uniform(0, 23400, n)
        # U-shaped: more volume at open/close
        s = timestamps / 23400
        # Parabola: high at 0 and 1, low at 0.5
        base_intensity = 1 + 3 * (2 * s - 1) ** 2
        volumes = (base_intensity * rng.exponential(100, n)).astype(float)

        pat = IntradayPattern(n_bins=20, n_harmonics=3)
        result = pat.estimate_volume_pattern(timestamps, volumes)
        # First and last bins should have more than middle
        mid = len(result) // 2
        assert result[0] > result[mid]
        assert result[-1] > result[mid]
