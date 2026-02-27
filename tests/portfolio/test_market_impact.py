"""Tests for market impact models."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.market_impact import (
    AlmgrenChrissModel,
    CompositeImpactModel,
    CostBreakdown,
    FixedCostModel,
    SquareRootModel,
    estimate_liquidity_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_data():
    """Synthetic OHLCV data for 3 tickers over 60 trading days."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=60)
    rows = []
    for ticker, base_price, base_vol in [
        ("AAPL", 180.0, 50_000_000),
        ("MSFT", 400.0, 30_000_000),
        ("TSLA", 250.0, 80_000_000),
    ]:
        price = base_price
        for d in dates:
            ret = np.random.normal(0.0005, 0.015)
            price *= 1 + ret
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            vol = int(base_vol * np.random.uniform(0.7, 1.3))
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * (1 + np.random.normal(0, 0.002)),
                "high": high,
                "low": low,
                "close": price,
                "volume": vol,
            })
    df = pd.DataFrame(rows)
    df.index = df["date"]
    return df


# ---------------------------------------------------------------------------
# CostBreakdown
# ---------------------------------------------------------------------------


class TestCostBreakdown:
    def test_total_computed(self):
        cb = CostBreakdown(
            commission_bps=1.0,
            spread_bps=1.0,
            temporary_impact_bps=5.0,
            permanent_impact_bps=2.0,
        )
        assert cb.total_bps == pytest.approx(9.0)

    def test_zero_cost(self):
        cb = CostBreakdown()
        assert cb.total_bps == 0.0
        assert cb.total_usd == 0.0


# ---------------------------------------------------------------------------
# FixedCostModel
# ---------------------------------------------------------------------------


class TestFixedCost:
    def test_returns_constant(self):
        model = FixedCostModel(total_bps=10.0)
        c1 = model.estimate_cost(100_000, adv=1e8, volatility=0.01)
        c2 = model.estimate_cost(500_000, adv=1e8, volatility=0.02)
        assert c1.total_bps == pytest.approx(c2.total_bps)

    def test_cost_positive(self):
        model = FixedCostModel(total_bps=10.0)
        cb = model.estimate_cost(100_000, adv=1e8, volatility=0.015)
        assert cb.total_bps > 0
        assert cb.total_usd > 0

    def test_usd_proportional_to_trade_value(self):
        model = FixedCostModel(total_bps=10.0)
        c1 = model.estimate_cost(100_000, adv=1e8, volatility=0.01)
        c2 = model.estimate_cost(200_000, adv=1e8, volatility=0.01)
        assert c2.total_usd == pytest.approx(c1.total_usd * 2, rel=0.01)

    def test_calibrate_returns_result(self, ohlcv_data):
        model = FixedCostModel()
        result = model.calibrate(ohlcv_data)
        assert result.estimation_method == "fixed_cost"


# ---------------------------------------------------------------------------
# SquareRootModel
# ---------------------------------------------------------------------------


class TestSquareRoot:
    def test_cost_increases_with_trade_size(self):
        model = SquareRootModel.default_sp500()
        c1 = model.estimate_cost(100_000, adv=1e8, volatility=0.015)
        c2 = model.estimate_cost(1_000_000, adv=1e8, volatility=0.015)
        assert c2.total_bps > c1.total_bps

    def test_cost_increases_with_volatility(self):
        model = SquareRootModel.default_sp500()
        c1 = model.estimate_cost(500_000, adv=1e8, volatility=0.01)
        c2 = model.estimate_cost(500_000, adv=1e8, volatility=0.03)
        assert c2.temporary_impact_bps > c1.temporary_impact_bps

    def test_cost_decreases_with_adv(self):
        model = SquareRootModel.default_sp500()
        c1 = model.estimate_cost(500_000, adv=5e7, volatility=0.015)
        c2 = model.estimate_cost(500_000, adv=5e8, volatility=0.015)
        assert c2.temporary_impact_bps < c1.temporary_impact_bps

    def test_participation_rate(self):
        model = SquareRootModel.default_sp500()
        cb = model.estimate_cost(1_000_000, adv=10_000_000, volatility=0.015)
        assert cb.participation_rate == pytest.approx(0.1)

    def test_capacity_warning(self):
        model = SquareRootModel(max_participation=0.05)
        cb = model.estimate_cost(1_000_000, adv=5_000_000, volatility=0.015)
        assert cb.capacity_warning is True

    def test_no_capacity_warning_small_trade(self):
        model = SquareRootModel.default_sp500()
        cb = model.estimate_cost(10_000, adv=1e8, volatility=0.015)
        assert cb.capacity_warning is False

    def test_conservative_higher_costs(self):
        default = SquareRootModel.default_sp500()
        conservative = SquareRootModel.conservative()
        cd = default.estimate_cost(500_000, adv=1e8, volatility=0.015)
        cc = conservative.estimate_cost(500_000, adv=1e8, volatility=0.015)
        assert cc.total_bps > cd.total_bps

    def test_calibrate_from_ohlcv(self, ohlcv_data):
        model = SquareRootModel.default_sp500()
        result = model.calibrate(ohlcv_data, lookback_days=30)
        assert result.n_assets == 3
        assert result.half_spread_bps > 0
        assert "parkinson" in result.estimation_method

    def test_estimate_cost_bps_shortcut(self):
        model = SquareRootModel.default_sp500()
        bps = model.estimate_cost_bps(500_000, adv=1e8, volatility=0.015)
        cb = model.estimate_cost(500_000, adv=1e8, volatility=0.015)
        assert bps == pytest.approx(cb.total_bps)


# ---------------------------------------------------------------------------
# AlmgrenChrissModel
# ---------------------------------------------------------------------------


class TestAlmgrenChriss:
    def test_has_temporary_and_permanent(self):
        model = AlmgrenChrissModel.default_sp500()
        cb = model.estimate_cost(500_000, adv=1e8, volatility=0.015)
        assert cb.temporary_impact_bps > 0
        assert cb.permanent_impact_bps > 0

    def test_cost_increases_with_size(self):
        model = AlmgrenChrissModel.default_sp500()
        c1 = model.estimate_cost(100_000, adv=1e8, volatility=0.015)
        c2 = model.estimate_cost(1_000_000, adv=1e8, volatility=0.015)
        assert c2.total_bps > c1.total_bps

    def test_conservative_higher_costs(self):
        default = AlmgrenChrissModel.default_sp500()
        conservative = AlmgrenChrissModel.conservative()
        cd = default.estimate_cost(500_000, adv=1e8, volatility=0.015)
        cc = conservative.estimate_cost(500_000, adv=1e8, volatility=0.015)
        assert cc.total_bps > cd.total_bps

    def test_calibrate_from_ohlcv(self, ohlcv_data):
        model = AlmgrenChrissModel.default_sp500()
        result = model.calibrate(ohlcv_data, lookback_days=30)
        assert result.n_assets == 3
        assert result.half_spread_bps > 0

    def test_execution_horizon_effect(self):
        fast = AlmgrenChrissModel(execution_horizon=0.5)
        slow = AlmgrenChrissModel(execution_horizon=2.0)
        cf = fast.estimate_cost(500_000, adv=1e8, volatility=0.015)
        cs = slow.estimate_cost(500_000, adv=1e8, volatility=0.015)
        # Faster execution = higher temporary impact
        assert cf.temporary_impact_bps > cs.temporary_impact_bps


# ---------------------------------------------------------------------------
# CompositeImpactModel
# ---------------------------------------------------------------------------


class TestComposite:
    def test_applies_floor(self):
        """Floor scaling increases impact component when total is below min."""
        base = FixedCostModel(total_bps=2.0, commission_bps=0.5)
        composite = CompositeImpactModel(base_model=base, min_cost_bps=5.0)
        cb_base = base.estimate_cost(100_000, adv=1e8, volatility=0.015)
        cb_comp = composite.estimate_cost(100_000, adv=1e8, volatility=0.015)
        assert cb_comp.temporary_impact_bps > cb_base.temporary_impact_bps
        assert cb_comp.total_bps > cb_base.total_bps

    def test_applies_cap(self):
        """Cap scaling reduces impact component when total exceeds max."""
        base = SquareRootModel(kappa=10.0)
        model = CompositeImpactModel(base_model=base, max_cost_bps=50.0)
        cb_base = base.estimate_cost(5_000_000, adv=1e7, volatility=0.03)
        cb_comp = model.estimate_cost(5_000_000, adv=1e7, volatility=0.03)
        assert cb_comp.temporary_impact_bps < cb_base.temporary_impact_bps
        assert cb_comp.total_bps < cb_base.total_bps

    def test_default_base_is_sqrt(self):
        model = CompositeImpactModel()
        assert isinstance(model.base_model, SquareRootModel)

    def test_calibrate_delegates(self, ohlcv_data):
        model = CompositeImpactModel()
        result = model.calibrate(ohlcv_data)
        assert result.n_assets == 3


# ---------------------------------------------------------------------------
# Portfolio-level methods
# ---------------------------------------------------------------------------


class TestPortfolioMethods:
    def test_estimate_portfolio_cost(self):
        model = SquareRootModel.default_sp500()
        trades = pd.Series({"AAPL": 200_000, "MSFT": 300_000, "GOOG": 0})
        adv = pd.Series({"AAPL": 1e8, "MSFT": 8e7, "GOOG": 5e7})
        vol = pd.Series({"AAPL": 0.015, "MSFT": 0.012, "GOOG": 0.018})
        result = model.estimate_portfolio_cost(trades, adv, vol)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "total_bps" in result.columns
        # GOOG should have zero cost (no trade)
        assert result.loc["GOOG", "total_bps"] == 0.0

    def test_tc_penalty_matrix(self):
        model = SquareRootModel.default_sp500()
        tickers = ["AAPL", "MSFT", "TSLA"]
        adv = pd.Series({"AAPL": 1e8, "MSFT": 8e7, "TSLA": 2e8})
        vol = pd.Series({"AAPL": 0.015, "MSFT": 0.012, "TSLA": 0.025})
        penalties = model.tc_penalty_matrix(
            tickers, adv, vol, portfolio_value=1_000_000
        )
        assert isinstance(penalties, pd.Series)
        assert len(penalties) == 3
        assert (penalties >= 0).all()
        # Higher vol + lower ADV → higher penalty
        assert penalties["TSLA"] > 0


# ---------------------------------------------------------------------------
# Liquidity profile
# ---------------------------------------------------------------------------


class TestLiquidityProfile:
    def test_returns_dataframe(self, ohlcv_data):
        result = estimate_liquidity_profile(ohlcv_data, lookback_days=30)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "adv_dollar" in result.columns
        assert "volatility_daily" in result.columns
        assert "spread_bps_est" in result.columns
        assert "liquidity_bucket" in result.columns

    def test_all_values_positive(self, ohlcv_data):
        result = estimate_liquidity_profile(ohlcv_data)
        assert (result["adv_dollar"] > 0).all()
        assert (result["volatility_daily"] > 0).all()
        assert (result["spread_bps_est"] > 0).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"ticker": ["AAPL"], "close": [180.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            estimate_liquidity_profile(df)
