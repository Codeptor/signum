"""Tests for transaction cost analysis."""

import numpy as np
import pytest

from python.execution.tca import (
    Fill,
    FillCostBreakdown,
    LinearImpact,
    Side,
    SquareRootImpact,
    TCAAnalyzer,
    TCAReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _buy_fill(**kwargs):
    defaults = dict(
        ticker="AAPL",
        side=Side.BUY,
        shares=1000,
        fill_price=152.0,
        arrival_price=151.0,
        vwap_benchmark=151.5,
        twap_benchmark=151.3,
        adv=1_000_000,
        spread_bps=5.0,
        volatility=0.02,
        execution_minutes=30,
        decision_price=150.5,
        close_price=152.5,
    )
    defaults.update(kwargs)
    return Fill(**defaults)


def _sell_fill(**kwargs):
    defaults = dict(
        ticker="MSFT",
        side=Side.SELL,
        shares=500,
        fill_price=310.0,
        arrival_price=312.0,
        vwap_benchmark=311.0,
        twap_benchmark=311.5,
        adv=800_000,
        spread_bps=4.0,
        volatility=0.018,
        execution_minutes=20,
        decision_price=312.5,
        close_price=309.0,
    )
    defaults.update(kwargs)
    return Fill(**defaults)


# ---------------------------------------------------------------------------
# Fill dataclass
# ---------------------------------------------------------------------------


class TestFill:
    def test_notional(self):
        f = _buy_fill(shares=100, fill_price=150.0)
        assert f.notional == 15000.0

    def test_participation_rate(self):
        f = _buy_fill(shares=10000, adv=1_000_000)
        assert f.participation_rate == pytest.approx(0.01)

    def test_zero_adv_participation(self):
        f = _buy_fill(adv=0)
        assert f.participation_rate == 0.0

    def test_side_enum(self):
        assert Side.BUY == "BUY"
        assert Side.SELL == "SELL"


# ---------------------------------------------------------------------------
# SquareRootImpact
# ---------------------------------------------------------------------------


class TestSquareRootImpact:
    def test_temporary_includes_half_spread(self):
        model = SquareRootImpact(eta=0.0)
        impact = model.temporary_impact_bps(0.01, 0.02, spread_bps=10.0)
        assert impact == pytest.approx(5.0)  # Half spread

    def test_temporary_increases_with_participation(self):
        model = SquareRootImpact(eta=0.25)
        low = model.temporary_impact_bps(0.01, 0.02, 5.0)
        high = model.temporary_impact_bps(0.05, 0.02, 5.0)
        assert high > low

    def test_permanent_increases_with_participation(self):
        model = SquareRootImpact(gamma=0.10, delta=0.5)
        low = model.permanent_impact_bps(0.01, 0.02)
        high = model.permanent_impact_bps(0.05, 0.02)
        assert high > low

    def test_total_is_sum(self):
        model = SquareRootImpact()
        temp = model.temporary_impact_bps(0.01, 0.02, 5.0)
        perm = model.permanent_impact_bps(0.01, 0.02)
        total = model.total_impact_bps(0.01, 0.02, 5.0)
        assert total == pytest.approx(temp + perm)

    def test_zero_participation_equals_half_spread(self):
        model = SquareRootImpact()
        temp = model.temporary_impact_bps(0.0, 0.02, 10.0)
        assert temp == pytest.approx(5.0)  # Just half-spread

    def test_higher_volatility_more_impact(self):
        model = SquareRootImpact()
        low_vol = model.total_impact_bps(0.01, 0.01, 5.0)
        high_vol = model.total_impact_bps(0.01, 0.04, 5.0)
        assert high_vol > low_vol


# ---------------------------------------------------------------------------
# LinearImpact
# ---------------------------------------------------------------------------


class TestLinearImpact:
    def test_temporary_includes_half_spread(self):
        model = LinearImpact(k_temp=0.0)
        impact = model.temporary_impact_bps(0.01, 0.02, spread_bps=8.0)
        assert impact == pytest.approx(4.0)

    def test_linear_scaling(self):
        model = LinearImpact(k_temp=0.5, k_perm=0.1)
        p1 = model.total_impact_bps(0.01, 0.02, 0.0)
        p2 = model.total_impact_bps(0.02, 0.02, 0.0)
        # Linear: doubling participation doubles impact
        assert p2 == pytest.approx(2 * p1)

    def test_permanent_positive(self):
        model = LinearImpact()
        perm = model.permanent_impact_bps(0.05, 0.02)
        assert perm > 0


# ---------------------------------------------------------------------------
# TCAAnalyzer - single fill
# ---------------------------------------------------------------------------


class TestAnalyzeFill:
    def test_buy_positive_is(self):
        """Buy at 152 vs arrival 151 → positive IS."""
        analyzer = TCAAnalyzer()
        fill = _buy_fill(fill_price=152.0, arrival_price=151.0)
        bd = analyzer.analyze_fill(fill)
        assert bd.total_is_bps > 0

    def test_sell_positive_is(self):
        """Sell at 310 vs arrival 312 → positive IS (sold cheaper)."""
        analyzer = TCAAnalyzer()
        fill = _sell_fill(fill_price=310.0, arrival_price=312.0)
        bd = analyzer.analyze_fill(fill)
        assert bd.total_is_bps > 0

    def test_zero_arrival_price(self):
        analyzer = TCAAnalyzer()
        fill = _buy_fill(arrival_price=0.0)
        bd = analyzer.analyze_fill(fill)
        assert bd.total_is_bps == 0.0

    def test_spread_cost_is_half_spread(self):
        analyzer = TCAAnalyzer()
        fill = _buy_fill(spread_bps=10.0)
        bd = analyzer.analyze_fill(fill)
        assert bd.spread_cost_bps == pytest.approx(5.0)

    def test_breakdown_has_predicted(self):
        analyzer = TCAAnalyzer()
        fill = _buy_fill()
        bd = analyzer.analyze_fill(fill)
        assert bd.predicted_impact_bps >= 0

    def test_vwap_slippage(self):
        analyzer = TCAAnalyzer()
        fill = _buy_fill(fill_price=152.0, vwap_benchmark=151.5)
        bd = analyzer.analyze_fill(fill)
        assert bd.vwap_slippage_bps > 0

    def test_returns_breakdown(self):
        analyzer = TCAAnalyzer()
        bd = analyzer.analyze_fill(_buy_fill())
        assert isinstance(bd, FillCostBreakdown)
        assert bd.ticker == "AAPL"
        assert bd.side == Side.BUY


# ---------------------------------------------------------------------------
# TCAAnalyzer - aggregate
# ---------------------------------------------------------------------------


class TestAnalyzeAggregate:
    def test_returns_report(self):
        analyzer = TCAAnalyzer()
        report = analyzer.analyze([_buy_fill(), _sell_fill()])
        assert isinstance(report, TCAReport)

    def test_empty_fills(self):
        analyzer = TCAAnalyzer()
        report = analyzer.analyze([])
        assert report.total_notional == 0
        assert report.avg_is_bps == 0
        assert len(report.fills) == 0

    def test_total_shares(self):
        analyzer = TCAAnalyzer()
        report = analyzer.analyze([_buy_fill(shares=1000), _sell_fill(shares=500)])
        assert report.total_shares == 1500

    def test_total_notional(self):
        analyzer = TCAAnalyzer()
        f1 = _buy_fill(shares=100, fill_price=150.0)
        f2 = _sell_fill(shares=200, fill_price=300.0)
        report = analyzer.analyze([f1, f2])
        assert report.total_notional == pytest.approx(100 * 150.0 + 200 * 300.0)

    def test_buy_sell_counts(self):
        analyzer = TCAAnalyzer()
        report = analyzer.analyze([_buy_fill(), _buy_fill(), _sell_fill()])
        assert report.n_buys == 2
        assert report.n_sells == 1

    def test_worst_best_fill(self):
        analyzer = TCAAnalyzer()
        # Large IS buy
        f1 = _buy_fill(fill_price=155.0, arrival_price=150.0)
        # Small IS buy
        f2 = _buy_fill(fill_price=150.5, arrival_price=150.0)
        report = analyzer.analyze([f1, f2])
        assert report.worst_fill_bps > report.best_fill_bps

    def test_summary_string(self):
        analyzer = TCAAnalyzer()
        report = analyzer.analyze([_buy_fill(), _sell_fill()])
        s = report.summary()
        assert "TCAReport" in s
        assert "bps" in s

    def test_prediction_rmse(self):
        analyzer = TCAAnalyzer()
        report = analyzer.analyze([_buy_fill(), _sell_fill()])
        assert report.prediction_rmse_bps >= 0


# ---------------------------------------------------------------------------
# Pre-trade cost estimation
# ---------------------------------------------------------------------------


class TestPreTradeCost:
    def test_returns_positive(self):
        analyzer = TCAAnalyzer()
        cost = analyzer.estimate_pretrade_cost(
            shares=5000, price=150.0, adv=1_000_000,
            volatility=0.02, spread_bps=5.0,
        )
        assert cost > 0

    def test_larger_order_higher_cost(self):
        analyzer = TCAAnalyzer()
        small = analyzer.estimate_pretrade_cost(
            1000, 150.0, 1_000_000, 0.02, 5.0
        )
        large = analyzer.estimate_pretrade_cost(
            50000, 150.0, 1_000_000, 0.02, 5.0
        )
        assert large > small

    def test_zero_adv_returns_half_spread(self):
        analyzer = TCAAnalyzer()
        cost = analyzer.estimate_pretrade_cost(
            1000, 150.0, 0, 0.02, 10.0
        )
        assert cost == pytest.approx(5.0)

    def test_custom_impact_model(self):
        analyzer = TCAAnalyzer(impact_model=LinearImpact())
        cost = analyzer.estimate_pretrade_cost(
            5000, 150.0, 1_000_000, 0.02, 5.0
        )
        assert cost > 0


# ---------------------------------------------------------------------------
# Optimal execution horizon
# ---------------------------------------------------------------------------


class TestOptimalHorizon:
    def test_returns_positive_minutes(self):
        analyzer = TCAAnalyzer()
        horizon = analyzer.optimal_execution_horizon(
            shares=10000, adv=1_000_000, volatility=0.02
        )
        assert horizon > 0

    def test_larger_order_longer(self):
        analyzer = TCAAnalyzer()
        small = analyzer.optimal_execution_horizon(1000, 1_000_000, 0.02)
        large = analyzer.optimal_execution_horizon(100000, 1_000_000, 0.02)
        assert large > small

    def test_urgency_shortens(self):
        analyzer = TCAAnalyzer()
        patient = analyzer.optimal_execution_horizon(
            10000, 1_000_000, 0.02, urgency=0.1
        )
        urgent = analyzer.optimal_execution_horizon(
            10000, 1_000_000, 0.02, urgency=0.9
        )
        assert urgent < patient

    def test_capped_at_full_day(self):
        analyzer = TCAAnalyzer()
        horizon = analyzer.optimal_execution_horizon(
            shares=500000, adv=100_000, volatility=0.02
        )
        assert horizon <= 390.0

    def test_min_5_minutes(self):
        analyzer = TCAAnalyzer()
        horizon = analyzer.optimal_execution_horizon(
            shares=10, adv=10_000_000, volatility=0.02
        )
        assert horizon >= 5.0

    def test_zero_adv(self):
        analyzer = TCAAnalyzer()
        horizon = analyzer.optimal_execution_horizon(
            shares=1000, adv=0, volatility=0.02
        )
        assert horizon == 390.0
