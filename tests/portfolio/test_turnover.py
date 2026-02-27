"""Tests for portfolio turnover optimization."""

import numpy as np
import pytest

from python.portfolio.turnover import (
    Trade,
    TradeList,
    TurnoverOptimizer,
    TurnoverReport,
    optimal_rebalance_frequency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_weights():
    return (
        {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25},
        {"AAPL": 0.30, "MSFT": 0.30, "GOOG": 0.20, "AMZN": 0.20},
    )


# ---------------------------------------------------------------------------
# adjust_weights
# ---------------------------------------------------------------------------


class TestAdjustWeights:
    def test_no_penalty_exact_target(self):
        opt = TurnoverOptimizer(turnover_penalty=0.0)
        current, target = _simple_weights()
        adjusted = opt.adjust_weights(current, target)
        for t in target:
            assert adjusted[t] == pytest.approx(target[t], abs=0.01)

    def test_penalty_reduces_changes(self):
        opt = TurnoverOptimizer(turnover_penalty=0.05)
        current, target = _simple_weights()
        adjusted = opt.adjust_weights(current, target)
        # Adjusted should be between current and target
        for t in current:
            delta_adj = abs(adjusted.get(t, 0) - current[t])
            delta_raw = abs(target.get(t, 0) - current[t])
            assert delta_adj <= delta_raw + 0.01

    def test_high_penalty_no_trades(self):
        opt = TurnoverOptimizer(turnover_penalty=0.2)
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.55, "B": 0.45}
        adjusted = opt.adjust_weights(current, target)
        # Deltas are 0.05, penalty is 0.20, so no trades
        for t in current:
            assert adjusted[t] == pytest.approx(current[t], abs=0.02)

    def test_max_turnover_constraint(self):
        opt = TurnoverOptimizer(turnover_penalty=0.0, max_turnover=0.05)
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.8, "B": 0.2}
        adjusted = opt.adjust_weights(current, target)
        turnover = sum(abs(adjusted[t] - current[t]) for t in current) / 2
        assert turnover <= 0.05 + 0.01

    def test_new_ticker_in_target(self):
        opt = TurnoverOptimizer(turnover_penalty=0.0)
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.3, "B": 0.3, "C": 0.4}
        adjusted = opt.adjust_weights(current, target)
        assert "C" in adjusted
        assert adjusted["C"] > 0

    def test_ticker_removed_from_target(self):
        opt = TurnoverOptimizer(turnover_penalty=0.0)
        current = {"A": 0.3, "B": 0.3, "C": 0.4}
        target = {"A": 0.5, "B": 0.5}
        adjusted = opt.adjust_weights(current, target)
        # C should be zero or absent
        assert adjusted.get("C", 0.0) < 0.01


# ---------------------------------------------------------------------------
# generate_trades
# ---------------------------------------------------------------------------


class TestGenerateTrades:
    def test_returns_trade_list(self):
        opt = TurnoverOptimizer()
        current, target = _simple_weights()
        tl = opt.generate_trades(current, target, portfolio_value=100000)
        assert isinstance(tl, TradeList)

    def test_trades_have_correct_sides(self):
        opt = TurnoverOptimizer(min_trade_pct=0.0)
        current = {"A": 0.3, "B": 0.7}
        target = {"A": 0.6, "B": 0.4}
        tl = opt.generate_trades(current, target, portfolio_value=100000)
        for trade in tl.trades:
            if trade.ticker == "A":
                assert trade.side == "BUY"
            elif trade.ticker == "B":
                assert trade.side == "SELL"

    def test_share_calculation(self):
        opt = TurnoverOptimizer(min_trade_pct=0.0)
        current = {"A": 0.0}
        target = {"A": 1.0}
        prices = {"A": 100.0}
        tl = opt.generate_trades(current, target, portfolio_value=10000, prices=prices)
        assert tl.trades[0].shares == 100

    def test_min_trade_filter(self):
        opt = TurnoverOptimizer(min_trade_pct=0.05)
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.52, "B": 0.48}  # Only 2% change each
        tl = opt.generate_trades(current, target, portfolio_value=100000)
        assert len(tl.trades) == 0

    def test_buy_and_sell_counts(self):
        opt = TurnoverOptimizer(min_trade_pct=0.0)
        current, target = _simple_weights()
        tl = opt.generate_trades(current, target, portfolio_value=100000)
        assert tl.n_buys >= 0
        assert tl.n_sells >= 0
        assert tl.n_buys + tl.n_sells == len(tl.trades)

    def test_summary_string(self):
        opt = TurnoverOptimizer()
        current, target = _simple_weights()
        tl = opt.generate_trades(current, target, portfolio_value=100000)
        s = tl.summary()
        assert "TradeList" in s
        assert "turnover" in s


# ---------------------------------------------------------------------------
# analyze_turnover
# ---------------------------------------------------------------------------


class TestAnalyzeTurnover:
    def test_returns_report(self):
        opt = TurnoverOptimizer()
        current, target = _simple_weights()
        report = opt.analyze_turnover(current, target)
        assert isinstance(report, TurnoverReport)

    def test_raw_turnover_correct(self):
        opt = TurnoverOptimizer()
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.7, "B": 0.3}
        report = opt.analyze_turnover(current, target)
        assert report.raw_turnover == pytest.approx(0.2)

    def test_penalty_reduces_optimized_turnover(self):
        opt = TurnoverOptimizer(turnover_penalty=0.05)
        current, target = _simple_weights()
        report = opt.analyze_turnover(current, target)
        assert report.optimized_turnover <= report.raw_turnover + 0.01

    def test_cost_saved_non_negative(self):
        opt = TurnoverOptimizer(turnover_penalty=0.05)
        current, target = _simple_weights()
        report = opt.analyze_turnover(current, target)
        assert report.cost_saved_bps >= -0.01


# ---------------------------------------------------------------------------
# optimal_rebalance_frequency
# ---------------------------------------------------------------------------


class TestOptimalRebalanceFrequency:
    def test_returns_positive_int(self):
        freq = optimal_rebalance_frequency(
            expected_alpha_decay=10, tc_bps=10, expected_turnover=0.20
        )
        assert isinstance(freq, int)
        assert freq >= 1

    def test_higher_costs_less_frequent(self):
        freq_low = optimal_rebalance_frequency(10, tc_bps=5, expected_turnover=0.20)
        freq_high = optimal_rebalance_frequency(10, tc_bps=50, expected_turnover=0.20)
        assert freq_high >= freq_low

    def test_faster_decay_more_frequent(self):
        freq_fast = optimal_rebalance_frequency(5, tc_bps=10, expected_turnover=0.20)
        freq_slow = optimal_rebalance_frequency(30, tc_bps=10, expected_turnover=0.20)
        assert freq_fast <= freq_slow


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_current(self):
        opt = TurnoverOptimizer(min_trade_pct=0.0)
        target = {"A": 0.5, "B": 0.5}
        tl = opt.generate_trades({}, target, portfolio_value=100000)
        assert tl.n_buys == 2
        assert tl.n_sells == 0

    def test_empty_target(self):
        opt = TurnoverOptimizer(min_trade_pct=0.0)
        current = {"A": 0.5, "B": 0.5}
        tl = opt.generate_trades(current, {}, portfolio_value=100000)
        assert tl.n_sells == 2
        assert tl.n_buys == 0

    def test_same_weights_no_trades(self):
        opt = TurnoverOptimizer()
        weights = {"A": 0.5, "B": 0.5}
        tl = opt.generate_trades(weights, weights, portfolio_value=100000)
        assert len(tl.trades) == 0

    def test_zero_portfolio_value(self):
        opt = TurnoverOptimizer(min_trade_pct=0.0)
        tl = opt.generate_trades({"A": 1.0}, {"B": 1.0}, portfolio_value=0)
        assert isinstance(tl, TradeList)
