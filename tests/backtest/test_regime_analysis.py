"""Tests for regime-specific backtesting (Phase 4, §2.4.2)."""

import numpy as np
import pandas as pd
import pytest

from python.backtest.regime_analysis import (
    DEFAULT_REGIMES,
    RegimeAnalysis,
    RegimeResult,
    backtest_by_regime,
    compute_regime_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(start="2020-01-01", periods=252, mean=0.0004, std=0.01, seed=42):
    """Create synthetic daily returns with DatetimeIndex."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, periods=periods)
    returns = rng.normal(mean, std, size=periods)
    return pd.Series(returns, index=dates, name="strategy_returns")


# ---------------------------------------------------------------------------
# Tests: RegimeResult
# ---------------------------------------------------------------------------


class TestRegimeResult:
    def test_profitable_positive_return(self):
        r = RegimeResult(
            regime="bull",
            start_date="2020-01-01",
            end_date="2020-12-31",
            total_return=0.15,
            annual_return=0.15,
            sharpe=1.2,
            max_drawdown=-0.05,
            volatility=0.12,
            n_days=252,
        )
        assert r.is_profitable is True

    def test_not_profitable_negative_return(self):
        r = RegimeResult(
            regime="bear",
            start_date="2020-01-01",
            end_date="2020-12-31",
            total_return=-0.10,
            annual_return=-0.10,
            sharpe=-0.5,
            max_drawdown=-0.15,
            volatility=0.20,
            n_days=252,
        )
        assert r.is_profitable is False


# ---------------------------------------------------------------------------
# Tests: RegimeAnalysis
# ---------------------------------------------------------------------------


class TestRegimeAnalysis:
    def test_profitable_count(self):
        analysis = RegimeAnalysis()
        analysis.results["bull"] = RegimeResult(
            "bull", "2020-01-01", "2020-12-31", 0.15, 0.15, 1.0, -0.05, 0.12, 252
        )
        analysis.results["bear"] = RegimeResult(
            "bear", "2021-01-01", "2021-12-31", -0.05, -0.05, -0.3, -0.10, 0.18, 252
        )
        assert analysis.profitable_regimes == 1
        assert analysis.total_regimes == 2

    def test_mean_sharpe(self):
        analysis = RegimeAnalysis()
        analysis.results["a"] = RegimeResult("a", "", "", 0.0, 0.0, 1.0, 0.0, 0.0, 100)
        analysis.results["b"] = RegimeResult("b", "", "", 0.0, 0.0, -0.5, 0.0, 0.0, 100)
        assert analysis.mean_sharpe == pytest.approx(0.25, abs=0.01)

    def test_worst_drawdown(self):
        analysis = RegimeAnalysis()
        analysis.results["a"] = RegimeResult("a", "", "", 0.0, 0.0, 0.0, -0.05, 0.0, 100)
        analysis.results["b"] = RegimeResult("b", "", "", 0.0, 0.0, 0.0, -0.20, 0.0, 100)
        assert analysis.worst_drawdown == pytest.approx(-0.20)

    def test_summary_dataframe(self):
        analysis = RegimeAnalysis()
        analysis.results["test"] = RegimeResult(
            "test", "2020-01-01", "2020-06-30", 0.10, 0.20, 1.5, -0.03, 0.10, 126
        )
        summary = analysis.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1
        assert "regime" in summary.columns
        assert "sharpe" in summary.columns

    def test_empty_analysis(self):
        analysis = RegimeAnalysis()
        assert analysis.profitable_regimes == 0
        assert analysis.mean_sharpe == 0.0
        assert analysis.worst_drawdown == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_regime_metrics
# ---------------------------------------------------------------------------


class TestComputeRegimeMetrics:
    def test_positive_returns(self):
        returns = _make_returns(mean=0.001, periods=252)
        result = compute_regime_metrics(returns, "bull", "2020-01-01", "2020-12-31")
        assert result.total_return > 0
        assert result.annual_return > 0
        assert result.sharpe > 0

    def test_negative_returns(self):
        returns = _make_returns(mean=-0.002, periods=252)
        result = compute_regime_metrics(returns, "bear", "2020-01-01", "2020-12-31")
        assert result.total_return < 0

    def test_max_drawdown_negative(self):
        returns = _make_returns(mean=0.0, periods=252)
        result = compute_regime_metrics(returns, "flat", "2020-01-01", "2020-12-31")
        assert result.max_drawdown <= 0  # Drawdowns are always <= 0

    def test_n_days_correct(self):
        returns = _make_returns(periods=100)
        result = compute_regime_metrics(returns, "test", "2020-01-01", "2020-05-01")
        assert result.n_days == 100

    def test_empty_returns(self):
        returns = pd.Series([], dtype=float)
        result = compute_regime_metrics(returns, "empty", "2020-01-01", "2020-01-01")
        assert result.total_return == 0.0
        assert result.n_days == 0

    def test_volatility_positive(self):
        returns = _make_returns(std=0.02, periods=252)
        result = compute_regime_metrics(returns, "volatile", "2020-01-01", "2020-12-31")
        assert result.volatility > 0


# ---------------------------------------------------------------------------
# Tests: backtest_by_regime
# ---------------------------------------------------------------------------


class TestBacktestByRegime:
    def test_custom_regimes(self):
        returns = _make_returns(start="2020-01-01", periods=500)
        regimes = {
            "period_a": ("2020-01-01", "2020-06-30"),
            "period_b": ("2020-07-01", "2020-12-31"),
        }
        analysis = backtest_by_regime(returns, regimes=regimes)
        assert len(analysis.results) == 2
        assert "period_a" in analysis.results
        assert "period_b" in analysis.results

    def test_skips_empty_regimes(self):
        returns = _make_returns(start="2020-01-01", periods=100)
        regimes = {
            "covered": ("2020-01-01", "2020-06-30"),
            "missing": ("2025-01-01", "2025-12-31"),  # No data for this period
        }
        analysis = backtest_by_regime(returns, regimes=regimes)
        assert len(analysis.results) == 1
        assert "covered" in analysis.results

    def test_returns_regime_analysis(self):
        returns = _make_returns(start="2020-01-01", periods=100)
        regimes = {"test": ("2020-01-01", "2020-06-30")}
        analysis = backtest_by_regime(returns, regimes=regimes)
        assert isinstance(analysis, RegimeAnalysis)

    def test_default_regimes_constant_exists(self):
        assert len(DEFAULT_REGIMES) >= 5  # Should have several regime periods
        for name, (start, end) in DEFAULT_REGIMES.items():
            assert start < end, f"Regime '{name}' has invalid date range"
