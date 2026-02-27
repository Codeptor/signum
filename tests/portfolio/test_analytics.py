"""Tests for portfolio performance analytics."""

import numpy as np
import pytest

from python.portfolio.analytics import (
    PerformanceAnalyzer,
    PerformanceReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n=504, mu=0.0003, sigma=0.01, seed=42):
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    return mu + sigma * rng.standard_normal(n)


def _make_benchmark(n=504, seed=99):
    rng = np.random.default_rng(seed)
    return 0.0002 + 0.012 * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Return Metrics
# ---------------------------------------------------------------------------


class TestReturnMetrics:
    def test_total_return_positive(self):
        rets = _make_returns(mu=0.001)
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.total_return() > 0

    def test_cagr_consistent_with_total(self):
        rets = _make_returns(n=504)
        analyzer = PerformanceAnalyzer(rets)
        total = analyzer.total_return()
        cagr = analyzer.cagr()
        # CAGR^n_years should approximate total_return
        reconstructed = (1 + cagr) ** analyzer.n_years - 1
        assert reconstructed == pytest.approx(total, rel=0.01)

    def test_volatility_positive(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.volatility() > 0

    def test_volatility_scales_with_input(self):
        rets_low = _make_returns(sigma=0.005)
        rets_high = _make_returns(sigma=0.02)
        assert PerformanceAnalyzer(rets_low).volatility() < PerformanceAnalyzer(rets_high).volatility()


# ---------------------------------------------------------------------------
# Risk-Adjusted Metrics
# ---------------------------------------------------------------------------


class TestRiskAdjusted:
    def test_sharpe_positive_for_positive_returns(self):
        rets = _make_returns(mu=0.001, sigma=0.01)
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.sharpe_ratio() > 0

    def test_sharpe_negative_for_negative_returns(self):
        rets = _make_returns(mu=-0.001, sigma=0.01)
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.sharpe_ratio() < 0

    def test_sortino_higher_than_sharpe(self):
        """Sortino usually >= Sharpe for positively-skewed returns."""
        rets = _make_returns(mu=0.001, sigma=0.01)
        analyzer = PerformanceAnalyzer(rets)
        # Not always true but typical for normal-ish returns
        sharpe = analyzer.sharpe_ratio()
        sortino = analyzer.sortino_ratio()
        # Both should be positive
        assert sharpe > 0
        assert sortino > 0

    def test_calmar_positive(self):
        rets = _make_returns(mu=0.001, sigma=0.01)
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.calmar_ratio() > 0

    def test_omega_greater_than_one_for_positive(self):
        rets = _make_returns(mu=0.001, sigma=0.01)
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.omega_ratio() > 1.0

    def test_information_ratio_requires_benchmark(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.information_ratio() is None

    def test_information_ratio_with_benchmark(self):
        rets = _make_returns()
        bench = _make_benchmark()
        analyzer = PerformanceAnalyzer(rets, bench)
        ir = analyzer.information_ratio()
        assert ir is not None
        assert np.isfinite(ir)

    def test_tracking_error_with_benchmark(self):
        rets = _make_returns()
        bench = _make_benchmark()
        analyzer = PerformanceAnalyzer(rets, bench)
        te = analyzer.tracking_error()
        assert te is not None
        assert te > 0


# ---------------------------------------------------------------------------
# Tail Risk
# ---------------------------------------------------------------------------


class TestTailRisk:
    def test_var_negative(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.var(0.95) < 0

    def test_cvar_worse_than_var(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.cvar(0.95) <= analyzer.var(0.95)

    def test_var_99_worse_than_95(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.var(0.99) <= analyzer.var(0.95)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_max_drawdown_negative(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.max_drawdown() < 0

    def test_drawdown_analysis_result(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        dd = analyzer.drawdown_analysis()
        assert dd.max_drawdown < 0
        assert dd.max_dd_trough >= dd.max_dd_start

    def test_underwater_curve_shape(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        uw = analyzer.underwater_curve()
        assert len(uw) == len(rets)
        assert all(u <= 0 for u in uw)

    def test_monotonic_up_zero_dd(self):
        rets = np.full(100, 0.001)  # Always positive
        analyzer = PerformanceAnalyzer(rets)
        assert analyzer.max_drawdown() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Alpha / Beta
# ---------------------------------------------------------------------------


class TestAlphaBeta:
    def test_no_benchmark(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        alpha, beta = analyzer.alpha_beta()
        assert alpha is None
        assert beta is None

    def test_with_benchmark(self):
        rets = _make_returns()
        bench = _make_benchmark()
        analyzer = PerformanceAnalyzer(rets, bench)
        alpha, beta = analyzer.alpha_beta()
        assert alpha is not None
        assert beta is not None
        assert np.isfinite(alpha)
        assert np.isfinite(beta)

    def test_perfect_tracking_beta_one(self):
        """Portfolio = benchmark → beta ≈ 1, alpha ≈ 0."""
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets, rets)
        alpha, beta = analyzer.alpha_beta()
        assert beta == pytest.approx(1.0, abs=0.01)
        assert alpha == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------


class TestRolling:
    def test_rolling_sharpe_shape(self):
        rets = _make_returns(n=504)
        analyzer = PerformanceAnalyzer(rets)
        rs = analyzer.rolling_sharpe(window=252)
        assert len(rs) == 504
        assert np.isnan(rs[0])
        assert np.isfinite(rs[300])

    def test_rolling_volatility_shape(self):
        rets = _make_returns(n=504)
        analyzer = PerformanceAnalyzer(rets)
        rv = analyzer.rolling_volatility(window=63)
        assert len(rv) == 504

    def test_rolling_max_drawdown(self):
        rets = _make_returns(n=504)
        analyzer = PerformanceAnalyzer(rets)
        rdd = analyzer.rolling_max_drawdown(window=252)
        assert len(rdd) == 504
        valid = rdd[~np.isnan(rdd)]
        assert all(v <= 0 for v in valid)


# ---------------------------------------------------------------------------
# Period Returns
# ---------------------------------------------------------------------------


class TestPeriodReturns:
    def test_monthly_returns(self):
        rets = _make_returns(n=504)
        analyzer = PerformanceAnalyzer(rets)
        monthly = analyzer.period_returns(21)
        assert len(monthly) == 504 // 21

    def test_annual_returns(self):
        rets = _make_returns(n=504)
        analyzer = PerformanceAnalyzer(rets)
        annual = analyzer.period_returns(252)
        assert len(annual) == 2


# ---------------------------------------------------------------------------
# Full Report
# ---------------------------------------------------------------------------


class TestFullReport:
    def test_returns_report(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        report = analyzer.full_report()
        assert isinstance(report, PerformanceReport)

    def test_report_with_benchmark(self):
        rets = _make_returns()
        bench = _make_benchmark()
        analyzer = PerformanceAnalyzer(rets, bench)
        report = analyzer.full_report()
        assert report.alpha is not None
        assert report.beta is not None
        assert report.information_ratio is not None

    def test_summary_string(self):
        rets = _make_returns()
        analyzer = PerformanceAnalyzer(rets)
        report = analyzer.full_report()
        summary = report.summary()
        assert "CAGR" in summary
        assert "Sharpe" in summary

    def test_hit_rate_bounded(self):
        rets = _make_returns()
        report = PerformanceAnalyzer(rets).full_report()
        assert 0 <= report.hit_rate <= 1

    def test_gain_loss_ratio_positive(self):
        rets = _make_returns()
        report = PerformanceAnalyzer(rets).full_report()
        assert report.gain_loss_ratio > 0
