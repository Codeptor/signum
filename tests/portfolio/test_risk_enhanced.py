"""Tests for enhanced risk metrics and new RiskEngine features."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.risk import RiskEngine


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.DataFrame(
        {
            "AAPL": np.random.normal(0.0005, 0.02, 252),
            "MSFT": np.random.normal(0.0003, 0.018, 252),
            "GOOGL": np.random.normal(0.0004, 0.022, 252),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def equal_weights():
    """Equal weights for testing."""
    return pd.Series({"AAPL": 0.333, "MSFT": 0.333, "GOOGL": 0.334})


class TestEnhancedRiskMetrics:
    """Test new risk metrics added in Phase 1."""

    def test_sortino_ratio(self, sample_returns, equal_weights):
        """Test Sortino ratio calculation."""
        engine = RiskEngine(sample_returns, equal_weights)
        sortino = engine.sortino_ratio()

        assert isinstance(sortino, float)
        # Sortino can be positive, negative, or zero depending on returns

    def test_calmar_ratio(self, sample_returns, equal_weights):
        """Test Calmar ratio calculation."""
        engine = RiskEngine(sample_returns, equal_weights)
        calmar = engine.calmar_ratio()

        assert isinstance(calmar, float)
        # Calmar can be negative if returns are negative

    def test_omega_ratio(self, sample_returns, equal_weights):
        """Test Omega ratio calculation."""
        engine = RiskEngine(sample_returns, equal_weights)
        omega = engine.omega_ratio()

        assert isinstance(omega, float)
        assert omega >= 0 or np.isinf(omega)

    def test_var_cornish_fisher(self, sample_returns, equal_weights):
        """Test Cornish-Fisher VaR."""
        engine = RiskEngine(sample_returns, equal_weights)
        var_cf = engine.var_cornish_fisher(0.95)
        var_hist = engine.var_historical(0.95)

        assert isinstance(var_cf, float)
        # CF VaR should be different from historical (accounts for skew/kurt)
        assert abs(var_cf - var_hist) < 0.1  # Within reasonable range

    def test_downside_deviation(self, sample_returns, equal_weights):
        """Test downside deviation calculation."""
        engine = RiskEngine(sample_returns, equal_weights)
        dd = engine.downside_deviation()

        assert isinstance(dd, float)
        assert dd >= 0
        # Downside dev should be <= total vol
        assert dd <= engine.volatility()

    def test_beta_calculation(self, sample_returns, equal_weights):
        """Test beta calculation with benchmark."""
        # Create synthetic benchmark
        benchmark = sample_returns.mean(axis=1)

        engine = RiskEngine(sample_returns, equal_weights, benchmark_returns=benchmark)
        beta = engine.beta()

        assert isinstance(beta, float)
        # Beta should be around 1 for diversified portfolio
        assert 0.5 <= beta <= 1.5

    def test_information_ratio(self, sample_returns, equal_weights):
        """Test Information ratio."""
        benchmark = sample_returns.mean(axis=1)
        engine = RiskEngine(sample_returns, equal_weights, benchmark_returns=benchmark)
        ir = engine.information_ratio()

        assert isinstance(ir, float)


class TestRollingMetrics:
    """Test rolling window metrics."""

    def test_rolling_sharpe(self, sample_returns, equal_weights):
        """Test rolling Sharpe ratio."""
        engine = RiskEngine(sample_returns, equal_weights)
        rolling_sharpe = engine.rolling_sharpe(window=63)

        assert isinstance(rolling_sharpe, pd.Series)
        assert len(rolling_sharpe) == len(sample_returns)
        assert rolling_sharpe.isna().sum() == 62  # First 62 are NaN

    def test_rolling_var(self, sample_returns, equal_weights):
        """Test rolling VaR."""
        engine = RiskEngine(sample_returns, equal_weights)
        rolling_var = engine.rolling_var(window=63)

        assert isinstance(rolling_var, pd.Series)
        assert len(rolling_var) == len(sample_returns)

    def test_rolling_max_drawdown(self, sample_returns, equal_weights):
        """Test rolling max drawdown."""
        engine = RiskEngine(sample_returns, equal_weights)
        rolling_dd = engine.rolling_max_drawdown(window=63)

        assert isinstance(rolling_dd, pd.Series)
        # Skip NaN values and check non-NaN drawdowns are negative
        valid_dd = rolling_dd.dropna()
        assert all(valid_dd <= 0)  # Drawdowns are negative

    def test_volatility_regime(self, sample_returns, equal_weights):
        """Test volatility regime classification."""
        engine = RiskEngine(sample_returns, equal_weights)
        regime = engine.volatility_regime(window=63)

        assert isinstance(regime, pd.Series)
        assert set(regime.dropna().unique()).issubset({"low", "normal", "high"})


class TestDrawdownAnalysis:
    """Test enhanced drawdown metrics."""

    def test_avg_drawdown(self, sample_returns, equal_weights):
        """Test average drawdown calculation."""
        engine = RiskEngine(sample_returns, equal_weights)
        avg_dd = engine.avg_drawdown()

        assert isinstance(avg_dd, float)
        assert -1 <= avg_dd <= 0  # Drawdowns are negative

    def test_drawdown_duration(self, sample_returns, equal_weights):
        """Test drawdown duration statistics."""
        engine = RiskEngine(sample_returns, equal_weights)
        dd_stats = engine.drawdown_duration()

        assert isinstance(dd_stats, dict)
        assert "max_duration" in dd_stats
        assert "avg_duration" in dd_stats
        assert "num_drawdowns" in dd_stats
        assert all(isinstance(v, (int, float)) for v in dd_stats.values())


class TestComprehensiveSummary:
    """Test the comprehensive summary method."""

    def test_summary_contains_all_metrics(self, sample_returns, equal_weights):
        """Test that summary includes all new metrics."""
        engine = RiskEngine(sample_returns, equal_weights)
        summary = engine.summary()

        # Check all new metrics are present
        assert "sortino_ratio" in summary
        assert "calmar_ratio" in summary
        assert "omega_ratio" in summary
        assert "var_95_cornish_fisher" in summary
        assert "downside_deviation" in summary
        assert "skewness" in summary
        assert "kurtosis" in summary
        assert "max_drawdown_duration" in summary
        assert "num_drawdowns" in summary

    def test_summary_with_benchmark(self, sample_returns, equal_weights):
        """Test summary when benchmark is provided."""
        benchmark = sample_returns.mean(axis=1)
        engine = RiskEngine(sample_returns, equal_weights, benchmark_returns=benchmark)
        summary = engine.summary()

        assert "information_ratio" in summary
        assert "beta" in summary
