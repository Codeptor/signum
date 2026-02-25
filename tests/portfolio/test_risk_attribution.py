"""Tests for RiskAttribution module."""

import numpy as np
import pandas as pd
import pytest

from python.portfolio.risk_attribution import (
    RiskAttribution,
    calculate_risk_parity_allocation,
)


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
            "AMZN": np.random.normal(0.0002, 0.025, 252),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def equal_weights():
    """Equal weights for testing."""
    return pd.Series({"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25})


@pytest.fixture
def concentrated_weights():
    """Concentrated weights for testing."""
    return pd.Series({"AAPL": 0.60, "MSFT": 0.20, "GOOGL": 0.15, "AMZN": 0.05})


class TestRiskAttributionInitialization:
    """Test RiskAttribution class initialization."""

    def test_init_with_weights(self, sample_returns, equal_weights):
        """Test initialization with provided weights."""
        attr = RiskAttribution(sample_returns, equal_weights)

        assert attr.tickers == list(sample_returns.columns)
        assert len(attr.weights) == 4
        assert attr.weights.sum() == pytest.approx(1.0)
        assert attr.port_vol > 0

    def test_init_without_weights(self, sample_returns):
        """Test initialization without weights (defaults to equal)."""
        attr = RiskAttribution(sample_returns)

        assert len(attr.weights) == 4
        assert attr.weights.sum() == pytest.approx(1.0)
        # Should be equal weights
        expected_weight = 1.0 / 4
        for w in attr.weights:
            assert w == pytest.approx(expected_weight)

    def test_covariance_matrix_calculation(self, sample_returns, equal_weights):
        """Test that covariance matrix is calculated correctly."""
        attr = RiskAttribution(sample_returns, equal_weights)

        assert attr.cov_matrix.shape == (4, 4)
        # Diagonal should be variances (positive)
        for i in range(4):
            assert attr.cov_matrix.iloc[i, i] > 0
        # Matrix should be symmetric
        pd.testing.assert_frame_equal(attr.cov_matrix, attr.cov_matrix.T)


class TestMarginalRiskContribution:
    """Test marginal risk contribution calculations."""

    def test_mrc_calculation(self, sample_returns, equal_weights):
        """Test MRC calculation."""
        attr = RiskAttribution(sample_returns, equal_weights)
        mrc = attr.marginal_risk_contribution()

        assert isinstance(mrc, pd.Series)
        assert len(mrc) == 4
        assert all(ticker in mrc.index for ticker in sample_returns.columns)
        # MRC can be positive or negative

    def test_mrc_with_zero_volatility(self):
        """Test MRC with zero volatility (degenerate case)."""
        returns = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        weights = pd.Series({"A": 0.5, "B": 0.5})
        attr = RiskAttribution(returns, weights)

        mrc = attr.marginal_risk_contribution()
        assert all(mrc == 0.0)


class TestComponentRisk:
    """Test component risk calculations."""

    def test_component_risk_calculation(self, sample_returns, equal_weights):
        """Test component risk calculation."""
        attr = RiskAttribution(sample_returns, equal_weights)
        component = attr.component_risk()

        assert isinstance(component, pd.Series)
        assert len(component) == 4
        # Sum of components should equal portfolio volatility
        assert component.sum() == pytest.approx(attr.port_vol, rel=1e-5)

    def test_component_vs_mrc(self, sample_returns, equal_weights):
        """Test that component = weight × MRC."""
        attr = RiskAttribution(sample_returns, equal_weights)
        mrc = attr.marginal_risk_contribution()
        component = attr.component_risk()

        expected_component = attr.weights * mrc
        pd.testing.assert_series_equal(component, expected_component)


class TestRiskContributionPercentage:
    """Test risk contribution percentage calculations."""

    def test_risk_contribution_pct(self, sample_returns, equal_weights):
        """Test risk contribution percentages."""
        attr = RiskAttribution(sample_returns, equal_weights)
        pct = attr.risk_contribution_pct()

        assert isinstance(pct, pd.Series)
        assert len(pct) == 4
        # Percentages should sum to 1 (or very close due to floating point)
        assert pct.sum() == pytest.approx(1.0, abs=1e-10)

    def test_pct_with_concentrated_weights(self, sample_returns, concentrated_weights):
        """Test that concentrated weights show different contributions."""
        attr = RiskAttribution(sample_returns, concentrated_weights)
        pct = attr.risk_contribution_pct()

        # AAPL has 60% weight, should have highest risk contribution
        assert pct["AAPL"] > pct["AMZN"]


class TestRiskParity:
    """Test risk parity optimization."""

    def test_risk_parity_optimization(self, sample_returns):
        """Test risk parity weight calculation."""
        attr = RiskAttribution(sample_returns)
        rp_weights = attr.risk_parity_weights()

        assert isinstance(rp_weights, pd.Series)
        assert len(rp_weights) == 4
        assert rp_weights.sum() == pytest.approx(1.0)
        assert all(rp_weights >= 0)

    def test_risk_parity_equal_contributions(self, sample_returns):
        """Test that risk parity produces near-equal risk contributions."""
        attr = RiskAttribution(sample_returns)
        rp_weights = attr.risk_parity_weights()

        # Create new attribution with RP weights
        attr_rp = RiskAttribution(sample_returns, rp_weights)
        pct = attr_rp.risk_contribution_pct()

        # All contributions should be close to 25% (equal)
        expected = 1.0 / 4
        for p in pct:
            assert p == pytest.approx(expected, abs=0.05)  # Within 5%

    def test_risk_parity_with_bounds(self, sample_returns):
        """Test risk parity with weight bounds."""
        attr = RiskAttribution(sample_returns)
        rp_weights = attr.risk_parity_weights(min_weight=0.05, max_weight=0.50)

        # Check bounds respected
        assert all(rp_weights >= 0.05)
        assert all(rp_weights <= 0.50)

    def test_risk_parity_convenience_function(self, sample_returns):
        """Test the convenience function for risk parity."""
        weights = calculate_risk_parity_allocation(sample_returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == 4
        assert weights.sum() == pytest.approx(1.0)


class TestDiversificationRatio:
    """Test diversification ratio calculations."""

    def test_diversification_ratio_greater_than_one(self, sample_returns, equal_weights):
        """Test that diversification ratio > 1 for diversified portfolio."""
        attr = RiskAttribution(sample_returns, equal_weights)
        dr = attr.diversification_ratio()

        assert isinstance(dr, float)
        assert dr > 1.0  # Diversified portfolio should benefit from diversification

    def test_diversification_ratio_single_asset(self, sample_returns):
        """Test diversification ratio for single asset (should be 1.0)."""
        single_returns = sample_returns[["AAPL"]]
        weights = pd.Series({"AAPL": 1.0})
        attr = RiskAttribution(single_returns, weights)

        dr = attr.diversification_ratio()
        assert dr == pytest.approx(1.0)

    def test_concentrated_vs_diversified(self, sample_returns):
        """Test that concentrated portfolio has lower diversification ratio."""
        equal_w = pd.Series({t: 0.25 for t in sample_returns.columns})
        conc_w = pd.Series({"AAPL": 0.7, "MSFT": 0.1, "GOOGL": 0.1, "AMZN": 0.1})

        attr_equal = RiskAttribution(sample_returns, equal_w)
        attr_conc = RiskAttribution(sample_returns, conc_w)

        # Equal weight should have better diversification
        assert attr_equal.diversification_ratio() > attr_conc.diversification_ratio()


class TestTrackingError:
    """Test tracking error calculations."""

    def test_tracking_error_calculation(self, sample_returns, equal_weights):
        """Test tracking error vs benchmark."""
        # Use average as benchmark
        benchmark = sample_returns.mean(axis=1)
        attr = RiskAttribution(sample_returns, equal_weights)

        te = attr.tracking_error(benchmark)

        assert isinstance(te, float)
        assert te >= 0


class TestConditionalCorrelation:
    """Test stress period correlation analysis."""

    def test_conditional_correlation(self, sample_returns, equal_weights):
        """Test correlation during stress periods."""
        attr = RiskAttribution(sample_returns, equal_weights)
        stress_corr = attr.conditional_correlation(threshold_percentile=10)

        assert isinstance(stress_corr, pd.DataFrame)
        assert stress_corr.shape == (4, 4)
        # Should be symmetric
        pd.testing.assert_frame_equal(stress_corr, stress_corr.T)

    def test_stress_correlation_computable(self, sample_returns, equal_weights):
        """Test that stress correlation can be computed."""
        attr = RiskAttribution(sample_returns, equal_weights)
        stress_corr = attr.conditional_correlation(threshold_percentile=10)

        # Just verify we can compute it - with random data, correlations may vary
        assert stress_corr.shape == (4, 4)
        assert not stress_corr.isna().all().all()


class TestRiskReport:
    """Test risk report generation."""

    def test_risk_report_structure(self, sample_returns, equal_weights):
        """Test risk report contains expected fields."""
        attr = RiskAttribution(sample_returns, equal_weights)
        report = attr.risk_report()

        assert isinstance(report, dict)
        assert "portfolio_volatility" in report
        assert "diversification_ratio" in report
        assert "marginal_risk_contribution" in report
        assert "component_risk" in report
        assert "risk_contribution_pct" in report
        assert "top_risk_contributors" in report
        assert "concentration_risk" in report

    def test_risk_report_values(self, sample_returns, equal_weights):
        """Test risk report values are reasonable."""
        attr = RiskAttribution(sample_returns, equal_weights)
        report = attr.risk_report()

        assert report["portfolio_volatility"] > 0
        assert report["diversification_ratio"] >= 1.0
        assert len(report["top_risk_contributors"]) <= 5


class TestOptimalWeightsComparison:
    """Test weight comparison across methods."""

    def test_weights_comparison(self, sample_returns, equal_weights):
        """Test comparison of different weighting schemes."""
        attr = RiskAttribution(sample_returns, equal_weights)
        comparison = attr.get_optimal_weights_comparison()

        assert isinstance(comparison, pd.DataFrame)
        assert "current" in comparison.columns
        assert "equal" in comparison.columns
        assert "risk_parity" in comparison.columns
        assert len(comparison) == 4

    def test_weights_comparison_custom_methods(self, sample_returns, equal_weights):
        """Test comparison with custom method list."""
        attr = RiskAttribution(sample_returns, equal_weights)
        comparison = attr.get_optimal_weights_comparison(methods=["current", "equal"])

        assert "current" in comparison.columns
        assert "equal" in comparison.columns
        assert "risk_parity" not in comparison.columns
