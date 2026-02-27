import numpy as np
import pandas as pd
import pytest

from python.portfolio.optimizer import PortfolioOptimizer


@pytest.fixture
def price_data():
    """Synthetic price data for 5 assets."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.randn(252, 5) * 0.01, axis=0)) * 100,
        index=dates,
        columns=tickers,
    )
    return prices


def test_hrp_allocation(price_data):
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.hrp()
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()


def test_cvar_allocation(price_data):
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.min_cvar()
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -0.01).all()  # Allow tiny numerical errors


def test_black_litterman_allocation(price_data):
    views = pd.Series({"AAPL": 0.02, "MSFT": 0.01, "GOOG": -0.005})
    view_confidences = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.black_litterman(views, view_confidences)
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_herc_allocation(price_data):
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.herc()
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()
    assert len(weights) == 5


def test_nco_allocation(price_data):
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.nco()
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -0.01).all()  # NCO can produce tiny negative weights
    assert len(weights) == 5


def test_herc_with_max_weight(price_data):
    optimizer = PortfolioOptimizer(price_data, max_weight=0.25)
    weights = optimizer.herc()
    assert weights.max() <= 0.25 + 1e-6
    assert abs(weights.sum() - 1.0) < 1e-6


def test_nco_with_max_weight(price_data):
    optimizer = PortfolioOptimizer(price_data, max_weight=0.25)
    weights = optimizer.nco()
    assert weights.max() <= 0.25 + 1e-6
    assert abs(weights.sum() - 1.0) < 1e-6


def test_compare_all_includes_herc_nco(price_data):
    optimizer = PortfolioOptimizer(price_data)
    comparison = optimizer.compare_all()
    assert "herc" in comparison.columns
    assert "nco" in comparison.columns
    assert "hrp" in comparison.columns
    assert "risk_parity" in comparison.columns


def test_turnover_penalty_herc(price_data):
    current = pd.Series(
        np.ones(5) / 5,
        index=price_data.columns,
    )
    optimizer = PortfolioOptimizer(
        price_data, current_weights=current, turnover_threshold=0.01
    )
    weights = optimizer.optimize_with_turnover_penalty(method="herc")
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_turnover_penalty_nco(price_data):
    current = pd.Series(
        np.ones(5) / 5,
        index=price_data.columns,
    )
    optimizer = PortfolioOptimizer(
        price_data, current_weights=current, turnover_threshold=0.01
    )
    weights = optimizer.optimize_with_turnover_penalty(method="nco")
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
