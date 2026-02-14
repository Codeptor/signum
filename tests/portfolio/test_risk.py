import numpy as np
import pandas as pd
import pytest

from python.portfolio.risk import RiskEngine


@pytest.fixture
def returns_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        index=dates,
        columns=["AAPL", "MSFT", "GOOG"],
    )
    weights = pd.Series({"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25})
    return returns, weights


def test_var_parametric(returns_data):
    returns, weights = returns_data
    engine = RiskEngine(returns, weights)
    var = engine.var_parametric(confidence=0.95)
    assert var < 0  # VaR should be negative (loss)


def test_cvar(returns_data):
    returns, weights = returns_data
    engine = RiskEngine(returns, weights)
    cvar = engine.cvar_historical(confidence=0.95)
    var = engine.var_historical(confidence=0.95)
    assert cvar <= var  # CVaR is always worse than VaR


def test_max_drawdown(returns_data):
    returns, weights = returns_data
    engine = RiskEngine(returns, weights)
    dd = engine.max_drawdown()
    assert dd <= 0
    assert dd >= -1
