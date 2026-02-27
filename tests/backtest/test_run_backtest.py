"""Smoke test for run_backtest (P2-28: zero test coverage)."""

import numpy as np
import pandas as pd
import pytest

from python.backtest.run import run_backtest


@pytest.fixture
def synthetic_prices():
    """Generate synthetic price data: 10 tickers x 500 trading days."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2021-01-01", periods=500)
    tickers = [f"TICK{i}" for i in range(10)]
    data = {}
    for t in tickers:
        base = 100 + rng.randn(500).cumsum() * 0.5
        # Ensure prices are always positive
        data[t] = np.maximum(base, 5.0)
    return pd.DataFrame(data, index=dates)


def test_run_backtest_smoke(synthetic_prices):
    """Basic smoke test: run_backtest returns dict with expected keys."""
    result = run_backtest(
        prices=synthetic_prices,
        n_splits=2,
        top_n=5,
        rebalance_days=5,
        optimizer_method="equal_weight",
        transaction_cost_bps=10.0,
        max_weight=0.30,
        macro_path=None,  # skip macro features
        liquidity_filter_pct=0.0,
        vix_scaling=False,
        enable_risk_manager=False,
    )

    assert isinstance(result, dict)
    # Check essential keys
    for key in ["annualized_return", "sharpe_ratio", "max_drawdown", "avg_turnover"]:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], (int, float)), f"{key} should be numeric"

    # Sharpe should be finite
    assert np.isfinite(result["sharpe_ratio"])
