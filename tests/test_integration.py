"""End-to-end integration test: data -> features -> model -> portfolio -> risk."""

import numpy as np
import pandas as pd
import pytest

from python.alpha.features import compute_alpha_features, compute_forward_returns
from python.alpha.model import CrossSectionalModel
from python.alpha.train import FEATURE_COLS
from python.bridge.bl_views import create_bl_views
from python.portfolio.optimizer import PortfolioOptimizer
from python.portfolio.risk import RiskEngine


@pytest.fixture
def synthetic_universe():
    """Generate a synthetic 10-stock universe with realistic-ish prices."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = [f"STOCK_{i}" for i in range(10)]
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.randn(252, 10) * 0.015, axis=0)) * 100,
        index=dates,
        columns=tickers,
    )
    ohlcv_frames = []
    for ticker in tickers:
        ohlcv_frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "open": prices[ticker] * (1 + np.random.randn(252) * 0.005),
                    "high": prices[ticker] * (1 + abs(np.random.randn(252)) * 0.01),
                    "low": prices[ticker] * (1 - abs(np.random.randn(252)) * 0.01),
                    "close": prices[ticker],
                    "volume": np.random.randint(100000, 1000000, 252).astype(float),
                },
                index=dates,
            )
        )
    return pd.concat(ohlcv_frames), prices


def test_full_pipeline(synthetic_universe):
    """Verify the complete signal -> portfolio -> risk pipeline works end-to-end."""
    ohlcv, prices = synthetic_universe

    # 1. Feature engineering
    features = compute_alpha_features(ohlcv)
    labeled = compute_forward_returns(features, horizon=5)
    labeled = labeled.dropna(subset=FEATURE_COLS + ["target_5d"])
    assert len(labeled) > 0

    # 2. Train model
    model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
    train = labeled.iloc[: int(len(labeled) * 0.8)]
    test = labeled.iloc[int(len(labeled) * 0.8) :]
    model.fit(train, target_col="target_5d")

    # 3. Generate predictions
    predictions = pd.Series(model.predict(test), index=test.index)
    # Get per-ticker averages for BL views
    test_preds = test.copy()
    test_preds["prediction"] = predictions.values
    ticker_preds = test_preds.groupby("ticker")["prediction"].mean()
    confidences = pd.Series(0.5, index=ticker_preds.index)

    # 4. Create BL views
    views, view_confs = create_bl_views(ticker_preds, confidences)
    assert len(views) > 0

    # 5. Portfolio optimization
    optimizer = PortfolioOptimizer(prices)
    hrp_weights = optimizer.hrp()
    assert abs(hrp_weights.sum() - 1.0) < 1e-6

    # 6. Risk metrics
    returns = prices.pct_change().dropna()
    engine = RiskEngine(returns, hrp_weights)
    summary = engine.summary()
    assert "sharpe" in summary
    assert "max_drawdown" in summary
    assert summary["max_drawdown"] <= 0
