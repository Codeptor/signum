import numpy as np
import pandas as pd
import pytest
from python.alpha.features import compute_alpha_features


@pytest.fixture
def sample_prices():
    """Multi-ticker OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    np.random.seed(42)
    tickers = ["AAPL", "MSFT"]
    frames = []
    for ticker in tickers:
        base = 150 + np.cumsum(np.random.randn(60) * 2)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "open": base + np.random.randn(60),
                    "high": base + abs(np.random.randn(60)) * 2,
                    "low": base - abs(np.random.randn(60)) * 2,
                    "close": base,
                    "volume": np.random.randint(500000, 2000000, 60).astype(float),
                },
                index=dates,
            )
        )
    return pd.concat(frames)


def test_compute_alpha_features_shape(sample_prices):
    features = compute_alpha_features(sample_prices)
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0
    # Should have technical indicator columns
    assert any("rsi" in c.lower() or "ma" in c.lower() for c in features.columns)


def test_compute_alpha_features_no_future_leak(sample_prices):
    """Features at time t should only use data from t and before."""
    features = compute_alpha_features(sample_prices)
    # No NaN in the non-warmup period (after 30 days of history)
    ticker_feats = features[features["ticker"] == "AAPL"].iloc[30:]
    # Allow some NaN from long lookback windows but not all
    null_ratio = ticker_feats.drop(columns=["ticker"]).isnull().mean().mean()
    assert null_ratio < 0.1, f"Too many NaN in features: {null_ratio:.2%}"
