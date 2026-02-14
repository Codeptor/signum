import numpy as np
import pandas as pd
import pytest

# Skip all tests if pytorch-forecasting is not installed
pf = pytest.importorskip("pytorch_forecasting")

from python.alpha.tft_model import TFTAlphaModel


@pytest.fixture
def time_series_data():
    """Multi-ticker time series data for TFT testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG"]
    frames = []
    for ticker in tickers:
        base = 150 + np.cumsum(np.random.randn(200) * 2)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "close": base,
                    "ret_1d": np.random.randn(200) * 0.02,
                    "rsi_14": 50 + np.random.randn(200) * 10,
                    "volume_ratio": 1 + np.random.randn(200) * 0.3,
                    "target_5d": np.random.randn(200) * 0.02,
                },
                index=dates,
            )
        )
    return pd.concat(frames).sort_index()


def test_tft_model_creates_dataset(time_series_data):
    model = TFTAlphaModel(
        feature_cols=["ret_1d", "rsi_14", "volume_ratio"],
        target_col="target_5d",
        max_encoder_length=30,
    )
    train_dl, val_dl = model.prepare_data(time_series_data, val_frac=0.2)
    assert train_dl is not None
    assert val_dl is not None
