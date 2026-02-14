import numpy as np
import pandas as pd
import pytest
from python.alpha.model import CrossSectionalModel


@pytest.fixture
def training_data():
    """Synthetic labeled features for model testing."""
    np.random.seed(42)
    n = 500
    feature_cols = [f"feat_{i}" for i in range(10)]
    data = pd.DataFrame(np.random.randn(n, 10), columns=feature_cols)
    data["ticker"] = np.random.choice(["AAPL", "MSFT", "GOOG", "AMZN", "META"], n)
    data["target_5d"] = np.random.randn(n) * 0.02
    dates = np.repeat(pd.date_range("2024-01-01", periods=100, freq="B"), 5)
    data.index = dates
    return data, feature_cols


def test_model_train_and_predict(training_data):
    data, feature_cols = training_data
    model = CrossSectionalModel(model_type="lightgbm", feature_cols=feature_cols)

    train = data.iloc[:400]
    test = data.iloc[400:]

    model.fit(train, target_col="target_5d")
    preds = model.predict(test)

    assert len(preds) == len(test)
    assert preds.dtype == np.float64


def test_model_rank_predictions(training_data):
    data, feature_cols = training_data
    model = CrossSectionalModel(model_type="lightgbm", feature_cols=feature_cols)
    model.fit(data.iloc[:400], target_col="target_5d")
    ranks = model.predict_ranks(data.iloc[400:])
    # Ranks should be between 0 and 1
    assert ranks.min() >= 0
    assert ranks.max() <= 1
