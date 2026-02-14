import numpy as np
import pandas as pd
import pytest
from python.monitoring.drift import DriftDetector


@pytest.fixture
def feature_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    reference = pd.DataFrame(
        np.random.randn(n, 3), index=dates[:n], columns=["feat_a", "feat_b", "feat_c"]
    )
    # Current data with drift in feat_a
    current = pd.DataFrame(
        np.random.randn(n, 3) + [2.0, 0.0, 0.0],
        index=dates[:n],
        columns=["feat_a", "feat_b", "feat_c"],
    )
    return reference, current


def test_detect_drift(feature_data):
    reference, current = feature_data
    detector = DriftDetector(reference, threshold=0.01)
    report = detector.detect(current)
    assert "feat_a" in report
    assert report["feat_a"]["drifted"] is True  # feat_a has a +2.0 shift
    assert report["feat_b"]["drifted"] is False
