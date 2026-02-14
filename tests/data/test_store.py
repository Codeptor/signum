import pandas as pd
import pytest
from python.data.store import DataStore


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "open": [150.0, 151.0, 152.0, 153.0, 154.0],
            "high": [155.0, 156.0, 157.0, 158.0, 159.0],
            "low": [149.0, 150.0, 151.0, 152.0, 153.0],
            "close": [153.0, 154.0, 155.0, 156.0, 157.0],
            "volume": [1000000] * 5,
        },
        index=dates,
    )


def test_datastore_roundtrip_sqlite(sample_ohlcv, tmp_path):
    """Test storing and retrieving data with SQLite fallback."""
    store = DataStore(f"sqlite:///{tmp_path}/test.db")
    store.init_db()
    store.upsert_ohlcv(sample_ohlcv)
    result = store.get_ohlcv("AAPL", "2024-01-01", "2024-01-10")
    assert len(result) == 5
    assert result.iloc[0]["close"] == 153.0
