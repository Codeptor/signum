"""Tests for data ingestion helpers: reshape_ohlcv_wide_to_long, extract_close_prices.

These pure-data functions sit on the critical path for every live and backtest
pipeline.  We test them with synthetic DataFrames — no network calls needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from python.data.ingestion import extract_close_prices, reshape_ohlcv_wide_to_long

# ---------------------------------------------------------------------------
# Helpers to build synthetic yfinance-style DataFrames
# ---------------------------------------------------------------------------


def _make_multiindex_ohlcv(
    tickers: list[str],
    n_days: int = 5,
    start: str = "2026-01-01",
) -> pd.DataFrame:
    """Build a DataFrame mimicking yfinance multi-ticker download (group_by='ticker').

    Columns are a MultiIndex: level 0 = ticker, level 1 = OHLCV field.
    """
    dates = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(42)

    tuples = []
    data = {}
    for ticker in tickers:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            col = (ticker, field)
            tuples.append(col)
            if field == "Volume":
                data[col] = rng.integers(100_000, 1_000_000, size=n_days).astype(float)
            else:
                data[col] = rng.uniform(100, 200, size=n_days)

    mi = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame(data, index=dates, columns=mi)
    return df


def _make_flat_ohlcv(n_days: int = 5) -> pd.DataFrame:
    """Build a flat (non-MultiIndex) DataFrame — single-ticker download."""
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Open": rng.uniform(100, 200, n_days),
            "High": rng.uniform(100, 200, n_days),
            "Low": rng.uniform(100, 200, n_days),
            "Close": rng.uniform(100, 200, n_days),
            "Volume": rng.integers(100_000, 1_000_000, n_days).astype(float),
        },
        index=dates,
    )


# ===========================================================================
# reshape_ohlcv_wide_to_long
# ===========================================================================


class TestReshapeOhlcvWideToLong:
    def test_basic_two_tickers(self):
        """Multi-ticker wide format → long format with 'ticker' column."""
        wide = _make_multiindex_ohlcv(["AAPL", "MSFT"], n_days=3)
        long = reshape_ohlcv_wide_to_long(wide)

        # Should have a 'ticker' column
        assert "ticker" in long.columns
        # Column names should be lowercased
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in long.columns

        # 2 tickers × 3 days = 6 rows
        assert len(long) == 6
        assert set(long["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_preserves_date_index(self):
        wide = _make_multiindex_ohlcv(["AAPL"], n_days=5)
        long = reshape_ohlcv_wide_to_long(wide)
        assert isinstance(long.index, pd.DatetimeIndex)
        assert len(long) == 5

    def test_flat_dataframe_passthrough(self):
        """A non-MultiIndex DataFrame is returned unchanged."""
        flat = _make_flat_ohlcv()
        result = reshape_ohlcv_wide_to_long(flat)
        pd.testing.assert_frame_equal(result, flat)

    def test_sorted_by_date(self):
        wide = _make_multiindex_ohlcv(["AAPL", "MSFT"], n_days=5)
        long = reshape_ohlcv_wide_to_long(wide)
        assert long.index.is_monotonic_increasing

    def test_single_ticker(self):
        wide = _make_multiindex_ohlcv(["AAPL"], n_days=3)
        long = reshape_ohlcv_wide_to_long(wide)
        assert len(long) == 3
        assert (long["ticker"] == "AAPL").all()

    def test_many_tickers(self):
        tickers = [f"T{i:03d}" for i in range(20)]
        wide = _make_multiindex_ohlcv(tickers, n_days=10)
        long = reshape_ohlcv_wide_to_long(wide)
        assert len(long) == 200  # 20 × 10
        assert set(long["ticker"].unique()) == set(tickers)


# ===========================================================================
# extract_close_prices
# ===========================================================================


class TestExtractClosePrices:
    def test_multiindex_extraction(self):
        """Extracts a (date × ticker) close matrix from MultiIndex DataFrame."""
        wide = _make_multiindex_ohlcv(["AAPL", "MSFT"], n_days=5)
        close = extract_close_prices(wide)

        assert isinstance(close, pd.DataFrame)
        assert set(close.columns) == {"AAPL", "MSFT"}
        assert len(close) == 5
        # Values should match the original Close column
        np.testing.assert_array_equal(close["AAPL"].values, wide[("AAPL", "Close")].values)
        np.testing.assert_array_equal(close["MSFT"].values, wide[("MSFT", "Close")].values)

    def test_flat_dataframe_passthrough(self):
        """Non-MultiIndex input returned as-is."""
        flat = _make_flat_ohlcv()
        result = extract_close_prices(flat)
        pd.testing.assert_frame_equal(result, flat)

    def test_preserves_date_index(self):
        wide = _make_multiindex_ohlcv(["AAPL"], n_days=3)
        close = extract_close_prices(wide)
        assert isinstance(close.index, pd.DatetimeIndex)

    def test_single_ticker(self):
        wide = _make_multiindex_ohlcv(["AAPL"], n_days=5)
        close = extract_close_prices(wide)
        assert list(close.columns) == ["AAPL"]
        assert len(close) == 5
