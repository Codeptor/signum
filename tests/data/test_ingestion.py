import pandas as pd
import pytest
from python.data.ingestion import fetch_sp500_tickers, fetch_ohlcv


def test_fetch_sp500_tickers_returns_list():
    tickers = fetch_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400
    assert "AAPL" in tickers


def test_fetch_ohlcv_returns_dataframe():
    df = fetch_ohlcv(["AAPL", "MSFT"], period="5d")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
