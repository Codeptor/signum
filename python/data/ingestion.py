"""Data ingestion from yfinance and other free sources."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from python.data.config import DEFAULT_INTERVAL, DEFAULT_PERIOD, RAW_DIR

logger = logging.getLogger(__name__)


def fetch_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, storage_options={"User-Agent": "quant-platform/0.1"})[0]
    return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())


def fetch_ohlcv(
    tickers: list[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Download OHLCV data for given tickers via yfinance."""
    logger.info(f"Fetching OHLCV for {len(tickers)} tickers, period={period}")
    df = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True)
    return df


def fetch_fred_macro() -> pd.DataFrame:
    """Fetch key macro indicators from FRED via yfinance."""
    macro_tickers = {
        "^VIX": "vix",
        "^TNX": "us10y",
        "^IRX": "us3m",
    }
    frames = {}
    for ticker, name in macro_tickers.items():
        data = yf.download(ticker, period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL)
        close = data["Close"]
        # yfinance may return DataFrame instead of Series; squeeze to 1-D
        if isinstance(close, pd.DataFrame):
            close = close.squeeze(axis=1)
        frames[name] = close
    return pd.DataFrame(frames)


def reshape_ohlcv_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance MultiIndex-column OHLCV to long format.

    yfinance with group_by='ticker' returns columns like (AAPL, Close), (AAPL, Open), ...
    This converts to rows with columns: [ticker, open, high, low, close, volume]
    and a DatetimeIndex.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df  # already flat

    frames = []
    tickers = df.columns.get_level_values(0).unique()
    for ticker in tickers:
        ticker_df = df[ticker].copy()
        ticker_df.columns = ticker_df.columns.str.lower()
        ticker_df["ticker"] = ticker
        frames.append(ticker_df)
    return pd.concat(frames).sort_index()


def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a simple (date x ticker) close-price matrix from yfinance output.

    Returns DataFrame with DatetimeIndex and one column per ticker.
    """
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique()
        return pd.DataFrame(
            {t: df[(t, "Close")].values for t in tickers},
            index=df.index,
        )
    return df


def save_raw_data(df: pd.DataFrame, name: str) -> Path:
    """Save DataFrame as Parquet in the raw data directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} rows to {path}")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickers = fetch_sp500_tickers()
    logger.info(f"Got {len(tickers)} S&P 500 tickers")
    ohlcv = fetch_ohlcv(tickers)
    save_raw_data(ohlcv, "sp500_ohlcv")
    macro = fetch_fred_macro()
    save_raw_data(macro, "macro_indicators")
