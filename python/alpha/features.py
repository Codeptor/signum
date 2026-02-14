"""Technical feature computation inspired by Qlib Alpha158."""

import numpy as np
import pandas as pd


def compute_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features per ticker.

    Input: DataFrame with columns [ticker, open, high, low, close, volume] and DatetimeIndex.
    Output: DataFrame with original columns plus computed features.
    """
    results = []
    for ticker, group in df.groupby("ticker"):
        feats = _compute_single_ticker(group.copy())
        feats["ticker"] = ticker
        results.append(feats)
    return pd.concat(results).sort_index()


def _compute_single_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single ticker's OHLCV data."""
    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    # Returns
    for d in [1, 5, 10, 20]:
        df[f"ret_{d}d"] = c.pct_change(d)

    # Moving averages
    for w in [5, 10, 20, 60]:
        df[f"ma_{w}"] = c.rolling(w).mean()
        df[f"ma_ratio_{w}"] = c / c.rolling(w).mean()

    # Volatility
    for w in [5, 10, 20]:
        df[f"vol_{w}d"] = c.pct_change().rolling(w).std()

    # RSI
    for w in [14]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Bollinger Bands
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_position"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volume features
    df["volume_ma_10"] = v.rolling(10).mean()
    df["volume_ratio"] = v / v.rolling(10).mean()

    # High-low range
    df["hl_range"] = (h - lo) / c
    df["oc_range"] = (c - o) / c

    return df


def compute_forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Compute forward returns as the prediction target.

    IMPORTANT: This uses future data and must only be used for label creation,
    never as a feature.
    """
    results = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        g[f"target_{horizon}d"] = g["close"].pct_change(horizon).shift(-horizon)
        results.append(g)
    return pd.concat(results).sort_index()
