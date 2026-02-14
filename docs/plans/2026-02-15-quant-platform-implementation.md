# Quantitative Equity Platform — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end quantitative equity platform from data ingestion through ML alpha generation, portfolio optimization, and a Rust matching engine — as a portfolio project for quant engineering roles.

**Architecture:** Vertical slice approach — every layer is thin but connected end-to-end. Python handles ML, portfolio optimization, data, and monitoring. Rust handles the matching engine. Docker Compose provides infrastructure (TimescaleDB, Redis, MLflow). The "bridge" module connects ML signals to portfolio construction via Black-Litterman views.

**Tech Stack:** Python (Qlib, LightGBM, CatBoost, PyTorch Forecasting TFT, skfolio, Dash, Evidently), Rust (BTreeMap order book, Criterion.rs), TimescaleDB, Redis, MLflow, DVC

**Reference:** See `docs/plans/2026-02-15-quant-platform-design.md` for the full approved design.

---

## Phase 1: Project Scaffolding & Infrastructure

### Task 1: Initialize Python project with pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `python/__init__.py` (already exists)
- Create: `python/alpha/__init__.py`
- Create: `python/portfolio/__init__.py`
- Create: `python/data/__init__.py`
- Create: `python/backtest/__init__.py`
- Create: `python/monitoring/__init__.py`
- Create: `python/bridge/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "quant-platform"
version = "0.1.0"
description = "Quantitative equity platform: ML alpha, portfolio optimization, and execution"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "lightgbm>=4.0",
    "catboost>=1.2",
    "yfinance>=0.2",
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "redis>=5.0",
    "mlflow>=2.10",
    "plotly>=5.18",
    "dash>=2.14",
]

[project.optional-dependencies]
ml = [
    "qlib>=0.9",
    "pytorch-forecasting>=1.0",
    "torch>=2.1",
    "lightning>=2.1",
]
portfolio = [
    "skfolio>=0.4",
    "cvxpy>=1.4",
    "riskfolio-lib>=6.0",
]
monitoring = [
    "evidently>=0.4",
    "shap>=0.43",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.3",
]
all = ["quant-platform[ml,portfolio,monitoring,dev]"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["."]
include = ["python*"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**Step 2: Create all __init__.py package files**

Create empty `__init__.py` in each directory: `python/alpha/`, `python/portfolio/`, `python/data/`, `python/backtest/`, `python/monitoring/`, `python/bridge/`, `tests/`.

**Step 3: Install in dev mode and verify**

Run: `pip install -e ".[all]"` (may need to install some deps separately if they conflict)
Expected: Successful installation

**Step 4: Commit**

```bash
git add pyproject.toml python/ tests/__init__.py
git commit -m "scaffold: Python project structure with pyproject.toml"
```

---

### Task 2: Initialize Rust workspace and matching-engine crate

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `rust/matching-engine/Cargo.toml`
- Create: `rust/matching-engine/src/lib.rs`

**Step 1: Create workspace Cargo.toml**

```toml
[workspace]
members = ["rust/matching-engine"]
resolver = "2"
```

**Step 2: Create matching-engine crate**

`rust/matching-engine/Cargo.toml`:
```toml
[package]
name = "matching-engine"
version = "0.1.0"
edition = "2021"

[dependencies]
ordered-float = "4"
thiserror = "2"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
rand = "0.8"

[[bench]]
name = "orderbook_bench"
harness = false
```

`rust/matching-engine/src/lib.rs`:
```rust
//! Lock-free matching engine with price-time priority order book.

pub mod orderbook;
pub mod types;
```

**Step 3: Verify it compiles**

Run: `cargo check`
Expected: Successful compilation

**Step 4: Commit**

```bash
git add Cargo.toml rust/
git commit -m "scaffold: Rust workspace with matching-engine crate"
```

---

### Task 3: Docker Compose infrastructure

**Files:**
- Create: `infra/docker-compose.yml`
- Create: `.env.example`

**Step 1: Create docker-compose.yml**

```yaml
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: quant
      POSTGRES_PASSWORD: quant_dev
      POSTGRES_DB: quant_platform
    volumes:
      - timescale_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.16.0
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlartifacts
    volumes:
      - mlflow_data:/mlartifacts

volumes:
  timescale_data:
  mlflow_data:
```

**Step 2: Create .env.example**

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=quant
POSTGRES_PASSWORD=quant_dev
POSTGRES_DB=quant_platform
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
target/
.venv/
*.db
data/raw/
data/processed/
mlruns/
```

**Step 4: Verify docker-compose is valid**

Run: `docker-compose -f infra/docker-compose.yml config`
Expected: Valid YAML output

**Step 5: Commit**

```bash
git add infra/ .env.example .gitignore
git commit -m "scaffold: Docker Compose stack (TimescaleDB, Redis, MLflow)"
```

---

### Task 4: Makefile for orchestration

**Files:**
- Create: `Makefile`

**Step 1: Create Makefile**

```makefile
.PHONY: setup infra down ingest train optimize backtest dashboard test lint format bench

setup:
	pip install -e ".[all]"

infra:
	docker-compose -f infra/docker-compose.yml up -d

down:
	docker-compose -f infra/docker-compose.yml down

ingest:
	python -m python.data.ingestion

train:
	python -m python.alpha.train

optimize:
	python -m python.portfolio.optimizer

backtest:
	python -m python.backtest.run

dashboard:
	python -m python.monitoring.dashboard

test:
	pytest tests/ -v

lint:
	ruff check python/ tests/

format:
	ruff format python/ tests/

bench:
	cd rust && cargo bench
```

**Step 2: Commit**

```bash
git add Makefile
git commit -m "scaffold: Makefile for build orchestration"
```

---

## Phase 2: Data Ingestion Layer

### Task 5: yfinance data connector

**Files:**
- Create: `python/data/ingestion.py`
- Create: `python/data/config.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_ingestion.py`

**Step 1: Write the failing test**

`tests/data/test_ingestion.py`:
```python
import pandas as pd
import pytest
from python.data.ingestion import fetch_sp500_tickers, fetch_ohlcv


def test_fetch_sp500_tickers_returns_list():
    tickers = fetch_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400  # S&P 500 should have ~500 tickers
    assert "AAPL" in tickers


def test_fetch_ohlcv_returns_dataframe():
    df = fetch_ohlcv(["AAPL", "MSFT"], period="5d")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Close" in df.columns or ("Close", "AAPL") in df.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_ingestion.py -v`
Expected: FAIL — module not found

**Step 3: Write config.py**

`python/data/config.py`:
```python
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SP500_UNIVERSE = "sp500"
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"
```

**Step 4: Write ingestion.py**

`python/data/ingestion.py`:
```python
"""Data ingestion from yfinance and other free sources."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from python.data.config import DEFAULT_INTERVAL, DEFAULT_PERIOD, RAW_DIR

logger = logging.getLogger(__name__)


def fetch_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia."""
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
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
        frames[name] = data["Close"]
    return pd.DataFrame(frames)


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
```

**Step 5: Run tests**

Run: `pytest tests/data/test_ingestion.py -v`
Expected: PASS (requires internet connection)

**Step 6: Commit**

```bash
git add python/data/ tests/data/
git commit -m "feat: data ingestion layer with yfinance and macro indicators"
```

---

### Task 6: TimescaleDB storage layer

**Files:**
- Create: `python/data/store.py`
- Create: `python/data/models.py`
- Create: `tests/data/test_store.py`

**Step 1: Write the failing test**

`tests/data/test_store.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_store.py -v`
Expected: FAIL — module not found

**Step 3: Write models.py**

`python/data/models.py`:
```python
"""Database table definitions."""

from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class OHLCVBar(Base):
    __tablename__ = "ohlcv_bars"
    __table_args__ = (UniqueConstraint("ticker", "timestamp", name="uq_ticker_timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
```

**Step 4: Write store.py**

`python/data/store.py`:
```python
"""Data storage layer. Uses TimescaleDB in production, SQLite for testing."""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from python.data.models import Base, OHLCVBar

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, connection_string: str = "sqlite:///data/quant.db"):
        self.engine = create_engine(connection_string)

    def init_db(self):
        """Create tables if they don't exist."""
        Base.metadata.create_all(self.engine)

    def upsert_ohlcv(self, df: pd.DataFrame) -> int:
        """Insert or update OHLCV bars from a DataFrame.

        Expects columns: ticker, open, high, low, close, volume
        with a DatetimeIndex.
        """
        records = []
        for ts, row in df.iterrows():
            records.append(
                OHLCVBar(
                    ticker=row["ticker"],
                    timestamp=ts,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
            )

        with Session(self.engine) as session:
            for record in records:
                existing = session.execute(
                    select(OHLCVBar).where(
                        OHLCVBar.ticker == record.ticker,
                        OHLCVBar.timestamp == record.timestamp,
                    )
                ).scalar_one_or_none()
                if existing:
                    existing.open = record.open
                    existing.high = record.high
                    existing.low = record.low
                    existing.close = record.close
                    existing.volume = record.volume
                else:
                    session.add(record)
            session.commit()

        logger.info(f"Upserted {len(records)} OHLCV bars")
        return len(records)

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Retrieve OHLCV bars for a ticker within a date range."""
        with Session(self.engine) as session:
            stmt = (
                select(OHLCVBar)
                .where(
                    OHLCVBar.ticker == ticker,
                    OHLCVBar.timestamp >= datetime.fromisoformat(start_date),
                    OHLCVBar.timestamp <= datetime.fromisoformat(end_date),
                )
                .order_by(OHLCVBar.timestamp)
            )
            rows = session.execute(stmt).scalars().all()

        return pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "ticker": r.ticker,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]
        )
```

**Step 5: Run tests**

Run: `pytest tests/data/test_store.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add python/data/models.py python/data/store.py tests/data/test_store.py
git commit -m "feat: data storage layer with SQLAlchemy (TimescaleDB/SQLite)"
```

---

## Phase 3: ML Alpha Generation

### Task 7: Feature engineering with Alpha158-style factors

**Files:**
- Create: `python/alpha/__init__.py`
- Create: `python/alpha/features.py`
- Create: `tests/alpha/__init__.py`
- Create: `tests/alpha/test_features.py`

**Step 1: Write the failing test**

`tests/alpha/test_features.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/alpha/test_features.py -v`
Expected: FAIL

**Step 3: Implement feature engineering**

`python/alpha/features.py`:
```python
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

    # Rank features (cross-sectional ranking happens in the training pipeline)
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
```

**Step 4: Run tests**

Run: `pytest tests/alpha/test_features.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/alpha/features.py tests/alpha/
git commit -m "feat: Alpha158-inspired technical feature engineering"
```

---

### Task 8: LightGBM/CatBoost cross-sectional model

**Files:**
- Create: `python/alpha/model.py`
- Create: `python/alpha/train.py`
- Create: `tests/alpha/test_model.py`

**Step 1: Write the failing test**

`tests/alpha/test_model.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/alpha/test_model.py -v`
Expected: FAIL

**Step 3: Implement model.py**

`python/alpha/model.py`:
```python
"""Cross-sectional return prediction models."""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossSectionalModel:
    """Wraps LightGBM or CatBoost for cross-sectional equity return prediction."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        feature_cols: list[str] | None = None,
        params: dict | None = None,
    ):
        self.model_type = model_type
        self.feature_cols = feature_cols
        self.model = None
        self.params = params or self._default_params()

    def _default_params(self) -> dict:
        if self.model_type == "lightgbm":
            return {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbose": -1,
                "n_estimators": 200,
            }
        elif self.model_type == "catboost":
            return {
                "iterations": 200,
                "learning_rate": 0.05,
                "depth": 6,
                "verbose": 0,
            }
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, df: pd.DataFrame, target_col: str = "target_5d") -> None:
        """Train on labeled cross-sectional data."""
        X = df[self.feature_cols].values
        y = df[target_col].values

        mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        X, y = X[mask], y[mask]

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X, y)
        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor

            self.model = CatBoostRegressor(**self.params)
            self.model.fit(X, y)

        logger.info(f"Trained {self.model_type} on {len(y)} samples")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict raw scores."""
        X = df[self.feature_cols].values
        return self.model.predict(X).astype(np.float64)

    def predict_ranks(self, df: pd.DataFrame) -> pd.Series:
        """Predict and rank within each date cross-section (0=worst, 1=best)."""
        preds = pd.Series(self.predict(df), index=df.index)
        ranks = preds.groupby(preds.index).rank(pct=True)
        return ranks

    def feature_importance(self) -> pd.Series:
        """Return feature importance scores."""
        if self.model_type == "lightgbm":
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
        elif self.model_type == "catboost":
            return pd.Series(
                self.model.get_feature_importance(),
                index=self.feature_cols,
            ).sort_values(ascending=False)
```

**Step 4: Write train.py entry point**

`python/alpha/train.py`:
```python
"""Training pipeline for alpha models with MLflow tracking."""

import logging
from pathlib import Path

import mlflow
import pandas as pd

from python.alpha.features import compute_alpha_features, compute_forward_returns
from python.alpha.model import CrossSectionalModel

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
    "vol_5d", "vol_10d", "vol_20d",
    "rsi_14", "macd", "macd_signal",
    "bb_position", "volume_ratio", "hl_range", "oc_range",
]


def run_training(data_path: str = "data/raw/sp500_ohlcv.parquet"):
    """Full training pipeline: load data → features → train → log to MLflow."""
    raw = pd.read_parquet(data_path)
    featured = compute_alpha_features(raw)
    labeled = compute_forward_returns(featured, horizon=5)
    labeled = labeled.dropna(subset=FEATURE_COLS + ["target_5d"])

    # Time-based split: last 20% for validation
    split_idx = int(len(labeled) * 0.8)
    train = labeled.iloc[:split_idx]
    val = labeled.iloc[split_idx:]

    with mlflow.start_run(run_name="lgbm_alpha158"):
        model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
        model.fit(train, target_col="target_5d")

        val_preds = model.predict(val)
        ic = pd.Series(val_preds, index=val.index).corr(val["target_5d"])

        mlflow.log_params(model.params)
        mlflow.log_metric("information_coefficient", ic)
        mlflow.log_metric("train_size", len(train))
        mlflow.log_metric("val_size", len(val))

        importance = model.feature_importance()
        logger.info(f"Top features:\n{importance.head(10)}")
        logger.info(f"Validation IC: {ic:.4f}")

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
```

**Step 5: Run tests**

Run: `pytest tests/alpha/test_model.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add python/alpha/model.py python/alpha/train.py tests/alpha/test_model.py
git commit -m "feat: LightGBM cross-sectional model with MLflow tracking"
```

---

### Task 9: Temporal Fusion Transformer model

**Files:**
- Create: `python/alpha/tft_model.py`
- Create: `tests/alpha/test_tft.py`

**Step 1: Write the failing test**

`tests/alpha/test_tft.py`:
```python
import numpy as np
import pandas as pd
import pytest
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/alpha/test_tft.py -v`
Expected: FAIL

**Step 3: Implement TFT model wrapper**

`python/alpha/tft_model.py`:
```python
"""Temporal Fusion Transformer wrapper for equity return prediction."""

import logging

import numpy as np
import pandas as pd
import pytorch_forecasting as pf
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

logger = logging.getLogger(__name__)


class TFTAlphaModel:
    """Wraps PyTorch Forecasting's TFT for equity prediction."""

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str = "target_5d",
        max_encoder_length: int = 30,
        max_prediction_length: int = 5,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.model = None
        self.training_dataset = None

    def prepare_data(
        self, df: pd.DataFrame, val_frac: float = 0.2
    ) -> tuple:
        """Create TimeSeriesDataSet from DataFrame.

        Expects: DatetimeIndex, columns [ticker, close, *features, target_col].
        """
        data = df.copy()
        data["time_idx"] = data.groupby("ticker").cumcount()
        data["date"] = data.index

        max_time = data["time_idx"].max()
        train_cutoff = int(max_time * (1 - val_frac))

        training = TimeSeriesDataSet(
            data[data["time_idx"] <= train_cutoff],
            time_idx="time_idx",
            target=self.target_col,
            group_ids=["ticker"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=self.feature_cols + [self.target_col],
            static_categoricals=["ticker"],
            target_normalizer=GroupNormalizer(groups=["ticker"]),
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            data[data["time_idx"] > train_cutoff],
            stop_randomization=True,
        )

        self.training_dataset = training

        train_dl = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dl = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

        return train_dl, val_dl

    def build_model(self) -> TemporalFusionTransformer:
        """Create TFT model from training dataset."""
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=0.001,
            hidden_size=32,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,  # Quantile predictions
            loss=pf.metrics.QuantileLoss(),
            reduce_on_plateau_patience=3,
        )
        return self.model
```

**Step 4: Run tests**

Run: `pytest tests/alpha/test_tft.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/alpha/tft_model.py tests/alpha/test_tft.py
git commit -m "feat: Temporal Fusion Transformer model wrapper"
```

---

## Phase 4: Portfolio Optimization

### Task 10: Black-Litterman bridge (ML signals → views)

**Files:**
- Create: `python/bridge/bl_views.py`
- Create: `tests/bridge/__init__.py`
- Create: `tests/bridge/test_bl_views.py`

**Step 1: Write the failing test**

`tests/bridge/test_bl_views.py`:
```python
import numpy as np
import pandas as pd
import pytest
from python.bridge.bl_views import create_bl_views


def test_create_bl_views():
    """ML predictions should convert to Black-Litterman absolute views."""
    predictions = pd.Series(
        {"AAPL": 0.02, "MSFT": 0.01, "GOOG": -0.005, "AMZN": 0.015, "META": -0.01},
        name="predicted_return",
    )
    confidences = pd.Series(
        {"AAPL": 0.8, "MSFT": 0.6, "GOOG": 0.3, "AMZN": 0.7, "META": 0.4},
        name="confidence",
    )

    views, view_confidences = create_bl_views(predictions, confidences)

    assert len(views) == 5
    assert len(view_confidences) == 5
    # Higher confidence → lower uncertainty
    assert view_confidences["AAPL"] < view_confidences["GOOG"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/bridge/test_bl_views.py -v`
Expected: FAIL

**Step 3: Implement BL views bridge**

`python/bridge/bl_views.py`:
```python
"""Bridge between ML model predictions and Black-Litterman views.

Converts ML return predictions + confidence scores into the views format
expected by portfolio optimization (skfolio / Riskfolio-Lib).
"""

import numpy as np
import pandas as pd


def create_bl_views(
    predictions: pd.Series,
    confidences: pd.Series,
    tau: float = 0.05,
    base_uncertainty: float = 0.1,
) -> tuple[pd.Series, pd.Series]:
    """Convert ML predictions to Black-Litterman absolute views.

    Args:
        predictions: Expected returns per ticker from ML model.
        confidences: Confidence scores per ticker (0-1 scale).
        tau: Scaling factor for view uncertainty (smaller = more certain).
        base_uncertainty: Base uncertainty when confidence is 0.

    Returns:
        views: Predicted returns (same as input predictions).
        view_confidences: Uncertainty per view (lower = more confident).
    """
    # Convert confidence (0=uncertain, 1=certain) to uncertainty (high=uncertain)
    # Inverse mapping: high confidence → low uncertainty
    uncertainties = base_uncertainty * (1 - confidences) * tau + 1e-6

    return predictions, uncertainties


def create_picking_matrix(
    tickers: list[str],
    views: pd.Series,
) -> np.ndarray:
    """Create the picking matrix P for Black-Litterman.

    For absolute views (each view is about one asset), P is an identity matrix
    over the assets with views.
    """
    n_views = len(views)
    n_assets = len(tickers)
    P = np.zeros((n_views, n_assets))
    for i, ticker in enumerate(views.index):
        j = tickers.index(ticker)
        P[i, j] = 1.0
    return P
```

**Step 4: Run tests**

Run: `pytest tests/bridge/test_bl_views.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/bridge/ tests/bridge/
git commit -m "feat: Black-Litterman bridge from ML predictions to portfolio views"
```

---

### Task 11: Portfolio optimizer (HRP, CVaR, Black-Litterman)

**Files:**
- Create: `python/portfolio/optimizer.py`
- Create: `tests/portfolio/__init__.py`
- Create: `tests/portfolio/test_optimizer.py`

**Step 1: Write the failing test**

`tests/portfolio/test_optimizer.py`:
```python
import numpy as np
import pandas as pd
import pytest
from python.portfolio.optimizer import PortfolioOptimizer


@pytest.fixture
def price_data():
    """Synthetic price data for 5 assets."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.randn(252, 5) * 0.01, axis=0)) * 100,
        index=dates,
        columns=tickers,
    )
    return prices


def test_hrp_allocation(price_data):
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.hrp()
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()


def test_cvar_allocation(price_data):
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.min_cvar()
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= -0.01).all()  # Allow tiny numerical errors


def test_black_litterman_allocation(price_data):
    views = pd.Series({"AAPL": 0.02, "MSFT": 0.01, "GOOG": -0.005})
    view_confidences = pd.Series({"AAPL": 0.001, "MSFT": 0.002, "GOOG": 0.005})
    optimizer = PortfolioOptimizer(price_data)
    weights = optimizer.black_litterman(views, view_confidences)
    assert isinstance(weights, pd.Series)
    assert abs(weights.sum() - 1.0) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/portfolio/test_optimizer.py -v`
Expected: FAIL

**Step 3: Implement optimizer**

`python/portfolio/optimizer.py`:
```python
"""Portfolio optimization: HRP, CVaR, Black-Litterman with ML views."""

import logging

import numpy as np
import pandas as pd
from skfolio import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity, MeanRisk
from skfolio.prior import BlackLitterman, EmpiricalPrior

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Multi-strategy portfolio optimizer using skfolio."""

    def __init__(self, prices: pd.DataFrame):
        """Initialize with a DataFrame of asset prices (columns=tickers, index=dates)."""
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.tickers = list(prices.columns)

    def hrp(self) -> pd.Series:
        """Hierarchical Risk Parity allocation."""
        model = HierarchicalRiskParity()
        model.fit(self.returns)
        return pd.Series(model.weights_, index=self.tickers, name="hrp_weights")

    def min_cvar(self, confidence_level: float = 0.95) -> pd.Series:
        """Minimum CVaR (Conditional Value at Risk) allocation."""
        model = MeanRisk(
            risk_measure=RiskMeasure.CVAR,
            min_weights=0.0,
        )
        model.fit(self.returns)
        return pd.Series(model.weights_, index=self.tickers, name="min_cvar_weights")

    def black_litterman(
        self,
        views: pd.Series,
        view_confidences: pd.Series,
    ) -> pd.Series:
        """Black-Litterman allocation with ML model views.

        Args:
            views: Expected returns for a subset of assets.
            view_confidences: Uncertainty per view (lower = more confident).
        """
        n_assets = len(self.tickers)
        n_views = len(views)

        # Build picking matrix (absolute views)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega = np.zeros((n_views, n_views))

        for i, (ticker, ret) in enumerate(views.items()):
            j = self.tickers.index(ticker)
            P[i, j] = 1.0
            Q[i] = ret
            omega[i, i] = view_confidences[ticker] ** 2

        prior_model = BlackLitterman(
            prior_estimator=EmpiricalPrior(),
            investor_views=P,
            expected_returns=Q,
            covariance=omega,
        )

        model = MeanRisk(
            risk_measure=RiskMeasure.CVAR,
            prior_estimator=prior_model,
            min_weights=0.0,
        )
        model.fit(self.returns)
        return pd.Series(model.weights_, index=self.tickers, name="bl_weights")

    def risk_parity(self) -> pd.Series:
        """Equal risk contribution allocation."""
        # Use HRP as a proxy for risk parity (simpler, similar concept)
        model = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
        model.fit(self.returns)
        return pd.Series(model.weights_, index=self.tickers, name="risk_parity_weights")

    def compare_all(
        self,
        views: pd.Series | None = None,
        view_confidences: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Run all optimization strategies and return weights comparison."""
        results = {"hrp": self.hrp(), "min_cvar": self.min_cvar()}
        if views is not None and view_confidences is not None:
            results["black_litterman"] = self.black_litterman(views, view_confidences)
        results["risk_parity"] = self.risk_parity()
        return pd.DataFrame(results)
```

**Step 4: Run tests**

Run: `pytest tests/portfolio/test_optimizer.py -v`
Expected: PASS

Note: skfolio's Black-Litterman API may differ slightly. Adjust parameter names if needed based on the installed version. Consult Context7 for the latest skfolio API docs if tests fail.

**Step 5: Commit**

```bash
git add python/portfolio/optimizer.py tests/portfolio/
git commit -m "feat: portfolio optimizer with HRP, CVaR, and Black-Litterman"
```

---

### Task 12: Risk metrics module

**Files:**
- Create: `python/portfolio/risk.py`
- Create: `tests/portfolio/test_risk.py`

**Step 1: Write the failing test**

`tests/portfolio/test_risk.py`:
```python
import numpy as np
import pandas as pd
import pytest
from python.portfolio.risk import RiskEngine


@pytest.fixture
def returns_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        index=dates,
        columns=["AAPL", "MSFT", "GOOG"],
    )
    weights = pd.Series({"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25})
    return returns, weights


def test_var_parametric(returns_data):
    returns, weights = returns_data
    engine = RiskEngine(returns, weights)
    var = engine.var_parametric(confidence=0.95)
    assert var < 0  # VaR should be negative (loss)


def test_cvar(returns_data):
    returns, weights = returns_data
    engine = RiskEngine(returns, weights)
    cvar = engine.cvar_historical(confidence=0.95)
    var = engine.var_historical(confidence=0.95)
    assert cvar <= var  # CVaR is always worse than VaR


def test_max_drawdown(returns_data):
    returns, weights = returns_data
    engine = RiskEngine(returns, weights)
    dd = engine.max_drawdown()
    assert dd <= 0
    assert dd >= -1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/portfolio/test_risk.py -v`
Expected: FAIL

**Step 3: Implement risk engine**

`python/portfolio/risk.py`:
```python
"""Risk metrics: VaR, CVaR, drawdown, concentration."""

import numpy as np
import pandas as pd
from scipy import stats


class RiskEngine:
    """Portfolio risk calculation engine."""

    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns
        self.weights = weights
        self.portfolio_returns = (returns * weights).sum(axis=1)

    def var_parametric(self, confidence: float = 0.95) -> float:
        """Parametric VaR assuming normal distribution."""
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        return stats.norm.ppf(1 - confidence, mu, sigma)

    def var_historical(self, confidence: float = 0.95) -> float:
        """Historical simulation VaR."""
        return float(np.percentile(self.portfolio_returns, (1 - confidence) * 100))

    def cvar_historical(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) via historical simulation."""
        var = self.var_historical(confidence)
        tail = self.portfolio_returns[self.portfolio_returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def max_drawdown(self) -> float:
        """Maximum drawdown of the portfolio."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        return float(drawdowns.min())

    def rolling_sharpe(self, window: int = 60, risk_free: float = 0.0) -> pd.Series:
        """Rolling Sharpe ratio."""
        excess = self.portfolio_returns - risk_free / 252
        rolling_mean = excess.rolling(window).mean() * 252
        rolling_std = excess.rolling(window).std() * np.sqrt(252)
        return rolling_mean / rolling_std

    def concentration(self) -> pd.Series:
        """Herfindahl-Hirschman Index and effective number of bets."""
        hhi = (self.weights ** 2).sum()
        return pd.Series({"hhi": hhi, "effective_n": 1 / hhi})

    def summary(self) -> dict:
        """Full risk summary."""
        return {
            "var_95_parametric": self.var_parametric(0.95),
            "var_95_historical": self.var_historical(0.95),
            "cvar_95": self.cvar_historical(0.95),
            "max_drawdown": self.max_drawdown(),
            "annualized_vol": self.portfolio_returns.std() * np.sqrt(252),
            "annualized_return": self.portfolio_returns.mean() * 252,
            "sharpe": (self.portfolio_returns.mean() * 252)
            / (self.portfolio_returns.std() * np.sqrt(252)),
            "hhi": self.concentration()["hhi"],
        }
```

**Step 4: Run tests**

Run: `pytest tests/portfolio/test_risk.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add python/portfolio/risk.py tests/portfolio/test_risk.py
git commit -m "feat: risk engine with VaR, CVaR, drawdown, and Sharpe"
```

---

## Phase 5: Rust Matching Engine

### Task 13: Order book types and data structures

**Files:**
- Create: `rust/matching-engine/src/types.rs`
- Modify: `rust/matching-engine/src/lib.rs`

**Step 1: Write the types**

`rust/matching-engine/src/types.rs`:
```rust
use ordered_float::OrderedFloat;
use std::fmt;

pub type Price = OrderedFloat<f64>;
pub type Quantity = u64;
pub type OrderId = u64;
pub type Timestamp = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    ImmediateOrCancel,
    FillOrKill,
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: OrderId,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub remaining: Quantity,
    pub order_type: OrderType,
    pub timestamp: Timestamp,
}

impl Order {
    pub fn new(
        id: OrderId,
        side: Side,
        price: f64,
        quantity: Quantity,
        order_type: OrderType,
        timestamp: Timestamp,
    ) -> Self {
        Self {
            id,
            side,
            price: OrderedFloat(price),
            quantity,
            remaining: quantity,
            order_type,
            timestamp,
        }
    }

    pub fn is_filled(&self) -> bool {
        self.remaining == 0
    }
}

#[derive(Debug, Clone)]
pub struct Execution {
    pub buy_order_id: OrderId,
    pub sell_order_id: OrderId,
    pub price: Price,
    pub quantity: Quantity,
    pub timestamp: Timestamp,
}

impl fmt::Display for Execution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trade: buy={} sell={} price={} qty={}",
            self.buy_order_id, self.sell_order_id, self.price, self.quantity
        )
    }
}

#[derive(Debug)]
pub struct L2Snapshot {
    pub bids: Vec<(Price, Quantity)>,
    pub asks: Vec<(Price, Quantity)>,
}
```

**Step 2: Verify it compiles**

Run: `cargo check`
Expected: PASS

**Step 3: Commit**

```bash
git add rust/
git commit -m "feat(rust): order book types and data structures"
```

---

### Task 14: Order book matching logic

**Files:**
- Create: `rust/matching-engine/src/orderbook.rs`
- Modify: `rust/matching-engine/src/lib.rs`

**Step 1: Write tests first in the orderbook module**

`rust/matching-engine/src/orderbook.rs`:
```rust
use std::collections::{BTreeMap, VecDeque};

use crate::types::*;

pub struct OrderBook {
    bids: BTreeMap<Price, VecDeque<Order>>,
    asks: BTreeMap<Price, VecDeque<Order>>,
    orders: std::collections::HashMap<OrderId, (Side, Price)>,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: std::collections::HashMap::new(),
        }
    }

    pub fn submit(&mut self, order: Order, timestamp: Timestamp) -> Vec<Execution> {
        match order.order_type {
            OrderType::Limit => self.process_limit(order, timestamp),
            OrderType::Market => self.process_market(order, timestamp),
            OrderType::ImmediateOrCancel => self.process_ioc(order, timestamp),
            OrderType::FillOrKill => self.process_fok(order, timestamp),
        }
    }

    pub fn cancel(&mut self, order_id: OrderId) -> bool {
        if let Some((side, price)) = self.orders.remove(&order_id) {
            let book = match side {
                Side::Buy => &mut self.bids,
                Side::Sell => &mut self.asks,
            };
            if let Some(level) = book.get_mut(&price) {
                level.retain(|o| o.id != order_id);
                if level.is_empty() {
                    book.remove(&price);
                }
            }
            true
        } else {
            false
        }
    }

    pub fn best_bid(&self) -> Option<Price> {
        self.bids.keys().next_back().copied()
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks.keys().next().copied()
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((ask - bid).into_inner()),
            _ => None,
        }
    }

    pub fn snapshot(&self, depth: usize) -> L2Snapshot {
        let bids: Vec<(Price, Quantity)> = self
            .bids
            .iter()
            .rev()
            .take(depth)
            .map(|(&p, q)| (p, q.iter().map(|o| o.remaining).sum()))
            .collect();

        let asks: Vec<(Price, Quantity)> = self
            .asks
            .iter()
            .take(depth)
            .map(|(&p, q)| (p, q.iter().map(|o| o.remaining).sum()))
            .collect();

        L2Snapshot { bids, asks }
    }

    fn process_limit(&mut self, mut order: Order, timestamp: Timestamp) -> Vec<Execution> {
        let mut executions = self.try_match(&mut order, timestamp);
        if order.remaining > 0 {
            self.add_to_book(order);
        }
        executions
    }

    fn process_market(&mut self, mut order: Order, timestamp: Timestamp) -> Vec<Execution> {
        self.try_match(&mut order, timestamp)
    }

    fn process_ioc(&mut self, mut order: Order, timestamp: Timestamp) -> Vec<Execution> {
        let executions = self.try_match(&mut order, timestamp);
        // IOC: any unfilled portion is cancelled (not added to book)
        executions
    }

    fn process_fok(&mut self, order: Order, timestamp: Timestamp) -> Vec<Execution> {
        // FOK: must fill entirely or not at all
        let available = self.available_quantity(&order);
        if available >= order.remaining {
            let mut order = order;
            self.try_match(&mut order, timestamp)
        } else {
            vec![]
        }
    }

    fn available_quantity(&self, order: &Order) -> Quantity {
        let book = match order.side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };

        let mut total = 0u64;
        for (&price, level) in book.iter() {
            let crosses = match order.side {
                Side::Buy => price <= order.price,
                Side::Sell => price >= order.price,
            };
            if !crosses && order.order_type != OrderType::Market {
                break;
            }
            total += level.iter().map(|o| o.remaining).sum::<u64>();
        }
        total
    }

    fn try_match(&mut self, order: &mut Order, timestamp: Timestamp) -> Vec<Execution> {
        let mut executions = Vec::new();
        let opposite = match order.side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };

        let mut empty_levels = Vec::new();

        let iter: Vec<Price> = match order.side {
            Side::Buy => opposite.keys().copied().collect(),
            Side::Sell => opposite.keys().rev().copied().collect(),
        };

        for price in iter {
            if order.remaining == 0 {
                break;
            }

            let crosses = match order.side {
                Side::Buy => price <= order.price || order.order_type == OrderType::Market,
                Side::Sell => price >= order.price || order.order_type == OrderType::Market,
            };
            if !crosses {
                break;
            }

            if let Some(level) = opposite.get_mut(&price) {
                while order.remaining > 0 && !level.is_empty() {
                    let resting = level.front_mut().unwrap();
                    let fill_qty = order.remaining.min(resting.remaining);

                    order.remaining -= fill_qty;
                    resting.remaining -= fill_qty;

                    let (buy_id, sell_id) = match order.side {
                        Side::Buy => (order.id, resting.id),
                        Side::Sell => (resting.id, order.id),
                    };

                    executions.push(Execution {
                        buy_order_id: buy_id,
                        sell_order_id: sell_id,
                        price,
                        quantity: fill_qty,
                        timestamp,
                    });

                    if resting.is_filled() {
                        let filled = level.pop_front().unwrap();
                        self.orders.remove(&filled.id);
                    }
                }
                if level.is_empty() {
                    empty_levels.push(price);
                }
            }
        }

        let opposite = match order.side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };
        for price in empty_levels {
            opposite.remove(&price);
        }

        executions
    }

    fn add_to_book(&mut self, order: Order) {
        let book = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        self.orders.insert(order.id, (order.side, order.price));
        book.entry(order.price).or_default().push_back(order);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limit_buy(id: u64, price: f64, qty: u64) -> Order {
        Order::new(id, Side::Buy, price, qty, OrderType::Limit, 0)
    }

    fn limit_sell(id: u64, price: f64, qty: u64) -> Order {
        Order::new(id, Side::Sell, price, qty, OrderType::Limit, 0)
    }

    #[test]
    fn test_limit_order_no_match() {
        let mut book = OrderBook::new();
        let execs = book.submit(limit_buy(1, 100.0, 10), 1);
        assert!(execs.is_empty());
        assert_eq!(book.best_bid(), Some(OrderedFloat(100.0)));
    }

    #[test]
    fn test_limit_order_exact_match() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 10), 1);
        let execs = book.submit(limit_buy(2, 100.0, 10), 2);
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].quantity, 10);
        assert_eq!(execs[0].price, OrderedFloat(100.0));
    }

    #[test]
    fn test_partial_fill() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 10), 1);
        let execs = book.submit(limit_buy(2, 100.0, 5), 2);
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].quantity, 5);
        // Remaining 5 should still be on the ask side
        assert_eq!(book.best_ask(), Some(OrderedFloat(100.0)));
    }

    #[test]
    fn test_price_time_priority() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 101.0, 10), 1);
        book.submit(limit_sell(2, 100.0, 5), 2);  // Better price
        book.submit(limit_sell(3, 100.0, 5), 3);  // Same price, later time

        let execs = book.submit(limit_buy(4, 101.0, 8), 4);
        // Should fill at 100.0 first (price priority)
        assert_eq!(execs[0].price, OrderedFloat(100.0));
        assert_eq!(execs[0].sell_order_id, 2); // Time priority
    }

    #[test]
    fn test_market_order() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 5), 1);
        book.submit(limit_sell(2, 101.0, 5), 2);

        let market = Order::new(3, Side::Buy, 0.0, 8, OrderType::Market, 3);
        let execs = book.submit(market, 3);
        assert_eq!(execs.len(), 2);
        assert_eq!(execs[0].quantity, 5);
        assert_eq!(execs[1].quantity, 3);
    }

    #[test]
    fn test_cancel_order() {
        let mut book = OrderBook::new();
        book.submit(limit_buy(1, 100.0, 10), 1);
        assert!(book.cancel(1));
        assert_eq!(book.best_bid(), None);
        assert!(!book.cancel(1)); // Already cancelled
    }

    #[test]
    fn test_ioc_partial_fill() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 5), 1);

        let ioc = Order::new(2, Side::Buy, 100.0, 10, OrderType::ImmediateOrCancel, 2);
        let execs = book.submit(ioc, 2);
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].quantity, 5);
        // Unfilled 5 should NOT be on the book
        assert_eq!(book.best_bid(), None);
    }

    #[test]
    fn test_fok_rejected() {
        let mut book = OrderBook::new();
        book.submit(limit_sell(1, 100.0, 5), 1);

        let fok = Order::new(2, Side::Buy, 100.0, 10, OrderType::FillOrKill, 2);
        let execs = book.submit(fok, 2);
        assert!(execs.is_empty()); // Not enough liquidity
        // Original sell should still be there
        assert_eq!(book.best_ask(), Some(OrderedFloat(100.0)));
    }

    #[test]
    fn test_spread() {
        let mut book = OrderBook::new();
        book.submit(limit_buy(1, 99.0, 10), 1);
        book.submit(limit_sell(2, 101.0, 10), 2);
        assert_eq!(book.spread(), Some(2.0));
    }

    #[test]
    fn test_l2_snapshot() {
        let mut book = OrderBook::new();
        book.submit(limit_buy(1, 99.0, 10), 1);
        book.submit(limit_buy(2, 98.0, 20), 2);
        book.submit(limit_sell(3, 101.0, 15), 3);

        let snap = book.snapshot(5);
        assert_eq!(snap.bids.len(), 2);
        assert_eq!(snap.asks.len(), 1);
        assert_eq!(snap.bids[0].0, OrderedFloat(99.0)); // Best bid first
    }
}
```

**Step 2: Run tests**

Run: `cargo test`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add rust/
git commit -m "feat(rust): lock-free order book with price-time priority matching"
```

---

### Task 15: Criterion benchmarks for the matching engine

**Files:**
- Create: `rust/matching-engine/benches/orderbook_bench.rs`

**Step 1: Write benchmarks**

`rust/matching-engine/benches/orderbook_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use matching_engine::orderbook::OrderBook;
use matching_engine::types::*;
use rand::Rng;

fn bench_insert_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_limit");

    group.bench_function("single_insert", |b| {
        let mut book = OrderBook::new();
        let mut id = 0u64;
        b.iter(|| {
            id += 1;
            let order = Order::new(id, Side::Buy, 100.0, 10, OrderType::Limit, id);
            black_box(book.submit(order, id));
        });
    });

    group.finish();
}

fn bench_match_market(c: &mut Criterion) {
    let mut group = c.benchmark_group("match_market");

    for depth in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            &depth,
            |b, &depth| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let mut book = OrderBook::new();
                        // Populate ask side
                        for i in 0..depth {
                            let order = Order::new(
                                i as u64,
                                Side::Sell,
                                100.0 + (i as f64) * 0.01,
                                100,
                                OrderType::Limit,
                                i as u64,
                            );
                            book.submit(order, i as u64);
                        }

                        let market = Order::new(
                            depth as u64 + 1,
                            Side::Buy,
                            0.0,
                            50,
                            OrderType::Market,
                            depth as u64 + 1,
                        );

                        let start = std::time::Instant::now();
                        black_box(book.submit(market, depth as u64 + 1));
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_cancel(c: &mut Criterion) {
    c.bench_function("cancel_order", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for i in 0..iters {
                let mut book = OrderBook::new();
                let order = Order::new(i, Side::Buy, 100.0, 10, OrderType::Limit, i);
                book.submit(order, i);

                let start = std::time::Instant::now();
                black_box(book.cancel(i));
                total += start.elapsed();
            }
            total
        });
    });
}

fn bench_throughput(c: &mut Criterion) {
    c.bench_function("mixed_workload_1000", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let mut book = OrderBook::new();
            for i in 0..1000u64 {
                let side = if rng.gen_bool(0.5) { Side::Buy } else { Side::Sell };
                let price = 100.0 + (rng.gen::<f64>() - 0.5) * 10.0;
                let order = Order::new(i, side, price, 100, OrderType::Limit, i);
                black_box(book.submit(order, i));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_insert_limit,
    bench_match_market,
    bench_cancel,
    bench_throughput,
);
criterion_main!(benches);
```

**Step 2: Run benchmarks**

Run: `cargo bench`
Expected: Benchmark results with timing. Target <1µs median for single operations.

**Step 3: Commit**

```bash
git add rust/matching-engine/benches/
git commit -m "feat(rust): Criterion benchmarks for order book operations"
```

---

## Phase 6: Monitoring & Dashboard

### Task 16: Evidently drift detection

**Files:**
- Create: `python/monitoring/drift.py`
- Create: `tests/monitoring/__init__.py`
- Create: `tests/monitoring/test_drift.py`

**Step 1: Write the failing test**

`tests/monitoring/test_drift.py`:
```python
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
    detector = DriftDetector(reference)
    report = detector.detect(current)
    assert "feat_a" in report
    assert report["feat_a"]["drifted"] is True  # feat_a has a +2.0 shift
    assert report["feat_b"]["drifted"] is False
```

**Step 2: Implement drift detection**

`python/monitoring/drift.py`:
```python
"""Feature and prediction drift detection."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects distribution drift between reference and current data."""

    def __init__(self, reference: pd.DataFrame, threshold: float = 0.05):
        self.reference = reference
        self.threshold = threshold

    def detect(self, current: pd.DataFrame) -> dict:
        """Run KS test for each feature and return drift report."""
        report = {}
        for col in self.reference.columns:
            if col not in current.columns:
                continue
            ref_vals = self.reference[col].dropna()
            cur_vals = current[col].dropna()

            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            psi = self._psi(ref_vals, cur_vals)

            report[col] = {
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "psi": psi,
                "drifted": p_value < self.threshold,
            }

        drifted = [k for k, v in report.items() if v["drifted"]]
        if drifted:
            logger.warning(f"Drift detected in features: {drifted}")

        return report

    @staticmethod
    def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Population Stability Index."""
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            bins + 1,
        )
        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

        # Avoid log(0)
        ref_counts = np.clip(ref_counts, 1e-6, None)
        cur_counts = np.clip(cur_counts, 1e-6, None)

        return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
```

**Step 3: Run tests**

Run: `pytest tests/monitoring/test_drift.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add python/monitoring/ tests/monitoring/
git commit -m "feat: drift detection with KS test and PSI"
```

---

### Task 17: Dash risk dashboard

**Files:**
- Create: `python/monitoring/dashboard.py`

**Step 1: Implement dashboard**

`python/monitoring/dashboard.py`:
```python
"""Risk monitoring dashboard with Dash + Plotly."""

import logging

import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_dashboard(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    risk_summary: dict,
    rolling_sharpe: pd.Series,
) -> dash.Dash:
    """Create and return a Dash app for risk monitoring."""
    app = dash.Dash(__name__)

    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()

    app.layout = html.Div(
        [
            html.H1("Quant Platform — Risk Dashboard"),
            # KPI cards
            html.Div(
                [
                    _kpi_card("Sharpe Ratio", f"{risk_summary.get('sharpe', 0):.2f}"),
                    _kpi_card("Max Drawdown", f"{risk_summary.get('max_drawdown', 0):.2%}"),
                    _kpi_card("Ann. Return", f"{risk_summary.get('annualized_return', 0):.2%}"),
                    _kpi_card("VaR (95%)", f"{risk_summary.get('var_95_historical', 0):.4f}"),
                    _kpi_card("CVaR (95%)", f"{risk_summary.get('cvar_95', 0):.4f}"),
                ],
                style={"display": "flex", "gap": "20px", "marginBottom": "30px"},
            ),
            # Charts row 1
            html.Div(
                [
                    dcc.Graph(
                        figure=_cumulative_returns_chart(cumulative),
                        style={"flex": "1"},
                    ),
                    dcc.Graph(
                        figure=_weights_chart(weights),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "20px"},
            ),
            # Charts row 2
            html.Div(
                [
                    dcc.Graph(
                        figure=_drawdown_chart(drawdown),
                        style={"flex": "1"},
                    ),
                    dcc.Graph(
                        figure=_rolling_sharpe_chart(rolling_sharpe),
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "20px"},
            ),
        ],
        style={"padding": "20px", "fontFamily": "system-ui"},
    )

    return app


def _kpi_card(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "14px", "color": "#666"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "bold"}),
        ],
        style={
            "padding": "15px 25px",
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "minWidth": "140px",
        },
    )


def _cumulative_returns_chart(cumulative: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values, mode="lines", name="Portfolio"))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1")
    return fig


def _weights_chart(weights: pd.Series) -> go.Figure:
    fig = go.Figure(
        data=[go.Treemap(labels=weights.index, parents=[""] * len(weights), values=weights.values)]
    )
    fig.update_layout(title="Portfolio Weights")
    return fig


def _drawdown_chart(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line={"color": "red"},
        )
    )
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown %")
    return fig


def _rolling_sharpe_chart(rolling_sharpe: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", name="Sharpe"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Rolling Sharpe Ratio (60d)", xaxis_title="Date", yaxis_title="Sharpe")
    return fig


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    weights = pd.Series({"AAPL": 0.3, "MSFT": 0.25, "GOOG": 0.2, "AMZN": 0.15, "META": 0.1})
    risk = {
        "sharpe": 1.2,
        "max_drawdown": -0.15,
        "annualized_return": 0.12,
        "var_95_historical": -0.018,
        "cvar_95": -0.025,
    }
    rolling = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)

    app = create_dashboard(returns, weights, risk, rolling)
    app.run(debug=True, port=8050)
```

**Step 2: Commit**

```bash
git add python/monitoring/dashboard.py
git commit -m "feat: Dash risk monitoring dashboard with KPIs and charts"
```

---

## Phase 7: Backtesting & Validation

### Task 18: Walk-forward backtesting with CPCV

**Files:**
- Create: `python/backtest/run.py`
- Create: `python/backtest/validation.py`
- Create: `tests/backtest/__init__.py`
- Create: `tests/backtest/test_validation.py`

**Step 1: Write the failing test**

`tests/backtest/test_validation.py`:
```python
import numpy as np
import pandas as pd
import pytest
from python.backtest.validation import walk_forward_split


def test_walk_forward_split():
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    df = pd.DataFrame({"value": range(1000)}, index=dates)

    splits = list(walk_forward_split(df, n_splits=5, train_pct=0.6, embargo_days=5))
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        # No overlap
        assert len(set(train_idx) & set(test_idx)) == 0
        # Train comes before test
        assert df.index[train_idx[-1]] < df.index[test_idx[0]]
        # Embargo gap exists
        gap = (df.index[test_idx[0]] - df.index[train_idx[-1]]).days
        assert gap >= 5
```

**Step 2: Implement validation module**

`python/backtest/validation.py`:
```python
"""Walk-forward and CPCV validation for financial models."""

import numpy as np
import pandas as pd


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.6,
    embargo_days: int = 5,
):
    """Walk-forward cross-validation with embargo period.

    Yields (train_indices, test_indices) tuples.
    Embargo period prevents data leakage from overlapping return windows.
    """
    n = len(df)
    test_size = n // n_splits
    train_size = int(test_size * train_pct / (1 - train_pct))

    for i in range(n_splits):
        test_start = n - (n_splits - i) * test_size
        test_end = test_start + test_size
        train_start = max(0, test_start - train_size - embargo_days)
        train_end = test_start - embargo_days

        if train_start >= train_end or test_start >= n:
            continue

        train_idx = list(range(train_start, train_end))
        test_idx = list(range(test_start, min(test_end, n)))
        yield train_idx, test_idx


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio per Bailey & López de Prado (2014).

    Adjusts for multiple testing by comparing against the expected maximum
    Sharpe under the null hypothesis.
    """
    from scipy import stats

    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    e_max = stats.norm.ppf(1 - 1 / n_trials) * (1 - 0.5772 / np.log(n_trials))

    # Standard error of Sharpe
    se = np.sqrt(
        (1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2) / n_observations
    )

    # Probability that the observed Sharpe exceeds the expected max under null
    dsr = stats.norm.cdf((sharpe - e_max) / se)
    return float(dsr)
```

`python/backtest/run.py`:
```python
"""Full backtesting pipeline: train → predict → allocate → evaluate."""

import logging

import numpy as np
import pandas as pd

from python.alpha.features import compute_alpha_features, compute_forward_returns
from python.alpha.model import CrossSectionalModel
from python.alpha.train import FEATURE_COLS
from python.backtest.validation import deflated_sharpe_ratio, walk_forward_split
from python.bridge.bl_views import create_bl_views
from python.portfolio.optimizer import PortfolioOptimizer
from python.portfolio.risk import RiskEngine

logger = logging.getLogger(__name__)


def run_backtest(
    prices: pd.DataFrame,
    n_splits: int = 5,
    top_n: int = 20,
    rebalance_days: int = 5,
) -> dict:
    """Walk-forward backtest with the full pipeline.

    Args:
        prices: Multi-column DataFrame of close prices (columns=tickers).
        n_splits: Number of walk-forward windows.
        top_n: Number of stocks to hold in portfolio.
        rebalance_days: Days between rebalancing.
    """
    # Prepare features and targets
    ohlcv_frames = []
    for ticker in prices.columns:
        ohlcv_frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "open": prices[ticker],
                    "high": prices[ticker] * 1.01,  # Approximate
                    "low": prices[ticker] * 0.99,
                    "close": prices[ticker],
                    "volume": 1e6,  # Placeholder
                },
                index=prices.index,
            )
        )
    ohlcv = pd.concat(ohlcv_frames)
    featured = compute_alpha_features(ohlcv)
    labeled = compute_forward_returns(featured, horizon=rebalance_days)
    labeled = labeled.dropna(subset=FEATURE_COLS + [f"target_{rebalance_days}d"])

    all_returns = []

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_split(labeled, n_splits=n_splits)
    ):
        logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

        train = labeled.iloc[train_idx]
        test = labeled.iloc[test_idx]

        # Train model
        model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
        model.fit(train, target_col=f"target_{rebalance_days}d")

        # Generate predictions
        preds = model.predict(test)
        ranks = pd.Series(preds, index=test.index).groupby(test.index).rank(pct=True)

        # Select top-N stocks per date
        test_with_preds = test.copy()
        test_with_preds["pred_rank"] = ranks.values

        fold_returns = []
        for date, group in test_with_preds.groupby(level=0):
            top = group.nlargest(top_n, "pred_rank")
            # Equal weight among top picks
            ret = top[f"target_{rebalance_days}d"].mean()
            fold_returns.append({"date": date, "return": ret})

        all_returns.extend(fold_returns)

    returns_df = pd.DataFrame(all_returns).set_index("date")
    portfolio_returns = returns_df["return"]

    # Compute metrics
    ann_return = portfolio_returns.mean() * (252 / rebalance_days)
    ann_vol = portfolio_returns.std() * np.sqrt(252 / rebalance_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    return {
        "portfolio_returns": portfolio_returns,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": ((1 + portfolio_returns).cumprod().cummax() - (1 + portfolio_returns).cumprod()).max(),
        "deflated_sharpe": deflated_sharpe_ratio(sharpe, n_trials=n_splits, n_observations=len(portfolio_returns)),
        "n_folds": n_splits,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices = pd.read_parquet("data/raw/sp500_ohlcv.parquet")
    results = run_backtest(prices)
    for k, v in results.items():
        if k != "portfolio_returns":
            logger.info(f"{k}: {v}")
```

**Step 3: Run tests**

Run: `pytest tests/backtest/test_validation.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add python/backtest/ tests/backtest/
git commit -m "feat: walk-forward backtesting with CPCV and deflated Sharpe ratio"
```

---

## Phase 8: DVC Pipeline & Integration

### Task 19: DVC pipeline configuration

**Files:**
- Create: `dvc.yaml`
- Create: `.dvcignore`

**Step 1: Initialize DVC**

Run: `pip install dvc && dvc init`

**Step 2: Create dvc.yaml**

```yaml
stages:
  ingest:
    cmd: python -m python.data.ingestion
    deps:
      - python/data/ingestion.py
    outs:
      - data/raw/sp500_ohlcv.parquet
      - data/raw/macro_indicators.parquet

  train_lgbm:
    cmd: python -m python.alpha.train
    deps:
      - python/alpha/train.py
      - python/alpha/model.py
      - python/alpha/features.py
      - data/raw/sp500_ohlcv.parquet
    metrics:
      - metrics/lgbm_metrics.json:
          cache: false

  backtest:
    cmd: python -m python.backtest.run
    deps:
      - python/backtest/run.py
      - python/alpha/model.py
      - python/portfolio/optimizer.py
      - data/raw/sp500_ohlcv.parquet
    metrics:
      - metrics/backtest_metrics.json:
          cache: false
```

**Step 3: Create .dvcignore**

```
# DVC ignore patterns
__pycache__
.git
.venv
target
```

**Step 4: Commit**

```bash
git add dvc.yaml .dvcignore .dvc/
git commit -m "feat: DVC pipeline for reproducible data and training"
```

---

### Task 20: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""End-to-end integration test: data → features → model → portfolio → risk."""

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
    """Verify the complete signal → portfolio → risk pipeline works end-to-end."""
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
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for full pipeline"
```

---

## Summary: Build Order

| Phase | Tasks | Est. Days | Description |
|-------|-------|-----------|-------------|
| 1. Scaffolding | 1-4 | 1 | pyproject.toml, Cargo.toml, Docker, Makefile |
| 2. Data | 5-6 | 1-2 | yfinance ingestion, TimescaleDB storage |
| 3. ML Alpha | 7-9 | 3-4 | Features, LightGBM, TFT |
| 4. Portfolio | 10-12 | 2-3 | BL bridge, optimizer, risk engine |
| 5. Rust Engine | 13-15 | 2-3 | Types, order book, benchmarks |
| 6. Monitoring | 16-17 | 1-2 | Drift detection, Dash dashboard |
| 7. Backtest | 18 | 1-2 | Walk-forward CPCV, deflated Sharpe |
| 8. Integration | 19-20 | 1 | DVC pipeline, integration test |
| **Total** | **20 tasks** | **~12-17 days** | |
