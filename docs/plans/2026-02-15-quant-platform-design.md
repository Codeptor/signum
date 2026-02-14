# Quantitative Equity Platform — Design Document

**Date:** 2026-02-15
**Approach:** Vertical Slice — end-to-end pipeline from data to execution
**Goal:** Portfolio project for quant engineering roles
**Timeline:** 2-4 weeks
**Focus:** ML alpha generation (deep) + portfolio optimization + Rust matching engine (core)

---

## Project Structure

```
quant-platform/
├── rust/
│   └── matching-engine/       # Lock-free order book, Criterion benchmarks
├── python/
│   ├── alpha/                 # ML signal generation
│   │   ├── qlib_lgbm/        # Qlib + LightGBM/CatBoost, Alpha158
│   │   └── tft/               # Temporal Fusion Transformer
│   ├── portfolio/             # Portfolio construction & risk
│   │   ├── optimizer.py       # skfolio: HRP, CVaR, BL with ML views
│   │   └── risk.py            # VaR, CVaR, drawdown, stress tests
│   ├── data/                  # Data ingestion & storage
│   │   ├── ingestion.py       # yfinance + SEC EDGAR connectors
│   │   └── store.py           # TimescaleDB interface
│   ├── backtest/              # Qlib CPCV backtesting
│   ├── monitoring/            # Evidently drift detection + Dash dashboard
│   └── bridge/                # ML signals → BL views → execution
├── infra/
│   └── docker-compose.yml     # TimescaleDB, Redis, MLflow
├── mlops/
│   ├── mlflow/                # Experiment tracking config
│   └── dvc/                   # Data versioning
├── tests/
├── benchmarks/                # Criterion.rs reports
├── pyproject.toml
├── Cargo.toml                 # Workspace root
└── Makefile                   # Build/run orchestration
```

---

## Layer 1: ML Alpha Generation

### Model 1 — Qlib + LightGBM/CatBoost (Tier 1)

- Microsoft Qlib with Alpha158 factor set (158 pre-computed technical features)
- LightGBM and CatBoost for cross-sectional return prediction
- Universe: S&P 500, daily frequency
- Target: 5-day forward returns
- Validation: Combinatorial Purged Cross-Validation (CPCV) via skfolio

### Model 2 — Temporal Fusion Transformer (Tier 2)

- PyTorch Forecasting TFT implementation
- Inputs: OHLCV + Alpha158 subset + sector encoding (static covariates)
- Interpretable attention weights for demo
- Same CPCV validation framework

### Signal Combination

- Weighted ensemble of LightGBM rank + TFT rank
- Output: ranked stock scores → Black-Litterman views

### Experiment Tracking

- MLflow: hyperparameters, IC, Rank IC, Sharpe, model artifacts
- DVC: dataset versioning

---

## Layer 2: Portfolio Optimization & Risk

### Optimization (skfolio)

- **HRP:** Primary allocation — avoids covariance inversion issues
- **CVaR optimization:** Minimize tail risk via linear program
- **Black-Litterman with ML views:** ML confidence → view uncertainties, ML returns → expected returns. Crown jewel bridge between ML and portfolio construction.
- **Risk Parity:** Benchmark comparison

### Risk Monitoring

- VaR: parametric + historical simulation
- CVaR / Expected Shortfall
- Rolling drawdown tracking
- Position concentration limits

### Dashboard (Dash + Plotly)

- Portfolio weights (treemap/stacked bar)
- Risk decomposition by factor/sector
- Rolling Sharpe ratio
- Drawdown chart
- Drift alerts panel

### Transaction Costs

- Proportional costs (bps per trade)
- Turnover constraints in optimizer

---

## Layer 3: Rust Matching Engine

### Order Book

- `BTreeMap<Price, VecDeque<Order>>` for bid/ask levels
- Price-time priority matching
- Order types: Limit, Market, IOC, FOK
- Single-threaded business logic (LMAX pattern)

### Data Structures

- `Order`: id, side, price, quantity, timestamp, order_type
- `OrderBook`: bids (descending), asks (ascending), match logic
- `Execution`: trade reports
- `L2Snapshot`: top-N price levels

### Benchmarks (Criterion.rs)

- Insert, cancel, match latency
- Throughput: orders/second
- HDR Histogram: p50/p95/p99/p99.9
- Target: <1µs median per operation

### Out of Scope (Phase 1)

- FIX gateway, UDP multicast, io_uring, ring buffers
- PyO3 bridge (stretch goal)

---

## Layer 4: Data Pipeline & Infrastructure

### Data Sources (Free)

- **yfinance:** Daily OHLCV, S&P 500 fundamentals
- **SEC EDGAR** (edgartools): 10-K/10-Q filings
- **FRED:** Macro indicators (yield curve, VIX, unemployment)

### Storage

- **TimescaleDB** (Docker): Daily bars, features, portfolio state
- **Parquet:** Raw data, DVC-versioned
- **Redis:** Cached predictions + portfolio weights

### Pipeline

- Python ingestion scripts
- Makefile orchestration: `make ingest`, `make train`, `make optimize`, `make backtest`
- DVC pipelines define the DAG

### Docker Compose Stack

- `timescaledb` — time series storage
- `redis` — caching
- `mlflow` — experiment tracking (localhost:5000)
- `dashboard` — Dash app (localhost:8050)

### Out of Scope

- Kafka, Airflow, Feast (overkill for daily-frequency portfolio project)

---

## Layer 5: MLOps & Monitoring

### MLflow

- All runs logged: hyperparams, metrics, artifacts
- Model registry with stage promotion

### DVC

- Raw data + features versioned alongside code
- Reproducible pipeline DAG

### Drift Detection (Evidently AI)

- Feature distribution monitoring (PSI, KS test)
- Prediction distribution shifts
- SHAP value tracking for explanation drift
- Rolling Sharpe on 30/60/90-day windows
- Alerts in Dash dashboard

### Backtesting (Qlib)

- Walk-forward with CPCV
- Deflated Sharpe ratio
- Transaction cost-adjusted returns
- Cross-model/allocation comparison

### Out of Scope

- Shadow/canary deployment, CI/CD, automated retraining triggers

---

## Key Interview Highlights

1. **Rust matching engine** with nanosecond benchmarks (<1µs median)
2. **ML → Black-Litterman bridge** connecting signals to portfolio construction
3. **Drift detection pipeline** flagging model degradation via SHAP + Evidently
4. **CPCV validation** showing understanding of financial CV pitfalls
5. **End-to-end system** demonstrating breadth across every quant platform layer
