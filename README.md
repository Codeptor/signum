# Quantitative Equity Platform

End-to-end quantitative equity platform: ML alpha generation, portfolio optimization, and a Rust matching engine.

## Architecture

```
ML Signals ──► Black-Litterman Bridge ──► Portfolio Optimizer ──► Risk Engine
    │                                            │
    ▼                                            ▼
LightGBM (Alpha158)                    HRP / CVaR / BL
TFT (PyTorch)                         skfolio + cvxpy
    │
    ▼
Drift Detection (KS + PSI) ──► Dash Dashboard
```

**Rust Matching Engine** — Lock-free order book with price-time priority, 4 order types (Limit, Market, IOC, FOK), sub-microsecond latency.

## Benchmark Results (Criterion.rs)

| Operation | Median Latency |
|-----------|---------------|
| Limit order insert | **132 ns** |
| Market order match (10 levels) | **84 ns** |
| Market order match (100 levels) | **231 ns** |
| Market order match (1000 levels) | **1.92 µs** |
| Cancel order | **71 ns** |
| Mixed workload (1000 orders) | **249 µs** |

## Project Structure

```
quant-platform/
├── python/
│   ├── alpha/           # ML signal generation (LightGBM, TFT)
│   ├── portfolio/       # HRP, CVaR, Black-Litterman optimizer + risk engine
│   ├── data/            # yfinance ingestion + TimescaleDB storage
│   ├── backtest/        # Walk-forward CPCV + deflated Sharpe ratio
│   ├── monitoring/      # Drift detection + Dash dashboard
│   └── bridge/          # ML predictions → Black-Litterman views
├── rust/
│   └── matching-engine/ # Lock-free order book with Criterion benchmarks
├── infra/
│   └── docker-compose.yml  # TimescaleDB, Redis, MLflow
├── tests/               # 17 tests (+ 1 skipped without PyTorch)
├── dvc.yaml             # Reproducible pipeline DAG
└── Makefile             # Build orchestration
```

## Quick Start

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[all]"

# Infrastructure
docker-compose -f infra/docker-compose.yml up -d

# Run pipeline
make ingest    # Fetch S&P 500 data via yfinance
make train     # Train LightGBM with MLflow tracking
make backtest  # Walk-forward backtest with CPCV
make dashboard # Launch Dash risk dashboard on :8050

# Tests
pytest tests/ -v       # Python (17 passed, 1 skipped)
cargo test             # Rust (10 passed)
cargo bench            # Criterion benchmarks
```

## Key Components

### ML Alpha Generation
- **LightGBM/CatBoost** cross-sectional model with 18 Alpha158-inspired features
- **Temporal Fusion Transformer** wrapper (requires `pip install 'quant-platform[ml]'`)
- MLflow experiment tracking with IC, Rank IC metrics

### Portfolio Optimization (skfolio)
- **Hierarchical Risk Parity (HRP)** — avoids covariance inversion
- **Minimum CVaR** — tail risk minimization via linear program
- **Black-Litterman with ML views** — ML confidence scores mapped to view uncertainties

### Risk Engine
- VaR (parametric + historical simulation)
- CVaR / Expected Shortfall
- Maximum drawdown tracking
- Rolling Sharpe ratio (60-day window)
- Herfindahl-Hirschman concentration index

### Rust Matching Engine
- `BTreeMap<Price, VecDeque<Order>>` for bid/ask levels
- Price-time priority matching (LMAX Disruptor pattern)
- Order types: Limit, Market, IOC, FOK
- All operations sub-microsecond at typical book depths

### Backtesting
- Walk-forward cross-validation with embargo period
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Transaction cost-adjusted framework

### Monitoring
- Feature drift detection (KS test + Population Stability Index)
- Dash dashboard: KPI cards, cumulative returns, drawdown, rolling Sharpe, weight treemap

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | LightGBM, CatBoost, PyTorch Forecasting (TFT) |
| Portfolio | skfolio, cvxpy |
| Data | yfinance, SQLAlchemy, TimescaleDB, Parquet |
| Risk | scipy, numpy |
| Monitoring | Evidently-style drift (scipy), Dash + Plotly |
| Execution | Rust, Criterion.rs |
| MLOps | MLflow, DVC |
| Infra | Docker Compose (TimescaleDB, Redis, MLflow) |
