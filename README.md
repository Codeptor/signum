# Signum

[![CI](https://github.com/Codeptor/signum/actions/workflows/ci.yml/badge.svg)](https://github.com/Codeptor/signum/actions/workflows/ci.yml)

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

## Live Backtest Results (S&P 500, 5yr history)

Walk-forward backtest on 503 S&P 500 constituents, LightGBM alpha (21 features), top-20 portfolio, 5-day rebalancing, 10 bps transaction costs, 15% max position weight:

| Optimizer | Sharpe (net) | Sharpe (gross) | Ann. Return | Max DD | Avg Turnover |
|-----------|-------------|----------------|-------------|--------|-------------|
| Equal Weight | **1.10** | 1.23 | 21.5% | 52.5% | 51% |
| HRP | 0.63 | 0.81 | 9.8% | 55.6% | 56% |
| Risk Parity | 0.63 | 0.81 | 9.8% | 55.6% | 56% |
| Black-Litterman | 0.36 | 0.59 | 5.3% | 58.8% | 67% |
| Min CVaR | 0.36 | 0.59 | 5.3% | 58.8% | 67% |

Alpha is concentrated in ML stock selection — equal-weight top-20 achieves the strongest risk-adjusted returns, consistent with the "1/N puzzle" in portfolio theory (DeMiguel et al., 2009). HRP provides the best risk-managed allocation among optimization methods.

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
├── tests/               # 18 tests
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
pytest tests/ -v       # Python (18 passed)
cargo test             # Rust (10 passed)
cargo bench            # Criterion benchmarks
```

## Key Components

### ML Alpha Generation
- **LightGBM/CatBoost** cross-sectional model with 21 Alpha158-inspired features (incl. liquidity)
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
- Transaction cost model (turnover-based, configurable bps)
- Position limits (max weight constraint)
- Turnover dampening (weight blending)
- Multi-strategy comparison

### Monitoring
- Feature drift detection (KS test + Population Stability Index)
- Dash dashboard: KPI cards, cumulative returns, drawdown, rolling Sharpe, turnover, concentration gauge

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
