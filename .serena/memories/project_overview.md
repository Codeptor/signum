# Quant Platform — Project Overview

## Purpose
Production-grade quantitative equity platform for portfolio showcase. Demonstrates end-to-end systems thinking: data ingestion → ML alpha generation → portfolio optimization → execution engine.

## Tech Stack
- **Python**: ML models (Qlib, LightGBM, CatBoost, TFT), portfolio optimization (skfolio), data pipeline, monitoring
- **Rust**: Lock-free matching engine with Criterion.rs benchmarks
- **Infrastructure**: Docker Compose (TimescaleDB, Redis, MLflow), DVC for data versioning

## Project Structure
```
quant-platform/
├── rust/matching-engine/     # Lock-free order book
├── python/
│   ├── alpha/               # ML signal generation (Qlib + TFT)
│   ├── portfolio/           # Optimization & risk (skfolio, HRP, CVaR, BL)
│   ├── data/                # Ingestion (yfinance, SEC EDGAR, FRED)
│   ├── backtest/            # Qlib CPCV backtesting
│   ├── monitoring/          # Evidently drift detection + Dash dashboard
│   └── bridge/              # ML signals → BL views → execution
├── infra/docker-compose.yml
├── mlops/                   # MLflow + DVC configs
├── tests/
└── benchmarks/
```

## Status
Greenfield project — design approved, implementation not yet started.
