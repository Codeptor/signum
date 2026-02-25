# Suggested Commands

## Build & Run
- `make ingest` — Run data ingestion pipeline
- `make train` — Train ML models
- `make optimize` — Run portfolio optimization
- `make backtest` — Run backtesting suite
- `docker-compose -f infra/docker-compose.yml up -d` — Start infrastructure services

## Rust
- `cargo build --release` — Build matching engine
- `cargo test` — Run Rust tests
- `cargo bench` — Run Criterion.rs benchmarks

## Python
- `pip install -e .` — Install Python package in dev mode
- `pytest tests/` — Run Python tests
- `ruff check python/` — Lint Python code
- `ruff format python/` — Format Python code

## MLOps
- `mlflow ui` — Start MLflow tracking UI
- `dvc repro` — Reproduce DVC pipeline

## Git
- `git status` — Check working tree status
- `git log --oneline` — View commit history

## System
- `ls`, `grep`, `find` — Standard Linux utilities
