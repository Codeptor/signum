# Building a production-grade quantitative equity platform

**The most impressive quant portfolio project in 2025 isn't a better model — it's a complete system.** Quant firms hiring engineers consistently report that candidates who demonstrate end-to-end systems thinking outperform those who show only ML skills. A model is roughly 20% of a production trading system; data infrastructure, deployment, monitoring, risk management, and execution fill the remaining 80%. This report maps the exact technologies, architectures, and design decisions needed to build a showcase project spanning deep learning, portfolio optimization, and low-latency execution — all grounded in what actually works rather than academic hype.

The project described here combines **Python for ML research and portfolio optimization**, **Rust for a high-performance execution engine**, and modern MLOps practices into a monorepo that demonstrates breadth across every layer of a quantitative trading system. Each technology choice reflects current industry practice at firms like Jane Street, Citadel, and Two Sigma.

---

## What actually predicts equity returns (and what doesn't)

The single most important finding from recent research is a reality check: **gradient boosting models (CatBoost, LightGBM, XGBoost) consistently outperform deep neural networks for cross-sectional equity return prediction.** A landmark 2024 paper found CatBoost achieved a Sharpe ratio of 6.79 while time series foundation models like Google's TimesFM and Amazon's Chronos produced negative returns on the same equity prediction task. Zero-shot foundation models, despite their promise in NLP, fail to transfer meaningfully to daily equity returns.

This doesn't mean deep learning has no role. The **Temporal Fusion Transformer (TFT)** offers the best balance of performance and interpretability among DL architectures. A 2024 TFT-GNN hybrid achieved R²=0.9645 on US equities by integrating graph neural network relational signals — but with an important caveat: simplified up/down GNN signals outperformed high-dimensional graph embeddings, suggesting compact relational cues carry more information than complex representations. The **xLSTM** (Beck et al., 2024) modernizes LSTMs with exponential gating and matrix memory, consistently outperforming standard LSTMs with the gap widening for longer prediction horizons. State space models like **Mamba** show promise for long-sequence modeling with linear complexity, though financial-domain validation remains thin.

A critical warning: a 2025 Nature paper demonstrated that most prominent LSTM studies create false positives through the "lagging effect" — models learn to copy the previous day's price rather than genuinely predict. Any model showing >90% accuracy on stock price levels should be treated with extreme skepticism.

For the portfolio project, the recommended ML architecture follows a tiered approach:

- **Tier 1 (proven):** Microsoft Qlib + LightGBM/CatBoost with Alpha158 features, validated with Combinatorial Purged Cross-Validation (CPCV)
- **Tier 2 (impressive):** TFT via PyTorch Forecasting with technical indicators + FinBERT/LLM sentiment features
- **Tier 3 (experimental):** GNN for stock correlation modeling using sector/supply-chain graphs, or FinRL PPO agent for dynamic allocation

For sentiment, an ensemble of FinBERT + RoBERTa + DeBERTa achieves approximately **80% accuracy** on financial text — better than any single model. LLaMA-3 fine-tuned on financial text outperforms FinBERT for sentiment estimation under turbulent market conditions. The open-source **FinGPT** project provides a complete financial LLM platform for experimentation.

### Key ML frameworks and libraries

**Microsoft Qlib** (15,500+ GitHub stars, MIT license) is the single best framework for a quant ML project. It provides 40+ built-in models, Alpha158/Alpha360 feature sets, walk-forward backtesting, and integrates with MLflow for experiment tracking. Its custom data format processes datasets in 7.4 seconds versus 184 seconds for HDF5. **FinRL** (12,000+ stars) provides the leading open-source deep RL framework for trading, implementing DQN, DDPG, PPO, SAC, A2C, and TD3 through Stable-Baselines3. For time series forecasting, **PyTorch Forecasting** offers production-ready TFT and DeepAR implementations, while **Darts** by Unit8 provides a unified API across dozens of model architectures.

Validation methodology matters more than model architecture. **Combinatorial Purged Cross-Validation (CPCV)**, the gold standard per Marcos López de Prado, creates multiple train-test paths from k folds instead of one, enabling statistical backtesting with deflated Sharpe ratios. The `skfolio` library implements this as `CombinatorialPurgedCV` with sklearn-compatible APIs including purge and embargo parameters.

---

## Portfolio optimization beyond Markowitz

Three Python libraries dominate modern portfolio optimization, each with distinct strengths. **Riskfolio-Lib** (v7.2.0, ~3,800 stars) is the most comprehensive, offering mean-risk optimization with **24 convex risk measures**, risk parity with 20 measures, HRP/HERC with 35 measures, plus Black-Litterman variants, worst-case mean-variance, and OWA optimization. **PyPortfolioOpt** (~5,300 stars) provides the simplest API with Efficient CVaR, CDaR, Black-Litterman, and HRP through a clean scikit-learn-inspired interface. **skfolio** (~1,800 stars, rising fast) is the newest contender, built entirely on the scikit-learn API with `fit`/`predict`/`transform` semantics, enabling full Pipeline integration with cross-validation and hyperparameter tuning — it also offers Entropy Pooling (a powerful generalization of Black-Litterman) and Vine Copula synthetic data generation for stress testing.

The key optimization techniques to implement, ordered by practical value:

- **Hierarchical Risk Parity (HRP):** Uses hierarchical clustering to avoid the covariance matrix inversion problems of mean-variance optimization. Consistently better out-of-sample performance than Markowitz.
- **CVaR optimization:** Minimizes expected tail loss beyond the VaR threshold. Reduces to a linear program via Rockafellar & Uryasev (2001), making it computationally efficient. All three major libraries support it natively.
- **Black-Litterman with ML views:** Feed ML model predictions as "views" into the Black-Litterman framework, blending with market equilibrium. This bridges the gap between ML signal generation and portfolio construction elegantly.
- **Risk Parity / Risk Budgeting:** Equalizes risk contribution across assets. Riskfolio-Lib supports relaxed risk parity via SDP formulation.
- **Deep RL allocation:** FinRL's PPO/SAC agents can learn dynamic allocation policies directly, bypassing explicit return estimation. Research shows periodic rebalancing outperforms continuous rebalancing due to transaction costs.

For real-time risk monitoring, implement **VaR** (parametric + historical simulation + Monte Carlo), **CVaR/Expected Shortfall**, rolling drawdown tracking, and stress testing via Entropy Pooling. Build the dashboard with **Dash + Plotly** for production or **Streamlit** for rapid prototyping, refreshing calculations as new data arrives and triggering alerts when thresholds are breached.

Transaction cost modeling is critical for realistic results. The Almgren-Chriss framework models market impact as proportional to the square root of trade size. **cvxpy** supports linear, quadratic, and square-root cost formulations natively. Research by Olivares-Nadal & DeMiguel showed that calibrated proportional transaction costs reduced monthly turnover from 80.84% to 1.54% for min-variance portfolios with minimal Sharpe degradation.

---

## A Rust-powered low-latency execution engine

The execution layer is where systems engineering skills shine brightest. Building a matching engine and order execution system in Rust demonstrates memory safety without garbage collection pauses, zero-cost abstractions at C++ performance levels, and compile-time data race prevention — all properties that trading firms prize.

The architecture should follow the **LMAX Disruptor pattern**: a single-threaded business logic processor handles all matching sequentially on one dedicated CPU core, while lock-free ring buffers manage inter-thread communication for I/O. LMAX achieves 6M+ transactions per second with this design. In Rust, `disruptor-rs` and `rusted-ring` (cache-aligned at 64 bytes, 175M events/sec) provide the ring buffer primitives.

The core components to build, in priority order:

**Lock-free Order Book** (target: <1µs per operation). Use `BTreeMap<Price, VecDeque<Order>>` for price levels with price-time priority matching. Support Limit, Market, IOC, and FOK order types. The `orderbook-rs` crate by joaquinbejar demonstrates a thread-safe implementation with atomics supporting 10+ order types. Benchmark with Criterion.rs and HDR Histogram for percentile latency analysis.

**FIX Protocol Gateway.** Use **FerrumFIX** (`fefix` on crates.io), the most comprehensive FIX implementation in Rust — zero-copy, zero-allocation hot paths, supporting FIX 4.x/5.x with Tokio integration. Implement New Order Single, Cancel, and Execution Report messages.

**Market Data Publisher.** UDP multicast of L2 order book snapshots with incremental updates. Use **SBE (Simple Binary Encoding)** or FlatBuffers for zero-copy, allocation-free serialization.

**Network I/O.** For a portfolio project, **io_uring** via `glommio` (io_uring-based, thread-per-core runtime) provides modern Linux async I/O without the complexity of kernel bypass. For advanced demonstrations, **AF_XDP** offers a middle ground between kernel stack and full DPDK bypass. True DPDK reduces UDP transfer latency from ~13.7µs (kernel) to near wire latency (~1.2µs) but requires taking over entire NICs.

System tuning techniques to showcase: CPU pinning with `isolcpus`, NUMA-aware memory allocation, 2MB huge pages (disable THP), interrupt affinity routing away from latency-critical cores, and memory pre-allocation at startup to eliminate malloc on the hot path. Target metrics: **median tick-to-trade <1µs**, throughput >1M orders/sec, with p99.9 tail latency control.

### How top firms actually build their stacks

**Jane Street** runs ~30M lines of OCaml, extending the language with modal types and data-race freedom guarantees. They design FPGA trading hardware in OCaml via HardCaml. **Citadel Securities** uses C++ (already deploying C++26 features in production under Technical Fellow Herb Sutter) with `std::execution` for pipelined async processing. **Two Sigma** operates primarily on Java with Kubernetes, processing 110,000+ simulations daily across 600+ PB of storage. The takeaway: language choice matters less than architecture quality. A well-built Rust system demonstrates the same engineering principles these firms value.

Reference projects worth studying include `exchange-core` (Java, 5M ops/sec with Disruptor), `liquibook` (C++, clean matching engine API), and **NautilusTrader** (~18,900 stars, Rust core + Python API, production-grade event-driven platform with zero code changes between backtest and live trading).

---

## Data pipeline and infrastructure architecture

The data pipeline should follow a **four-layer architecture**: sources → ingestion → storage → serving. For ingestion, use **Apache Kafka** as the backbone for reliable, persistent streaming (millisecond latency, millions of messages per second, exactly-once semantics) and **Redis** for sub-millisecond last-value caching downstream. Orchestrate batch jobs with **Apache Airflow** for daily factor computation and data quality checks.

For time series storage, the choice depends on the use case. **QuestDB** leads benchmarks for tick data — computing 5-minute OHLCV bars from 100M tick records in **25ms** versus 547ms for ClickHouse and 1,021ms for TimescaleDB. **TimescaleDB** provides full PostgreSQL compatibility and ACID compliance, ideal for aggregated bars and metadata. **ClickHouse** excels at multi-TB analytical warehousing with 10:1 compression ratios, used by Deutsche Bank and Uber. For DataFrame storage, **ArcticDB** by Man Group offers a Pandas-native API with versioned data and petabyte-scale processing, now integrated into Bloomberg's BQuant platform. Store raw data in **Parquet** format on S3/GCS, partitioned by date for efficient time-range queries.

Build a feature store using **Feast** (Linux Foundation, open-source) to provide point-in-time correctness that prevents feature leakage — critical for financial ML. Feast supports pluggable backends (BigQuery, Redis, Spark) and serves features consistently for both training and live inference. The emerging **DuckDB** serves as an excellent embedded analytical engine for quant research on Parquet files.

### Data sources and costs

| Source | Type | Cost | Best For |
|--------|------|------|----------|
| **yfinance** | Historical OHLCV, fundamentals | Free | Prototyping, learning |
| **Polygon.io** (now Massive) | Institutional-grade trades/quotes | $29/mo starter | Serious project data |
| **SEC EDGAR** | 10-K, 10-Q, 13F, insider trades | Free | Fundamental analysis |
| **FRED** | 800K+ economic time series | Free | Macro factor models |
| **Alpha Vantage** | Prices + 60 technical indicators | Free (25 calls/day) | API integration demo |
| **Finnhub** | News sentiment, alternative data | Free tier | Sentiment signals |

For SEC EDGAR parsing, the **edgartools** Python library extracts financial statements in three lines of code and includes an AI/MCP server. Reddit sentiment via **PRAW** remains viable for research at the free tier (100 queries/min). Twitter/X API is effectively unusable for portfolio projects at $200-5,000/month.

---

## Backtesting with realistic market simulation

**QuantConnect LEAN** (16,300+ stars, Apache 2.0) is the gold standard for production-grade backtesting. Its Algorithm Framework separates Universe Selection → Alpha → Portfolio Construction → Execution → Risk into modular, pluggable components. It supports equities, options, futures, forex, and crypto simultaneously with sophisticated slippage, transaction cost, and margin models. LEAN runs locally via Docker or on QuantConnect's cloud with 40+ broker integrations including Interactive Brokers and Alpaca.

For rapid quantitative research, **VectorBT** is unmatched in speed — testing thousands of parameter combinations simultaneously via NumPy/Numba vectorization. It excels at hypothesis generation but lacks built-in look-ahead bias prevention. **Zipline-reloaded** (v3.1.1, maintained by Stefan Jansen) offers the cleanest Pythonic API with a Pipeline API for factor-based screening, though it's backtesting-only with no live trading support. **NautilusTrader** (Rust core, ~18,900 stars) bridges both worlds with zero code changes between backtest and live trading, streaming up to 5M rows/second.

The recommended approach: use **VectorBT** for fast signal research and parameter scanning, **Qlib** for ML model evaluation with CPCV validation, and **LEAN** or **NautilusTrader** for production-grade strategy validation with full transaction cost and market impact modeling.

---

## MLOps: keeping models alive in non-stationary markets

Financial ML models face a uniquely hostile deployment environment. Markets are non-stationary by nature, adversarial dynamics erode edges as other participants adapt, and regime changes invalidate learned patterns without warning. A MIT/Harvard study found **91% of ML models degrade over time**, and McKinsey reported 56% of financial institutions experienced significant model drift in 2023.

The deployment pipeline should follow this pattern:

**Git push → Data validation → Feature engineering → Model training → Evaluation gates → Registry promotion → Shadow/paper trading → Production deployment → Continuous monitoring**

Use **DVC** for data versioning alongside code in Git, **MLflow** for experiment tracking and model registry with stage promotion (Development → Staging → Production → Archived), and **Evidently AI** (open-source) for drift detection using Population Stability Index, Kolmogorov-Smirnov tests, and Jensen-Shannon divergence on both input features and prediction distributions.

The **shadow mode** deployment pattern is the gold standard for financial ML: run the new model alongside the production champion, generating signals but not executing trades, then compare performance over a statistically meaningful window before promotion. Paper trading via Alpaca's free paper trading API provides execution-level validation without capital risk. Implement canary deployments that gradually increase allocation from 5% → 25% → 100% while monitoring Sharpe ratio, max drawdown, and P&L attribution.

Monitor **explanation drift** via SHAP value distributions over time — changes in feature importance indicate the model's reasoning has shifted even if aggregate performance hasn't yet degraded. Track rolling Sharpe ratios on 30/60/90-day windows to detect early signs of alpha decay. Set retraining triggers on both performance thresholds (e.g., Sharpe < 0.5) and drift detection signals, with guard rails preventing retraining more than once per week.

---

## Recommended project structure and technology map

The project should be organized as a monorepo demonstrating the complete engineering surface of a trading system:

```
quant-platform/
├── rust/
│   ├── matching-engine/        # Lock-free order book, price-time priority
│   ├── fix-gateway/            # FerrumFIX-based FIX 4.4 gateway
│   ├── market-data/            # UDP multicast publisher, SBE encoding
│   └── ring-buffer/            # LMAX Disruptor-style SPSC ring buffer
├── python/
│   ├── alpha/
│   │   ├── factor_models/      # LightGBM, CatBoost via Qlib + Alpha158
│   │   ├── deep_learning/      # TFT via PyTorch Forecasting
│   │   ├── sentiment/          # FinBERT + LLM ensemble
│   │   └── graph_nets/         # GNN for stock correlation
│   ├── portfolio/
│   │   ├── optimizer/          # skfolio/Riskfolio-Lib (HRP, CVaR, BL)
│   │   └── risk/               # VaR, CVaR, stress testing, drawdown
│   ├── data/
│   │   ├── ingestion/          # yfinance, Polygon, SEC EDGAR connectors
│   │   ├── feature_store/      # Feast definitions and transformations
│   │   └── pipeline/           # Airflow DAGs for batch processing
│   ├── backtest/               # Qlib CPCV + LEAN/NautilusTrader integration
│   └── monitoring/
│       ├── drift_detection/    # Evidently AI monitors
│       └── dashboard/          # Dash + Plotly risk dashboard
├── infra/
│   ├── docker/                 # Multi-stage Dockerfiles per service
│   ├── k8s/                    # Kubernetes manifests, Helm charts
│   └── kafka/                  # Kafka + Redis Streams configuration
├── mlops/
│   ├── mlflow/                 # Experiment tracking, model registry
│   ├── dvc/                    # Data versioning pipelines
│   └── ci/                     # GitHub Actions workflows
├── tests/                      # Unit, integration, backtest regression
├── docs/                       # Architecture diagrams, runbooks
└── benchmarks/                 # Criterion.rs latency reports, HDR histograms
```

### Complete technology stack at a glance

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Models** | Qlib + LightGBM/CatBoost, PyTorch Forecasting (TFT), FinBERT | Alpha signal generation |
| **Portfolio Optimization** | skfolio, Riskfolio-Lib, cvxpy | HRP, CVaR, Black-Litterman |
| **Execution Engine** | Rust, FerrumFIX, disruptor-rs, glommio | Low-latency order matching |
| **Data Pipeline** | Kafka, Redis, Airflow | Streaming and batch ingestion |
| **Storage** | TimescaleDB/QuestDB, ArcticDB, Parquet/S3 | Time series + DataFrame storage |
| **Feature Store** | Feast + Redis | Point-in-time feature serving |
| **Backtesting** | VectorBT (research), LEAN/NautilusTrader (production) | Strategy validation |
| **MLOps** | MLflow, DVC, Evidently AI, GitHub Actions | Model lifecycle management |
| **Monitoring** | Prometheus, Grafana, Dash + Plotly | System and risk observability |
| **Serialization** | SBE, FlatBuffers, Cap'n Proto | Zero-copy message encoding |

## Conclusion

The strongest design choice in this project is the deliberate contrast between what's academically impressive and what actually works. Starting the ML layer with gradient boosting via Qlib — rather than a transformer — signals intellectual honesty and practical knowledge that experienced quant practitioners immediately recognize. Adding TFT and GNN layers demonstrates deep learning fluency while the Rust execution engine proves systems engineering depth. The CPCV validation framework, shadow mode deployment, and drift monitoring infrastructure show production awareness that separates this from toy projects.

Three elements will have the highest impact at interview time: the **lock-free matching engine in Rust** with nanosecond-resolution benchmarks (target <1µs median tick-to-trade), the **ML-to-portfolio-optimization bridge** using Black-Litterman with model predictions as views, and the **drift detection pipeline** that automatically flags model degradation. Together, these demonstrate that you understand not just how to build models, but how to build systems that keep models working in adversarial, non-stationary markets — which is the actual hard problem in quantitative finance.
