# Signum Quant Platform — Paper-Trading Readiness Assessment

**Date:** 2026-02-26
**Scope:** Full re-audit after completing all 45 findings from the initial audit
**Platform:** ML-driven live trading bot (LightGBM ranking + HRP optimization) targeting Alpaca paper trading
**Test baseline:** 141 tests passing, 0 failures, 3 warnings (pre-existing sklearn)

---

## Part 1: Summary of Completed Work

All **45 findings** from the initial audit (`docs/AUDIT_REPORT.md`) have been resolved across 34 commits. The fixes span 4 sprints covering execution correctness, ML pipeline integrity, data quality, broker reliability, risk management, and operational robustness.

### Sprint 1 — Execution & Core Trading (10 fixes)

| # | Fix | File(s) | Commit |
|---|-----|---------|--------|
| 1-2 | Per-position pricing in equity calculation | `execution.py` | `c573f1e` |
| 3 | Close stale positions not in target weights | `execution.py` | `c9d40c8` |
| 4 | Snapshot equity before reconciliation loop | `execution.py` | `fe6e3c4` |
| 5 | Negate weight_change for sells in risk tracking | `execution.py` | `5f22c9a` |
| 6 | Handle target_weight=0, close stale positions | `base.py` | `37fa263` |
| 10 | Initialize risk engine with historical data | `live_bot.py` | `2790c80` |
| 11 | Preserve bracket SL/TP when cancelling stale orders | `live_bot.py`, `base.py`, `alpaca_broker.py` | `09ca691` |
| 12 | Persist ExecutionBridge across trading cycles | `live_bot.py` | `248dd1c` |
| 13 | Fill verification with polling and timeout | `live_bot.py` | `ea06e31` |
| 14 | Remove `* 0.0` debug code zeroing spread costs | `run.py` | `bcd848b` |

### Sprint 2 — ML Pipeline Integrity (9 fixes)

| # | Fix | File(s) | Commit |
|---|-----|---------|--------|
| 7 | Date-based train/val split with 5-day embargo | `train.py` | `73e29f6` |
| 8 | Live model validation with last-20% holdout | `predict.py` | `401ac5e` |
| 9 | Early stopping with `lgb.early_stopping(10)` | `model.py` | `401ac5e` |
| 22 | Huber loss instead of MSE | `model.py` | `401ac5e` |
| 23 | Null guard on `predict()` when model is None | `model.py` | `401ac5e` |
| 24 | Fixed random seeds (42) for reproducibility | `model.py` | `401ac5e` |
| 19 | Removed duplicate `hl_range` feature | `features.py`, `train.py` | `ea867de` |
| 20 | Winsorization at 1st/99th percentiles | `features.py` | `ea867de` |
| 21 | Log returns for volatility features | `features.py` | `ea867de` |

### Sprint 3 — Data & Broker Reliability (11 fixes)

| # | Fix | File(s) | Commit |
|---|-----|---------|--------|
| 27-28 | Retry/backoff + NaN validation in ingestion | `ingestion.py` | `f6faa28` |
| 29 | Survivorship bias documented | `ingestion.py` | `f6faa28` |
| 30-31 | Retry + deterministic `client_order_id` for Alpaca | `alpaca_broker.py` | `d6a2386` |
| 32-33 | Fractional shares + fill-price anchored SL/TP | `base.py`, `live_bot.py` | `6fb7cf6` |
| 34 | Dynamic sleep using Alpaca clock API | `live_bot.py` | `e658b0e` |
| 35 | Broker extras in CI | `ci.yml` | `6023b23` |
| 36-37 | Log rotation + webhook alerting on fatal crash | `live_bot.py` | `a16ee44` |

### Sprint 4 — Risk, Optimization & Quality (15 fixes)

| # | Fix | File(s) | Commit |
|---|-----|---------|--------|
| 15 | Leverage double-count fix | `risk_manager.py` | `cbdcb0a` |
| 16 | Turnover limit enforcement | `risk_manager.py` | `cbdcb0a` |
| 17 | Sector weight constraint | `risk_manager.py` | `cbdcb0a` |
| 18 | Max-weight cap with iterative redistribution | `optimizer.py` | `7eca407` |
| 25 | `np.random.default_rng(42)` for Monte Carlo | `robustness.py` | `2b0095c` |
| 26 | Bulk `INSERT ON CONFLICT DO UPDATE` | `store.py` | `c66b4e1` |
| 38 | Actual portfolio weights for risk metrics | `run.py` | `68ae5c3` |
| 39 | Removed dead `train_start` assignment | `validation.py` | `9fddc7d` |
| 40 | Geometric annualization for ratios | `risk.py` | `9fddc7d` |
| 41 | PSI bins from reference quantiles | `drift.py` | `7446405` |
| 42 | Idzorek confidence passthrough | `bl_views.py` | `0ade213` |
| 43 | Type hints on public APIs | multiple | `4edc521` |
| 44 | 12 integration tests for live path | `test_live_integration.py` | `173cc3f` |
| 45 | Model versioning with pruning and rollback | `predict.py`, `test_predict.py` | `5e83d2c` |

---

## Part 2: Fresh Re-Audit Findings

A complete re-audit of every Python source file identified **new issues not covered by the original 45 findings**. These are organized by severity and module.

---

### P0 — Critical (Must Fix Before Paper Trading)

#### P0-1. SL/TP Order Accumulation Creates Naked Short Risk

**File:** `examples/live_bot.py:211-216, 360-397`

Old bracket legs (SL/TP) are preserved across cycles (line 212: `if order.parent_order_id: continue`). New SL/TP orders are attached after every buy fill (line 335: `needs_sl_tp: side == "buy"`). If a position is topped up on a subsequent cycle, the old SL/TP legs remain while new ones are added. When any stop triggers, it sells shares that may also be covered by other SL/TP orders, potentially selling more shares than held — creating a naked short.

**Example:** Cycle 1 buys 50 AAPL, attaches SL for 50 shares. Cycle 2 buys 10 more AAPL (rebalance), attaches SL for 10 shares. Now there are two SL orders totaling 60 shares against a 60-share position. If both trigger at different prices, the second fill sells shares already sold by the first.

**Fix:** Before attaching new SL/TP for a symbol, cancel all existing SL/TP orders for that symbol. Or track SL/TP order IDs per symbol and cancel them before reattaching.

---

#### P0-2. `rolling_beta` Will Crash — `apply()` Works Column-Wise

**File:** `python/portfolio/risk.py:237-256`

`DataFrame.rolling(window).apply()` calls the function on each **column** independently, not on the 2D window. The lambda `calc_beta(x.to_frame())` receives a single-column DataFrame, then tries `window_data.iloc[:, 1]` which raises `IndexError`. This method will crash on any call.

**Fix:** Replace with manual rolling covariance/variance: `cov = combined.rolling(window).cov()`, or compute `rolling_cov / rolling_var` directly.

---

#### P0-3. VaR Sign Convention Inconsistency

**File:** `python/portfolio/risk.py:48-60`

Three VaR methods return values with inconsistent signs:
- `var_parametric(0.95)` → negative number
- `var_historical(0.95)` → negative number
- `var_cornish_fisher(0.95)` → **positive** number (negated)

The risk manager uses `abs(var_95)` to work around this (line 288), but any direct comparison between VaR methods produces wrong results. If someone adds a "use Cornish-Fisher VaR for risk checks" option, the sign flip would silently pass risk checks that should fail.

**Fix:** Standardize all VaR methods to return the same sign (conventionally positive for loss amounts).

---

#### P0-4. No Crash Recovery — Bot Dies Permanently on Any Error

**File:** `run_live_bot.sh`

The shell script runs `uv run python examples/live_bot.py` with no restart mechanism. Any crash (OOM, network timeout, uncaught exception) terminates the bot permanently. There is no systemd service, Docker restart policy, supervisor, or even a `while true` loop. The bot will not survive overnight.

**Fix:** Add a process supervisor. Options: systemd unit file, Docker with `restart: unless-stopped`, or at minimum a bash restart loop with exponential backoff.

---

#### P0-5. No Duplicate Execution Guard on Restart

**File:** `examples/live_bot.py:524-544`

If the bot crashes after submitting orders but before sleeping, the restart will re-enter the `while True` loop and call `run_trading_cycle` again. With no "already traded today" flag or idempotency check, the bot submits duplicate orders on the same session, doubling position sizes.

**Fix:** Persist a `last_traded_date` to disk (or check Alpaca's recent orders) before running a new cycle. Skip if already traded today.

---

#### P0-6. Division by Zero in Feature Engineering

**File:** `python/alpha/features.py:101, 105, 116`

Multiple features have unguarded division:
- `bb_position`: `(c - bb_lower) / (bb_upper - bb_lower)` — zero when `std20 == 0` (halted stock)
- `volume_ratio`: `v / v.rolling(10).mean()` — zero when 10+ consecutive zero-volume days
- `oc_range`: `(c - o) / c` — zero if close price is 0

Any `inf`/`NaN` propagates through the model and can produce extreme or undefined predictions.

**Fix:** Add `np.where` guards or `.replace([np.inf, -np.inf], np.nan)` after each computation.

---

### P1 — High (Should Fix Before Paper Trading)

#### P1-1. Short Position Handling Is Broken

**File:** `python/bridge/execution.py:64-76`

If a SELL order exceeds the position quantity (going short by rounding error or strategy intent), `avg_cost` is never updated for the short entry. Subsequent BUY to cover uses the old long `avg_cost`, producing garbage P&L. Additionally, `reconcile_target_weights` (line 311) only closes positions where `quantity > 1e-6`, so negative (short) stale positions are never closed.

**Fix:** Handle short positions explicitly in `Position.update()`. Use `abs(pos.quantity) > 1e-6` in stale position checks.

---

#### P1-2. Equity Calculation Is Stale Between MTM Updates

**File:** `python/bridge/execution.py:235`

`_update_equity` falls back to `avg_cost` for positions without a price in the dict:
```python
price = prices.get(ticker, pos.avg_cost)
```

In `_update_state`, only the filled ticker's price is passed: `_update_equity({fill.order.ticker: current_price})`. All other positions are valued at cost basis, making `self.equity` wrong between full mark-to-market updates.

**Fix:** Store last-known prices and use them as fallback instead of `avg_cost`.

---

#### P1-3. `get_latest_prices` Makes N Sequential API Calls

**File:** `python/brokers/alpaca_broker.py:391-396`

For each ticker, a separate `get_latest_trade(sym)` HTTP call is made. For 50 assets, this is 50 sequential requests. Alpaca's free tier allows 200 req/min; a rebalance could exhaust the quota.

**Fix:** Use Alpaca's batch endpoint `get_latest_trades(symbols)` for a single call.

---

#### P1-4. `get_position` and `list_orders` Swallow Exceptions

**File:** `python/brokers/alpaca_broker.py:310-311, 329-331`

Network errors return `[]` or `None` — indistinguishable from "no data exists." This can cause `reconcile_portfolio` to open duplicate positions (thinking none exist) or skip cancellation of existing orders.

**Fix:** Let network errors propagate (or raise a specific exception) so callers can distinguish "no data" from "API failure."

---

#### P1-5. `risk_parity_weights` Raises on Failure With No Fallback

**File:** `python/portfolio/risk_attribution.py:158`

`risk_parity_weights` raises `RuntimeError` if SLSQP optimization fails. Unlike `optimizer.py` which has an equal-weight fallback, this method crashes the caller. If called during a live rebalance, it halts the entire trading cycle.

**Fix:** Add equal-weight fallback consistent with `optimizer.py`.

---

#### P1-6. No State Persistence Across Restarts

**File:** `examples/live_bot.py`

All state is in-memory: `bridge` (equity curve, positions, P&L), `risk_manager` (daily trades, daily turnover, current weights). A restart wipes the risk manager's trade count, allowing the bot to exceed daily limits. Equity curve history is lost.

**Fix:** Persist daily trade state to a file or database. Reload on startup.

---

#### P1-7. Alerting Only on Fatal Crash

**File:** `examples/live_bot.py:585`

`_send_alert()` is only called in the final `except`. Individual cycle failures (ML pipeline error, order rejections, all-critical risk violations) don't trigger alerts. The bot could silently fail to trade for days with no notification.

**Fix:** Add alerting for cycle-level failures and risk violations, not just fatal crashes.

---

#### P1-8. No Staleness Check on Market Data

**File:** `python/data/ingestion.py:52`

`yf.download()` can return data that is days old (weekend, holiday, yfinance cache). There is no check that the latest timestamp is within an acceptable recency window. The bot could train on stale data and submit orders based on Friday's prices on Monday.

**Fix:** Validate that the latest data point is within 1 trading day of the current date.

---

#### P1-9. Drift Detection Never Called From Live Bot

**File:** `python/monitoring/drift.py`

The `DriftDetector` class is implemented but never invoked from `live_bot.py`. Feature drift (distribution shift in inputs) goes undetected indefinitely during live trading, potentially causing the model to make predictions on out-of-distribution data.

**Fix:** Call `DriftDetector` before each trading cycle and log/alert on significant drift.

---

#### P1-10. Macro Features Computed But Never Used

**File:** `python/alpha/train.py:20-44`

`merge_macro_features()` is called during training, merging `vix`, `vix_ma_ratio`, `term_spread`, etc. into the DataFrame. However, these columns are **not** in `FEATURE_COLS`, so the model never sees them. This is wasted computation and a likely incomplete feature integration.

**Fix:** Either add macro features to `FEATURE_COLS` or remove the `merge_macro_features` call.

---

#### P1-11. `optimize_weights` Equal-Weight Fallback Uses Original Tickers

**File:** `python/alpha/predict.py:340-341`

When the optimizer fails and falls back to equal weights, it uses the original `tickers` list — which may include tickers that were dropped due to NaN data (line 335). This assigns weight to tickers with no valid price data.

**Fix:** Use only the surviving tickers after NaN filtering for the fallback.

---

### P2 — Medium (Fix After Initial Paper Trading Launch)

| # | File | Issue |
|---|------|-------|
| P2-1 | `features.py:42` | `winsorize()` mutates DataFrame in place — caller's original data is modified |
| P2-2 | `model.py:120` | `feature_importance()` has no None guard — crashes if called before `fit()` |
| P2-3 | `model.py:95` | `best_iteration_` check uses falsy `if best_iter:` — fails when `best_iteration_ == 0` |
| P2-4 | `model.py:117` | `predict_ranks` broken for MultiIndex — groups by full tuple, not date. Never called in live path. |
| P2-5 | `risk_manager.py:207-222` | Leverage check emits no passing `RiskCheck` — inconsistent audit trail |
| P2-6 | `risk_manager.py:271` | `check_portfolio_risk(returns)` ignores the `returns` parameter entirely |
| P2-7 | `risk_manager.py:396` | `record_trade` can push weights negative — unintended short exposure |
| P2-8 | `risk_manager.py:490` | `risk_based_size` returns dollar amount, not fraction, when `portfolio_value != 1.0` |
| P2-9 | `optimizer.py:53` | `pct_change().dropna()` drops entire rows on any single NaN — can silently destroy return history |
| P2-10 | `risk.py:31` | Portfolio returns computed with potentially misaligned weights — silent NaN |
| P2-11 | `robustness.py:326` | Inconsistent drawdown sign convention — negative vs positive across modules |
| P2-12 | `robustness.py:375` | `regime_stress_tests` mutates input Series index in place |
| P2-13 | `validation.py:55` | `deflated_sharpe_ratio` divides by zero when `n_trials=1` |
| P2-14 | `store.py:78,94,114` | No error handling on `session.commit()` — data loss on connection drop |
| P2-15 | `predict.py:78` | `datetime.utcnow()` deprecated since Python 3.12 |
| P2-16 | `predict.py:133` | `list_model_versions` loads full model files to read metadata — wasteful |
| P2-17 | `alpaca_broker.py:119` | Idempotency key has minute-boundary race — retry can get different key |
| P2-18 | `alpaca_broker.py:185` | Fractional qty submitted for non-fractional-eligible assets — API rejection risk |
| P2-19 | `ingestion.py:68` | `dropna(how="all")` too lenient for wide DataFrames — NaN propagation |
| P2-20 | `live_bot.py:475` | `_seconds_until` minimum 60s can miss market open or delay unnecessarily |

### P3 — Low (Improvements / Technical Debt)

| # | Issue |
|---|-------|
| P3-1 | Hardcoded magic numbers throughout (RSI window, MACD spans, VIX thresholds, etc.) |
| P3-2 | Relative file paths (`data/models`, `data/raw/`, `data/processed/`) break if CWD changes |
| P3-3 | `paper_trading = True` hardcoded in source, not configurable via env var |
| P3-4 | `ann_factor = 252` hardcoded in `risk.py` — breaks for crypto or non-US markets |
| P3-5 | CatBoost path lacks early stopping parity with LightGBM |
| P3-6 | `run_live_bot.sh` has placeholder credentials and no env validation |
| P3-7 | No PID file or lockfile to prevent multiple bot instances |
| P3-8 | No heartbeat / health-check endpoint for external monitoring |
| P3-9 | `alpaca-trade-api` package is deprecated — Alpaca recommends `alpaca-py` |
| P3-10 | `information_ratio` uses arithmetic annualization while Sharpe/Sortino use geometric |

---

## Part 3: Paper-Trading Readiness Verdict

### Assessment: NOT READY — 6 P0 Blockers Remaining

The original 45 audit findings have been successfully resolved, significantly improving execution correctness, ML pipeline integrity, and operational robustness. However, the fresh re-audit uncovered **6 critical (P0)** and **11 high-priority (P1)** issues that were outside the scope of the original audit.

### What Works Well

- **ML pipeline fundamentals** — date-based splits with embargo, early stopping, huber loss, reproducible seeds
- **Portfolio optimization** — HRP with max-weight capping and equal-weight fallback
- **Risk checks** — leverage, turnover, sector, drawdown, and VaR limits are wired in
- **Broker integration** — retry/backoff, deterministic order IDs, fill verification with polling
- **Operational basics** — log rotation, webhook alerting (on fatal), dynamic sleep timing
- **Test coverage** — 141 tests covering unit, integration, and live path

### Blockers (Must Fix)

| Priority | Issue | Risk If Unfixed |
|----------|-------|-----------------|
| **P0-1** | SL/TP accumulation | Naked short positions from duplicate stop orders |
| **P0-2** | `rolling_beta` crash | Runtime crash if risk dashboard calls this method |
| **P0-3** | VaR sign inconsistency | Risk checks produce wrong results with Cornish-Fisher |
| **P0-4** | No crash recovery | Bot dies permanently on first transient error |
| **P0-5** | No duplicate execution guard | Double-trades on restart |
| **P0-6** | Division by zero in features | `inf`/`NaN` predictions on halted or zero-volume stocks |

### Recommended Fix Order

**Phase 1 — Immediate (before first paper trade):**
1. P0-4: Add process supervisor (systemd or restart loop)
2. P0-5: Add `last_traded_date` persistence
3. P0-1: Cancel existing SL/TP before attaching new ones
4. P0-6: Add division-by-zero guards in `features.py`
5. P0-3: Standardize VaR sign convention
6. P0-2: Fix `rolling_beta` implementation

**Phase 2 — First week of paper trading:**
7. P1-3: Batch price fetching
8. P1-4: Stop swallowing exceptions in broker
9. P1-6: State persistence across restarts
10. P1-7: Cycle-level alerting
11. P1-1: Short position handling
12. P1-2: Last-known price cache for equity

**Phase 3 — Ongoing improvement:**
13. P1-8 through P1-11 and all P2 items

### Estimated Effort to Reach Paper-Trading Ready

| Phase | Items | Estimated Effort |
|-------|-------|-----------------|
| Phase 1 (P0 blockers) | 6 fixes | 4-6 hours |
| Phase 2 (P1 reliability) | 6 fixes | 6-8 hours |
| Phase 3 (P2 quality) | 20+ items | 2-3 days |

**Bottom line:** The platform needs ~4-6 hours of work on the 6 P0 blockers before it can safely run in paper trading mode. The P1 items should follow within the first week to ensure reliability.
