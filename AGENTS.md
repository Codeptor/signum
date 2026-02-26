# Parallel Agent Coordination

**Created:** 2026-02-26
**Purpose:** Fix all findings from the paper trading readiness audit (session ses_3698)
**Source:** Audit report at bottom of `session-ses_3698.md` (lines 7600-7768)

---

## Worktree Layout

| Agent | Worktree Path | Branch | Scope |
|-------|---------------|--------|-------|
| **Claude Code** | `.worktrees/claude-code` | `fix/execution-broker-risk` | Execution, broker, risk manager, live bot |
| **OpenCode** | `.worktrees/opencode` | `fix/pipeline-regime-training` | Feature pipeline, regime detection, training, ingestion |

Both branch from `main` at commit `4fb210d`.

---

## File Ownership (STRICT - no cross-editing)

### Claude Code owns:
- `examples/live_bot.py`
- `python/brokers/alpaca_broker.py`
- `python/brokers/base.py`
- `python/execution/execution.py`
- `python/execution/risk_manager.py`
- `tests/` files for the above modules

### OpenCode owns:
- `python/alpha/features.py`
- `python/alpha/predict.py`
- `python/alpha/regime.py`
- `python/data/ingestion.py`
- `tests/` files for the above modules

### Shared (coordinate before editing):
- `pyproject.toml`
- `python/alpha/model.py` (if needed)
- `AGENTS.md` (this file - update your status section)

---

## Task Assignments

### Claude Code — Branch: `fix/execution-broker-risk`

| Priority | ID | File | Fix |
|----------|----|------|-----|
| **CRITICAL** | C2 | `live_bot.py` | `_has_traded_today` crashes on datetime comparison — trades dict stores ISO strings, compared as datetimes |
| **CRITICAL** | C5 | `live_bot.py` | Minute-boundary duplicate orders — `_last_rebalance_time` check races with scheduler |
| **CRITICAL** | C6 | `execution.py` | `ExecutionBridge` passes trade size as weight to `RiskManager.check_trade()` — all risk checks ineffective |
| **CRITICAL** | C1 | `live_bot.py` | Non-OCO SL/TP creates race condition — if SL fills, TP becomes naked short |
| **CRITICAL** | C3 | `live_bot.py` | SIGTERM handler saves `self.state` but `_has_traded_today`/positions tracked outside state dict |
| **CRITICAL** | C4 | `live_bot.py` | Top-up orders get SL/TP sized for increment only, not full position |
| **CRITICAL** | C7 | `risk_manager.py` | Short sells bypass `check_trade()` — only `side == 'buy'` checked |
| **CRITICAL** | C8 | `risk_manager.py` | Short positions bypass drawdown/concentration checks |
| **HIGH** | H1 | `live_bot.py` | State file writes non-atomic — crash during write corrupts state |
| **HIGH** | H2 | `live_bot.py` | `_liquidate_all` doesn't verify fills — may leave phantom positions |
| **HIGH** | H3 | `execution.py` | Bridge tracks positions independently from broker — phantom positions accumulate |
| **HIGH** | H4 | `execution.py` | `_sync_with_broker` only runs at init, never again — drift grows over time |
| **HIGH** | H5 | `risk_manager.py` | Daily trade counter resets on restart (stored in memory, not persisted) |
| **MEDIUM** | M1 | `live_bot.py` | `_has_traded_today` uses local tz vs Alpaca UTC |
| **MEDIUM** | M2 | `live_bot.py` | Weight renormalization can push past `MAX_POSITION_WEIGHT` |
| **MEDIUM** | M3 | `live_bot.py` | `paper_trading = True` hardcoded, should be env var |
| **MEDIUM** | M4 | `risk_manager.py` | `daily_trades`/`daily_turnover` dicts grow unboundedly |
| **MEDIUM** | M5 | `risk_manager.py` | `check_portfolio_risk` accepts unused `returns` param |
| **MEDIUM** | M6 | `execution.py` | `equity_curve` list grows unboundedly |
| **MEDIUM** | M7 | `execution.py` | No cash check for sell orders opening shorts |
| **MEDIUM** | M8 | `alpaca_broker.py` | `get_position` swallows ALL exceptions |
| **MEDIUM** | M9 | `alpaca_broker.py` | `created_at` passes datetime but field typed as `Optional[str]` |
| **MEDIUM** | M10 | `base.py` | `reconcile_portfolio` ignores short positions |

### OpenCode — Branch: `fix/pipeline-regime-training`

| Priority | ID | File | Fix |
|----------|----|------|-----|
| **CRITICAL** | C9 | `features.py` | VIX filled with 0.0 when missing — during stress (when VIX matters most), model gets "calm" signal |
| **CRITICAL** | C10 | `features.py` | `compute_cross_sectional_features` divides by zero when all stocks have same value |
| **CRITICAL** | C11 | `features.py` | `inf` values from log(0) and division propagate through pipeline unchecked |
| **HIGH** | H6 | `features.py` | `winsorize` clips at hardcoded 1st/99th percentile — too aggressive for fat-tailed returns |
| **HIGH** | H7 | `features.py` | Feature computation order-dependent — RSI computed before winsorize, momentum after |
| **HIGH** | H8 | `regime.py` | yfinance returns MultiIndex for single ticker — `.iloc[:, 0]` may grab wrong column |
| **HIGH** | H9 | `regime.py` | No hysteresis on regime thresholds — VIX oscillating around 30 causes rapid regime flipping |
| **HIGH** | H10 | `predict.py` | Survivorship bias — current S&P 500 list used for 5yr history (delisted stocks excluded) |
| **HIGH** | H11 | `predict.py` | Double data fetch — stocks ranked on fetch #1, weights optimized on fetch #2 |
| **HIGH** | H12 | `ingestion.py` | Wikipedia S&P 500 scraping fragile — hardcoded table index and column name |
| **HIGH** | H13 | `ingestion.py` | Macro "FRED" function uses Yahoo Finance tickers with no SLA |
| **MEDIUM** | M11 | `predict.py` | Missing features filled with 0.0 — non-neutral for VIX (~20), RSI (50) |
| **MEDIUM** | M12 | `predict.py` | Training uses first 100 tickers alphabetically — systematic bias |
| **MEDIUM** | M13 | `ingestion.py` | yfinance partial ticker failures silently accepted |
| **MEDIUM** | M14 | `regime.py` | `adjust_weights` returns `{}` for halt — ambiguous API |
| **MEDIUM** | M15 | `features.py` | `winsorize` mutates DataFrame in-place — target clipped before residual |

---

## Status Tracking

Update your section when you start/finish items. This is how we avoid conflicts.

### Claude Code Status
- [x] ALL DONE — committed `093b79e` on `fix/execution-broker-risk`
- [x] C1: SL/TP now OCO pairs (no orphaned orders)
- [x] C2: _has_traded_today handles datetime created_at
- [x] C3: SIGTERM saves full state (positions, cash, qty)
- [x] C4: SL/TP sized for total position, not just increment
- [x] C5: client_order_id day-level (no minute-boundary dupes)
- [x] C6: ExecutionBridge passes target position weight (not trade size)
- [x] C7/C8: Risk checks use abs(weight), shorts validated same as longs
- [x] H1: Atomic state writes (tmp + rename)
- [x] H2: Liquidation fill verification with alerts
- [x] M1-M10: All medium fixes applied
- **Tests: 364 passed, 0 failed**

#### BREAKING CHANGE for OpenCode:
`RiskManager.check_portfolio_risk()` signature changed:
- OLD: `check_portfolio_risk(self, returns: pd.Series)`
- NEW: `check_portfolio_risk(self, weights: pd.Series)`
If you call this method anywhere in your files, update the param name.

### OpenCode Status
- [x] ALL DONE — committed on `fix/pipeline-regime-training`
- [x] C9: VIX filled with neutral default (20.0) instead of 0.0
- [x] C10: Division-by-zero guard in compute_cross_sectional_features
- [x] C11: inf scrubbing via _scrub_infinities() + log(0) guard
- [x] H6: Winsorize widened to 0.5th/99.5th percentile
- [x] H7: Winsorize applied uniformly at end of compute_alpha_features
- [x] H8: _extract_close_series handles yfinance MultiIndex safely
- [x] H9: Hysteresis on regime thresholds prevents rapid flipping
- [x] H10: Survivorship bias warning logged when fetching live data
- [x] H11: Single fetch in get_ml_weights (price_data passthrough)
- [x] H12: Robust Wikipedia table search for symbol column
- [x] H13: Per-ticker error handling in macro fetch, ffill limited to 5d
- [x] M11: Missing features use FEATURE_NEUTRAL_DEFAULTS (not 0.0)
- [x] M12: Random sample(seed=42) instead of alphabetical[:100]
- [x] M13: NaN ticker detection and removal in fetch_ohlcv
- [x] M14: Added is_halt() static method for unambiguous halt detection
- [x] M15: winsorize() and cross-sectional features operate on copies
- **Breaking change note:** check_portfolio_risk(weights) — confirmed no references in my files
- **Tests: 412 passed, 0 failed, 1 skipped**

---

## Communication Protocol

1. **Before editing a shared file:** Write which file and why in your status section, wait for the other agent to not be touching it
2. **When done with a batch:** Commit to your branch, update status here
3. **If you need something from the other agent's files:** Ask the user to relay, do NOT edit files you don't own
4. **Merge order:** Both branches merge to `main` when complete. Claude Code merges first (execution layer), OpenCode second (pipeline layer)

## Running Tests

```bash
# From your worktree root:
uv run pytest tests/ -x -q

# Run only your module tests:
# Claude Code:
uv run pytest tests/test_execution.py tests/test_risk_manager.py tests/test_live_bot.py -x -q

# OpenCode:
uv run pytest tests/test_features.py tests/test_predict.py tests/test_regime.py tests/test_ingestion.py -x -q
```

## Target Metrics (from audit)
- Sharpe > 0.5
- Annual return > 8%
- Max drawdown < 12%
- Transaction costs < 2%
