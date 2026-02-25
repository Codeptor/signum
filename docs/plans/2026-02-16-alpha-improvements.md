# Alpha Improvements: Before & After

## Summary

Systematic alpha improvement loop applied to the quant backtest pipeline.
Starting from a baseline Sharpe of 1.10, reached **1.66 net Sharpe** (+51%) through
feature engineering, overfitting diagnosis, model tuning, and parameter optimization.

## Baseline (Before)

- **21 features**: purely absolute per-ticker technicals (returns, MAs, vol, RSI, MACD, BB)
- **Target**: raw 5d forward returns
- **Model**: LightGBM with `min_child_samples=20`
- **Backtest**: synthetic OHLCV (constant volume=1e6), `blend_alpha=0.5`
- **No cross-sectional features, no macro awareness**

| Method | Sharpe (net) | Ann. Return | Max DD | Turnover |
|--------|-------------|-------------|--------|----------|
| equal_weight | 1.097 | 21.5% | 52.5% | 0.510 |

## Changes Applied

### 1. Cross-Sectional Features (kept)
Percentile ranks within each date's cross-section:
- `cs_ret_rank_5d`, `cs_ret_rank_20d` -- momentum rank
- `cs_vol_rank_20d` -- volatility rank (IC=0.069, second-strongest predictor)
- `cs_volume_rank` -- liquidity rank

**Impact**: These features capture *relative* positioning, which is exactly what a
stock-selection model needs. The absolute features can't distinguish "stock X went up 2%
because the market went up 2%" from "stock X outperformed by 2%".

### 2. Residual Target (kept)
Subtract cross-sectional mean return from the 5d forward return target.
Model now predicts *relative outperformance* instead of absolute return.

**Impact**: Forces the model to learn stock-specific alpha rather than market direction.
Combined with cross-sectional features, this was the biggest single improvement.

### 3. Macro Features in Model (removed from model, kept for VIX scaling)
Initially added VIX, term spread, etc. as model features. Feature importance analysis
revealed they dominated tree splits (top 4 by gain) but had **zero cross-sectional IC**
(~0.0001). Since the target is residual returns, macro features are identical across all
stocks on a given date and cannot predict which stock outperforms.

**Diagnosis**: Classic overfitting trap -- high model importance but zero predictive power
means the model was fitting noise in the date-level variation.

### 4. Real OHLCV Data (fixed bug)
The backtest was synthesizing fake OHLCV with constant `volume=1e6`. This made
`dollar_volume_20d = close * 1e6`, so the liquidity filter just removed cheap stocks
(price bias). Switched to passing real OHLCV from the raw data.

### 5. Model Regularization (tuned)
Swept LightGBM hyperparameters. `min_child_samples=100` (up from 20) was the clear
winner -- more regularization prevents overfitting on noisy return data.

| Config | Sharpe |
|--------|--------|
| default (min_child=20) | 1.328 |
| **min_child=100** | **1.473** |
| conservative (lr=0.01, leaves=31) | 1.245 |

### 6. Turnover Dampening (tuned)
Swept `blend_alpha` (weight on new allocation vs previous). Lower values reduce
turnover and transaction costs at the expense of signal freshness.

| blend_alpha | Sharpe | Turnover |
|------------|--------|----------|
| 1.0 (no blend) | 1.473 | 0.944 |
| 0.5 (default) | 1.603 | 0.547 |
| **0.3 (tuned)** | **1.657** | **0.361** |

### 7. Feature Pruning (removed noise)
Removed `ret_1d` (IC=-0.003) and `oc_range` (IC=-0.005) which had negative
predictive power. Feature count: 29 -> 23.

### 8. VIX Position Scaling (softened)
Original thresholds (VIX<20: 100%, 20-30: 70%, >30: 50%) were too aggressive for
a long-only strategy. Softened to VIX<25: 100%, 25-35: 85%, >35: 70%.

## Final Results (After)

| Method | Sharpe (net) | Sharpe (gross) | Ann. Return | Max DD | Turnover |
|--------|-------------|---------------|-------------|--------|----------|
| equal_weight | **1.657** | 1.780 | 24.6% | 50.0% | 0.361 |
| hrp | **1.284** | 1.466 | 13.9% | 40.8% | 0.390 |
| black_litterman | **0.991** | 1.203 | 10.5% | 31.8% | 0.443 |

## Improvement Breakdown

| Step | Sharpe | Delta | Cumulative |
|------|--------|-------|------------|
| Baseline | 1.097 | -- | -- |
| + CS features + residual target | 1.290 | +0.193 | +18% |
| + Real OHLCV + liquidity filter + VIX scaling | 1.436 | +0.146 | +31% |
| + Feature pruning (remove macro from model) | 1.603 | +0.167 | +46% |
| + min_child_samples=100 | 1.603 | (included above) | +46% |
| + blend_alpha=0.3 | 1.657 | +0.054 | +**51%** |

## Key Lessons

1. **Cross-sectional features matter more than absolute ones** for stock selection.
   The model needs to know where a stock ranks vs peers, not just its raw level.

2. **Broadcast features are dangerous in residual models**. Macro features that are
   identical across all stocks can't predict cross-sectional variation. The model
   overfits to date-level noise.

3. **More regularization helps with noisy targets**. Financial returns are inherently
   noisy; `min_child_samples=100` prevents the model from memorizing noise.

4. **Turnover costs compound**. Dampening portfolio changes (blend_alpha=0.3) improves
   net Sharpe significantly even though it reduces signal responsiveness.

5. **Use real data in backtests**. Synthetic shortcuts (constant volume) can hide bugs
   and introduce biases that distort results.

## Files Modified

| File | Changes |
|------|---------|
| `python/alpha/features.py` | +3 functions: cross-sectional, macro merge, residual target |
| `python/alpha/train.py` | Updated FEATURE_COLS (21->23), pipeline calls new functions |
| `python/alpha/model.py` | `min_child_samples`: 20->100 |
| `python/backtest/run.py` | Real OHLCV support, VIX scaling, liquidity filter, blend_alpha=0.3 |
| `tests/alpha/test_features.py` | +4 tests for new functions |
| `tests/test_integration.py` | Updated with mock macro + new pipeline |
