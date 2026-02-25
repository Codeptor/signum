# Signum Quant Platform - Improvement Plan

**Generated:** 2026-02-26  
**Current State:** Production-grade backtester with 51% Sharpe improvement (1.097 → 1.657)

## Executive Summary

After analyzing the current codebase against industry best practices from quant finance skills, I've identified **5 priority improvement areas** that will enhance robustness, performance, and production readiness.

---

## Phase 1: Enhanced Risk Metrics (Priority: HIGH)

### Current Gap
Your `RiskEngine` only has basic metrics: VaR, CVaR, Sharpe, max drawdown, HHI.

### Missing Critical Metrics

| Metric | Why It Matters | Implementation Effort |
|--------|---------------|----------------------|
| **Sortino Ratio** | Measures downside risk only (better than Sharpe for asymmetric returns) | Low |
| **Calmar Ratio** | Return / max drawdown (capital preservation focus) | Low |
| **Omega Ratio** | Gains vs losses ratio (entire distribution) | Low |
| **Information Ratio** | Alpha per unit tracking error vs benchmark | Medium |
| **Cornish-Fisher VaR** | Accounts for skewness & kurtosis (fat tails) | Medium |
| **Rolling Metrics** | 63-day rolling Sharpe, VaR, drawdown | Medium |

### Implementation Plan

**File:** `python/portfolio/risk.py`

Add methods to `RiskEngine` class:
```python
def sortino_ratio(self, threshold: float = 0) -> float:
    """Downside deviation only."""
    downside_returns = self.portfolio_returns[self.portfolio_returns < threshold]
    downside_vol = downside_returns.std() * np.sqrt(252)
    excess_return = self.portfolio_returns.mean() * 252 - self.rf_rate
    return excess_return / downside_vol if downside_vol > 0 else 0

def calmar_ratio(self) -> float:
    """Annual return / max drawdown."""
    annual_return = self.portfolio_returns.mean() * 252
    max_dd = abs(self.max_drawdown())
    return annual_return / max_dd if max_dd > 0 else 0

def rolling_sharpe(self, window: int = 63) -> pd.Series:
    """Rolling Sharpe ratio."""
    excess = self.portfolio_returns - self.rf_rate / 252
    rolling_mean = excess.rolling(window).mean() * 252
    rolling_std = excess.rolling(window).std() * np.sqrt(252)
    return rolling_mean / rolling_std
```

**Tests:** Add to `tests/portfolio/test_risk.py`

**Timeline:** 1-2 days

---

## Phase 2: Risk Decomposition & Attribution (Priority: HIGH)

### Current Gap
No visibility into which assets contribute most to portfolio risk.

### What's Missing

1. **Marginal Risk Contribution (MRC)** - How much each asset adds to total volatility
2. **Component Risk** - MRC × weight = contribution to total risk
3. **Risk Parity Weights** - True equal risk contribution (different from HRP)
4. **Diversification Ratio** - Weighted avg vol / portfolio vol

### Implementation Plan

**New File:** `python/portfolio/risk_attribution.py`

```python
class RiskAttribution:
    """Portfolio risk decomposition and attribution."""
    
    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns
        self.weights = weights
        self.cov_matrix = returns.cov() * 252
        self.port_vol = self._portfolio_volatility()
    
    def marginal_risk_contribution(self) -> pd.Series:
        """MRC = (cov_matrix @ weights) / port_vol."""
        return (self.cov_matrix @ self.weights) / self.port_vol
    
    def component_risk(self) -> pd.Series:
        """Component = weight × MRC."""
        return self.weights * self.marginal_risk_contribution()
    
    def risk_parity_weights(self) -> pd.Series:
        """Optimize for equal risk contribution."""
        from scipy.optimize import minimize
        
        def objective(w):
            port_vol = np.sqrt(w @ self.cov_matrix @ w)
            mrc = (self.cov_matrix @ w) / port_vol
            rc = w * mrc
            target = port_vol / len(w)
            return np.sum((rc - target) ** 2)
        
        # Optimize with constraints...
```

**Integration:** Use in dashboard to show "Risk Contributors" chart

**Timeline:** 2-3 days

---

## Phase 3: Advanced Stress Testing (Priority: MEDIUM)

### Current Gap
Basic regime tests (2022-2025). Missing hypothetical scenarios.

### Enhancements

1. **Hypothetical Shock Testing** - What if Tech drops 20%?
2. **Monte Carlo Stress** - Elevated volatility simulations
3. **Correlation Breakdown** - Stress period correlation matrix

### Implementation Plan

**File:** `python/backtest/robustness.py` (extend existing)

```python
class StressTester:
    """Enhanced stress testing."""
    
    def hypothetical_shock(self, shocks: Dict[str, float]) -> float:
        """Test portfolio against hypothetical returns."""
        total_impact = sum(
            self.weights.get(asset, 0) * shock 
            for asset, shock in shocks.items()
        )
        return total_impact
    
    def monte_carlo_stress(
        self, 
        n_sims: int = 10000, 
        vol_mult: float = 2.0,
        horizon: int = 21
    ) -> Dict[str, float]:
        """MC with elevated volatility."""
        mean = self.returns.mean()
        vol = self.returns.std() * vol_mult
        
        sims = np.random.normal(mean, vol, (n_sims, horizon))
        total_rets = (1 + sims).prod(axis=1) - 1
        
        return {
            'var_95': -np.percentile(total_rets, 5),
            'var_99': -np.percentile(total_rets, 1),
            'prob_10pct_loss': (total_rets < -0.10).mean()
        }
```

**Timeline:** 1-2 days

---

## Phase 4: Performance Optimization with C Extensions (Priority: MEDIUM)

### Current Gap
Pure Python for covariance calculations - slow for large portfolios (1000+ assets).

### Optimization Targets

1. **Covariance matrix operations** - Matrix-vector multiply for portfolio vol
2. **Risk contribution calculations** - Repeated in optimization loops
3. **Monte Carlo simulations** - Parallel random number generation

### Implementation Plan

**New Directory:** `python/portfolio/_optimized/`

Files:
- `portfolio_risk.c` - C extension for risk calculations
- `setup.py` - Build configuration
- `optimized.py` - Python wrapper

```c
// portfolio_risk.c
static PyObject* portfolio_volatility(PyObject* self, PyObject* args) {
    PyArrayObject *cov_matrix, *weights;
    
    if (!PyArg_ParseTuple(args, "OO", &cov_matrix, &weights))
        return NULL;
    
    // Direct memory access, no bounds checking
    double *cov_data = (double*)PyArray_DATA(cov_matrix);
    double *w_data = (double*)PyArray_DATA(weights);
    
    // Calculate w^T @ cov @ w
    // ... implementation
    
    return PyFloat_FromDouble(volatility);
}
```

**Expected Speedup:** 5-10x for large portfolios (n > 500)

**Timeline:** 3-4 days

---

## Phase 5: Risk Management Rules Integration (Priority: MEDIUM)

### Current Gap
No automated risk checks during backtest execution.

### Risk Rules from Analysis

From the risk-management skill patterns:

1. **Position Size Limits** - Max 25% equity per position (you have 15%)
2. **Trade Frequency Adaptation** - Reduce trades in choppy markets
3. **Risk Validation** - Check risk/reward before every trade
4. **Loss Limits** - Close positions proactively with stop-losses

### Implementation Plan

**New File:** `python/portfolio/risk_manager.py`

```python
class RiskManager:
    """Real-time risk monitoring and enforcement."""
    
    def __init__(self, max_position_pct: float = 0.25, 
                 max_daily_trades: int = 10):
        self.max_position = max_position_pct
        self.max_trades = max_daily_trades
        self.daily_trades = 0
    
    def check_trade(self, signal: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """Validate trade against risk rules."""
        
        # Check position size
        if signal['weight'] > self.max_position:
            return False, f"Position size {signal['weight']:.1%} exceeds limit"
        
        # Check daily trade limit
        if self.daily_trades >= self.max_trades:
            return False, "Daily trade limit reached"
        
        # Check risk/reward
        if signal.get('risk_reward', 0) < 2.0:
            return False, "Risk/reward ratio below 2:1"
        
        return True, "OK"
    
    def check_portfolio_risk(self, returns: pd.Series) -> Dict:
        """Daily portfolio risk check."""
        metrics = RiskMetrics(returns)
        
        alerts = []
        if metrics.var_historical(0.95) > 0.05:  # 5% daily VaR
            alerts.append("HIGH_VAR")
        if metrics.max_drawdown() < -0.20:  # 20% drawdown
            alerts.append("HIGH_DRAWDOWN")
        
        return {'alerts': alerts, 'metrics': metrics.summary()}
```

**Integration:** Call in `run_backtest()` before each rebalance

**Timeline:** 2-3 days

---

## Implementation Priority Matrix

| Phase | Impact | Effort | Risk | Priority |
|-------|--------|--------|------|----------|
| 1: Enhanced Risk Metrics | High | Low | Low | **P0** |
| 2: Risk Attribution | High | Medium | Low | **P0** |
| 3: Stress Testing | Medium | Low | Low | **P1** |
| 4: C Extensions | Medium | High | Medium | **P2** |
| 5: Risk Manager | High | Medium | Low | **P1** |

---

## Success Metrics

**Phase 1:**
- [ ] Sortino, Calmar, Omega ratios implemented
- [ ] All metrics tested with 100% coverage
- [ ] Dashboard updated with new metrics

**Phase 2:**
- [ ] Risk attribution charts in dashboard
- [ ] Top 5 risk contributors identified per rebalance
- [ ] Risk parity optimizer working

**Phase 3:**
- [ ] Hypothetical stress tests runnable
- [ ] MC stress with confidence intervals
- [ ] Stress test results in backtest output

**Phase 4:**
- [ ] C extension builds successfully
- [ ] 5x+ speedup for n=500 portfolio
- [ ] Numerical accuracy within 1e-10

**Phase 5:**
- [ ] Risk checks run on every trade
- [ ] Alerts logged when limits breached
- [ ] Configurable risk parameters

---

## Next Steps

1. **Start with Phase 1** - Quick wins, immediate value
2. **Then Phase 2** - Adds significant analytical capability
3. **Phase 3 & 5** - Can be done in parallel
4. **Phase 4** - Save for when you need scale (1000+ assets)

Would you like me to start implementing Phase 1 (Enhanced Risk Metrics)?
