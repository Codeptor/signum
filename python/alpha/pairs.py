"""Cointegration-based pairs trading signal generation.

Implements the statistical arbitrage framework:
  1. Screen universe for cointegrated pairs (Engle-Granger / Augmented Dickey-Fuller).
  2. Estimate the hedge ratio (OLS or Kalman filter).
  3. Compute the spread and its z-score.
  4. Generate trading signals from z-score mean reversion.
  5. Estimate half-life via Ornstein-Uhlenbeck regression.

Usage::

    scanner = PairScanner(prices)
    pairs = scanner.find_cointegrated_pairs(pvalue_threshold=0.05)
    for pair in pairs:
        signal = pair.generate_signal()
        # signal > entry_z → short spread, signal < -entry_z → long spread

References:
  - Engle & Granger (1987), "Co-Integration and Error Correction"
  - Vidyamurthy (2004), "Pairs Trading"
  - Avellaneda & Lee (2010), "Statistical Arbitrage in the US Equity Market"
"""

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PairResult:
    """Analysis result for a single pair."""

    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    adf_statistic: float
    adf_pvalue: float
    is_cointegrated: bool
    half_life: float  # Ornstein-Uhlenbeck half-life in days
    spread_mean: float
    spread_std: float
    correlation: float
    hurst_exponent: float

    @property
    def mean_reversion_speed(self) -> float:
        """Kappa = ln(2) / half_life."""
        return np.log(2) / max(self.half_life, 0.1)


@dataclass
class PairSignal:
    """Trading signal for a pair."""

    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    z_score: pd.Series  # Full z-score history
    current_z: float
    signal: int  # +1 = long spread, -1 = short spread, 0 = flat
    entry_z: float
    exit_z: float


# ---------------------------------------------------------------------------
# ADF test (avoid statsmodels dependency)
# ---------------------------------------------------------------------------


def _adf_test(y: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """Simplified ADF test using OLS regression.

    Tests H0: series has a unit root (non-stationary).

    Returns (t_statistic, approximate_pvalue).
    """
    n = len(y)
    if n < 20:
        return 0.0, 1.0

    dy = np.diff(y)
    y_lag = y[:-1]

    # ADF regression: dy_t = alpha + beta * y_{t-1} + sum(gamma_i * dy_{t-i}) + eps
    X = np.column_stack([np.ones(len(dy) - max_lags), y_lag[max_lags:]])
    for lag in range(1, max_lags + 1):
        X = np.column_stack([X, dy[max_lags - lag : len(dy) - lag]])

    Y = dy[max_lags:]

    try:
        beta, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    # T-statistic for the coefficient on y_{t-1}
    if len(residuals) > 0:
        sse = residuals[0]
    else:
        sse = np.sum((Y - X @ beta) ** 2)

    n_obs = len(Y)
    k = X.shape[1]
    mse = sse / max(n_obs - k, 1)
    var_beta = mse * np.linalg.pinv(X.T @ X)
    se_beta1 = np.sqrt(max(var_beta[1, 1], 1e-20))
    t_stat = beta[1] / se_beta1

    # Approximate p-value using MacKinnon critical values (n=100 interpolation)
    # Critical values: 1%=-3.51, 5%=-2.89, 10%=-2.58
    if t_stat < -3.51:
        p_approx = 0.005
    elif t_stat < -2.89:
        p_approx = 0.025
    elif t_stat < -2.58:
        p_approx = 0.075
    elif t_stat < -1.95:
        p_approx = 0.15
    else:
        p_approx = 0.5

    return float(t_stat), p_approx


# ---------------------------------------------------------------------------
# Half-life estimation
# ---------------------------------------------------------------------------


def _estimate_half_life(spread: np.ndarray) -> float:
    """Estimate OU half-life from spread time series.

    Regression: d(spread) = kappa * (mu - spread) * dt + noise
    → d(spread) = a + b * spread_{t-1}
    Half-life = -ln(2) / b
    """
    y = np.diff(spread)
    x = spread[:-1]

    if len(y) < 10:
        return float("inf")

    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")

    b = beta[1]
    if b >= 0:
        return float("inf")  # Not mean-reverting

    half_life = -np.log(2) / b
    return float(max(half_life, 0.1))


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------


def _hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """Estimate Hurst exponent using rescaled range (R/S) analysis.

    H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending.
    """
    n = len(ts)
    if n < 20:
        return 0.5

    lags = range(2, min(max_lag + 1, n // 2))
    tau = []
    rs = []

    for lag in lags:
        # Split into sub-series of length lag
        n_sub = n // lag
        if n_sub < 1:
            continue

        rs_vals = []
        for i in range(n_sub):
            sub = ts[i * lag : (i + 1) * lag]
            mean_sub = np.mean(sub)
            cumdev = np.cumsum(sub - mean_sub)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(sub)
            if S > 1e-10:
                rs_vals.append(R / S)

        if rs_vals:
            tau.append(lag)
            rs.append(np.mean(rs_vals))

    if len(tau) < 3:
        return 0.5

    log_tau = np.log(tau)
    log_rs = np.log(rs)

    try:
        slope, _, _, _, _ = stats.linregress(log_tau, log_rs)
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Pair Scanner
# ---------------------------------------------------------------------------


class PairScanner:
    """Scan a universe of price series for cointegrated pairs.

    Parameters
    ----------
    prices : pd.DataFrame
        Price series (columns=tickers, index=dates).
    min_history : int
        Minimum overlapping days required.
    """

    def __init__(self, prices: pd.DataFrame, min_history: int = 120):
        self.prices = prices.dropna(axis=1, how="all")
        self.min_history = min_history
        self.tickers = list(self.prices.columns)

    def _compute_spread(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute spread = log(A) - beta * log(B) via OLS."""
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        X = np.column_stack([np.ones(len(log_b)), log_b])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, log_a, rcond=None)
        except np.linalg.LinAlgError:
            return log_a - log_b, 1.0

        hedge_ratio = beta[1]
        spread = log_a - hedge_ratio * log_b
        return spread, float(hedge_ratio)

    def analyze_pair(self, ticker_a: str, ticker_b: str) -> PairResult:
        """Analyze a single pair for cointegration."""
        a = self.prices[ticker_a].dropna()
        b = self.prices[ticker_b].dropna()
        common = a.index.intersection(b.index)

        if len(common) < self.min_history:
            return PairResult(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                hedge_ratio=1.0,
                adf_statistic=0.0,
                adf_pvalue=1.0,
                is_cointegrated=False,
                half_life=float("inf"),
                spread_mean=0.0,
                spread_std=0.0,
                correlation=0.0,
                hurst_exponent=0.5,
            )

        pa = a.loc[common].values
        pb = b.loc[common].values
        spread, hedge_ratio = self._compute_spread(pa, pb)
        adf_stat, adf_p = _adf_test(spread)
        half_life = _estimate_half_life(spread)
        hurst = _hurst_exponent(spread)
        corr = float(np.corrcoef(pa, pb)[0, 1])

        return PairResult(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            hedge_ratio=hedge_ratio,
            adf_statistic=adf_stat,
            adf_pvalue=adf_p,
            is_cointegrated=adf_p < 0.05,
            half_life=half_life,
            spread_mean=float(np.mean(spread)),
            spread_std=float(np.std(spread)),
            correlation=corr,
            hurst_exponent=hurst,
        )

    def find_cointegrated_pairs(
        self,
        pvalue_threshold: float = 0.05,
        max_half_life: float = 60,
        min_half_life: float = 1,
        max_pairs: int = 50,
    ) -> list[PairResult]:
        """Screen all pairs for cointegration.

        Parameters
        ----------
        pvalue_threshold : float
            ADF p-value threshold for cointegration.
        max_half_life : float
            Maximum half-life in days (longer = slower reversion).
        min_half_life : float
            Minimum half-life (shorter might be microstructure noise).
        max_pairs : int
            Maximum number of pairs to return.

        Returns
        -------
        list[PairResult]
            Pairs sorted by ADF p-value (best first).
        """
        results = []
        n_tickers = len(self.tickers)

        logger.info(
            f"Scanning {n_tickers * (n_tickers - 1) // 2} pairs "
            f"from {n_tickers} tickers"
        )

        for a, b in combinations(self.tickers, 2):
            pair = self.analyze_pair(a, b)
            if (
                pair.adf_pvalue < pvalue_threshold
                and min_half_life <= pair.half_life <= max_half_life
            ):
                results.append(pair)

        results.sort(key=lambda p: p.adf_pvalue)
        results = results[:max_pairs]

        logger.info(f"Found {len(results)} cointegrated pairs")
        return results

    def generate_signal(
        self,
        pair: PairResult,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> PairSignal:
        """Generate trading signal for a pair.

        Parameters
        ----------
        pair : PairResult
        lookback : int
            Rolling window for z-score computation.
        entry_z : float
            Z-score threshold for entry.
        exit_z : float
            Z-score threshold for exit.

        Returns
        -------
        PairSignal
        """
        a = self.prices[pair.ticker_a].dropna()
        b = self.prices[pair.ticker_b].dropna()
        common = a.index.intersection(b.index)

        pa = a.loc[common]
        pb = b.loc[common]

        log_a = np.log(pa)
        log_b = np.log(pb)
        spread = log_a - pair.hedge_ratio * log_b

        rolling_mean = spread.rolling(lookback).mean()
        rolling_std = spread.rolling(lookback).std().clip(lower=1e-8)
        z_score = (spread - rolling_mean) / rolling_std
        z_score = z_score.dropna()

        current_z = float(z_score.iloc[-1]) if len(z_score) > 0 else 0.0

        if current_z > entry_z:
            signal = -1  # Short spread (spread is expensive)
        elif current_z < -entry_z:
            signal = 1  # Long spread (spread is cheap)
        elif abs(current_z) < exit_z:
            signal = 0  # Close position
        else:
            signal = 0  # In no-trade zone

        return PairSignal(
            ticker_a=pair.ticker_a,
            ticker_b=pair.ticker_b,
            hedge_ratio=pair.hedge_ratio,
            z_score=z_score,
            current_z=current_z,
            signal=signal,
            entry_z=entry_z,
            exit_z=exit_z,
        )
