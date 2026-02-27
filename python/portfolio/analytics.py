"""Comprehensive portfolio performance analytics.

Implements institutional-grade performance measurement:
  1. Return metrics: CAGR, total return, best/worst periods.
  2. Risk-adjusted: Sharpe, Sortino, Calmar, Omega, Information ratio.
  3. Drawdown analytics: max DD, duration, recovery, underwater curve.
  4. Alpha/Beta decomposition vs benchmark.
  5. Rolling performance windows for regime analysis.
  6. Monthly/annual return heatmap data.

Usage::

    analyzer = PerformanceAnalyzer(returns, benchmark_returns)
    report = analyzer.full_report()
    rolling = analyzer.rolling_sharpe(window=252)
    monthly = analyzer.monthly_returns()

References:
  - Bacon (2008), "Practical Portfolio Performance Measurement and Attribution"
  - Lo (2002), "The Statistics of Sharpe Ratios"
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DrawdownInfo:
    """Drawdown analysis result."""

    max_drawdown: float
    max_dd_start: int  # Index of peak before max DD
    max_dd_trough: int  # Index of trough
    max_dd_recovery: int | None  # Index of recovery (None if not recovered)
    max_dd_duration: int  # Days from peak to trough
    max_dd_recovery_duration: int | None  # Days from trough to recovery
    avg_drawdown: float
    avg_drawdown_duration: float
    n_drawdowns: int  # Number of drawdowns > threshold


@dataclass
class PerformanceReport:
    """Full performance analytics."""

    # Returns
    total_return: float
    cagr: float
    mean_daily_return: float
    volatility: float  # Annualized

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float | None  # Requires benchmark
    tracking_error: float | None

    # Tail risk
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    var_99: float

    # Drawdown
    drawdown: DrawdownInfo

    # Alpha/Beta
    alpha: float | None  # Requires benchmark
    beta: float | None

    # Descriptive
    n_observations: int
    n_years: float
    hit_rate: float  # Fraction of positive return days
    gain_loss_ratio: float  # Average gain / average loss
    best_day: float
    worst_day: float

    def summary(self) -> str:
        lines = [
            f"Performance Report ({self.n_years:.1f} years, {self.n_observations} obs)",
            f"  CAGR:       {self.cagr:.2%}",
            f"  Volatility: {self.volatility:.2%}",
            f"  Sharpe:     {self.sharpe_ratio:.2f}",
            f"  Sortino:    {self.sortino_ratio:.2f}",
            f"  Calmar:     {self.calmar_ratio:.2f}",
            f"  Max DD:     {self.drawdown.max_drawdown:.2%}",
            f"  VaR 95%:    {self.var_95:.2%}",
            f"  Hit Rate:   {self.hit_rate:.2%}",
        ]
        if self.alpha is not None:
            lines.append(f"  Alpha:      {self.alpha:.4f}")
            lines.append(f"  Beta:       {self.beta:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Performance Analyzer
# ---------------------------------------------------------------------------


class PerformanceAnalyzer:
    """Comprehensive portfolio performance analysis.

    Parameters
    ----------
    returns : np.ndarray
        Daily returns (simple, not log).
    benchmark_returns : np.ndarray, optional
        Benchmark daily returns (same length as returns).
    risk_free_rate : float
        Annualized risk-free rate.
    """

    def __init__(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray | None = None,
        risk_free_rate: float = 0.0,
    ):
        self.returns = np.asarray(returns, dtype=float).ravel()
        self.benchmark = (
            np.asarray(benchmark_returns, dtype=float).ravel()
            if benchmark_returns is not None
            else None
        )
        self.rf = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        self.n = len(self.returns)
        self.n_years = self.n / TRADING_DAYS_PER_YEAR

        # Precompute wealth curve
        self._wealth = np.cumprod(1 + self.returns)

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        return float(self._wealth[-1] - 1) if self.n > 0 else 0.0

    def cagr(self) -> float:
        if self.n_years < 1e-6:
            return 0.0
        total = self._wealth[-1]
        return float(total ** (1 / self.n_years) - 1)

    def volatility(self) -> float:
        return float(np.std(self.returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        excess = self.returns - self.rf_daily
        std = np.std(excess, ddof=1)
        if std < 1e-12:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS_PER_YEAR))

    def sortino_ratio(self) -> float:
        excess = self.returns - self.rf_daily
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf") if np.mean(excess) > 0 else 0.0
        downside_std = np.sqrt(np.mean(downside**2))
        if downside_std < 1e-12:
            return 0.0
        return float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))

    def calmar_ratio(self) -> float:
        dd = self.max_drawdown()
        if abs(dd) < 1e-12:
            return 0.0
        return float(self.cagr() / abs(dd))

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """Omega ratio: probability-weighted gain/loss ratio."""
        excess = self.returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess <= 0].sum())
        if losses < 1e-12:
            return float("inf") if gains > 0 else 1.0
        return float(1 + gains / losses)

    def information_ratio(self) -> float | None:
        if self.benchmark is None:
            return None
        active = self.returns - self.benchmark
        te = np.std(active, ddof=1)
        if te < 1e-12:
            return 0.0
        return float(np.mean(active) / te * np.sqrt(TRADING_DAYS_PER_YEAR))

    def tracking_error(self) -> float | None:
        if self.benchmark is None:
            return None
        active = self.returns - self.benchmark
        return float(np.std(active, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------

    def var(self, confidence: float = 0.95) -> float:
        """Historical Value at Risk (negative number)."""
        return float(np.percentile(self.returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        """Historical CVaR / Expected Shortfall."""
        threshold = self.var(confidence)
        tail = self.returns[self.returns <= threshold]
        return float(tail.mean()) if len(tail) > 0 else threshold

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self._wealth)
        dd = (self._wealth - peak) / np.maximum(peak, 1e-10)
        return float(dd.min())

    def drawdown_analysis(self, threshold: float = 0.01) -> DrawdownInfo:
        """Comprehensive drawdown analysis."""
        wealth = self._wealth
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / np.maximum(peak, 1e-10)

        # Find max drawdown
        trough_idx = int(np.argmin(dd))
        peak_idx = int(np.argmax(wealth[:trough_idx + 1])) if trough_idx > 0 else 0

        # Recovery
        recovery_idx = None
        for i in range(trough_idx, self.n):
            if wealth[i] >= peak[trough_idx]:
                recovery_idx = i
                break

        # Count all drawdowns > threshold
        in_dd = False
        n_drawdowns = 0
        dd_durations = []
        dd_start = 0
        for i in range(self.n):
            if dd[i] < -threshold and not in_dd:
                in_dd = True
                n_drawdowns += 1
                dd_start = i
            elif dd[i] >= -threshold / 10 and in_dd:
                in_dd = False
                dd_durations.append(i - dd_start)

        return DrawdownInfo(
            max_drawdown=float(dd.min()),
            max_dd_start=peak_idx,
            max_dd_trough=trough_idx,
            max_dd_recovery=recovery_idx,
            max_dd_duration=trough_idx - peak_idx,
            max_dd_recovery_duration=(
                recovery_idx - trough_idx if recovery_idx is not None else None
            ),
            avg_drawdown=float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
            avg_drawdown_duration=(
                float(np.mean(dd_durations)) if dd_durations else 0.0
            ),
            n_drawdowns=n_drawdowns,
        )

    def underwater_curve(self) -> np.ndarray:
        """Drawdown curve (always <= 0)."""
        peak = np.maximum.accumulate(self._wealth)
        return (self._wealth - peak) / np.maximum(peak, 1e-10)

    # ------------------------------------------------------------------
    # Alpha / Beta
    # ------------------------------------------------------------------

    def alpha_beta(self) -> tuple[float | None, float | None]:
        """CAPM alpha and beta vs benchmark."""
        if self.benchmark is None:
            return None, None

        excess_port = self.returns - self.rf_daily
        excess_bench = self.benchmark - self.rf_daily

        cov = np.cov(excess_port, excess_bench)
        var_bench = cov[1, 1]
        if var_bench < 1e-16:
            return None, None

        beta = float(cov[0, 1] / var_bench)
        alpha = float(
            np.mean(excess_port) - beta * np.mean(excess_bench)
        ) * TRADING_DAYS_PER_YEAR

        return alpha, beta

    # ------------------------------------------------------------------
    # Rolling
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window: int = 252) -> np.ndarray:
        """Rolling annualized Sharpe ratio."""
        n = self.n
        result = np.full(n, np.nan)
        excess = self.returns - self.rf_daily
        for i in range(window, n):
            chunk = excess[i - window : i]
            std = np.std(chunk, ddof=1)
            if std > 1e-12:
                result[i] = np.mean(chunk) / std * np.sqrt(TRADING_DAYS_PER_YEAR)
        return result

    def rolling_volatility(self, window: int = 63) -> np.ndarray:
        """Rolling annualized volatility."""
        n = self.n
        result = np.full(n, np.nan)
        for i in range(window, n):
            result[i] = np.std(self.returns[i - window : i], ddof=1) * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        return result

    def rolling_max_drawdown(self, window: int = 252) -> np.ndarray:
        """Rolling max drawdown over window."""
        n = self.n
        result = np.full(n, np.nan)
        for i in range(window, n):
            chunk_wealth = np.cumprod(1 + self.returns[i - window : i])
            peak = np.maximum.accumulate(chunk_wealth)
            dd = (chunk_wealth - peak) / np.maximum(peak, 1e-10)
            result[i] = dd.min()
        return result

    # ------------------------------------------------------------------
    # Monthly / Annual Returns
    # ------------------------------------------------------------------

    def period_returns(self, period_length: int = 21) -> np.ndarray:
        """Returns aggregated over fixed-length periods.

        Parameters
        ----------
        period_length : int
            Number of trading days per period (21 ≈ monthly, 252 ≈ annual).

        Returns
        -------
        np.ndarray
            Compounded return per period.
        """
        n_periods = self.n // period_length
        results = np.zeros(n_periods)
        for i in range(n_periods):
            start = i * period_length
            end = start + period_length
            results[i] = np.prod(1 + self.returns[start:end]) - 1
        return results

    # ------------------------------------------------------------------
    # Full Report
    # ------------------------------------------------------------------

    def full_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        alpha, beta = self.alpha_beta()

        pos_rets = self.returns[self.returns > 0]
        neg_rets = self.returns[self.returns < 0]
        avg_gain = float(np.mean(pos_rets)) if len(pos_rets) > 0 else 0.0
        avg_loss = float(np.mean(np.abs(neg_rets))) if len(neg_rets) > 0 else 1e-12
        gl_ratio = avg_gain / max(avg_loss, 1e-12)

        return PerformanceReport(
            total_return=self.total_return(),
            cagr=self.cagr(),
            mean_daily_return=float(np.mean(self.returns)),
            volatility=self.volatility(),
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            calmar_ratio=self.calmar_ratio(),
            omega_ratio=self.omega_ratio(),
            information_ratio=self.information_ratio(),
            tracking_error=self.tracking_error(),
            skewness=float(_skew(self.returns)),
            kurtosis=float(_kurtosis(self.returns)),
            var_95=self.var(0.95),
            cvar_95=self.cvar(0.95),
            var_99=self.var(0.99),
            drawdown=self.drawdown_analysis(),
            alpha=alpha,
            beta=beta,
            n_observations=self.n,
            n_years=self.n_years,
            hit_rate=float(np.mean(self.returns > 0)),
            gain_loss_ratio=gl_ratio,
            best_day=float(np.max(self.returns)),
            worst_day=float(np.min(self.returns)),
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _skew(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * np.sum(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    m4 = np.mean(((x - m) / s) ** 4)
    return float(m4 - 3)
