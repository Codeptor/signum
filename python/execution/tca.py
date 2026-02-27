"""Transaction Cost Analysis (TCA) — measure and decompose execution costs.

Implements the standard TCA framework used by institutional traders:
  1. Implementation Shortfall (IS) decomposition into delay, timing, and market impact.
  2. Spread cost estimation from quote data or statistical models.
  3. Market impact modeling (temporary + permanent) using square-root model.
  4. Arrival price benchmarks and VWAP/TWAP slippage.
  5. Cost attribution across trades for performance analysis.

Usage::

    analyzer = TCAAnalyzer(impact_model=SquareRootImpact())
    fills = [Fill("AAPL", "BUY", 500, 152.30, arrival_price=151.80, ...)]
    report = analyzer.analyze(fills, portfolio_value=1_000_000)
    print(report.summary())

References:
  - Kissell & Glantz (2003), "Optimal Trading Strategies"
  - Almgren et al. (2005), "Direct Estimation of Equity Market Impact"
  - Frazzini, Israel & Moskowitz (2018), "Trading Costs"
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Fill:
    """A single execution fill."""

    ticker: str
    side: Side
    shares: int
    fill_price: float
    arrival_price: float  # Price at decision time
    vwap_benchmark: float = 0.0  # Market VWAP over execution window
    twap_benchmark: float = 0.0  # Market TWAP over execution window
    adv: float = 0.0  # Average daily volume
    spread_bps: float = 0.0  # Bid-ask spread at arrival
    volatility: float = 0.0  # Daily volatility (decimal)
    execution_minutes: float = 0.0  # Time to complete fill
    decision_price: float = 0.0  # Price when trading decision was made
    close_price: float = 0.0  # Close price on execution day

    @property
    def notional(self) -> float:
        return self.shares * self.fill_price

    @property
    def participation_rate(self) -> float:
        """Fraction of ADV traded."""
        if self.adv <= 0:
            return 0.0
        return self.shares / self.adv


@dataclass
class FillCostBreakdown:
    """Cost breakdown for a single fill."""

    ticker: str
    side: Side
    shares: int
    notional: float

    # Implementation shortfall components (all in bps)
    total_is_bps: float  # Total implementation shortfall
    spread_cost_bps: float  # Half-spread cost
    market_impact_bps: float  # Price impact
    timing_cost_bps: float  # Delay / timing cost
    opportunity_cost_bps: float  # Unfilled portion cost (if partial)

    # Benchmark slippage
    arrival_slippage_bps: float  # vs arrival price
    vwap_slippage_bps: float  # vs VWAP
    twap_slippage_bps: float  # vs TWAP

    # Predicted vs actual
    predicted_impact_bps: float
    prediction_error_bps: float


@dataclass
class TCAReport:
    """Aggregate TCA report across fills."""

    fills: list[FillCostBreakdown]
    total_notional: float
    total_shares: int

    # Aggregate costs (notional-weighted, in bps)
    avg_is_bps: float
    avg_spread_cost_bps: float
    avg_impact_bps: float
    avg_timing_cost_bps: float

    # Benchmark slippage
    avg_arrival_slippage_bps: float
    avg_vwap_slippage_bps: float

    # Model quality
    avg_prediction_error_bps: float
    prediction_rmse_bps: float

    # Distribution stats
    is_std_bps: float
    worst_fill_bps: float
    best_fill_bps: float

    n_buys: int
    n_sells: int

    def summary(self) -> str:
        return (
            f"TCAReport: {len(self.fills)} fills "
            f"({self.n_buys} buys, {self.n_sells} sells), "
            f"notional=${self.total_notional:,.0f}\n"
            f"  Avg IS: {self.avg_is_bps:.1f}bps, "
            f"spread: {self.avg_spread_cost_bps:.1f}bps, "
            f"impact: {self.avg_impact_bps:.1f}bps, "
            f"timing: {self.avg_timing_cost_bps:.1f}bps\n"
            f"  VWAP slippage: {self.avg_vwap_slippage_bps:.1f}bps, "
            f"arrival slippage: {self.avg_arrival_slippage_bps:.1f}bps\n"
            f"  Model RMSE: {self.prediction_rmse_bps:.1f}bps"
        )


# ---------------------------------------------------------------------------
# Market Impact Models
# ---------------------------------------------------------------------------


class ImpactModel(ABC):
    """Base class for market impact models."""

    @abstractmethod
    def temporary_impact_bps(
        self,
        participation_rate: float,
        volatility: float,
        spread_bps: float,
    ) -> float:
        """Estimate temporary (transient) impact in bps."""

    @abstractmethod
    def permanent_impact_bps(
        self,
        participation_rate: float,
        volatility: float,
    ) -> float:
        """Estimate permanent impact in bps."""

    def total_impact_bps(
        self,
        participation_rate: float,
        volatility: float,
        spread_bps: float,
    ) -> float:
        return self.temporary_impact_bps(
            participation_rate, volatility, spread_bps
        ) + self.permanent_impact_bps(participation_rate, volatility)


class SquareRootImpact(ImpactModel):
    """Square-root market impact model (Almgren et al., 2005).

    Temporary impact ≈ spread/2 + eta * sigma * sqrt(Q/V)
    Permanent impact ≈ gamma * sigma * (Q/V)^delta

    Parameters
    ----------
    eta : float
        Temporary impact coefficient (typical: 0.1 - 0.5).
    gamma : float
        Permanent impact coefficient (typical: 0.05 - 0.3).
    delta : float
        Permanent impact exponent (typical: 0.5 - 0.7).
    """

    def __init__(
        self,
        eta: float = 0.25,
        gamma: float = 0.10,
        delta: float = 0.5,
    ):
        self.eta = eta
        self.gamma = gamma
        self.delta = delta

    def temporary_impact_bps(
        self,
        participation_rate: float,
        volatility: float,
        spread_bps: float,
    ) -> float:
        # Half-spread + temporary market impact
        half_spread = spread_bps / 2
        impact = self.eta * volatility * 10_000 * np.sqrt(
            max(participation_rate, 0)
        )
        return half_spread + impact

    def permanent_impact_bps(
        self,
        participation_rate: float,
        volatility: float,
    ) -> float:
        return self.gamma * volatility * 10_000 * max(
            participation_rate, 0
        ) ** self.delta


class LinearImpact(ImpactModel):
    """Simple linear impact model for low-participation scenarios.

    Temporary impact = spread/2 + k_temp * participation_rate * sigma
    Permanent impact = k_perm * participation_rate * sigma
    """

    def __init__(self, k_temp: float = 0.5, k_perm: float = 0.1):
        self.k_temp = k_temp
        self.k_perm = k_perm

    def temporary_impact_bps(
        self,
        participation_rate: float,
        volatility: float,
        spread_bps: float,
    ) -> float:
        half_spread = spread_bps / 2
        impact = self.k_temp * participation_rate * volatility * 10_000
        return half_spread + impact

    def permanent_impact_bps(
        self,
        participation_rate: float,
        volatility: float,
    ) -> float:
        return self.k_perm * participation_rate * volatility * 10_000


# ---------------------------------------------------------------------------
# Implementation Shortfall Decomposition
# ---------------------------------------------------------------------------


def _compute_is_bps(fill: Fill, side_sign: int) -> float:
    """Total implementation shortfall in bps."""
    if fill.arrival_price <= 0:
        return 0.0
    slippage = (fill.fill_price - fill.arrival_price) / fill.arrival_price
    return slippage * side_sign * 10_000


def _compute_spread_cost_bps(fill: Fill) -> float:
    """Spread cost component (half the bid-ask spread)."""
    return fill.spread_bps / 2


def _compute_timing_cost_bps(fill: Fill, side_sign: int) -> float:
    """Timing cost = price movement between decision and arrival."""
    if fill.decision_price <= 0 or fill.arrival_price <= 0:
        return 0.0
    drift = (fill.arrival_price - fill.decision_price) / fill.decision_price
    return drift * side_sign * 10_000


def _compute_opportunity_cost_bps(fill: Fill, side_sign: int) -> float:
    """Opportunity cost from unfilled shares (close vs arrival)."""
    if fill.close_price <= 0 or fill.arrival_price <= 0:
        return 0.0
    # This captures what you missed by not trading instantly
    drift = (fill.close_price - fill.arrival_price) / fill.arrival_price
    return max(drift * side_sign * 10_000, 0.0)


def _compute_benchmark_slippage(
    fill_price: float, benchmark: float, side_sign: int
) -> float:
    """Slippage vs a benchmark price in bps."""
    if benchmark <= 0:
        return 0.0
    return (fill_price - benchmark) / benchmark * side_sign * 10_000


# ---------------------------------------------------------------------------
# TCA Analyzer
# ---------------------------------------------------------------------------


class TCAAnalyzer:
    """Analyze execution quality across a set of fills.

    Parameters
    ----------
    impact_model : ImpactModel
        Model for predicting market impact.
    """

    def __init__(self, impact_model: ImpactModel | None = None):
        self.impact_model = impact_model or SquareRootImpact()

    def analyze_fill(self, fill: Fill) -> FillCostBreakdown:
        """Decompose costs for a single fill."""
        side_sign = 1 if fill.side == Side.BUY else -1

        total_is = _compute_is_bps(fill, side_sign)
        spread_cost = _compute_spread_cost_bps(fill)
        timing_cost = _compute_timing_cost_bps(fill, side_sign)
        opportunity_cost = _compute_opportunity_cost_bps(fill, side_sign)

        # Market impact = residual after spread and timing
        market_impact = max(total_is - spread_cost - timing_cost, 0.0)

        # Benchmark slippage
        arrival_slip = _compute_benchmark_slippage(
            fill.fill_price, fill.arrival_price, side_sign
        )
        vwap_slip = _compute_benchmark_slippage(
            fill.fill_price, fill.vwap_benchmark, side_sign
        )
        twap_slip = _compute_benchmark_slippage(
            fill.fill_price, fill.twap_benchmark, side_sign
        )

        # Predicted impact from model
        predicted = self.impact_model.total_impact_bps(
            fill.participation_rate, fill.volatility, fill.spread_bps
        )
        prediction_error = abs(total_is - predicted)

        return FillCostBreakdown(
            ticker=fill.ticker,
            side=fill.side,
            shares=fill.shares,
            notional=fill.notional,
            total_is_bps=total_is,
            spread_cost_bps=spread_cost,
            market_impact_bps=market_impact,
            timing_cost_bps=timing_cost,
            opportunity_cost_bps=opportunity_cost,
            arrival_slippage_bps=arrival_slip,
            vwap_slippage_bps=vwap_slip,
            twap_slippage_bps=twap_slip,
            predicted_impact_bps=predicted,
            prediction_error_bps=prediction_error,
        )

    def analyze(self, fills: list[Fill]) -> TCAReport:
        """Produce aggregate TCA report from a list of fills."""
        if not fills:
            return TCAReport(
                fills=[],
                total_notional=0,
                total_shares=0,
                avg_is_bps=0,
                avg_spread_cost_bps=0,
                avg_impact_bps=0,
                avg_timing_cost_bps=0,
                avg_arrival_slippage_bps=0,
                avg_vwap_slippage_bps=0,
                avg_prediction_error_bps=0,
                prediction_rmse_bps=0,
                is_std_bps=0,
                worst_fill_bps=0,
                best_fill_bps=0,
                n_buys=0,
                n_sells=0,
            )

        breakdowns = [self.analyze_fill(f) for f in fills]
        total_notional = sum(b.notional for b in breakdowns)
        total_shares = sum(b.shares for b in breakdowns)

        # Notional-weighted average costs
        def _wavg(field: str) -> float:
            if total_notional <= 0:
                return 0.0
            return sum(
                getattr(b, field) * b.notional for b in breakdowns
            ) / total_notional

        is_values = [b.total_is_bps for b in breakdowns]
        errors = [b.prediction_error_bps for b in breakdowns]

        return TCAReport(
            fills=breakdowns,
            total_notional=total_notional,
            total_shares=total_shares,
            avg_is_bps=_wavg("total_is_bps"),
            avg_spread_cost_bps=_wavg("spread_cost_bps"),
            avg_impact_bps=_wavg("market_impact_bps"),
            avg_timing_cost_bps=_wavg("timing_cost_bps"),
            avg_arrival_slippage_bps=_wavg("arrival_slippage_bps"),
            avg_vwap_slippage_bps=_wavg("vwap_slippage_bps"),
            avg_prediction_error_bps=float(np.mean(errors)) if errors else 0.0,
            prediction_rmse_bps=float(np.sqrt(np.mean(np.array(errors) ** 2)))
            if errors
            else 0.0,
            is_std_bps=float(np.std(is_values)) if len(is_values) > 1 else 0.0,
            worst_fill_bps=max(is_values) if is_values else 0.0,
            best_fill_bps=min(is_values) if is_values else 0.0,
            n_buys=sum(1 for b in breakdowns if b.side == Side.BUY),
            n_sells=sum(1 for b in breakdowns if b.side == Side.SELL),
        )

    def estimate_pretrade_cost(
        self,
        shares: int,
        price: float,
        adv: float,
        volatility: float,
        spread_bps: float,
    ) -> float:
        """Pre-trade cost estimate in bps.

        Use before execution to estimate expected costs and choose algo.
        """
        if adv <= 0 or price <= 0:
            return spread_bps / 2
        participation = shares / adv
        return self.impact_model.total_impact_bps(
            participation, volatility, spread_bps
        )

    def optimal_execution_horizon(
        self,
        shares: int,
        adv: float,
        volatility: float,
        urgency: float = 0.5,
    ) -> float:
        """Estimate optimal execution time in minutes.

        Based on the Almgren-Chriss framework: balance impact (trade fast)
        against timing risk (trade slow).

        Parameters
        ----------
        shares : int
            Order size.
        adv : float
            Average daily volume.
        volatility : float
            Daily volatility (decimal).
        urgency : float
            0 = patient, 1 = urgent.

        Returns
        -------
        float
            Suggested execution time in minutes.
        """
        if adv <= 0:
            return 390.0  # Full day

        participation = shares / adv

        # Higher participation → longer horizon
        # Higher urgency → shorter horizon
        # Base: trade 5% of ADV per hour ≈ ~78 min per 1% participation
        base_minutes = participation * 390 * 10  # Minutes per unit participation
        urgency_factor = max(1.0 - urgency * 0.8, 0.2)

        horizon = base_minutes * urgency_factor
        return float(np.clip(horizon, 5.0, 390.0))
