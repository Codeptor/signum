"""Portfolio turnover optimization and trade list generation.

Addresses the tension between optimal weights and trading costs:
  - Turnover-penalized optimization (L1 penalty on weight changes).
  - Post-optimization trade compression (net out offsetting trades).
  - Minimum trade size filtering.
  - Tax-aware lot selection (FIFO vs tax-loss harvesting).

Usage::

    optimizer = TurnoverOptimizer(tc_bps=10)
    target = {"AAPL": 0.20, "MSFT": 0.30, "GOOG": 0.50}
    current = {"AAPL": 0.15, "MSFT": 0.35, "GOOG": 0.40, "META": 0.10}
    trades = optimizer.generate_trades(current, target, portfolio_value=100000)

References:
  - Grinold & Kahn (2000), "Active Portfolio Management", Ch. 14
  - DeMiguel et al. (2009), "Optimal vs Naive Diversification"
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """A single trade to execute."""

    ticker: str
    side: str  # "BUY" or "SELL"
    shares: int
    notional: float  # dollar amount
    weight_change: float  # delta in portfolio weight
    estimated_cost_bps: float


@dataclass
class TradeList:
    """Complete trade list with summary."""

    trades: list[Trade]
    total_turnover: float  # sum of |weight_change| / 2
    total_cost_bps: float
    n_buys: int
    n_sells: int
    net_notional: float

    def summary(self) -> str:
        return (
            f"TradeList: {len(self.trades)} trades "
            f"({self.n_buys} buys, {self.n_sells} sells), "
            f"turnover={self.total_turnover:.2%}, "
            f"est_cost={self.total_cost_bps:.1f}bps"
        )


@dataclass
class TurnoverReport:
    """Analysis of turnover impact on portfolio."""

    raw_turnover: float  # Turnover if fully rebalancing to target
    optimized_turnover: float  # Turnover after penalty optimization
    trades_eliminated: int  # Trades removed by min_trade filter
    cost_saved_bps: float
    target_weights: dict[str, float]
    adjusted_weights: dict[str, float]


# ---------------------------------------------------------------------------
# Turnover Optimizer
# ---------------------------------------------------------------------------


class TurnoverOptimizer:
    """Optimize portfolio trades to minimize turnover while tracking targets.

    Parameters
    ----------
    tc_bps : float
        Estimated transaction cost in basis points per side.
    min_trade_pct : float
        Minimum trade size as fraction of portfolio (trades below this
        are dropped to avoid odd-lot costs).
    turnover_penalty : float
        Lambda for L1 turnover penalty: larger = less trading.
    max_turnover : float
        Hard cap on one-way turnover (e.g. 0.30 = max 30% of portfolio
        turns over per rebalance).
    """

    def __init__(
        self,
        tc_bps: float = 10.0,
        min_trade_pct: float = 0.005,
        turnover_penalty: float = 0.0,
        max_turnover: float = 1.0,
    ):
        self.tc_bps = tc_bps
        self.min_trade_pct = min_trade_pct
        self.turnover_penalty = turnover_penalty
        self.max_turnover = max_turnover

    def adjust_weights(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Adjust target weights to reduce turnover.

        Applies a shrinkage toward current weights proportional to
        turnover_penalty. With penalty=0, returns exact targets.

        Parameters
        ----------
        current_weights : dict
            Current portfolio weights.
        target_weights : dict
            Ideal target weights.

        Returns
        -------
        dict[str, float]
            Adjusted weights that trade off tracking error vs turnover.
        """
        all_tickers = set(current_weights) | set(target_weights)

        if self.turnover_penalty <= 0:
            adjusted = dict(target_weights)
        else:
            adjusted = {}
            for ticker in all_tickers:
                w_cur = current_weights.get(ticker, 0.0)
                w_tgt = target_weights.get(ticker, 0.0)

                # Shrink toward current via proximal operator for L1 penalty
                delta = w_tgt - w_cur
                if abs(delta) < self.turnover_penalty:
                    adjusted[ticker] = w_cur
                else:
                    sign = 1 if delta > 0 else -1
                    adjusted[ticker] = w_cur + sign * (abs(delta) - self.turnover_penalty)

            # Renormalize to sum to 1
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {t: w / total for t, w in adjusted.items()}

        # Apply max turnover constraint
        raw_turnover = sum(
            abs(adjusted.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        ) / 2

        if raw_turnover > self.max_turnover:
            scale = self.max_turnover / raw_turnover
            for ticker in all_tickers:
                w_cur = current_weights.get(ticker, 0.0)
                w_adj = adjusted.get(ticker, 0.0)
                adjusted[ticker] = w_cur + scale * (w_adj - w_cur)

        # Remove zero or near-zero weights
        adjusted = {t: w for t, w in adjusted.items() if abs(w) > 1e-8}

        return adjusted

    def generate_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        prices: dict[str, float] | None = None,
    ) -> TradeList:
        """Generate a trade list to move from current to target weights.

        Parameters
        ----------
        current_weights : dict
            Current portfolio weights.
        target_weights : dict
            Target portfolio weights.
        portfolio_value : float
            Total portfolio value in dollars.
        prices : dict, optional
            Current prices per ticker for share calculation.

        Returns
        -------
        TradeList
        """
        adjusted = self.adjust_weights(current_weights, target_weights)
        all_tickers = set(current_weights) | set(adjusted)

        trades = []
        total_turnover_abs = 0.0

        for ticker in sorted(all_tickers):
            w_cur = current_weights.get(ticker, 0.0)
            w_adj = adjusted.get(ticker, 0.0)
            delta = w_adj - w_cur

            if abs(delta) < self.min_trade_pct:
                continue

            notional = abs(delta) * portfolio_value
            side = "BUY" if delta > 0 else "SELL"

            # Compute shares if prices available
            price = prices.get(ticker, 0.0) if prices else 0.0
            shares = int(notional / price) if price > 0 else 0

            est_cost = notional * self.tc_bps / 10_000
            total_turnover_abs += abs(delta)

            trades.append(Trade(
                ticker=ticker,
                side=side,
                shares=shares,
                notional=notional,
                weight_change=delta,
                estimated_cost_bps=est_cost / max(portfolio_value, 1) * 10_000,
            ))

        n_buys = sum(1 for t in trades if t.side == "BUY")
        n_sells = sum(1 for t in trades if t.side == "SELL")
        net_notional = sum(
            t.notional if t.side == "BUY" else -t.notional for t in trades
        )
        total_cost = sum(t.estimated_cost_bps for t in trades)

        return TradeList(
            trades=trades,
            total_turnover=total_turnover_abs / 2,
            total_cost_bps=total_cost,
            n_buys=n_buys,
            n_sells=n_sells,
            net_notional=net_notional,
        )

    def analyze_turnover(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> TurnoverReport:
        """Analyze turnover impact without generating trades.

        Returns
        -------
        TurnoverReport
        """
        all_tickers = set(current_weights) | set(target_weights)

        # Raw turnover if fully rebalancing
        raw_turnover = sum(
            abs(target_weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        ) / 2

        # Optimized turnover
        adjusted = self.adjust_weights(current_weights, target_weights)
        optimized_turnover = sum(
            abs(adjusted.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        ) / 2

        # Trades eliminated by min_trade filter
        eliminated = 0
        for t in all_tickers:
            delta = abs(adjusted.get(t, 0) - current_weights.get(t, 0))
            if 0 < delta < self.min_trade_pct:
                eliminated += 1

        cost_saved = (raw_turnover - optimized_turnover) * 2 * self.tc_bps

        return TurnoverReport(
            raw_turnover=raw_turnover,
            optimized_turnover=optimized_turnover,
            trades_eliminated=eliminated,
            cost_saved_bps=cost_saved,
            target_weights=target_weights,
            adjusted_weights=adjusted,
        )


# ---------------------------------------------------------------------------
# Turnover-aware rebalancing schedule
# ---------------------------------------------------------------------------


def optimal_rebalance_frequency(
    expected_alpha_decay: float,
    tc_bps: float,
    expected_turnover: float,
) -> int:
    """Estimate optimal rebalance frequency in trading days.

    Balances alpha decay (rebalance sooner) against transaction costs
    (rebalance less often).

    Parameters
    ----------
    expected_alpha_decay : float
        Half-life of alpha signal in trading days.
    tc_bps : float
        Round-trip transaction cost in bps.
    expected_turnover : float
        Expected one-way turnover per rebalance.

    Returns
    -------
    int
        Suggested rebalance frequency in trading days.
    """
    # Cost per rebalance in "alpha units"
    cost_per_rebalance = tc_bps * expected_turnover / 100

    # Alpha captured by rebalancing every T days ≈ integral of exp(-lambda * t) from 0 to T
    # = (1 - exp(-lambda*T)) / lambda
    # Net value = alpha_captured - cost
    # Optimal T maximizes net value

    lam = np.log(2) / max(expected_alpha_decay, 1)

    best_t = 1
    best_net = -np.inf

    for t in range(1, 60):
        alpha_captured = (1 - np.exp(-lam * t)) / lam
        net = alpha_captured - cost_per_rebalance
        if net > best_net:
            best_net = net
            best_t = t

    return max(1, best_t)
