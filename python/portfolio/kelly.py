"""Kelly criterion position sizing for portfolio optimization.

Implements:
  - ``full_kelly``: Optimal growth-rate maximizing allocation (K = Sigma^{-1} mu)
  - ``fractional_kelly``: Scaled-down allocation for practical use (fraction * K)
  - ``kelly_from_predictions``: Converts model predictions + confidence into Kelly weights
  - ``kelly_edge_sizing``: Per-asset edge-based sizing (f* = edge / variance)

The Kelly criterion maximizes long-run geometric growth by allocating
proportional to edge/variance. In practice, full Kelly is too aggressive
(~30-50% annual drawdowns), so fractional Kelly (0.25-0.5x) is standard.

References:
  - Kelly, J.L. (1956), "A New Interpretation of Information Rate"
  - Thorp, E.O. (2006), "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
  - Lopez de Prado, M. (2018), Ch. 10 — "Bet Sizing"
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def full_kelly(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Full Kelly allocation: w* = Sigma^{-1} * (mu - r_f).

    This maximizes the expected log-growth rate. WARNING: Full Kelly is
    extremely aggressive and not recommended for production use.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected excess returns per asset (annualized).
    covariance : pd.DataFrame
        Covariance matrix of asset returns (annualized).
    risk_free_rate : float
        Risk-free rate (annualized).

    Returns
    -------
    pd.Series
        Kelly optimal weights (can be > 1 or < 0).
    """
    tickers = expected_returns.index
    mu = expected_returns.values - risk_free_rate
    sigma = covariance.loc[tickers, tickers].values

    try:
        sigma_inv = np.linalg.inv(sigma)
        w = sigma_inv @ mu
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix singular; using pseudoinverse")
        sigma_inv = np.linalg.pinv(sigma)
        w = sigma_inv @ mu

    return pd.Series(w, index=tickers, name="kelly_weights")


def fractional_kelly(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    fraction: float = 0.25,
    risk_free_rate: float = 0.0,
    max_weight: Optional[float] = None,
    long_only: bool = True,
) -> pd.Series:
    """Fractional Kelly allocation: w = fraction * Sigma^{-1} * (mu - r_f).

    Standard practice uses 0.25x Kelly (quarter Kelly) for robustness
    to estimation error in expected returns.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected excess returns per asset (annualized).
    covariance : pd.DataFrame
        Covariance matrix (annualized).
    fraction : float
        Kelly fraction (0.25 = quarter Kelly, 0.5 = half Kelly).
    risk_free_rate : float
        Risk-free rate.
    max_weight : float, optional
        Cap on individual weight (e.g. 0.25).
    long_only : bool
        If True, clip negative weights to 0 and renormalize.

    Returns
    -------
    pd.Series
        Scaled Kelly weights summing to <= 1.
    """
    w = full_kelly(expected_returns, covariance, risk_free_rate)
    w = w * fraction

    if long_only:
        w = w.clip(lower=0.0)

    if max_weight is not None:
        w = w.clip(upper=max_weight)

    # Normalize to sum to 1 (or less if all weights are small)
    total = w.sum()
    if total > 1.0:
        w = w / total
    elif total <= 0:
        # All weights are zero or negative — fall back to equal weight
        w = pd.Series(1.0 / len(w), index=w.index, name="kelly_weights")

    w.name = "kelly_weights"
    return w


def kelly_from_predictions(
    predictions: pd.Series,
    prediction_std: pd.Series,
    covariance: pd.DataFrame,
    fraction: float = 0.25,
    confidence_scaling: bool = True,
    max_weight: Optional[float] = None,
) -> pd.Series:
    """Convert model predictions + uncertainty into Kelly weights.

    Uses predictions as expected returns and optionally scales by
    prediction confidence (1/std) to reduce allocation to uncertain bets.

    Parameters
    ----------
    predictions : pd.Series
        Model predicted returns per asset.
    prediction_std : pd.Series
        Standard deviation of prediction (from conformal intervals or ensemble).
    covariance : pd.DataFrame
        Asset covariance matrix.
    fraction : float
        Kelly fraction.
    confidence_scaling : bool
        If True, scale predictions by confidence (1/std).
    max_weight : float, optional
        Cap per asset.

    Returns
    -------
    pd.Series
        Kelly-sized portfolio weights.
    """
    tickers = predictions.index

    if confidence_scaling:
        # Higher confidence (lower std) → bigger bet
        confidence = 1.0 / prediction_std.clip(lower=1e-6)
        confidence = confidence / confidence.mean()  # normalize to mean=1
        mu = predictions * confidence
    else:
        mu = predictions

    return fractional_kelly(
        expected_returns=mu,
        covariance=covariance,
        fraction=fraction,
        max_weight=max_weight,
        long_only=True,
    )


def kelly_edge_sizing(
    edges: pd.Series,
    variances: pd.Series,
    fraction: float = 0.25,
    max_weight: float = 0.20,
) -> pd.Series:
    """Simple per-asset Kelly sizing: f* = fraction * edge / variance.

    For independent bets, the Kelly fraction for each asset is
    f_i = edge_i / variance_i. This is useful when you don't have
    a full covariance matrix.

    Parameters
    ----------
    edges : pd.Series
        Expected edge (alpha) per asset.
    variances : pd.Series
        Return variance per asset.
    fraction : float
        Kelly fraction.
    max_weight : float
        Maximum allocation per asset.

    Returns
    -------
    pd.Series
        Position sizes (fraction of portfolio).
    """
    safe_var = variances.clip(lower=1e-8)
    raw_kelly = edges / safe_var
    w = raw_kelly * fraction
    w = w.clip(lower=0.0, upper=max_weight)

    total = w.sum()
    if total > 1.0:
        w = w / total

    w.name = "kelly_edge_weights"
    return w


def kelly_growth_rate(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> float:
    """Compute the expected log-growth rate for given weights.

    g(w) = w'mu - 0.5 * w'Sigma w

    This is the objective that Kelly criterion maximizes.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.
    expected_returns : pd.Series
        Expected returns.
    covariance : pd.DataFrame
        Covariance matrix.

    Returns
    -------
    float
        Expected log-growth rate.
    """
    tickers = weights.index
    w = weights.values
    mu = expected_returns.reindex(tickers, fill_value=0.0).values
    sigma = covariance.loc[tickers, tickers].values

    growth = w @ mu - 0.5 * w @ sigma @ w
    return float(growth)
