"""Market microstructure analysis and order flow signals.

Implements core microstructure metrics:
  1. Order book imbalance (OBI) with multi-level depth weighting.
  2. Volume-weighted microprice for improved fair value estimation.
  3. VPIN (Volume-synchronized Probability of Informed Trading) for toxicity.
  4. Kyle's lambda estimation for price impact.
  5. Roll's effective spread estimator.
  6. Intraday volatility and volume seasonality (U-shape pattern).

Usage::

    obi = OrderBookImbalance(depth=5)
    imbalance = obi.compute(bid_volumes, ask_volumes)

    vpin = VPINEstimator(bucket_volume=10000, n_buckets=50)
    toxicity = vpin.compute(prices, volumes)

References:
  - Kyle (1985), "Continuous Auctions and Insider Trading"
  - Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread"
  - Easley, Lopez de Prado, O'Hara (2012), "Flow Toxicity and Liquidity"
  - Cartea, Jaimungal, Penalva (2015), "Algorithmic and HFT"
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order Book Imbalance
# ---------------------------------------------------------------------------


class WeightScheme(str, Enum):
    UNIFORM = "uniform"
    INVERSE_DISTANCE = "inverse_distance"
    EXPONENTIAL = "exponential"


@dataclass
class OrderBookImbalance:
    """Compute order book imbalance from depth data.

    Parameters
    ----------
    depth : int
        Number of price levels to use.
    scheme : WeightScheme
        How to weight levels by distance from best.
    decay : float
        Decay rate for exponential scheme.
    """

    depth: int = 5
    scheme: WeightScheme = WeightScheme.EXPONENTIAL
    decay: float = 1.0

    def _weights(self, actual_depth: int) -> np.ndarray:
        d = min(self.depth, actual_depth)
        levels = np.arange(1, d + 1, dtype=float)
        if self.scheme == WeightScheme.UNIFORM:
            return np.ones(d)
        if self.scheme == WeightScheme.INVERSE_DISTANCE:
            return 1.0 / levels
        if self.scheme == WeightScheme.EXPONENTIAL:
            return np.exp(-self.decay * (levels - 1))
        return np.ones(d)

    def compute(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> np.ndarray:
        """Compute OBI for each snapshot.

        Parameters
        ----------
        bid_volumes : np.ndarray
            (n_snapshots, depth) bid volumes.
        ask_volumes : np.ndarray
            (n_snapshots, depth) ask volumes.

        Returns
        -------
        np.ndarray
            OBI values in [-1, 1].
        """
        actual_depth = bid_volumes.shape[1] if bid_volumes.ndim == 2 else 1
        w = self._weights(actual_depth)

        if bid_volumes.ndim == 1:
            bid_volumes = bid_volumes.reshape(-1, 1)
            ask_volumes = ask_volumes.reshape(-1, 1)

        d = min(len(w), bid_volumes.shape[1])
        w = w[:d]

        wb = bid_volumes[:, :d] @ w
        wa = ask_volumes[:, :d] @ w
        denom = wb + wa
        safe_denom = np.where(denom > 0, denom, 1.0)
        return np.where(denom > 0, (wb - wa) / safe_denom, 0.0)


# ---------------------------------------------------------------------------
# Microprice
# ---------------------------------------------------------------------------


def microprice(
    bid_price: np.ndarray,
    ask_price: np.ndarray,
    bid_volume: np.ndarray,
    ask_volume: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Volume-weighted microprice (fair value estimator).

    Parameters
    ----------
    bid_price, ask_price : np.ndarray
        Best bid and ask prices.
    bid_volume, ask_volume : np.ndarray
        Volumes at best bid and ask.
    alpha : float
        Power parameter (1.0 = linear microprice).

    Returns
    -------
    np.ndarray
        Microprice estimates.
    """
    vb = np.power(np.maximum(bid_volume, 0), alpha)
    va = np.power(np.maximum(ask_volume, 0), alpha)
    denom = vb + va + 1e-12
    imb = vb / denom
    return bid_price + imb * (ask_price - bid_price)


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------


@dataclass
class VPINResult:
    """VPIN computation result."""

    vpin: np.ndarray  # VPIN at each bucket boundary
    bucket_indices: np.ndarray  # Trade indices for bucket boundaries
    buy_fractions: np.ndarray  # Fraction classified as buy per bucket


class VPINEstimator:
    """Volume-synchronized Probability of Informed Trading.

    Parameters
    ----------
    bucket_volume : float
        Volume per bucket (e.g. ADV / 50).
    n_buckets : int
        Lookback window in buckets.
    sigma_window : int
        Window for bulk classification sigma estimate.
    """

    def __init__(
        self,
        bucket_volume: float,
        n_buckets: int = 50,
        sigma_window: int = 50,
    ):
        self.bucket_volume = bucket_volume
        self.n_buckets = n_buckets
        self.sigma_window = sigma_window

    def compute(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> VPINResult:
        """Compute VPIN from trade-level data.

        Parameters
        ----------
        prices : np.ndarray
            Trade prices.
        volumes : np.ndarray
            Trade volumes.

        Returns
        -------
        VPINResult
        """
        cumvol = np.cumsum(volumes)
        total_vol = cumvol[-1]

        if total_vol < self.bucket_volume:
            return VPINResult(
                vpin=np.array([0.5]),
                bucket_indices=np.array([len(prices) - 1]),
                buy_fractions=np.array([0.5]),
            )

        bucket_edges = np.arange(
            self.bucket_volume, total_vol + 1, self.bucket_volume
        )
        bucket_idx = np.searchsorted(cumvol, bucket_edges)
        bucket_idx = np.clip(bucket_idx, 0, len(prices) - 1)

        # Price change per bucket
        bucket_prices = prices[bucket_idx]
        delta_p = np.diff(bucket_prices, prepend=bucket_prices[0])

        # Rolling sigma of price changes
        n_b = len(delta_p)
        sigma = np.full(n_b, np.nan)
        for t in range(min(self.sigma_window, n_b), n_b):
            start = max(0, t - self.sigma_window)
            sigma[t] = max(np.std(delta_p[start:t], ddof=1), 1e-10)
        # Fill early values
        first_valid = np.argmax(~np.isnan(sigma))
        sigma[:first_valid] = sigma[first_valid] if first_valid < n_b else 1e-10

        # Bulk volume classification
        z = delta_p / np.maximum(sigma, 1e-10)
        buy_frac = norm.cdf(z)
        order_imbalance = np.abs(2 * buy_frac - 1)

        # Rolling VPIN
        vpin = np.full(n_b, np.nan)
        for t in range(self.n_buckets, n_b):
            vpin[t] = np.mean(order_imbalance[t - self.n_buckets : t])
        # Fill early
        valid_start = np.argmax(~np.isnan(vpin))
        vpin[:valid_start] = vpin[valid_start] if valid_start < n_b else 0.5

        return VPINResult(
            vpin=vpin,
            bucket_indices=bucket_idx,
            buy_fractions=buy_frac,
        )


# ---------------------------------------------------------------------------
# Kyle's Lambda
# ---------------------------------------------------------------------------


@dataclass
class KyleLambdaResult:
    """Kyle's lambda estimation result."""

    lambdas: np.ndarray  # Rolling lambda estimates
    mean_lambda: float
    t_statistic: float


class KyleLambdaEstimator:
    """Estimate Kyle's lambda (price impact per unit order flow).

    Parameters
    ----------
    window : int
        Rolling OLS window.
    min_obs : int
        Minimum observations for estimation.
    """

    def __init__(self, window: int = 100, min_obs: int = 30):
        self.window = window
        self.min_obs = min_obs

    def estimate(
        self,
        price_changes: np.ndarray,
        signed_order_flow: np.ndarray,
    ) -> KyleLambdaResult:
        """Estimate rolling Kyle's lambda.

        Parameters
        ----------
        price_changes : np.ndarray
            delta_P per bar.
        signed_order_flow : np.ndarray
            Buy volume minus sell volume per bar.

        Returns
        -------
        KyleLambdaResult
        """
        n = len(price_changes)
        lambdas = np.full(n, np.nan)

        for i in range(self.min_obs, n):
            start = max(0, i - self.window)
            x = signed_order_flow[start:i]
            y = price_changes[start:i]

            x_dm = x - x.mean()
            var_x = np.dot(x_dm, x_dm)
            if var_x < 1e-12:
                continue
            lambdas[i] = np.dot(x_dm, y - y.mean()) / var_x

        valid = lambdas[~np.isnan(lambdas)]
        if len(valid) > 1:
            mean_lam = float(np.mean(valid))
            se = float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
            t_stat = mean_lam / max(se, 1e-12)
        else:
            mean_lam = 0.0
            t_stat = 0.0

        return KyleLambdaResult(
            lambdas=lambdas,
            mean_lambda=mean_lam,
            t_statistic=t_stat,
        )


# ---------------------------------------------------------------------------
# Roll's Effective Spread
# ---------------------------------------------------------------------------


def roll_effective_spread(
    price_changes: np.ndarray, min_obs: int = 100
) -> dict[str, float]:
    """Estimate effective spread using Roll's model.

    Returns
    -------
    dict
        half_spread, effective_spread, autocovariance.
    """
    n = len(price_changes)
    if n < min_obs:
        return {
            "half_spread": float("nan"),
            "effective_spread": float("nan"),
            "autocovariance": float("nan"),
        }

    dm = price_changes - price_changes.mean()
    autocov = float(np.dot(dm[:-1], dm[1:]) / (n - 1))

    if autocov >= 0:
        half_spread = 0.0
    else:
        half_spread = float(np.sqrt(-autocov))

    return {
        "half_spread": half_spread,
        "effective_spread": 2 * half_spread,
        "autocovariance": autocov,
    }


def roll_spread_rolling(
    price_changes: np.ndarray, window: int = 500
) -> np.ndarray:
    """Rolling Roll spread estimate."""
    n = len(price_changes)
    spreads = np.full(n, np.nan)
    for t in range(window, n):
        result = roll_effective_spread(
            price_changes[t - window : t], min_obs=50
        )
        spreads[t] = result["effective_spread"]
    return spreads


# ---------------------------------------------------------------------------
# Intraday Seasonality
# ---------------------------------------------------------------------------


class IntradayPattern:
    """Estimate and apply intraday volatility / volume seasonality.

    Parameters
    ----------
    n_bins : int
        Number of intraday bins (e.g. 78 for 5-min bins in 6.5h session).
    n_harmonics : int
        Fourier harmonics for smoothing.
    session_seconds : float
        Trading session length in seconds.
    """

    def __init__(
        self,
        n_bins: int = 78,
        n_harmonics: int = 3,
        session_seconds: float = 23400.0,
    ):
        self.n_bins = n_bins
        self.n_harmonics = n_harmonics
        self.session_seconds = session_seconds

    def estimate_volatility_pattern(
        self,
        timestamps: np.ndarray,
        squared_returns: np.ndarray,
    ) -> np.ndarray:
        """Estimate average volatility pattern by time-of-day.

        Parameters
        ----------
        timestamps : np.ndarray
            Seconds from market open.
        squared_returns : np.ndarray
            r_t^2 values.

        Returns
        -------
        np.ndarray
            (n_bins,) normalized pattern (mean = 1).
        """
        s = timestamps / self.session_seconds
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_idx = np.clip(np.digitize(s, bin_edges) - 1, 0, self.n_bins - 1)

        pattern = np.zeros(self.n_bins)
        counts = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = bin_idx == i
            if mask.sum() > 0:
                pattern[i] = squared_returns[mask].mean()
                counts[i] = mask.sum()

        # Fill empty bins
        mean_val = pattern[pattern > 0].mean() if (pattern > 0).any() else 1e-8
        pattern[pattern == 0] = mean_val

        # Smooth with Fourier
        return self._fourier_smooth(pattern)

    def estimate_volume_pattern(
        self,
        timestamps: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Estimate average volume pattern by time-of-day.

        Returns (n_bins,) pattern that sums to 1.0.
        """
        s = timestamps / self.session_seconds
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_idx = np.clip(np.digitize(s, bin_edges) - 1, 0, self.n_bins - 1)

        pattern = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            mask = bin_idx == i
            if mask.sum() > 0:
                pattern[i] = volumes[mask].mean()

        total = pattern.sum()
        if total > 0:
            pattern /= total
        else:
            pattern = np.ones(self.n_bins) / self.n_bins

        return pattern

    def _fourier_smooth(self, pattern: np.ndarray) -> np.ndarray:
        """Smooth pattern using truncated Fourier series."""
        n = len(pattern)
        s = np.linspace(0, 1, n, endpoint=False)

        X = [np.ones(n)]
        for p in range(1, self.n_harmonics + 1):
            X.append(np.cos(2 * np.pi * p * s))
            X.append(np.sin(2 * np.pi * p * s))
        X = np.column_stack(X)

        log_pattern = np.log(np.maximum(pattern, 1e-12))
        beta, _, _, _ = np.linalg.lstsq(X, log_pattern, rcond=None)
        smoothed = np.exp(X @ beta)

        # Normalize: mean = 1
        smoothed /= smoothed.mean()
        return smoothed
