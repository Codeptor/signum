"""Market regime detection and state classification.

Implements multiple approaches to identify market regimes:
  1. Hidden Markov Model (Gaussian emissions) for return-based regime detection.
  2. Volatility regime classification via rolling realized vol + percentile thresholds.
  3. Structural break detection with online CUSUM.
  4. Correlation regime detection via rolling eigenvalue analysis.

Usage::

    detector = RegimeDetector(n_regimes=3)
    regimes = detector.fit_predict(returns)
    current = detector.current_regime(returns)

References:
  - Hamilton (1989), "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
  - Ang & Bekaert (2002), "Regime Switches in Interest Rates"
  - Page (1954), "Continuous Inspection Schemes" (CUSUM)
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class MarketRegime(IntEnum):
    """Named market regimes (ordered by volatility)."""

    LOW_VOL = 0
    NORMAL = 1
    HIGH_VOL = 2
    CRISIS = 3


@dataclass
class RegimeState:
    """Current regime classification."""

    regime: int
    regime_name: str
    confidence: float  # Probability of being in this regime
    regime_probs: np.ndarray  # Full probability vector
    duration: int  # Days in current regime
    vol_percentile: float  # Current vol vs history


@dataclass
class HMMResult:
    """Result from Hidden Markov Model fitting."""

    n_regimes: int
    means: np.ndarray  # Emission means per regime
    stds: np.ndarray  # Emission stds per regime
    transition_matrix: np.ndarray  # n_regimes x n_regimes
    stationary_probs: np.ndarray  # Long-run regime probabilities
    regime_sequence: np.ndarray  # Viterbi path
    regime_probs: np.ndarray  # T x n_regimes smoothed probabilities
    log_likelihood: float
    n_iterations: int


@dataclass
class BreakpointResult:
    """Structural break detection result."""

    breakpoints: list[int]  # Indices where breaks detected
    cusum_values: np.ndarray
    threshold: float
    n_breaks: int


@dataclass
class CorrelationRegimeResult:
    """Correlation regime analysis result."""

    absorption_ratio: np.ndarray  # Time series of absorption ratio
    eigenvalue_ratio: np.ndarray  # Largest / sum of eigenvalues
    regime_indicator: np.ndarray  # 0=normal, 1=concentrated
    current_absorption: float
    is_crisis: bool


# ---------------------------------------------------------------------------
# Gaussian HMM (from scratch)
# ---------------------------------------------------------------------------


class GaussianHMM:
    """Hidden Markov Model with Gaussian emissions.

    Implements Baum-Welch (EM) for parameter estimation and Viterbi
    for most-likely state sequence decoding.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold on log-likelihood change.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol

        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.transition_: np.ndarray | None = None
        self.initial_: np.ndarray | None = None

    def _init_params(self, y: np.ndarray) -> None:
        """Initialize via quantile-based heuristic."""
        k = self.n_regimes
        # Sort observations and split into k groups
        sorted_y = np.sort(y)
        n = len(sorted_y)
        self.means_ = np.array([
            sorted_y[int(i * n / k) : int((i + 1) * n / k)].mean()
            for i in range(k)
        ])
        self.stds_ = np.array([
            max(sorted_y[int(i * n / k) : int((i + 1) * n / k)].std(), 1e-6)
            for i in range(k)
        ])
        # Sticky transition matrix (high self-transition probability)
        self.transition_ = np.full((k, k), 0.05 / (k - 1))
        np.fill_diagonal(self.transition_, 0.95)
        self.initial_ = np.ones(k) / k

    def _emission_probs(self, y: np.ndarray) -> np.ndarray:
        """Compute emission probabilities: T x K."""
        T = len(y)
        K = self.n_regimes
        probs = np.zeros((T, K))
        for k in range(K):
            probs[:, k] = sp_stats.norm.pdf(y, self.means_[k], self.stds_[k])
        return np.clip(probs, 1e-300, None)

    def _forward(self, emission: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (alpha, scaling_factors)."""
        T, K = emission.shape
        alpha = np.zeros((T, K))
        c = np.zeros(T)

        alpha[0] = self.initial_ * emission[0]
        c[0] = alpha[0].sum()
        alpha[0] /= max(c[0], 1e-300)

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transition_) * emission[t]
            c[t] = alpha[t].sum()
            alpha[t] /= max(c[t], 1e-300)

        return alpha, c

    def _backward(
        self, emission: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """Backward pass."""
        T, K = emission.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.transition_ @ (emission[t + 1] * beta[t + 1])
            beta[t] /= max(c[t + 1], 1e-300)

        return beta

    def fit(self, y: np.ndarray) -> HMMResult:
        """Fit HMM via Baum-Welch (EM) algorithm."""
        T = len(y)
        K = self.n_regimes

        if T < K * 5:
            logger.warning("Very short series for HMM fitting")

        self._init_params(y)
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            emission = self._emission_probs(y)

            # E-step
            alpha, c = self._forward(emission)
            beta = self._backward(emission, c)

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-300)

            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                xi_t = np.outer(alpha[t], emission[t + 1] * beta[t + 1])
                xi_t *= self.transition_
                xi_total = xi_t.sum()
                if xi_total > 1e-300:
                    xi[t] = xi_t / xi_total

            # Log-likelihood
            ll = np.sum(np.log(np.clip(c, 1e-300, None)))

            if abs(ll - prev_ll) < self.tol:
                logger.info(f"HMM converged after {iteration + 1} iterations")
                break
            prev_ll = ll

            # M-step
            self.initial_ = gamma[0]

            for k in range(K):
                gamma_k = gamma[:, k]
                total = gamma_k.sum()
                if total > 1e-8:
                    self.means_[k] = (gamma_k * y).sum() / total
                    diff = y - self.means_[k]
                    self.stds_[k] = np.sqrt(
                        max((gamma_k * diff**2).sum() / total, 1e-8)
                    )

            xi_sum = xi.sum(axis=0)
            for i in range(K):
                row_sum = xi_sum[i].sum()
                if row_sum > 1e-8:
                    self.transition_[i] = xi_sum[i] / row_sum

        # Viterbi decoding
        viterbi_path = self._viterbi(y)

        # Sort regimes by volatility (low → high)
        order = np.argsort(self.stds_)
        self.means_ = self.means_[order]
        self.stds_ = self.stds_[order]
        self.transition_ = self.transition_[order][:, order]
        self.initial_ = self.initial_[order]
        gamma = gamma[:, order]

        inv_order = np.zeros(K, dtype=int)
        for new_idx, old_idx in enumerate(order):
            inv_order[old_idx] = new_idx
        viterbi_path = inv_order[viterbi_path]

        # Stationary distribution: left eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.abs(eigenvectors[:, idx])
        stationary /= stationary.sum()

        return HMMResult(
            n_regimes=K,
            means=self.means_.copy(),
            stds=self.stds_.copy(),
            transition_matrix=self.transition_.copy(),
            stationary_probs=stationary,
            regime_sequence=viterbi_path,
            regime_probs=gamma,
            log_likelihood=prev_ll,
            n_iterations=iteration + 1,
        )

    def _viterbi(self, y: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most-likely state sequence."""
        T = len(y)
        K = self.n_regimes
        emission = self._emission_probs(y)

        log_A = np.log(np.clip(self.transition_, 1e-300, None))
        log_pi = np.log(np.clip(self.initial_, 1e-300, None))
        log_B = np.log(emission)

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(K):
                scores = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + log_B[t, j]

        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])

        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path


# ---------------------------------------------------------------------------
# Volatility Regime Classification
# ---------------------------------------------------------------------------


class VolatilityRegimeClassifier:
    """Classify regimes based on realized volatility percentiles.

    Parameters
    ----------
    lookback : int
        Window for realized vol calculation.
    history_window : int
        Window for percentile ranking.
    thresholds : tuple
        Percentile thresholds for (low, normal, high, crisis).
    """

    def __init__(
        self,
        lookback: int = 21,
        history_window: int = 252,
        thresholds: tuple[float, ...] = (25, 50, 85),
    ):
        self.lookback = lookback
        self.history_window = history_window
        self.thresholds = thresholds

    def classify(self, returns: np.ndarray) -> np.ndarray:
        """Classify each point into a volatility regime.

        Returns array of MarketRegime values.
        """
        n = len(returns)
        regimes = np.full(n, MarketRegime.NORMAL, dtype=int)

        if n < self.lookback + 1:
            return regimes

        # Rolling realized volatility (annualized)
        rvol = np.zeros(n)
        for t in range(self.lookback, n):
            window = returns[t - self.lookback : t]
            rvol[t] = np.std(window) * np.sqrt(252)

        # Classify based on percentile rank within history
        for t in range(self.lookback + self.history_window, n):
            hist_start = max(self.lookback, t - self.history_window)
            hist = rvol[hist_start:t]
            if len(hist) == 0:
                continue
            pctl = sp_stats.percentileofscore(hist, rvol[t])

            if pctl < self.thresholds[0]:
                regimes[t] = MarketRegime.LOW_VOL
            elif pctl < self.thresholds[1]:
                regimes[t] = MarketRegime.NORMAL
            elif pctl < self.thresholds[2]:
                regimes[t] = MarketRegime.HIGH_VOL
            else:
                regimes[t] = MarketRegime.CRISIS

        return regimes

    def current_state(self, returns: np.ndarray) -> RegimeState:
        """Get current regime state."""
        regimes = self.classify(returns)
        current = regimes[-1]

        # Compute duration (days in current regime)
        duration = 1
        for i in range(len(regimes) - 2, -1, -1):
            if regimes[i] == current:
                duration += 1
            else:
                break

        # Vol percentile
        if len(returns) >= self.lookback:
            rvol = np.std(returns[-self.lookback :]) * np.sqrt(252)
            hist_start = max(0, len(returns) - self.history_window)
            hist_vols = []
            for t in range(max(self.lookback, hist_start), len(returns)):
                w = returns[t - self.lookback : t]
                hist_vols.append(np.std(w) * np.sqrt(252))
            pctl = sp_stats.percentileofscore(hist_vols, rvol) if hist_vols else 50.0
        else:
            pctl = 50.0

        names = {
            MarketRegime.LOW_VOL: "Low Volatility",
            MarketRegime.NORMAL: "Normal",
            MarketRegime.HIGH_VOL: "High Volatility",
            MarketRegime.CRISIS: "Crisis",
        }

        # Compute regime probabilities as soft assignment from vol percentile
        regime_probs = np.zeros(4)
        regime_probs[current] = 0.7
        # Spread remaining probability to neighbors
        if current > 0:
            regime_probs[current - 1] = 0.15
        if current < 3:
            regime_probs[current + 1] = 0.15
        regime_probs /= regime_probs.sum()

        return RegimeState(
            regime=int(current),
            regime_name=names.get(MarketRegime(current), "Unknown"),
            confidence=float(regime_probs[current]),
            regime_probs=regime_probs,
            duration=duration,
            vol_percentile=float(pctl),
        )


# ---------------------------------------------------------------------------
# CUSUM Structural Break Detection
# ---------------------------------------------------------------------------


class CUSUMDetector:
    """Online CUSUM detector for structural breaks.

    Detects shifts in mean of a time series using Page's CUSUM test.

    Parameters
    ----------
    threshold : float
        CUSUM threshold for declaring a break (in standard deviations).
    drift : float
        Allowance parameter (typically 0.5 * expected shift size / sigma).
    min_spacing : int
        Minimum observations between declared breakpoints.
    """

    def __init__(
        self,
        threshold: float = 4.0,
        drift: float = 0.5,
        min_spacing: int = 20,
    ):
        self.threshold = threshold
        self.drift = drift
        self.min_spacing = min_spacing

    def detect(self, y: np.ndarray) -> BreakpointResult:
        """Detect structural breaks in the time series."""
        n = len(y)
        if n < 10:
            return BreakpointResult(
                breakpoints=[], cusum_values=np.zeros(n),
                threshold=self.threshold, n_breaks=0,
            )

        # Standardize
        mu = np.mean(y)
        sigma = max(np.std(y), 1e-10)
        z = (y - mu) / sigma

        # Two-sided CUSUM
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        cusum = np.zeros(n)

        breakpoints = []
        last_break = -self.min_spacing

        for t in range(1, n):
            s_pos[t] = max(0, s_pos[t - 1] + z[t] - self.drift)
            s_neg[t] = max(0, s_neg[t - 1] - z[t] - self.drift)
            cusum[t] = max(s_pos[t], s_neg[t])

            if cusum[t] > self.threshold and (t - last_break) >= self.min_spacing:
                breakpoints.append(t)
                last_break = t
                # Reset after detection
                s_pos[t] = 0
                s_neg[t] = 0

        return BreakpointResult(
            breakpoints=breakpoints,
            cusum_values=cusum,
            threshold=self.threshold,
            n_breaks=len(breakpoints),
        )


# ---------------------------------------------------------------------------
# Correlation Regime Detection
# ---------------------------------------------------------------------------


class CorrelationRegimeDetector:
    """Detect correlation regime changes via eigenvalue analysis.

    When the absorption ratio (fraction of variance explained by top
    eigenvalues) rises sharply, it indicates correlation concentration
    — a sign of crisis / risk-off behavior.

    Parameters
    ----------
    lookback : int
        Rolling window for correlation estimation.
    n_top_factors : int
        Number of top eigenvalues to track.
    crisis_threshold : float
        Absorption ratio above this → crisis indicator.
    """

    def __init__(
        self,
        lookback: int = 63,
        n_top_factors: int = 1,
        crisis_threshold: float = 0.60,
    ):
        self.lookback = lookback
        self.n_top_factors = n_top_factors
        self.crisis_threshold = crisis_threshold

    def analyze(self, returns: np.ndarray) -> CorrelationRegimeResult:
        """Analyze correlation regimes from a returns matrix (T x N).

        Parameters
        ----------
        returns : np.ndarray
            T x N matrix of asset returns.

        Returns
        -------
        CorrelationRegimeResult
        """
        T, N = returns.shape
        if N < 2:
            return CorrelationRegimeResult(
                absorption_ratio=np.zeros(T),
                eigenvalue_ratio=np.zeros(T),
                regime_indicator=np.zeros(T, dtype=int),
                current_absorption=0.0,
                is_crisis=False,
            )

        n_factors = min(self.n_top_factors, N)
        absorption = np.full(T, np.nan)
        eig_ratio = np.full(T, np.nan)

        for t in range(self.lookback, T):
            window = returns[t - self.lookback : t]
            # Correlation matrix
            corr = np.corrcoef(window.T)
            # Handle NaN
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)

            eigenvalues = np.linalg.eigvalsh(corr)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
            eigenvalues = np.clip(eigenvalues, 0, None)

            total_var = eigenvalues.sum()
            if total_var > 1e-10:
                absorption[t] = eigenvalues[:n_factors].sum() / total_var
                eig_ratio[t] = eigenvalues[0] / total_var
            else:
                absorption[t] = 0.0
                eig_ratio[t] = 0.0

        # Fill NaN at start
        first_valid = self.lookback
        absorption[:first_valid] = absorption[first_valid] if first_valid < T else 0.0
        eig_ratio[:first_valid] = eig_ratio[first_valid] if first_valid < T else 0.0

        # Crisis indicator
        regime_indicator = (absorption > self.crisis_threshold).astype(int)

        current_abs = float(absorption[-1]) if T > 0 else 0.0

        return CorrelationRegimeResult(
            absorption_ratio=absorption,
            eigenvalue_ratio=eig_ratio,
            regime_indicator=regime_indicator,
            current_absorption=current_abs,
            is_crisis=current_abs > self.crisis_threshold,
        )


# ---------------------------------------------------------------------------
# Unified Regime Detector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Unified regime detection combining multiple signals.

    Parameters
    ----------
    n_regimes : int
        Number of HMM regimes.
    vol_lookback : int
        Window for volatility-based classification.
    cusum_threshold : float
        CUSUM threshold for structural breaks.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        vol_lookback: int = 21,
        cusum_threshold: float = 4.0,
    ):
        self.hmm = GaussianHMM(n_regimes=n_regimes)
        self.vol_classifier = VolatilityRegimeClassifier(lookback=vol_lookback)
        self.cusum = CUSUMDetector(threshold=cusum_threshold)
        self.n_regimes = n_regimes

    def fit_predict(self, returns: np.ndarray) -> HMMResult:
        """Fit HMM and return full result with regime sequence."""
        return self.hmm.fit(returns)

    def detect_breaks(self, returns: np.ndarray) -> BreakpointResult:
        """Detect structural breaks."""
        return self.cusum.detect(returns)

    def classify_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Classify into volatility regimes."""
        return self.vol_classifier.classify(returns)

    def current_regime(self, returns: np.ndarray) -> RegimeState:
        """Get current regime state from volatility classifier."""
        return self.vol_classifier.current_state(returns)
