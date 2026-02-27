"""Signal combination and alpha blending with adaptive weighting.

Combines multiple alpha signals into a single composite signal using
sophisticated weighting methods that adapt to signal quality, market
regime, and inter-signal crowding.

Weighting methods:
  - **Inverse-variance**: Weight proportional to 1/var(IC), giving
    more weight to signals with stable predictive power.
  - **Bayesian model averaging (BMA)**: Posterior model weights from
    exponentially-decayed marginal likelihoods.
  - **Mean-variance optimization**: Markowitz-style optimization over
    the signal-level IC stream.

Key features:
  - Exponential decay tracking of signal IC, Sharpe, and turnover
  - Signal crowding detection via rolling correlation matrix
  - Alpha decay analysis (half-life estimation)
  - Regime-conditional allocation via existing HMM detector

References:
  - Bates & Granger (1969), "Combination of Forecasts"
  - Kakushadze (2016), "101 Formulaic Alphas"
  - Lopez de Prado (2018), Ch. 10 — "Bet Sizing"
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class CombineMethod(str, Enum):
    INVERSE_VARIANCE = "inverse_variance"
    BAYESIAN_MODEL_AVG = "bayesian_model_avg"
    MEAN_VARIANCE_OPT = "mean_variance_opt"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class SignalMetadata:
    """Static metadata for a registered alpha signal."""

    name: str
    description: str = ""
    lookback_days: int = 20
    expected_horizon: int = 5
    category: str = "momentum"


@dataclass
class SignalSnapshot:
    """Point-in-time performance snapshot for one signal."""

    name: str
    timestamp: pd.Timestamp
    rolling_ic: float = 0.0
    rolling_ic_std: float = 1.0
    rolling_sharpe: float = 0.0
    rolling_turnover: float = 0.0
    weight: float = 0.0

    @property
    def ic_variance(self) -> float:
        return max(self.rolling_ic_std**2, 1e-8)


@dataclass
class CrowdingReport:
    """Results of signal crowding analysis."""

    correlation_matrix: pd.DataFrame
    mean_pairwise_corr: float
    max_pairwise_corr: float
    crowded_pairs: list[tuple[str, str, float]]
    is_crowded: bool
    eigenvalue_ratio: float


@dataclass
class DecayProfile:
    """Alpha decay analysis for a single signal."""

    signal_name: str
    autocorrelations: np.ndarray
    half_life_days: float
    decay_rate: float
    is_fast_decay: bool
    lag_ics: np.ndarray


# ---------------------------------------------------------------------------
# Signal Performance Tracker
# ---------------------------------------------------------------------------


class SignalPerformanceTracker:
    """Track individual signal performance with exponential decay.

    Maintains rolling IC observations per signal and computes EW-weighted
    statistics (mean IC, IC variance, Sharpe, turnover) for the combiner.

    Parameters
    ----------
    halflife : int
        Exponential decay half-life in trading days (default 63 = 1 quarter).
    min_observations : int
        Minimum IC observations before signal is eligible for weighting.
    """

    def __init__(self, halflife: int = 63, min_observations: int = 20):
        self.halflife = halflife
        self.min_observations = min_observations
        self._ic_history: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        self._rank_history: dict[str, list[tuple[pd.Timestamp, np.ndarray]]] = {}

    def update(
        self,
        signal_name: str,
        date: pd.Timestamp,
        signal_values: np.ndarray,
        realized_returns: np.ndarray,
    ) -> None:
        """Record one date's IC observation."""
        mask = ~(np.isnan(signal_values) | np.isnan(realized_returns))
        if mask.sum() < 10:
            return

        rho, _ = spearmanr(signal_values[mask], realized_returns[mask])
        ic = float(rho) if not np.isnan(rho) else 0.0
        self._ic_history.setdefault(signal_name, []).append((date, ic))

        ranks = np.argsort(np.argsort(-signal_values)).astype(float)
        self._rank_history.setdefault(signal_name, []).append((date, ranks))

    def get_snapshot(self, signal_name: str, as_of: pd.Timestamp) -> SignalSnapshot:
        """Compute exponentially-weighted performance snapshot."""
        history = self._ic_history.get(signal_name, [])
        if len(history) < 2:
            return SignalSnapshot(name=signal_name, timestamp=as_of)

        dates, ics = zip(*history)
        ic_series = pd.Series(ics, index=pd.DatetimeIndex(dates))

        ewm = ic_series.ewm(halflife=self.halflife, min_periods=1)
        ew_mean = float(ewm.mean().iloc[-1])
        ew_std = max(float(ewm.std().iloc[-1]), 1e-6)

        ic_sharpe = (ew_mean / ew_std) * np.sqrt(252)

        # Signal turnover
        rank_hist = self._rank_history.get(signal_name, [])
        turnover = 0.0
        if len(rank_hist) >= 2:
            _, prev = rank_hist[-2]
            _, curr = rank_hist[-1]
            rho, _ = spearmanr(prev, curr)
            turnover = 1.0 - (float(rho) if not np.isnan(rho) else 0.0)

        return SignalSnapshot(
            name=signal_name,
            timestamp=as_of,
            rolling_ic=ew_mean,
            rolling_ic_std=ew_std,
            rolling_sharpe=ic_sharpe,
            rolling_turnover=turnover,
        )

    @property
    def signal_names(self) -> list[str]:
        return list(self._ic_history.keys())


# ---------------------------------------------------------------------------
# Signal Crowding Detector
# ---------------------------------------------------------------------------


class SignalCrowdingDetector:
    """Detect excessive correlation between alpha signals.

    Parameters
    ----------
    correlation_threshold : float
        Pairwise correlation above which signals are crowded (default 0.70).
    lookback_days : int
        Rolling window for correlation computation.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.70,
        lookback_days: int = 63,
    ):
        self.correlation_threshold = correlation_threshold
        self.lookback_days = lookback_days

    def analyze(self, signal_matrix: pd.DataFrame) -> CrowdingReport:
        """Analyze signal crowding from daily signal values.

        Parameters
        ----------
        signal_matrix : pd.DataFrame
            DatetimeIndex, columns = signal names, values = signal z-scores.

        Returns
        -------
        CrowdingReport
        """
        recent = signal_matrix.tail(self.lookback_days).dropna(axis=1, how="all")

        if recent.shape[1] < 2:
            return CrowdingReport(
                correlation_matrix=pd.DataFrame(),
                mean_pairwise_corr=0.0,
                max_pairwise_corr=0.0,
                crowded_pairs=[],
                is_crowded=False,
                eigenvalue_ratio=1.0,
            )

        corr = recent.corr(method="spearman")
        n = len(corr)
        mask = ~np.eye(n, dtype=bool)
        off_diag = corr.values[mask]

        mean_corr = float(np.nanmean(off_diag))
        max_corr = float(np.nanmax(np.abs(off_diag)))

        crowded_pairs = []
        cols = list(corr.columns)
        for i in range(n):
            for j in range(i + 1, n):
                rho = abs(corr.iloc[i, j])
                if rho > self.correlation_threshold:
                    crowded_pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))

        eigenvalues = np.linalg.eigvalsh(corr.values)
        eigenvalues = np.sort(eigenvalues)[::-1]
        ev_ratio = (
            float(eigenvalues[0] / max(eigenvalues[1], 1e-8))
            if len(eigenvalues) > 1
            else 1.0
        )

        return CrowdingReport(
            correlation_matrix=corr,
            mean_pairwise_corr=mean_corr,
            max_pairwise_corr=max_corr,
            crowded_pairs=crowded_pairs,
            is_crowded=len(crowded_pairs) > 0,
            eigenvalue_ratio=ev_ratio,
        )


# ---------------------------------------------------------------------------
# Signal Decay Analyzer
# ---------------------------------------------------------------------------


class SignalDecayAnalyzer:
    """Measure how fast alpha decays after signal generation.

    Parameters
    ----------
    max_lag : int
        Maximum forward lag in trading days (default 20).
    min_samples : int
        Minimum cross-sectional samples per lag.
    """

    def __init__(self, max_lag: int = 20, min_samples: int = 30):
        self.max_lag = max_lag
        self.min_samples = min_samples

    def analyze(
        self,
        signal_series: pd.DataFrame,
        returns_series: pd.DataFrame,
        signal_name: str = "signal",
    ) -> DecayProfile:
        """Analyze alpha decay for a single signal.

        Parameters
        ----------
        signal_series : pd.DataFrame
            DatetimeIndex, columns = tickers, values = signal at generation time.
        returns_series : pd.DataFrame
            Same structure, values = daily returns.
        """
        lag_ics = np.zeros(self.max_lag)

        for lag in range(1, self.max_lag + 1):
            fwd_ret = returns_series.rolling(lag).sum().shift(-lag)
            common_dates = signal_series.index.intersection(
                fwd_ret.dropna(how="all").index
            )
            if len(common_dates) < self.min_samples:
                continue

            ics = []
            for dt in common_dates:
                sig = signal_series.loc[dt].values
                ret = fwd_ret.loc[dt].values
                m = ~(np.isnan(sig) | np.isnan(ret))
                if m.sum() < 10:
                    continue
                rho, _ = spearmanr(sig[m], ret[m])
                if not np.isnan(rho):
                    ics.append(rho)

            lag_ics[lag - 1] = float(np.mean(ics)) if ics else 0.0

        half_life, decay_rate = self._fit_exponential_decay(lag_ics)

        return DecayProfile(
            signal_name=signal_name,
            autocorrelations=np.array([]),  # filled by caller if needed
            half_life_days=half_life,
            decay_rate=decay_rate,
            is_fast_decay=half_life < 5.0,
            lag_ics=lag_ics,
        )

    @staticmethod
    def _fit_exponential_decay(lag_ics: np.ndarray) -> tuple[float, float]:
        """Fit IC(lag) ~ A * exp(-lambda * lag) via log-linear regression."""
        lags = np.arange(1, len(lag_ics) + 1, dtype=float)
        positive_mask = lag_ics > 1e-6
        if positive_mask.sum() < 3:
            return float("inf"), 0.0

        log_ics = np.log(lag_ics[positive_mask])
        valid_lags = lags[positive_mask]

        A = np.vstack([valid_lags, np.ones(len(valid_lags))]).T
        try:
            result = np.linalg.lstsq(A, log_ics, rcond=None)
            slope = result[0][0]
            decay_rate = max(-slope, 1e-8)
            half_life = np.log(2) / decay_rate
            return float(half_life), float(decay_rate)
        except (np.linalg.LinAlgError, ValueError):
            return float("inf"), 0.0


# ---------------------------------------------------------------------------
# Main Signal Combiner
# ---------------------------------------------------------------------------


class SignalCombiner:
    """Combine multiple alpha signals with adaptive, regime-aware weighting.

    Parameters
    ----------
    method : CombineMethod
        Default combination method.
    tracker_halflife : int
        Half-life for performance tracker (trading days).
    crowding_threshold : float
        Correlation threshold for crowding detection.
    regime_method_map : dict, optional
        Override combination method per regime.
    min_weight : float
        Floor weight for any active signal.
    max_weight : float
        Cap on single signal weight.
    """

    def __init__(
        self,
        method: CombineMethod = CombineMethod.INVERSE_VARIANCE,
        tracker_halflife: int = 63,
        crowding_threshold: float = 0.70,
        regime_method_map: dict[str, CombineMethod] | None = None,
        min_weight: float = 0.05,
        max_weight: float = 0.60,
    ):
        self.method = method
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.regime_method_map = regime_method_map or {
            "high_vol": CombineMethod.EQUAL_WEIGHT,
            "normal": CombineMethod.INVERSE_VARIANCE,
            "low_vol": CombineMethod.MEAN_VARIANCE_OPT,
        }

        self.tracker = SignalPerformanceTracker(halflife=tracker_halflife)
        self.crowding_detector = SignalCrowdingDetector(
            correlation_threshold=crowding_threshold
        )
        self.decay_analyzer = SignalDecayAnalyzer()

        self._signals: dict[str, SignalMetadata] = {}
        self._last_crowding: CrowdingReport | None = None

    def register_signal(self, meta: SignalMetadata) -> None:
        """Register a new alpha signal."""
        self._signals[meta.name] = meta
        logger.info(f"Registered signal: {meta.name} ({meta.category})")

    def update_performance(
        self,
        date: pd.Timestamp,
        signal_values: dict[str, np.ndarray],
        realized_returns: np.ndarray,
    ) -> None:
        """Update performance tracker with one day's observations."""
        for name, values in signal_values.items():
            self.tracker.update(name, date, values, realized_returns)

    def combine(
        self,
        signal_values: dict[str, np.ndarray],
        as_of: pd.Timestamp,
        regime: str | None = None,
        signal_matrix: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Produce a blended composite signal from individual alpha signals.

        Parameters
        ----------
        signal_values : dict
            signal_name -> cross-sectional values (n_stocks,).
        as_of : pd.Timestamp
            Current timestamp for snapshot computation.
        regime : str, optional
            Current market regime label (e.g. "high_vol", "normal", "low_vol").
        signal_matrix : pd.DataFrame, optional
            Historical signal matrix for crowding detection.

        Returns
        -------
        (composite_signal, weights_dict)
        """
        active_signals = [n for n in signal_values if n in self._signals]
        if not active_signals:
            raise ValueError("No registered signals found in signal_values")

        # Determine method based on regime
        method = self.method
        if regime is not None:
            method = self.regime_method_map.get(regime, self.method)
            logger.info(f"Regime={regime} -> method={method.value}")

        # Crowding check
        if signal_matrix is not None:
            self._last_crowding = self.crowding_detector.analyze(signal_matrix)
            if self._last_crowding.is_crowded:
                logger.warning(
                    f"Signal crowding: {len(self._last_crowding.crowded_pairs)} "
                    f"pairs above {self.crowding_detector.correlation_threshold:.0%}"
                )

        # Performance snapshots
        snapshots = {
            name: self.tracker.get_snapshot(name, as_of)
            for name in active_signals
        }

        # Compute weights
        if method == CombineMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(active_signals)
        elif method == CombineMethod.INVERSE_VARIANCE:
            weights = self._inverse_variance(snapshots)
        elif method == CombineMethod.BAYESIAN_MODEL_AVG:
            weights = self._bayesian_model_avg(snapshots)
        elif method == CombineMethod.MEAN_VARIANCE_OPT:
            weights = self._mean_variance_opt(snapshots, signal_values)
        else:
            weights = self._equal_weight(active_signals)

        # Constraints
        weights = self._constrain_weights(weights)

        # Crowding penalty
        if self._last_crowding is not None and self._last_crowding.is_crowded:
            weights = self._apply_crowding_penalty(weights, self._last_crowding)

        # Blend: z-score normalize each signal before combining
        n_stocks = len(next(iter(signal_values.values())))
        composite = np.zeros(n_stocks)
        for name, w in weights.items():
            sig = signal_values[name].copy()
            sig_std = np.nanstd(sig)
            if sig_std > 1e-8:
                sig = (sig - np.nanmean(sig)) / sig_std
            composite += w * np.nan_to_num(sig, nan=0.0)

        return composite, weights

    # --- Weighting methods ---

    @staticmethod
    def _equal_weight(names: list[str]) -> dict[str, float]:
        n = len(names)
        return {name: 1.0 / n for name in names}

    @staticmethod
    def _inverse_variance(snapshots: dict[str, SignalSnapshot]) -> dict[str, float]:
        """Weight inversely proportional to IC variance (Bates & Granger 1969)."""
        inv_vars = {n: 1.0 / s.ic_variance for n, s in snapshots.items()}
        total = sum(inv_vars.values())
        if total < 1e-12:
            n = len(snapshots)
            return {name: 1.0 / n for name in snapshots}
        return {n: iv / total for n, iv in inv_vars.items()}

    @staticmethod
    def _bayesian_model_avg(snapshots: dict[str, SignalSnapshot]) -> dict[str, float]:
        """Posterior weight ~ exp(IC^2 / (2 * var_IC))."""
        log_ev = {n: (s.rolling_ic**2) / (2 * s.ic_variance) for n, s in snapshots.items()}
        max_le = max(log_ev.values())
        exp_ev = {n: np.exp(le - max_le) for n, le in log_ev.items()}
        total = sum(exp_ev.values())
        if total < 1e-12:
            n = len(snapshots)
            return {name: 1.0 / n for name in snapshots}
        return {n: e / total for n, e in exp_ev.items()}

    def _mean_variance_opt(
        self,
        snapshots: dict[str, SignalSnapshot],
        signal_values: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Markowitz optimization over signal IC streams."""
        names = list(snapshots.keys())
        n = len(names)
        mu = np.array([snapshots[name].rolling_ic for name in names])

        sig_matrix = np.column_stack([signal_values[name] for name in names])
        valid = ~np.any(np.isnan(sig_matrix), axis=1)
        sig_clean = sig_matrix[valid]

        if len(sig_clean) < n + 1:
            return self._inverse_variance(snapshots)

        ranks = np.apply_along_axis(
            lambda x: np.argsort(np.argsort(x)).astype(float), 0, sig_clean
        )
        ic_stds = np.array([snapshots[name].rolling_ic_std for name in names])
        D = np.diag(ic_stds)
        corr = np.corrcoef(ranks.T)
        Sigma = D @ corr @ D + np.eye(n) * 1e-4

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ Sigma @ w)
            return -port_ret / max(port_vol, 1e-8)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            return {name: float(w) for name, w in zip(names, result.x)}
        logger.warning(f"MVO failed ({result.message}), fallback to inverse-variance")
        return self._inverse_variance(snapshots)

    def _constrain_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply floor/cap constraints and renormalize."""
        constrained = {
            n: np.clip(w, self.min_weight, self.max_weight)
            for n, w in weights.items()
        }
        total = sum(constrained.values())
        if total > 0:
            constrained = {n: w / total for n, w in constrained.items()}
        return constrained

    @staticmethod
    def _apply_crowding_penalty(
        weights: dict[str, float],
        crowding: CrowdingReport,
        shrinkage: float = 0.3,
    ) -> dict[str, float]:
        """Shrink weights of crowded pairs toward equal weight."""
        crowded_names = set()
        for s1, s2, _ in crowding.crowded_pairs:
            crowded_names.add(s1)
            crowded_names.add(s2)

        if not crowded_names:
            return weights

        adjusted = dict(weights)
        for name in crowded_names:
            if name in adjusted:
                adjusted[name] *= 1.0 - shrinkage

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {n: w / total for n, w in adjusted.items()}
        return adjusted
