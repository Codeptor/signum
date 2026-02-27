"""Regime-conditional portfolio optimizer.

Selects portfolio construction method based on the current market regime
detected by the HMM regime detector:

  - **high_vol** → HRP (hierarchy-based, no covariance inversion → robust
    when correlations spike and the covariance matrix is most unreliable).
  - **normal** → min-CVaR (mean-risk optimization with tail-risk control).
  - **low_vol** → HERC or Black-Litterman (can exploit return estimates
    when estimation error is lower).

This adaptive approach is the cutting-edge theme in 2025 quant research:
static optimizers assume stationarity, but financial markets exhibit clear
regime shifts.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from python.monitoring.hmm_regime import HMMRegimeDetector, HMMRegimeState
from python.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

# Default optimization method per regime
DEFAULT_REGIME_METHODS: dict[str, str] = {
    "high_vol": "hrp",
    "normal": "min_cvar",
    "low_vol": "herc",
}


@dataclass
class RegimeConditionalOptimizer:
    """Selects portfolio optimizer based on HMM-detected market regime.

    Args:
        prices: Price DataFrame (columns=tickers, index=dates).
        market_returns: Broad market returns (e.g. SPY) for regime detection.
        regime_methods: Mapping of regime label to optimizer method name.
            Valid methods: 'hrp', 'min_cvar', 'herc', 'nco', 'risk_parity'.
        max_weight: Optional cap on individual asset weight.
        current_weights: Current portfolio weights for turnover penalty.
        turnover_threshold: Minimum turnover to justify rebalancing.
        views: Optional BL views for Black-Litterman (used if method is 'bl').
        view_confidences: Optional BL view confidences.
        hmm_detector: Pre-fitted HMM detector (auto-created if None).
    """

    prices: pd.DataFrame
    market_returns: pd.Series
    regime_methods: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_REGIME_METHODS)
    )
    max_weight: Optional[float] = None
    current_weights: Optional[pd.Series] = None
    turnover_threshold: float = 0.20
    views: Optional[pd.Series] = None
    view_confidences: Optional[pd.Series] = None
    hmm_detector: Optional[HMMRegimeDetector] = None

    def __post_init__(self):
        self._optimizer = PortfolioOptimizer(
            prices=self.prices,
            max_weight=self.max_weight,
            current_weights=self.current_weights,
            turnover_threshold=self.turnover_threshold,
        )
        if self.hmm_detector is None:
            self.hmm_detector = HMMRegimeDetector()
            self.hmm_detector.fit(self.market_returns)

    def detect_regime(self) -> HMMRegimeState:
        """Detect the current market regime from market returns."""
        return self.hmm_detector.predict_regime(self.market_returns)

    def optimize(self) -> tuple[pd.Series, HMMRegimeState]:
        """Run regime-conditional optimization.

        Returns:
            (weights, regime_state) — optimized weights and the detected
            regime state that drove the method selection.
        """
        state = self.detect_regime()
        method = self.regime_methods.get(state.regime, "hrp")

        logger.info(
            f"Regime: {state.regime} (p={state.probabilities.get(state.regime, 0):.2f}) "
            f"→ using {method} optimizer"
        )

        # Dispatch to the appropriate optimizer method
        if method == "hrp":
            weights = self._optimizer.hrp()
        elif method == "min_cvar":
            weights = self._optimizer.min_cvar()
        elif method == "herc":
            weights = self._optimizer.herc()
        elif method == "nco":
            weights = self._optimizer.nco()
        elif method == "risk_parity":
            weights = self._optimizer.risk_parity()
        elif method == "black_litterman":
            if self.views is not None and self.view_confidences is not None:
                weights = self._optimizer.black_litterman(
                    self.views, self.view_confidences
                )
            else:
                logger.warning(
                    "BL selected but no views provided — falling back to HRP"
                )
                weights = self._optimizer.hrp()
        else:
            logger.warning(f"Unknown method '{method}' — falling back to HRP")
            weights = self._optimizer.hrp()

        # Apply exposure multiplier from regime
        weights = weights * state.exposure_multiplier
        # Re-normalize if exposure_multiplier < 1 (cash allocation implicit)
        remaining = 1.0 - weights.sum()
        if remaining > 0.01:
            logger.info(
                f"Regime exposure {state.exposure_multiplier:.0%} → "
                f"{remaining:.1%} implicit cash"
            )

        return weights, state

    def optimize_with_turnover(self) -> tuple[pd.Series, HMMRegimeState]:
        """Regime-conditional optimization with turnover penalty.

        If turnover from current weights is below the threshold, maintains
        current positions.
        """
        weights, state = self.optimize()

        if self.current_weights is None or self.current_weights.empty:
            return weights, state

        all_tickers = sorted(set(weights.index) | set(self.current_weights.index))
        w_new = weights.reindex(all_tickers, fill_value=0.0)
        w_old = self.current_weights.reindex(all_tickers, fill_value=0.0)
        turnover = (w_new - w_old).abs().sum() / 2

        if turnover < self.turnover_threshold:
            logger.info(
                f"Regime-conditional: low turnover ({turnover:.1%} < "
                f"{self.turnover_threshold:.1%}) — maintaining positions"
            )
            return self.current_weights, state

        logger.info(
            f"Regime-conditional: turnover {turnover:.1%} → rebalancing with {state.regime}"
        )
        return weights, state

    def compare_methods(self) -> pd.DataFrame:
        """Run all methods and return a comparison DataFrame with regime annotation."""
        state = self.detect_regime()
        selected_method = self.regime_methods.get(state.regime, "hrp")

        results = self._optimizer.compare_all(
            views=self.views, view_confidences=self.view_confidences
        )
        results.attrs["regime"] = state.regime
        results.attrs["selected_method"] = selected_method
        results.attrs["exposure_multiplier"] = state.exposure_multiplier
        return results

    def to_json(self) -> dict:
        """Serialize regime optimizer state for dashboard."""
        state = self.detect_regime()
        method = self.regime_methods.get(state.regime, "hrp")
        return {
            "regime": state.regime,
            "regime_probabilities": state.probabilities,
            "exposure_multiplier": state.exposure_multiplier,
            "selected_method": method,
            "regime_methods": self.regime_methods,
        }
