"""Portfolio optimization: HRP, CVaR, Black-Litterman with ML views."""

import logging

import pandas as pd
from skfolio import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity, MeanRisk
from skfolio.prior import BlackLitterman, EmpiricalPrior

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Multi-strategy portfolio optimizer using skfolio."""

    def __init__(self, prices: pd.DataFrame):
        """Initialize with a DataFrame of asset prices (columns=tickers, index=dates)."""
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.tickers = list(prices.columns)

    def hrp(self) -> pd.Series:
        """Hierarchical Risk Parity allocation."""
        model = HierarchicalRiskParity()
        model.fit(self.returns)
        return pd.Series(model.weights_, index=self.tickers, name="hrp_weights")

    def min_cvar(self, confidence_level: float = 0.95) -> pd.Series:
        """Minimum CVaR (Conditional Value at Risk) allocation."""
        model = MeanRisk(
            risk_measure=RiskMeasure.CVAR,
            min_weights=0.0,
            cvar_beta=confidence_level,
        )
        model.fit(self.returns)
        return pd.Series(model.weights_, index=self.tickers, name="min_cvar_weights")

    def black_litterman(
        self,
        views: pd.Series,
        view_confidences: pd.Series,
    ) -> pd.Series:
        """Black-Litterman allocation with analyst/ML model views.

        Parameters
        ----------
        views : pd.Series
            Mapping of ticker to expected return (e.g. {"AAPL": 0.02}).
        view_confidences : pd.Series
            Mapping of ticker to confidence level between 0 and 1
            (Idzorek's method).
        """
        # Sanitize ticker names: skfolio's BL parser treats hyphens as
        # subtraction and chokes on scientific notation in view strings.
        sanitize = {t: t.replace("-", "_") for t in self.tickers if "-" in t}
        if sanitize:
            returns = self.returns.rename(columns=sanitize)
        else:
            returns = self.returns
        safe_tickers = [sanitize.get(t, t) for t in self.tickers]

        view_strings = [
            f"{sanitize.get(ticker, ticker)} = {ret:.10f}"
            for ticker, ret in views.items()
        ]
        confidence_array = [view_confidences[ticker] for ticker in views.index]

        prior_model = BlackLitterman(
            views=view_strings,
            view_confidences=confidence_array,
            prior_estimator=EmpiricalPrior(),
        )

        model = MeanRisk(
            risk_measure=RiskMeasure.CVAR,
            prior_estimator=prior_model,
            min_weights=0.0,
        )
        model.fit(returns)
        # Map sanitized names back to original tickers
        unsanitize = {v: k for k, v in sanitize.items()}
        original_tickers = [unsanitize.get(t, t) for t in safe_tickers]
        return pd.Series(model.weights_, index=original_tickers, name="bl_weights")

    def risk_parity(self) -> pd.Series:
        """Equal risk contribution allocation via HRP with variance risk measure."""
        model = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
        model.fit(self.returns)
        return pd.Series(
            model.weights_, index=self.tickers, name="risk_parity_weights"
        )

    def compare_all(
        self,
        views: pd.Series | None = None,
        view_confidences: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Run all optimization strategies and return weights comparison."""
        results = {"hrp": self.hrp(), "min_cvar": self.min_cvar()}
        if views is not None and view_confidences is not None:
            results["black_litterman"] = self.black_litterman(views, view_confidences)
        results["risk_parity"] = self.risk_parity()
        return pd.DataFrame(results)
