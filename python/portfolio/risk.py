"""Risk metrics: VaR, CVaR, drawdown, concentration."""

import numpy as np
import pandas as pd
from scipy import stats


class RiskEngine:
    """Portfolio risk calculation engine."""

    def __init__(self, returns: pd.DataFrame, weights: pd.Series):
        self.returns = returns
        self.weights = weights
        self.portfolio_returns = (returns * weights).sum(axis=1)

    def var_parametric(self, confidence: float = 0.95) -> float:
        """Parametric VaR assuming normal distribution."""
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        return stats.norm.ppf(1 - confidence, mu, sigma)

    def var_historical(self, confidence: float = 0.95) -> float:
        """Historical simulation VaR."""
        return float(np.percentile(self.portfolio_returns, (1 - confidence) * 100))

    def cvar_historical(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall) via historical simulation."""
        var = self.var_historical(confidence)
        tail = self.portfolio_returns[self.portfolio_returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def max_drawdown(self) -> float:
        """Maximum drawdown of the portfolio."""
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        return float(drawdowns.min())

    def rolling_sharpe(self, window: int = 60, risk_free: float = 0.0) -> pd.Series:
        """Rolling Sharpe ratio."""
        excess = self.portfolio_returns - risk_free / 252
        rolling_mean = excess.rolling(window).mean() * 252
        rolling_std = excess.rolling(window).std() * np.sqrt(252)
        return rolling_mean / rolling_std

    def concentration(self) -> pd.Series:
        """Herfindahl-Hirschman Index and effective number of bets."""
        hhi = (self.weights ** 2).sum()
        return pd.Series({"hhi": hhi, "effective_n": 1 / hhi})

    def summary(self) -> dict:
        """Full risk summary."""
        return {
            "var_95_parametric": self.var_parametric(0.95),
            "var_95_historical": self.var_historical(0.95),
            "cvar_95": self.cvar_historical(0.95),
            "max_drawdown": self.max_drawdown(),
            "annualized_vol": self.portfolio_returns.std() * np.sqrt(252),
            "annualized_return": self.portfolio_returns.mean() * 252,
            "sharpe": (self.portfolio_returns.mean() * 252)
            / (self.portfolio_returns.std() * np.sqrt(252)),
            "hhi": self.concentration()["hhi"],
        }
