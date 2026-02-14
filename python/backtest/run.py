"""Full backtesting pipeline: train -> predict -> allocate -> evaluate."""

import logging

import numpy as np
import pandas as pd

from python.alpha.features import compute_alpha_features, compute_forward_returns
from python.alpha.model import CrossSectionalModel
from python.alpha.train import FEATURE_COLS
from python.backtest.validation import deflated_sharpe_ratio, walk_forward_split
from python.bridge.bl_views import create_bl_views
from python.portfolio.optimizer import PortfolioOptimizer
from python.portfolio.risk import RiskEngine

logger = logging.getLogger(__name__)


def run_backtest(
    prices: pd.DataFrame,
    n_splits: int = 5,
    top_n: int = 20,
    rebalance_days: int = 5,
) -> dict:
    """Walk-forward backtest with the full pipeline."""
    ohlcv_frames = []
    for ticker in prices.columns:
        ohlcv_frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "open": prices[ticker],
                    "high": prices[ticker] * 1.01,
                    "low": prices[ticker] * 0.99,
                    "close": prices[ticker],
                    "volume": 1e6,
                },
                index=prices.index,
            )
        )
    ohlcv = pd.concat(ohlcv_frames)
    featured = compute_alpha_features(ohlcv)
    labeled = compute_forward_returns(featured, horizon=rebalance_days)
    labeled = labeled.dropna(subset=FEATURE_COLS + [f"target_{rebalance_days}d"])

    all_returns = []

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_split(labeled, n_splits=n_splits)
    ):
        logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

        train = labeled.iloc[train_idx]
        test = labeled.iloc[test_idx]

        model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
        model.fit(train, target_col=f"target_{rebalance_days}d")

        preds = model.predict(test)
        ranks = pd.Series(preds, index=test.index).groupby(test.index).rank(pct=True)

        test_with_preds = test.copy()
        test_with_preds["pred_rank"] = ranks.values

        fold_returns = []
        for date, group in test_with_preds.groupby(level=0):
            top = group.nlargest(top_n, "pred_rank")
            ret = top[f"target_{rebalance_days}d"].mean()
            fold_returns.append({"date": date, "return": ret})

        all_returns.extend(fold_returns)

    returns_df = pd.DataFrame(all_returns).set_index("date")
    portfolio_returns = returns_df["return"]

    ann_return = portfolio_returns.mean() * (252 / rebalance_days)
    ann_vol = portfolio_returns.std() * np.sqrt(252 / rebalance_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    return {
        "portfolio_returns": portfolio_returns,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": ((1 + portfolio_returns).cumprod().cummax() - (1 + portfolio_returns).cumprod()).max(),
        "deflated_sharpe": deflated_sharpe_ratio(sharpe, n_trials=n_splits, n_observations=len(portfolio_returns)),
        "n_folds": n_splits,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prices = pd.read_parquet("data/raw/sp500_ohlcv.parquet")
    results = run_backtest(prices)
    for k, v in results.items():
        if k != "portfolio_returns":
            logger.info(f"{k}: {v}")
