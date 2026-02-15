"""Full backtesting pipeline: train -> predict -> optimize -> evaluate."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from python.alpha.features import compute_alpha_features, compute_forward_returns
from python.alpha.model import CrossSectionalModel
from python.alpha.train import FEATURE_COLS
from python.backtest.validation import deflated_sharpe_ratio, walk_forward_split
from python.bridge.bl_views import create_bl_views
from python.data.ingestion import extract_close_prices
from python.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/processed")

# Minimum trailing days needed for covariance estimation
MIN_PRICE_HISTORY = 60


def _allocate(
    top_tickers: list[str],
    raw_preds: pd.Series,
    prices: pd.DataFrame,
    date,
    method: str,
    max_weight: float,
) -> pd.Series:
    """Compute portfolio weights for a single rebalance date."""
    # Get trailing price history for selected tickers
    available = [t for t in top_tickers if t in prices.columns]
    prices_window = prices.loc[:date, available].dropna(axis=1).tail(252)

    # Drop tickers with insufficient history
    prices_window = prices_window.loc[:, prices_window.count() >= MIN_PRICE_HISTORY]
    valid_tickers = list(prices_window.columns)

    if len(valid_tickers) < 3 or method == "equal_weight":
        weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)
        return _apply_max_weight(weights, max_weight)

    try:
        optimizer = PortfolioOptimizer(prices_window)

        if method == "black_litterman":
            # Use raw ML predictions as BL views, rank-based confidence
            preds_valid = raw_preds.reindex(valid_tickers).dropna()
            if len(preds_valid) < 3:
                weights = optimizer.hrp()
            else:
                # Normalize predictions to reasonable return scale
                pred_range = preds_valid.max() - preds_valid.min()
                if pred_range > 0:
                    conf = (preds_valid - preds_valid.min()) / pred_range
                    conf = conf.clip(0.1, 0.9)
                else:
                    conf = pd.Series(0.5, index=preds_valid.index)
                views, view_confs = create_bl_views(preds_valid, conf)
                weights = optimizer.black_litterman(views, view_confs)
        elif method == "hrp":
            weights = optimizer.hrp()
        elif method == "min_cvar":
            weights = optimizer.min_cvar()
        elif method == "risk_parity":
            weights = optimizer.risk_parity()
        else:
            raise ValueError(f"Unknown optimizer method: {method}")

        # Reindex to all top tickers (zero-fill any missing)
        weights = weights.reindex(top_tickers, fill_value=0.0)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)

    except Exception as e:
        logger.warning(f"Optimizer failed ({e}), falling back to equal-weight")
        weights = pd.Series(1.0 / len(top_tickers), index=top_tickers)

    return _apply_max_weight(weights, max_weight)


def _apply_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    """Clip weights to max and renormalize."""
    if max_weight >= 1.0:
        return weights
    weights = weights.clip(upper=max_weight)
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def run_backtest(
    prices: pd.DataFrame,
    n_splits: int = 5,
    top_n: int = 20,
    rebalance_days: int = 5,
    optimizer_method: str = "black_litterman",
    transaction_cost_bps: float = 10.0,
    max_weight: float = 0.15,
    blend_alpha: float = 0.5,
) -> dict:
    """Walk-forward backtest with ML alpha, portfolio optimization, and transaction costs.

    Parameters
    ----------
    prices : DataFrame with DatetimeIndex, columns = tickers, values = close prices
    n_splits : number of walk-forward folds
    top_n : number of top-ranked tickers to hold each period
    rebalance_days : holding period (forward return horizon)
    optimizer_method : 'equal_weight', 'black_litterman', 'hrp', 'min_cvar', 'risk_parity'
    transaction_cost_bps : one-way transaction cost in basis points
    max_weight : maximum single-position weight (e.g. 0.15 = 15%)
    blend_alpha : weight on new allocation vs previous (1.0 = fully new, 0.5 = 50/50 blend)
    """
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
    weight_history = []
    prev_weights = pd.Series(dtype=float)
    total_turnover = 0.0
    n_rebalances = 0

    for fold, (train_idx, test_idx) in enumerate(
        walk_forward_split(labeled, n_splits=n_splits)
    ):
        logger.info(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

        train = labeled.iloc[train_idx]
        test = labeled.iloc[test_idx]

        model = CrossSectionalModel(model_type="lightgbm", feature_cols=FEATURE_COLS)
        model.fit(train, target_col=f"target_{rebalance_days}d")

        preds = model.predict(test)
        raw_pred_series = pd.Series(preds, index=test.index)
        ranks = raw_pred_series.groupby(test.index).rank(pct=True)

        test_with_preds = test.copy()
        test_with_preds["pred_rank"] = ranks.values
        test_with_preds["raw_pred"] = preds

        for date, group in test_with_preds.groupby(level=0):
            top = group.nlargest(top_n, "pred_rank")
            top_tickers = top["ticker"].tolist()
            raw_preds_top = top.set_index("ticker")["raw_pred"]

            # Portfolio optimization
            weights = _allocate(
                top_tickers, raw_preds_top, prices, date,
                method=optimizer_method, max_weight=max_weight,
            )

            # Blend with previous weights to dampen turnover
            if len(prev_weights) > 0 and blend_alpha < 1.0:
                common = sorted(set(weights.index) | set(prev_weights.index))
                w_target = weights.reindex(common, fill_value=0.0)
                w_prev = prev_weights.reindex(common, fill_value=0.0)
                weights = blend_alpha * w_target + (1 - blend_alpha) * w_prev
                weights = weights[weights > 1e-6]
                weights /= weights.sum()

            # Transaction costs (two-way: sell old + buy new)
            # Align weights to common index for turnover calculation
            all_tickers = sorted(set(weights.index) | set(prev_weights.index))
            w_new = weights.reindex(all_tickers, fill_value=0.0)
            w_old = prev_weights.reindex(all_tickers, fill_value=0.0)
            turnover = (w_new - w_old).abs().sum()
            cost = turnover * transaction_cost_bps / 10_000
            total_turnover += turnover
            n_rebalances += 1

            # Weighted return net of costs
            target_col = f"target_{rebalance_days}d"
            ticker_returns = top.set_index("ticker")[target_col]
            aligned_w = weights.reindex(ticker_returns.index, fill_value=0.0)
            ret_gross = (ticker_returns * aligned_w).sum()
            ret_net = ret_gross - cost

            all_returns.append({"date": date, "return": ret_net, "return_gross": ret_gross,
                                "turnover": turnover, "cost": cost})

            # Record weight snapshot
            weight_history.append({"date": date, **weights.to_dict()})
            prev_weights = weights

    returns_df = pd.DataFrame(all_returns).set_index("date")
    portfolio_returns = returns_df["return"]
    gross_returns = returns_df["return_gross"]

    ann_return = portfolio_returns.mean() * (252 / rebalance_days)
    ann_vol = portfolio_returns.std() * np.sqrt(252 / rebalance_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    ann_return_gross = gross_returns.mean() * (252 / rebalance_days)
    sharpe_gross = ann_return_gross / ann_vol if ann_vol > 0 else 0

    avg_turnover = total_turnover / max(n_rebalances, 1)
    cumulative = (1 + portfolio_returns).cumprod()
    max_dd = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()

    # Latest weights
    latest_weights = prev_weights

    # Weight history DataFrame
    weight_history_df = pd.DataFrame(weight_history).set_index("date").fillna(0.0)

    return {
        "portfolio_returns": portfolio_returns,
        "gross_returns": gross_returns,
        "weights": latest_weights,
        "weight_history": weight_history_df,
        "turnover_series": returns_df["turnover"],
        "annualized_return": ann_return,
        "annualized_return_gross": ann_return_gross,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sharpe_ratio_gross": sharpe_gross,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "total_cost_bps": transaction_cost_bps,
        "deflated_sharpe": deflated_sharpe_ratio(
            sharpe, n_trials=n_splits, n_observations=len(portfolio_returns)
        ),
        "optimizer_method": optimizer_method,
        "n_folds": n_splits,
    }


def save_results(results: dict, output_dir: Path = RESULTS_DIR) -> None:
    """Persist backtest results to disk for the dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results["portfolio_returns"].to_frame("return").to_parquet(
        output_dir / "backtest_returns.parquet"
    )
    results["weights"].to_json(output_dir / "backtest_weights.json")
    results["weight_history"].to_parquet(output_dir / "backtest_weight_history.parquet")
    results["turnover_series"].to_frame("turnover").to_parquet(
        output_dir / "backtest_turnover.parquet"
    )

    serializable_keys = {
        "annualized_return", "annualized_return_gross", "annualized_volatility",
        "sharpe_ratio", "sharpe_ratio_gross", "max_drawdown", "avg_turnover",
        "total_cost_bps", "deflated_sharpe", "optimizer_method", "n_folds",
    }
    metrics = {}
    for k, v in results.items():
        if k in serializable_keys:
            metrics[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
    with open(output_dir / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved backtest results to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = pd.read_parquet("data/raw/sp500_ohlcv.parquet")
    prices = extract_close_prices(raw)

    # Strategy comparison
    methods = ["equal_weight", "hrp", "risk_parity", "black_litterman", "min_cvar"]
    comparison = []
    for method in methods:
        r = run_backtest(
            prices, optimizer_method=method,
            transaction_cost_bps=10.0, max_weight=0.15, blend_alpha=0.5,
        )
        comparison.append({
            "method": method,
            "sharpe_net": r["sharpe_ratio"],
            "sharpe_gross": r["sharpe_ratio_gross"],
            "ann_return": r["annualized_return"],
            "max_dd": r["max_drawdown"],
            "avg_turnover": r["avg_turnover"],
        })
        # Save the HRP run as the primary result (best risk-adjusted optimizer)
        if method == "hrp":
            save_results(r)

    comp_df = pd.DataFrame(comparison).set_index("method")
    logger.info(f"\nStrategy Comparison:\n{comp_df.to_string(float_format='%.3f')}")

    # Save comparison
    comp_df.to_csv(RESULTS_DIR / "strategy_comparison.csv")
    logger.info(f"Saved strategy comparison to {RESULTS_DIR / 'strategy_comparison.csv'}")
