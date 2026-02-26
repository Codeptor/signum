"""Walk-forward, purged k-fold, and CPCV validation for financial models.

Phase 4 additions:
  - ``purged_kfold_split`` — generator yielding (train, test) index pairs with
    purge and embargo gaps to eliminate look-ahead bias from overlapping
    return windows.
  - ``purged_kfold_cv`` — convenience function that trains + evaluates a model
    across purged k-fold splits, returning mean and std IC.
"""

import logging
from typing import Generator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.6,
    embargo_days: int = 5,
):
    """Walk-forward cross-validation with embargo period.

    Yields (train_indices, test_indices) tuples.
    Embargo period prevents data leakage from overlapping return windows.
    """
    n = len(df)
    # Reserve a minimum training window, then divide the rest into test folds
    min_train = max(1, int(n * train_pct / (n_splits + train_pct * (1 - n_splits))))
    # Compute test_size so that n_splits folds fit after the first training window
    total_test = n - min_train - embargo_days
    test_size = total_test // n_splits

    for i in range(n_splits):
        test_start = min_train + embargo_days + i * test_size
        test_end = test_start + test_size if i < n_splits - 1 else n
        train_end = test_start - embargo_days
        # Expanding window: use all available history (Fix #39)
        train_start = 0

        if train_start >= train_end or test_start >= n:
            continue

        train_idx = list(range(train_start, train_end))
        test_idx = list(range(test_start, min(test_end, n)))
        yield train_idx, test_idx


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

    Adjusts for multiple testing by comparing against the expected maximum
    Sharpe under the null hypothesis.
    """
    from scipy import stats

    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    e_max = stats.norm.ppf(1 - 1 / n_trials) * (1 - 0.5772 / np.log(n_trials))

    # Standard error of Sharpe
    se = np.sqrt((1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2) / n_observations)

    # Probability that the observed Sharpe exceeds the expected max under null
    dsr = stats.norm.cdf((sharpe - e_max) / se)
    return float(dsr)


# ---------------------------------------------------------------------------
# Phase 4: Purged k-fold cross-validation
# ---------------------------------------------------------------------------


def purged_kfold_split(
    n_samples: int,
    n_splits: int = 5,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Purged k-fold split with embargo (Lopez de Prado, 2018).

    Unlike standard k-fold, this accounts for temporal dependence in
    financial data by:

    1. **Purge**: Removing ``purge_pct`` of the fold size *before* each
       test fold from the training set to eliminate label overlap.
    2. **Embargo**: Removing ``embargo_pct`` of the fold size *after*
       each test fold from the training set to prevent information leakage
       from auto-correlated features.

    Args:
        n_samples: Total number of observations (assumed time-ordered).
        n_splits: Number of folds (default 5).
        purge_pct: Fraction of fold size to purge before test set.
        embargo_pct: Fraction of fold size to embargo after test set.

    Yields:
        (train_indices, test_indices) tuples for each fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_samples < n_splits:
        raise ValueError(f"n_samples ({n_samples}) must be >= n_splits ({n_splits})")

    fold_size = n_samples // n_splits
    purge_size = max(1, int(fold_size * purge_pct))
    embargo_size = max(1, int(fold_size * embargo_pct))

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples

        # Training indices: everything except (purge + test + embargo)
        purge_start = max(0, test_start - purge_size)
        embargo_end = min(n_samples, test_end + embargo_size)

        train_before = list(range(0, purge_start))
        train_after = list(range(embargo_end, n_samples))
        train_idx = train_before + train_after
        test_idx = list(range(test_start, test_end))

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield train_idx, test_idx


def purged_kfold_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_5d",
    n_splits: int = 5,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    model_factory: Optional[object] = None,
) -> tuple[float, float, list[float]]:
    """Run purged k-fold cross-validation and return IC statistics.

    Trains a fresh model per fold and evaluates Information Coefficient
    (Spearman rank correlation between predictions and true targets).

    Args:
        df: DataFrame with feature columns and target, assumed time-ordered.
        feature_cols: List of feature column names.
        target_col: Target column name.
        n_splits: Number of CV folds.
        purge_pct: Purge fraction (see ``purged_kfold_split``).
        embargo_pct: Embargo fraction.
        model_factory: Callable that returns a fresh model instance with
            ``.fit(df, target_col)`` and ``.predict(df)`` methods.
            Defaults to ``CrossSectionalModel(feature_cols=feature_cols)``.

    Returns:
        (mean_ic, std_ic, fold_ics) — mean IC, standard deviation, and
        per-fold IC values.
    """
    from python.alpha.model import CrossSectionalModel

    n_samples = len(df)
    fold_ics: list[float] = []

    for fold_num, (train_idx, test_idx) in enumerate(
        purged_kfold_split(n_samples, n_splits, purge_pct, embargo_pct)
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Create a fresh model for each fold
        if model_factory is not None:
            model = model_factory()  # type: ignore[operator]
        else:
            model = CrossSectionalModel(feature_cols=feature_cols)

        # Train
        model.fit(train_df, target_col=target_col)

        # Predict and compute IC
        preds = model.predict(test_df)
        y_true = test_df[target_col].values

        # Handle NaN in predictions or targets
        mask = ~np.isnan(preds) & ~np.isnan(y_true)
        if mask.sum() < 5:
            logger.warning(f"Fold {fold_num + 1}: too few valid predictions, skipping")
            continue

        ic = float(np.corrcoef(preds[mask], y_true[mask])[0, 1])
        fold_ics.append(ic)
        logger.info(
            f"Fold {fold_num + 1}/{n_splits}: IC = {ic:.4f} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

    if not fold_ics:
        logger.error("No valid folds — returning zero IC")
        return 0.0, 0.0, []

    mean_ic = float(np.mean(fold_ics))
    std_ic = float(np.std(fold_ics))
    logger.info(f"Purged {n_splits}-fold CV: IC = {mean_ic:.4f} ± {std_ic:.4f}")

    return mean_ic, std_ic, fold_ics
