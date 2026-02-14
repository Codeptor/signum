"""Walk-forward and CPCV validation for financial models."""

import numpy as np
import pandas as pd


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
        train_start = max(0, train_end - int(train_end * train_pct / (1 - train_pct + 1e-9)))
        # For expanding window, use all available history
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
    se = np.sqrt(
        (1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2) / n_observations
    )

    # Probability that the observed Sharpe exceeds the expected max under null
    dsr = stats.norm.cdf((sharpe - e_max) / se)
    return float(dsr)
