"""Tests for purged k-fold cross-validation (Phase 4, §2.4.1)."""

import numpy as np
import pandas as pd
import pytest

from python.backtest.validation import purged_kfold_split, purged_kfold_cv


# ---------------------------------------------------------------------------
# Tests: purged_kfold_split
# ---------------------------------------------------------------------------


class TestPurgedKfoldSplit:
    def test_yields_correct_number_of_folds(self):
        folds = list(purged_kfold_split(n_samples=1000, n_splits=5))
        assert len(folds) == 5

    def test_test_indices_cover_all_samples(self):
        """Every sample should appear in exactly one test fold."""
        n = 1000
        folds = list(purged_kfold_split(n, n_splits=5, purge_pct=0.0, embargo_pct=0.0))
        all_test = []
        for _, test_idx in folds:
            all_test.extend(test_idx)
        assert sorted(all_test) == list(range(n))

    def test_no_overlap_between_train_and_test(self):
        """Train and test sets should never share indices."""
        for train_idx, test_idx in purged_kfold_split(500, n_splits=5):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_purge_removes_before_test(self):
        """With purge, indices right before the test fold should be excluded from train."""
        n = 100
        folds = list(purged_kfold_split(n, n_splits=5, purge_pct=0.10, embargo_pct=0.0))

        # For fold 1 (test starts at index 20), purge_size = max(1, int(20*0.10)) = 2
        # So indices 18, 19 should NOT be in training
        _, test_idx = folds[1]
        train_idx_set = set(folds[1][0])
        test_start = test_idx[0]
        purge_size = max(1, int(20 * 0.10))

        for i in range(max(0, test_start - purge_size), test_start):
            assert i not in train_idx_set, f"Index {i} should be purged but found in train"

    def test_embargo_removes_after_test(self):
        """With embargo, indices right after the test fold should be excluded from train."""
        n = 100
        folds = list(purged_kfold_split(n, n_splits=5, purge_pct=0.0, embargo_pct=0.10))

        # For fold 0 (test ends at 20), embargo_size = max(1, int(20*0.10)) = 2
        # So indices 20, 21 should NOT be in training
        _, test_idx = folds[0]
        train_idx_set = set(folds[0][0])
        test_end = test_idx[-1] + 1
        embargo_size = max(1, int(20 * 0.10))

        for i in range(test_end, min(n, test_end + embargo_size)):
            assert i not in train_idx_set, f"Index {i} should be embargoed but found in train"

    def test_train_size_smaller_with_purge_embargo(self):
        """Purge + embargo should reduce training set size compared to no purge/embargo."""
        n = 500
        folds_clean = list(purged_kfold_split(n, n_splits=5, purge_pct=0.0, embargo_pct=0.0))
        folds_purged = list(purged_kfold_split(n, n_splits=5, purge_pct=0.05, embargo_pct=0.02))

        for (clean_train, _), (purged_train, _) in zip(folds_clean, folds_purged):
            assert len(purged_train) <= len(clean_train)

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            list(purged_kfold_split(100, n_splits=1))

    def test_insufficient_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples.*must be >= n_splits"):
            list(purged_kfold_split(3, n_splits=5))

    def test_three_splits(self):
        folds = list(purged_kfold_split(300, n_splits=3))
        assert len(folds) == 3
        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ---------------------------------------------------------------------------
# Tests: purged_kfold_cv
# ---------------------------------------------------------------------------


def _make_cv_data(n_rows=500, n_features=8, seed=42):
    """Create synthetic data for CV testing."""
    rng = np.random.RandomState(seed)
    feature_names = [f"f{i}" for i in range(n_features)]
    X = rng.randn(n_rows, n_features)
    coef = rng.randn(n_features)
    y = X @ coef + rng.randn(n_rows) * 2.0

    df = pd.DataFrame(X, columns=feature_names)
    df["target_5d"] = y
    return df, feature_names


class TestPurgedKfoldCV:
    def test_returns_three_values(self):
        df, cols = _make_cv_data()
        mean_ic, std_ic, fold_ics = purged_kfold_cv(df, cols, n_splits=3)
        assert isinstance(mean_ic, float)
        assert isinstance(std_ic, float)
        assert isinstance(fold_ics, list)

    def test_fold_count_matches_n_splits(self):
        df, cols = _make_cv_data()
        _, _, fold_ics = purged_kfold_cv(df, cols, n_splits=5)
        assert len(fold_ics) == 5

    def test_mean_ic_is_average_of_folds(self):
        df, cols = _make_cv_data()
        mean_ic, _, fold_ics = purged_kfold_cv(df, cols, n_splits=3)
        assert mean_ic == pytest.approx(np.mean(fold_ics), abs=1e-6)

    def test_std_ic_is_std_of_folds(self):
        df, cols = _make_cv_data()
        _, std_ic, fold_ics = purged_kfold_cv(df, cols, n_splits=3)
        assert std_ic == pytest.approx(np.std(fold_ics), abs=1e-6)

    def test_ics_are_finite(self):
        df, cols = _make_cv_data()
        _, _, fold_ics = purged_kfold_cv(df, cols, n_splits=3)
        for ic in fold_ics:
            assert np.isfinite(ic)

    def test_positive_ic_with_signal(self):
        """With a real linear signal, ICs should be positive on average."""
        df, cols = _make_cv_data(n_rows=2000, seed=42)
        mean_ic, _, _ = purged_kfold_cv(df, cols, n_splits=5)
        # LightGBM should learn the linear signal — IC should be positive
        assert mean_ic > 0.0

    def test_custom_model_factory(self):
        """model_factory parameter should be used to create models."""
        from python.alpha.model import CrossSectionalModel

        df, cols = _make_cv_data()
        call_count = [0]

        def factory():
            call_count[0] += 1
            return CrossSectionalModel(feature_cols=cols)

        purged_kfold_cv(df, cols, n_splits=3, model_factory=factory)
        assert call_count[0] == 3  # One model per fold
