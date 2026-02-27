"""Tests for feature importance methods (MDI, MDA, SFI, CluFI)."""

import numpy as np
import pytest

from python.alpha.feature_importance import (
    ClusteredImportanceResult,
    FeatureImportanceAnalyzer,
    ImportanceResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(T=500, N=8, n_informative=3, seed=42):
    """Generate data with known informative features."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))
    # y depends on first n_informative features
    coefs = np.zeros(N)
    coefs[:n_informative] = rng.uniform(0.5, 2.0, n_informative)
    y = X @ coefs + rng.standard_normal(T) * 0.5
    feature_names = [f"f{i}" for i in range(N)]
    return X, y, feature_names, n_informative


# ---------------------------------------------------------------------------
# MDI
# ---------------------------------------------------------------------------


class TestMDI:
    def test_returns_result(self):
        X, y, names, _ = _make_data()
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y, names)
        assert isinstance(result, ImportanceResult)

    def test_importances_sum_to_one(self):
        X, y, names, _ = _make_data()
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y, names)
        total = sum(result.importances.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_importances_non_negative(self):
        X, y, names, _ = _make_data()
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y, names)
        for v in result.importances.values():
            assert v >= 0

    def test_informative_features_rank_higher(self):
        """Informative features should generally rank in top positions."""
        X, y, names, n_info = _make_data(T=1000)
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y, names, n_trees=100)
        top_features = result.ranking[:n_info + 1]
        informative = set(names[:n_info])
        overlap = set(top_features) & informative
        assert len(overlap) >= 2  # At least 2 of 3 informative in top 4

    def test_ranking_order(self):
        X, y, names, _ = _make_data()
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y, names)
        ranking = result.ranking
        for i in range(len(ranking) - 1):
            assert result.importances[ranking[i]] >= result.importances[ranking[i + 1]]

    def test_top_n(self):
        X, y, names, _ = _make_data()
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y, names)
        top3 = result.top_n(3)
        assert len(top3) == 3

    def test_auto_feature_names(self):
        X, y, _, _ = _make_data()
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y)
        assert "f0" in result.importances

    def test_std_computed(self):
        X, y, names, _ = _make_data()
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.mdi(X, y, names, n_trees=50)
        assert all(v >= 0 for v in result.std.values())


# ---------------------------------------------------------------------------
# MDA
# ---------------------------------------------------------------------------


class TestMDA:
    def test_returns_result(self):
        X, y, names, _ = _make_data(T=200, N=4)
        analyzer = FeatureImportanceAnalyzer(n_splits=3, n_repeats=2, seed=42)
        result = analyzer.mda(X, y, names, n_trees=10)
        assert isinstance(result, ImportanceResult)

    def test_informative_features_positive_importance(self):
        """Permuting informative features should decrease score more."""
        X, y, names, n_info = _make_data(T=500, N=5, n_informative=2)
        analyzer = FeatureImportanceAnalyzer(n_splits=3, n_repeats=3, seed=42)
        result = analyzer.mda(X, y, names, n_trees=15)
        # At least one informative feature should have positive importance
        informative_imp = [result.importances[names[i]] for i in range(n_info)]
        assert max(informative_imp) > 0

    def test_all_features_have_scores(self):
        X, y, names, _ = _make_data(T=200, N=4)
        analyzer = FeatureImportanceAnalyzer(n_splits=3, n_repeats=2, seed=42)
        result = analyzer.mda(X, y, names, n_trees=10)
        assert len(result.importances) == 4

    def test_std_across_folds(self):
        X, y, names, _ = _make_data(T=200, N=4)
        analyzer = FeatureImportanceAnalyzer(n_splits=3, n_repeats=2, seed=42)
        result = analyzer.mda(X, y, names, n_trees=10)
        for f in names:
            assert f in result.std


# ---------------------------------------------------------------------------
# SFI
# ---------------------------------------------------------------------------


class TestSFI:
    def test_returns_result(self):
        X, y, names, _ = _make_data(T=200, N=4)
        analyzer = FeatureImportanceAnalyzer(n_splits=3, seed=42)
        result = analyzer.sfi(X, y, names, n_trees=10)
        assert isinstance(result, ImportanceResult)

    def test_all_features_scored(self):
        X, y, names, _ = _make_data(T=200, N=6)
        analyzer = FeatureImportanceAnalyzer(n_splits=3, seed=42)
        result = analyzer.sfi(X, y, names, n_trees=10)
        assert len(result.importances) == 6

    def test_informative_better_than_noise(self):
        """Informative features should have better single-feature scores."""
        X, y, names, n_info = _make_data(T=500, N=6, n_informative=2)
        analyzer = FeatureImportanceAnalyzer(n_splits=3, seed=42)
        result = analyzer.sfi(X, y, names, n_trees=15)
        info_scores = [result.importances[names[i]] for i in range(n_info)]
        noise_scores = [result.importances[names[i]] for i in range(n_info, len(names))]
        # Best informative should beat average noise
        assert max(info_scores) > np.mean(noise_scores)


# ---------------------------------------------------------------------------
# Clustered Feature Importance
# ---------------------------------------------------------------------------


class TestClusteredImportance:
    def test_returns_result(self):
        X, y, names, _ = _make_data(T=300, N=8)
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.clustered_importance(X, y, names, n_clusters=3, n_trees=20)
        assert isinstance(result, ClusteredImportanceResult)

    def test_all_features_assigned(self):
        X, y, names, _ = _make_data(T=300, N=8)
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.clustered_importance(X, y, names, n_clusters=3, n_trees=20)
        assert len(result.feature_to_cluster) == 8

    def test_cluster_count(self):
        X, y, names, _ = _make_data(T=300, N=8)
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.clustered_importance(X, y, names, n_clusters=4, n_trees=20)
        assert len(result.cluster_importance) <= 4

    def test_representative_features(self):
        X, y, names, _ = _make_data(T=300, N=8)
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.clustered_importance(X, y, names, n_clusters=3, n_trees=20)
        for c, rep in result.representative_features.items():
            assert rep in result.cluster_members[c]

    def test_silhouette_bounded(self):
        X, y, names, _ = _make_data(T=300, N=8)
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.clustered_importance(X, y, names, n_clusters=3, n_trees=20)
        assert -1 <= result.silhouette_score <= 1

    def test_cluster_importance_positive(self):
        X, y, names, _ = _make_data(T=300, N=8)
        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.clustered_importance(X, y, names, n_clusters=3, n_trees=20)
        for v in result.cluster_importance.values():
            assert v >= 0

    def test_correlated_features_grouped(self):
        """Highly correlated features should end up in the same cluster."""
        rng = np.random.default_rng(42)
        T, N = 300, 6
        base = rng.standard_normal((T, 2))
        # Features 0,1,2 are correlated; 3,4,5 are correlated
        X = np.column_stack([
            base[:, 0] + rng.standard_normal(T) * 0.1,
            base[:, 0] + rng.standard_normal(T) * 0.1,
            base[:, 0] + rng.standard_normal(T) * 0.1,
            base[:, 1] + rng.standard_normal(T) * 0.1,
            base[:, 1] + rng.standard_normal(T) * 0.1,
            base[:, 1] + rng.standard_normal(T) * 0.1,
        ])
        y = X[:, 0] + rng.standard_normal(T) * 0.5
        names = [f"f{i}" for i in range(N)]

        analyzer = FeatureImportanceAnalyzer(seed=42)
        result = analyzer.clustered_importance(X, y, names, n_clusters=2, n_trees=20)
        # Features 0,1,2 should be in same cluster
        c0 = result.feature_to_cluster["f0"]
        assert result.feature_to_cluster["f1"] == c0
        assert result.feature_to_cluster["f2"] == c0
