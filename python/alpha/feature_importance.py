"""Feature importance methods for financial ML (de Prado 2018).

Implements multiple importance metrics that reveal different aspects
of feature relevance:
  1. MDI — Mean Decrease Impurity (tree-based, fast, biased to cardinality).
  2. MDA — Mean Decrease Accuracy (permutation-based, unbiased, model-agnostic).
  3. SFI — Single Feature Importance (trains separate model per feature).
  4. Clustered Feature Importance (CluFI) — groups redundant features, avoids
     importance dilution from multicollinearity.

Usage::

    analyzer = FeatureImportanceAnalyzer(n_splits=5, scoring='neg_mse')
    result = analyzer.mda(model, X, y)
    # result.importances → {feature: importance_score}

    clustered = analyzer.clustered_importance(model, X, y, n_clusters=5)
    # clustered.cluster_importance → {cluster_id: importance}

References:
  - de Prado (2018), "Advances in Financial Machine Learning", Ch. 8
  - Breiman (2001), "Random Forests"
  - Strobl et al. (2007), "Bias in Random Forest Variable Importance"
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ImportanceResult:
    """Feature importance scores with statistics."""

    importances: dict[str, float]  # feature → mean importance
    std: dict[str, float]  # feature → std across folds/repeats
    feature_names: list[str]

    @property
    def ranking(self) -> list[str]:
        """Features sorted by importance (highest first)."""
        return sorted(self.feature_names, key=lambda f: self.importances[f], reverse=True)

    def top_n(self, n: int = 10) -> dict[str, float]:
        return {f: self.importances[f] for f in self.ranking[:n]}


@dataclass
class ClusteredImportanceResult:
    """Clustered feature importance result."""

    cluster_importance: dict[int, float]  # cluster_id → importance
    cluster_members: dict[int, list[str]]  # cluster_id → feature names
    feature_to_cluster: dict[str, int]  # feature → cluster_id
    representative_features: dict[int, str]  # cluster_id → best feature
    silhouette_score: float


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative MSE (higher = better)."""
    return -float(np.mean((y_true - y_pred) ** 2))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(y_true == y_pred))


def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank information coefficient (Spearman correlation)."""
    n = len(y_true)
    if n < 3:
        return 0.0
    rank_true = np.argsort(np.argsort(y_true)).astype(float)
    rank_pred = np.argsort(np.argsort(y_pred)).astype(float)
    d = rank_true - rank_pred
    return float(1 - 6 * np.sum(d**2) / (n * (n**2 - 1)))


_SCORERS = {
    "neg_mse": _neg_mse,
    "accuracy": _accuracy,
    "rank_ic": _rank_ic,
}


# ---------------------------------------------------------------------------
# Simple decision tree for MDI (avoid sklearn dependency)
# ---------------------------------------------------------------------------


class _SimpleDecisionTree:
    """Minimal decision tree for MDI computation.

    Single-split (depth-1) trees that track impurity decrease per feature.
    """

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 5, seed: int = 42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.default_rng(seed)
        self.feature_importances_ = None
        self._n_features = 0
        self._tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SimpleDecisionTree":
        self._n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self._n_features)
        total_var = np.var(y) * len(y)
        self._tree = self._build(X, y, depth=0, total_var=total_var)
        if total_var > 0:
            self.feature_importances_ /= total_var
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._tree is None:
            return np.zeros(len(X))
        return np.array([self._predict_one(x, self._tree) for x in X])

    def _build(self, X, y, depth, total_var):
        node = {"value": float(np.mean(y)), "n": len(y)}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        parent_var = np.var(y) * len(y)

        # Random feature subset (sqrt(n_features))
        n_try = max(1, int(np.sqrt(self._n_features)))
        features = self.rng.choice(self._n_features, n_try, replace=False)

        for feat in features:
            vals = np.unique(X[:, feat])
            if len(vals) < 2:
                continue
            # Try a subset of thresholds
            n_thresholds = min(10, len(vals) - 1)
            thresholds = np.quantile(vals, np.linspace(0.1, 0.9, n_thresholds))

            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                child_var = np.var(left) * len(left) + np.var(right) * len(right)
                gain = parent_var - child_var
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_feat < 0:
            return node

        self.feature_importances_[best_feat] += best_gain
        mask = X[:, best_feat] <= best_thresh
        node["feature"] = best_feat
        node["threshold"] = best_thresh
        node["left"] = self._build(X[mask], y[mask], depth + 1, total_var)
        node["right"] = self._build(X[~mask], y[~mask], depth + 1, total_var)
        return node

    def _predict_one(self, x, node):
        if "feature" not in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])


# ---------------------------------------------------------------------------
# Feature Importance Analyzer
# ---------------------------------------------------------------------------


class FeatureImportanceAnalyzer:
    """Compute feature importance via MDI, MDA, SFI, and CluFI.

    Parameters
    ----------
    n_splits : int
        Number of CV folds for MDA/SFI.
    n_repeats : int
        Number of permutation repeats for MDA.
    scoring : str
        Scoring function ('neg_mse', 'accuracy', 'rank_ic').
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 5,
        scoring: str = "neg_mse",
        seed: int = 42,
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _get_scorer(self):
        if self.scoring in _SCORERS:
            return _SCORERS[self.scoring]
        raise ValueError(f"Unknown scoring: {self.scoring}")

    def _cv_splits(self, n: int):
        """Generate simple k-fold indices."""
        indices = np.arange(n)
        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            test_mask = np.zeros(n, dtype=bool)
            test_mask[start:end] = True
            yield indices[~test_mask], indices[test_mask]

    # ------------------------------------------------------------------
    # MDI: Mean Decrease Impurity
    # ------------------------------------------------------------------

    def mdi(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        n_trees: int = 50,
    ) -> ImportanceResult:
        """Compute MDI feature importance using an ensemble of decision trees.

        Parameters
        ----------
        X : np.ndarray (T, N)
        y : np.ndarray (T,)
        feature_names : list[str], optional
        n_trees : int
            Number of trees in the ensemble.

        Returns
        -------
        ImportanceResult
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        all_importances = np.zeros((n_trees, n_features))

        for t in range(n_trees):
            # Bootstrap sample
            idx = self.rng.choice(len(X), size=len(X), replace=True)
            tree = _SimpleDecisionTree(max_depth=6, min_samples_leaf=5, seed=self.seed + t)
            tree.fit(X[idx], y[idx])
            all_importances[t] = tree.feature_importances_

        mean_imp = np.mean(all_importances, axis=0)
        std_imp = np.std(all_importances, axis=0)

        # Normalize
        total = mean_imp.sum()
        if total > 0:
            mean_imp /= total
            std_imp /= total

        return ImportanceResult(
            importances={f: float(mean_imp[i]) for i, f in enumerate(feature_names)},
            std={f: float(std_imp[i]) for i, f in enumerate(feature_names)},
            feature_names=feature_names,
        )

    # ------------------------------------------------------------------
    # MDA: Mean Decrease Accuracy (Permutation Importance)
    # ------------------------------------------------------------------

    def mda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        n_trees: int = 30,
    ) -> ImportanceResult:
        """Compute MDA feature importance via permutation.

        For each fold:
          1. Train model on train set.
          2. Compute baseline score on test set.
          3. For each feature, permute it and measure score decrease.

        Parameters
        ----------
        X, y : arrays
        feature_names : list[str], optional
        n_trees : int
            Trees per fold model.

        Returns
        -------
        ImportanceResult
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        scorer = self._get_scorer()
        fold_importances = []

        for train_idx, test_idx in self._cv_splits(len(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Train ensemble
            preds = np.zeros(len(X_test))
            for t in range(n_trees):
                idx = self.rng.choice(len(X_train), size=len(X_train), replace=True)
                tree = _SimpleDecisionTree(max_depth=6, seed=self.seed + t)
                tree.fit(X_train[idx], y_train[idx])
                preds += tree.predict(X_test)
            preds /= n_trees

            baseline_score = scorer(y_test, preds)
            imp = np.zeros(n_features)

            for feat in range(n_features):
                perm_scores = []
                for _ in range(self.n_repeats):
                    X_perm = X_test.copy()
                    X_perm[:, feat] = self.rng.permutation(X_perm[:, feat])
                    preds_perm = np.zeros(len(X_perm))
                    for t in range(n_trees):
                        idx = self.rng.choice(len(X_train), size=len(X_train), replace=True)
                        tree = _SimpleDecisionTree(max_depth=6, seed=self.seed + t)
                        tree.fit(X_train[idx], y_train[idx])
                        preds_perm += tree.predict(X_perm)
                    preds_perm /= n_trees
                    perm_scores.append(scorer(y_test, preds_perm))
                imp[feat] = baseline_score - np.mean(perm_scores)

            fold_importances.append(imp)

        all_imp = np.array(fold_importances)
        mean_imp = np.mean(all_imp, axis=0)
        std_imp = np.std(all_imp, axis=0)

        return ImportanceResult(
            importances={f: float(mean_imp[i]) for i, f in enumerate(feature_names)},
            std={f: float(std_imp[i]) for i, f in enumerate(feature_names)},
            feature_names=feature_names,
        )

    # ------------------------------------------------------------------
    # SFI: Single Feature Importance
    # ------------------------------------------------------------------

    def sfi(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        n_trees: int = 20,
    ) -> ImportanceResult:
        """Compute SFI by training a separate model per feature.

        Parameters
        ----------
        X, y : arrays
        feature_names : list[str], optional
        n_trees : int
            Trees per single-feature model.

        Returns
        -------
        ImportanceResult
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        scorer = self._get_scorer()
        mean_scores = np.zeros(n_features)
        std_scores = np.zeros(n_features)

        for feat in range(n_features):
            X_single = X[:, feat : feat + 1]
            fold_scores = []

            for train_idx, test_idx in self._cv_splits(len(X)):
                X_tr, y_tr = X_single[train_idx], y[train_idx]
                X_te, y_te = X_single[test_idx], y[test_idx]

                preds = np.zeros(len(X_te))
                for t in range(n_trees):
                    idx = self.rng.choice(len(X_tr), size=len(X_tr), replace=True)
                    tree = _SimpleDecisionTree(max_depth=3, seed=self.seed + t)
                    tree.fit(X_tr[idx], y_tr[idx])
                    preds += tree.predict(X_te)
                preds /= n_trees
                fold_scores.append(scorer(y_te, preds))

            mean_scores[feat] = np.mean(fold_scores)
            std_scores[feat] = np.std(fold_scores)

        return ImportanceResult(
            importances={f: float(mean_scores[i]) for i, f in enumerate(feature_names)},
            std={f: float(std_scores[i]) for i, f in enumerate(feature_names)},
            feature_names=feature_names,
        )

    # ------------------------------------------------------------------
    # CluFI: Clustered Feature Importance
    # ------------------------------------------------------------------

    def clustered_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        n_clusters: int = 5,
        n_trees: int = 30,
    ) -> ClusteredImportanceResult:
        """Compute clustered feature importance (CluFI).

        Groups correlated features into clusters, then measures
        importance at the cluster level to avoid dilution.

        Parameters
        ----------
        X, y : arrays
        feature_names : list[str], optional
        n_clusters : int
            Number of feature clusters.
        n_trees : int
            Trees for importance estimation.

        Returns
        -------
        ClusteredImportanceResult
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]

        n_clusters = min(n_clusters, n_features)

        # Cluster features by correlation distance
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)
        dist = (dist + dist.T) / 2

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        # Build cluster membership
        cluster_members = {}
        feature_to_cluster = {}
        for i, f in enumerate(feature_names):
            c = int(labels[i])
            feature_to_cluster[f] = c
            cluster_members.setdefault(c, []).append(f)

        # Get MDI importance for all features
        mdi_result = self.mdi(X, y, feature_names, n_trees=n_trees)

        # Aggregate importance per cluster
        cluster_importance = {}
        representative_features = {}
        for c, members in cluster_members.items():
            member_imp = [(f, mdi_result.importances[f]) for f in members]
            cluster_importance[c] = sum(imp for _, imp in member_imp)
            representative_features[c] = max(member_imp, key=lambda x: x[1])[0]

        # Silhouette score (simplified)
        sil = self._silhouette(dist, labels)

        return ClusteredImportanceResult(
            cluster_importance=cluster_importance,
            cluster_members=cluster_members,
            feature_to_cluster=feature_to_cluster,
            representative_features=representative_features,
            silhouette_score=sil,
        )

    def _silhouette(self, dist: np.ndarray, labels: np.ndarray) -> float:
        """Compute mean silhouette score."""
        n = len(labels)
        if n < 2:
            return 0.0
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        scores = np.zeros(n)
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            if same.sum() == 0:
                scores[i] = 0.0
                continue
            a = np.mean(dist[i, same])
            b = float("inf")
            for c in unique_labels:
                if c == labels[i]:
                    continue
                other = labels == c
                if other.sum() > 0:
                    b = min(b, np.mean(dist[i, other]))
            scores[i] = (b - a) / max(a, b, 1e-12)

        return float(np.mean(scores))
