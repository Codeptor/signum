"""Tests for shrinkage covariance estimators."""

import numpy as np
import pytest

from python.portfolio.covariance import (
    constant_correlation_shrinkage,
    denoise_covariance,
    ledoit_wolf,
    oas_shrinkage,
    shrink_covariance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n=500, p=10, seed=42):
    """Generate synthetic return matrix with known factor structure."""
    rng = np.random.default_rng(seed)
    # 2-factor model + idiosyncratic noise
    factors = rng.standard_normal((n, 2))
    loadings = rng.standard_normal((2, p)) * 0.5
    noise = rng.standard_normal((n, p)) * 0.3
    return factors @ loadings + noise


def _is_positive_definite(matrix):
    """Check if a matrix is positive definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues > -1e-10)


def _is_symmetric(matrix, tol=1e-10):
    return np.allclose(matrix, matrix.T, atol=tol)


# ---------------------------------------------------------------------------
# Ledoit-Wolf
# ---------------------------------------------------------------------------


class TestLedoitWolf:
    def test_output_shape(self):
        X = _make_returns(n=100, p=5)
        cov, alpha = ledoit_wolf(X)
        assert cov.shape == (5, 5)

    def test_shrinkage_bounded(self):
        X = _make_returns()
        _, alpha = ledoit_wolf(X)
        assert 0.0 <= alpha <= 1.0

    def test_positive_definite(self):
        X = _make_returns()
        cov, _ = ledoit_wolf(X)
        assert _is_positive_definite(cov)

    def test_symmetric(self):
        X = _make_returns()
        cov, _ = ledoit_wolf(X)
        assert _is_symmetric(cov)

    def test_diagonal_positive(self):
        X = _make_returns()
        cov, _ = ledoit_wolf(X)
        assert np.all(np.diag(cov) > 0)

    def test_more_shrinkage_with_fewer_samples(self):
        """Fewer observations → more shrinkage needed."""
        # Use data with factor structure so true cov != identity
        X_large = _make_returns(n=500, p=10, seed=42)
        X_small = _make_returns(n=30, p=10, seed=42)
        _, alpha_large = ledoit_wolf(X_large)
        _, alpha_small = ledoit_wolf(X_small)
        # Small sample needs more shrinkage (both < 1)
        assert alpha_small > alpha_large

    def test_identity_like_for_independent(self):
        """Independent returns → shrunk cov should be near diagonal."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 5))
        cov, alpha = ledoit_wolf(X)
        # Off-diagonals should be small relative to diagonals
        diag = np.diag(cov)
        off_diag = cov - np.diag(diag)
        assert np.mean(np.abs(off_diag)) < np.mean(diag) * 0.5


# ---------------------------------------------------------------------------
# OAS
# ---------------------------------------------------------------------------


class TestOAS:
    def test_output_shape(self):
        X = _make_returns(n=100, p=5)
        cov, alpha = oas_shrinkage(X)
        assert cov.shape == (5, 5)

    def test_shrinkage_bounded(self):
        X = _make_returns()
        _, alpha = oas_shrinkage(X)
        assert 0.0 <= alpha <= 1.0

    def test_positive_definite(self):
        X = _make_returns()
        cov, _ = oas_shrinkage(X)
        assert _is_positive_definite(cov)

    def test_symmetric(self):
        X = _make_returns()
        cov, _ = oas_shrinkage(X)
        assert _is_symmetric(cov)

    def test_different_from_ledoit_wolf(self):
        """OAS should use a different formula than LW, producing different covariances."""
        X = _make_returns(n=50, p=10)
        cov_lw, alpha_lw = ledoit_wolf(X)
        cov_oas, alpha_oas = oas_shrinkage(X)
        # Both are valid shrinkage estimators — different alphas (may be close)
        # The covariance matrices should differ even if alphas are similar
        diff = np.max(np.abs(cov_lw - cov_oas))
        assert diff > 0  # Not exactly the same


# ---------------------------------------------------------------------------
# Constant Correlation
# ---------------------------------------------------------------------------


class TestConstantCorrelation:
    def test_output_shape(self):
        X = _make_returns(n=100, p=5)
        cov, alpha = constant_correlation_shrinkage(X)
        assert cov.shape == (5, 5)

    def test_shrinkage_bounded(self):
        X = _make_returns()
        _, alpha = constant_correlation_shrinkage(X)
        assert 0.0 <= alpha <= 1.0

    def test_positive_definite(self):
        X = _make_returns()
        cov, _ = constant_correlation_shrinkage(X)
        assert _is_positive_definite(cov)

    def test_preserves_variances(self):
        """Constant-corr target preserves diagonal elements."""
        X = _make_returns()
        cov, alpha = constant_correlation_shrinkage(X)
        sample_cov = np.cov(X, rowvar=False)
        # Diagonal should be close to sample diagonal (shrinkage doesn't change variances)
        np.testing.assert_allclose(np.diag(cov), np.diag(sample_cov), rtol=0.3)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


class TestDenoising:
    def test_output_shape(self):
        X = _make_returns(n=100, p=5)
        cov = denoise_covariance(X)
        assert cov.shape == (5, 5)

    def test_positive_definite(self):
        X = _make_returns()
        cov = denoise_covariance(X)
        assert _is_positive_definite(cov)

    def test_symmetric(self):
        X = _make_returns()
        cov = denoise_covariance(X)
        assert _is_symmetric(cov)

    def test_explicit_factors(self):
        X = _make_returns()
        cov = denoise_covariance(X, n_factors=2)
        assert _is_positive_definite(cov)

    def test_fewer_noise_eigenvalues(self):
        """Denoised matrix should have less eigenvalue dispersion in noise."""
        X = _make_returns(n=200, p=20)
        sample = np.cov(X, rowvar=False)
        denoised = denoise_covariance(X)
        # Bottom eigenvalues should be more equal after denoising
        eig_sample = np.sort(np.linalg.eigvalsh(sample))
        eig_denoised = np.sort(np.linalg.eigvalsh(denoised))
        noise_std_sample = np.std(eig_sample[:15])
        noise_std_denoised = np.std(eig_denoised[:15])
        assert noise_std_denoised < noise_std_sample


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestShrinkCovariance:
    def test_all_methods(self):
        X = _make_returns(n=100, p=5)
        for method in ["ledoit_wolf", "oas", "constant_correlation", "denoise"]:
            cov, alpha = shrink_covariance(X, method=method)
            assert cov.shape == (5, 5)

    def test_invalid_method(self):
        X = _make_returns()
        with pytest.raises(ValueError, match="Unknown method"):
            shrink_covariance(X, method="bad")


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_asset(self):
        """Single asset should return scalar-like covariance."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 1))
        cov, alpha = ledoit_wolf(X)
        assert cov.shape == (1, 1)

    def test_two_assets(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))
        cov, alpha = ledoit_wolf(X)
        assert cov.shape == (2, 2)
        assert _is_positive_definite(cov)

    def test_very_short_series(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 3))
        cov, alpha = ledoit_wolf(X)
        assert cov.shape == (3, 3)

    def test_highly_correlated(self):
        """Highly correlated assets should still produce valid covariance."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(200)
        X = np.column_stack([
            base + rng.standard_normal(200) * 0.01,
            base + rng.standard_normal(200) * 0.01,
            base + rng.standard_normal(200) * 0.01,
        ])
        cov, alpha = ledoit_wolf(X)
        assert _is_positive_definite(cov)
        cov_oas, _ = oas_shrinkage(X)
        assert _is_positive_definite(cov_oas)
