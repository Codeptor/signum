"""Shrinkage covariance estimators for portfolio optimization.

Implements structured covariance estimation without sklearn dependency:
  1. Ledoit-Wolf (2004): Optimal shrinkage toward scaled identity.
  2. Oracle Approximating Shrinkage (OAS, Chen et al. 2010): Improved shrinkage
     that converges faster with fewer observations.
  3. Constant-correlation shrinkage: Target is equicorrelated matrix.
  4. Denoised covariance via Marchenko-Pastur random matrix theory.

These estimators reduce estimation error in the covariance matrix used by
HRP, minimum-variance, and risk-parity optimizers. Sample covariance is
notoriously noisy for portfolio optimization — shrinkage toward a
structured target produces more stable and out-of-sample reliable weights.

Usage::

    from python.portfolio.covariance import ledoit_wolf, oas_shrinkage

    returns = ...  # (T, N) array of asset returns
    cov, shrinkage = ledoit_wolf(returns)
    cov_oas, alpha_oas = oas_shrinkage(returns)

References:
  - Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional
    covariance matrices"
  - Chen, Wiesel, Eldar & Hero (2010), "Shrinkage Algorithms for MMSE
    Covariance Estimation"
  - Marchenko & Pastur (1967), distribution of eigenvalues of random matrices
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _sample_covariance(X: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix with 1/(n-1) normalization."""
    n = X.shape[0]
    X_centered = X - X.mean(axis=0, keepdims=True)
    return (X_centered.T @ X_centered) / max(n - 1, 1)


# ---------------------------------------------------------------------------
# Ledoit-Wolf Shrinkage
# ---------------------------------------------------------------------------


def ledoit_wolf(
    returns: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf optimal shrinkage toward scaled identity target.

    Computes the optimal shrinkage intensity analytically using the
    formula from Ledoit & Wolf (2004). The target is mu * I where
    mu = trace(S) / p (average eigenvalue).

    Parameters
    ----------
    returns : np.ndarray (T, N)
        Asset return matrix.

    Returns
    -------
    tuple[np.ndarray, float]
        (shrunk_covariance, shrinkage_intensity)
        where shrinkage_intensity in [0, 1].
    """
    X = np.asarray(returns, dtype=float)
    n, p = X.shape

    if n < 2 or p < 2:
        S = _sample_covariance(X)
        return S, 0.0

    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    S = (X_c.T @ X_c) / (n - 1)

    # Target: scaled identity (mu * I)
    mu = np.trace(S) / p
    target = mu * np.eye(p)

    # Compute optimal shrinkage intensity
    # delta = S - target
    delta = S - target

    # Sum of squared Frobenius norms needed for the formula
    # ||S||_F^2
    sum_S2 = np.sum(S**2)

    # Estimate of asymptotic squared loss of sample covariance
    # using the formula: beta = 1/(n^2) * sum_k ||x_k x_k^T - S||_F^2
    X_outer_sum = 0.0
    for k in range(n):
        x_k = X_c[k : k + 1]  # (1, p)
        outer = x_k.T @ x_k  # (p, p)
        diff = outer - S
        X_outer_sum += np.sum(diff**2)
    beta_hat = X_outer_sum / (n**2)

    # ||delta||_F^2
    delta_sq = np.sum(delta**2)

    # Optimal shrinkage
    if delta_sq > 0:
        alpha = min(beta_hat / delta_sq, 1.0)
    else:
        alpha = 1.0

    alpha = max(0.0, min(alpha, 1.0))

    shrunk = alpha * target + (1 - alpha) * S

    logger.debug(f"Ledoit-Wolf: shrinkage={alpha:.4f} (n={n}, p={p})")
    return shrunk, float(alpha)


# ---------------------------------------------------------------------------
# Oracle Approximating Shrinkage (OAS)
# ---------------------------------------------------------------------------


def oas_shrinkage(
    returns: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage estimator (Chen et al. 2010).

    More accurate than Ledoit-Wolf for small samples. Uses an improved
    formula that better approximates the oracle shrinkage.

    Parameters
    ----------
    returns : np.ndarray (T, N)

    Returns
    -------
    tuple[np.ndarray, float]
        (shrunk_covariance, shrinkage_intensity)
    """
    X = np.asarray(returns, dtype=float)
    n, p = X.shape

    if n < 2 or p < 2:
        S = _sample_covariance(X)
        return S, 0.0

    X_c = X - X.mean(axis=0, keepdims=True)
    S = (X_c.T @ X_c) / (n - 1)

    # Target: scaled identity
    mu = np.trace(S) / p

    # OAS formula (Chen et al. 2010, Theorem 1)
    # rho_hat = (1 - 2/p) * tr(S^2) + tr(S)^2
    tr_S = np.trace(S)
    tr_S2 = np.sum(S**2)

    rho_numerator = (1.0 - 2.0 / p) * tr_S2 + tr_S**2
    rho_denominator = (n + 1.0 - 2.0 / p) * (tr_S2 - tr_S**2 / p)

    if abs(rho_denominator) < 1e-16:
        alpha = 1.0
    else:
        alpha = rho_numerator / rho_denominator
        alpha = max(0.0, min(alpha, 1.0))

    target = mu * np.eye(p)
    shrunk = alpha * target + (1 - alpha) * S

    logger.debug(f"OAS: shrinkage={alpha:.4f} (n={n}, p={p})")
    return shrunk, float(alpha)


# ---------------------------------------------------------------------------
# Constant-Correlation Shrinkage
# ---------------------------------------------------------------------------


def constant_correlation_shrinkage(
    returns: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Shrinkage toward constant-correlation target (Ledoit & Wolf 2004).

    Target is the equicorrelated matrix where all off-diagonal
    correlations are set to the average pairwise correlation.

    Parameters
    ----------
    returns : np.ndarray (T, N)

    Returns
    -------
    tuple[np.ndarray, float]
        (shrunk_covariance, shrinkage_intensity)
    """
    X = np.asarray(returns, dtype=float)
    n, p = X.shape

    if n < 2 or p < 2:
        S = _sample_covariance(X)
        return S, 0.0

    X_c = X - X.mean(axis=0, keepdims=True)
    S = (X_c.T @ X_c) / (n - 1)

    # Compute correlation matrix
    stds = np.sqrt(np.diag(S))
    stds_safe = np.where(stds > 0, stds, 1.0)
    corr = S / np.outer(stds_safe, stds_safe)

    # Average off-diagonal correlation
    mask = ~np.eye(p, dtype=bool)
    r_bar = corr[mask].mean() if p > 1 else 0.0

    # Target: constant-correlation matrix
    target = r_bar * np.outer(stds, stds)
    np.fill_diagonal(target, np.diag(S))

    # Compute optimal shrinkage
    delta = S - target

    # Estimate beta using leave-one-out-like formula
    X_outer_sum = 0.0
    for k in range(n):
        x_k = X_c[k : k + 1]
        outer = x_k.T @ x_k
        diff = outer - S
        X_outer_sum += np.sum(diff**2)
    beta_hat = X_outer_sum / (n**2)

    delta_sq = np.sum(delta**2)

    if delta_sq > 0:
        alpha = min(beta_hat / delta_sq, 1.0)
    else:
        alpha = 1.0

    alpha = max(0.0, min(alpha, 1.0))
    shrunk = alpha * target + (1 - alpha) * S

    logger.debug(f"Const-corr shrinkage: alpha={alpha:.4f}, r_bar={r_bar:.4f}")
    return shrunk, float(alpha)


# ---------------------------------------------------------------------------
# Marchenko-Pastur Denoising
# ---------------------------------------------------------------------------


def denoise_covariance(
    returns: np.ndarray,
    n_factors: int | None = None,
) -> np.ndarray:
    """Denoise covariance matrix using random matrix theory.

    Applies the Marchenko-Pastur distribution to identify eigenvalues
    consistent with random noise, then shrinks them toward the mean
    noise eigenvalue. Signal eigenvalues are preserved.

    Parameters
    ----------
    returns : np.ndarray (T, N)
    n_factors : int, optional
        Number of significant factors to retain. If None, determined
        by Marchenko-Pastur upper bound.

    Returns
    -------
    np.ndarray
        Denoised covariance matrix.
    """
    X = np.asarray(returns, dtype=float)
    n, p = X.shape

    S = _sample_covariance(X)

    if n < 2 or p < 2:
        return S

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    idx = np.argsort(eigenvalues)[::-1]  # Descending
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marchenko-Pastur bounds
    q = p / n  # Ratio
    sigma2 = np.median(eigenvalues)  # Noise variance estimate
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2  # Upper MP bound

    if n_factors is None:
        # Count eigenvalues above MP upper bound
        n_factors = int(np.sum(eigenvalues > lambda_plus))
        n_factors = max(1, n_factors)

    # Denoise: shrink noise eigenvalues toward their mean
    noise_eigs = eigenvalues[n_factors:]
    noise_mean = noise_eigs.mean() if len(noise_eigs) > 0 else 0.0

    denoised_eigs = eigenvalues.copy()
    denoised_eigs[n_factors:] = noise_mean

    # Ensure positive definite
    denoised_eigs = np.maximum(denoised_eigs, 1e-10)

    # Reconstruct
    denoised = eigenvectors @ np.diag(denoised_eigs) @ eigenvectors.T

    # Symmetrize (numerical safety)
    denoised = (denoised + denoised.T) / 2

    logger.debug(
        f"Denoised covariance: {n_factors} signal factors, "
        f"MP upper bound={lambda_plus:.6f}"
    )
    return denoised


# ---------------------------------------------------------------------------
# Convenience: best estimator
# ---------------------------------------------------------------------------


def shrink_covariance(
    returns: np.ndarray,
    method: str = "oas",
) -> tuple[np.ndarray, float]:
    """Compute shrunk covariance matrix using the specified method.

    Parameters
    ----------
    returns : np.ndarray (T, N)
    method : str
        One of 'ledoit_wolf', 'oas', 'constant_correlation', 'denoise'.

    Returns
    -------
    tuple[np.ndarray, float]
        (covariance, shrinkage_intensity)
    """
    if method == "ledoit_wolf":
        return ledoit_wolf(returns)
    elif method == "oas":
        return oas_shrinkage(returns)
    elif method == "constant_correlation":
        return constant_correlation_shrinkage(returns)
    elif method == "denoise":
        cov = denoise_covariance(returns)
        return cov, 0.0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ledoit_wolf', 'oas', 'constant_correlation', or 'denoise'.")
