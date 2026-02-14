"""Bridge between ML model predictions and Black-Litterman views.

Converts ML return predictions + confidence scores into the views format
expected by portfolio optimization (skfolio / Riskfolio-Lib).
"""

import numpy as np
import pandas as pd


def create_bl_views(
    predictions: pd.Series,
    confidences: pd.Series,
    tau: float = 0.05,
    base_uncertainty: float = 0.1,
) -> tuple[pd.Series, pd.Series]:
    """Convert ML predictions to Black-Litterman absolute views.

    Args:
        predictions: Expected returns per ticker from ML model.
        confidences: Confidence scores per ticker (0-1 scale).
        tau: Scaling factor for view uncertainty (smaller = more certain).
        base_uncertainty: Base uncertainty when confidence is 0.

    Returns:
        views: Predicted returns (same as input predictions).
        view_confidences: Uncertainty per view (lower = more confident).
    """
    # Convert confidence (0=uncertain, 1=certain) to uncertainty (high=uncertain)
    # Inverse mapping: high confidence → low uncertainty
    uncertainties = base_uncertainty * (1 - confidences) * tau + 1e-6

    return predictions, uncertainties


def create_picking_matrix(
    tickers: list[str],
    views: pd.Series,
) -> np.ndarray:
    """Create the picking matrix P for Black-Litterman.

    For absolute views (each view is about one asset), P is an identity matrix
    over the assets with views.
    """
    n_views = len(views)
    n_assets = len(tickers)
    picking = np.zeros((n_views, n_assets))
    for i, ticker in enumerate(views.index):
        j = tickers.index(ticker)
        picking[i, j] = 1.0
    return picking
