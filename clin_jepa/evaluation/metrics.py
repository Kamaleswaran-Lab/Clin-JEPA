"""Evaluation metrics + clustered bootstrap CIs.

Provides:
- Basic metrics: MAE, RMSE, MASE, Spearman rho, R squared
- Clustered bootstrap confidence intervals (resample by stay_id)
- Paired bootstrap for model comparisons

"""

import logging
from typing import Callable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error, skipping NaN values."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(y_pred[valid] - y_true[valid])))


def compute_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error, skipping NaN values."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_pred[valid] - y_true[valid]) ** 2)))


def compute_mase(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_baseline: np.ndarray,
) -> float:
    """Mean Absolute Scaled Error: model MAE / baseline MAE.

    MASE < 1.0 means the model beats the baseline.
    Standard metric in forecasting literature (Hyndman & Koehler 2006).

    Args:
        y_pred: Model predictions.
        y_true: Ground truth.
        y_baseline: Baseline predictions (e.g., copy-forward).

    Returns:
        MASE value (NaN if baseline MAE is zero or no valid samples).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_baseline = np.asarray(y_baseline, dtype=np.float64)

    # Shared valid mask: require all three to be non-NaN.
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true) & ~np.isnan(y_baseline)
    if not valid.any():
        return float("nan")

    mae_model = float(np.mean(np.abs(y_pred[valid] - y_true[valid])))
    mae_baseline = float(np.mean(np.abs(y_baseline[valid] - y_true[valid])))

    if mae_baseline <= 0:
        return float("nan")
    return mae_model / mae_baseline


def compute_spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank correlation, skipping NaN values."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if valid.sum() < 3:
        return float("nan")
    rho, _ = stats.spearmanr(y_pred[valid], y_true[valid])
    return float(rho) if not np.isnan(rho) else float("nan")


def compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination (R squared).

    Returns fraction of variance explained by predictions.
    R2 = 1 - SS_res / SS_tot
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if valid.sum() < 2:
        return float("nan")

    pv = y_pred[valid]
    tv = y_true[valid]
    ss_res = float(np.sum((tv - pv) ** 2))
    ss_tot = float(np.sum((tv - np.mean(tv)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Clustered bootstrap CI (resample by stay_id to respect data dependence)
# ---------------------------------------------------------------------------

def bootstrap_ci_clustered(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    stay_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Clustered bootstrap confidence interval.

    Resamples stay_ids with replacement to respect within-patient
    dependence (multiple timesteps per patient are not independent).
    This gives wider, more honest CIs than naive row-level bootstrap.

    Args:
        y_pred: (N,) model predictions.
        y_true: (N,) ground truth.
        stay_ids: (N,) stay_id for each sample (int).
        metric_fn: Function taking (y_pred, y_true) and returning a float metric.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        confidence: Confidence level (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    stay_ids = np.asarray(stay_ids)

    assert len(y_pred) == len(y_true) == len(stay_ids), "array length mismatch"

    # Point estimate on full data
    point_est = metric_fn(y_pred, y_true)

    if np.isnan(point_est) or len(y_pred) == 0:
        return point_est, float("nan"), float("nan")

    # Build sample-to-stay lookup
    unique_stays = np.unique(stay_ids)
    n_stays = len(unique_stays)
    if n_stays < 2:
        # Not enough stays to bootstrap meaningfully
        return point_est, float("nan"), float("nan")

    # Group sample indices by stay_id for efficient resampling
    stay_to_indices: dict[int, np.ndarray] = {}
    for sid in unique_stays:
        stay_to_indices[int(sid)] = np.where(stay_ids == sid)[0]

    rng = np.random.default_rng(seed)
    boot_values = []

    for _ in range(n_bootstrap):
        # Resample stays with replacement
        sampled_stays = rng.choice(unique_stays, size=n_stays, replace=True)
        # Gather all sample indices for the resampled stays
        indices_list = [stay_to_indices[int(s)] for s in sampled_stays]
        indices = np.concatenate(indices_list)

        # Compute metric on resampled data
        boot_val = metric_fn(y_pred[indices], y_true[indices])
        if not np.isnan(boot_val):
            boot_values.append(boot_val)

    if len(boot_values) < 10:
        return point_est, float("nan"), float("nan")

    alpha = 1.0 - confidence
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_values, lo_pct))
    ci_upper = float(np.percentile(boot_values, hi_pct))

    return point_est, ci_lower, ci_upper


def bootstrap_ci_mae_clustered(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    stay_ids: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Convenience wrapper for MAE with clustered bootstrap CI."""
    return bootstrap_ci_clustered(
        y_pred, y_true, stay_ids,
        metric_fn=compute_mae,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


def bootstrap_ci_mase_clustered(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_baseline: np.ndarray,
    stay_ids: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Clustered bootstrap CI for MASE (requires three arrays).

    Computes MASE = MAE_model / MAE_baseline on each bootstrap resample.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_baseline = np.asarray(y_baseline)
    stay_ids = np.asarray(stay_ids)

    n = len(y_pred)
    assert len(y_true) == n and len(y_baseline) == n and len(stay_ids) == n

    # Point estimate
    point_est = compute_mase(y_pred, y_true, y_baseline)
    if np.isnan(point_est):
        return point_est, float("nan"), float("nan")

    unique_stays = np.unique(stay_ids)
    n_stays = len(unique_stays)
    if n_stays < 2:
        return point_est, float("nan"), float("nan")

    stay_to_indices: dict[int, np.ndarray] = {}
    for sid in unique_stays:
        stay_to_indices[int(sid)] = np.where(stay_ids == sid)[0]

    rng = np.random.default_rng(seed)
    boot_values = []

    for _ in range(n_bootstrap):
        sampled_stays = rng.choice(unique_stays, size=n_stays, replace=True)
        indices = np.concatenate([stay_to_indices[int(s)] for s in sampled_stays])
        val = compute_mase(y_pred[indices], y_true[indices], y_baseline[indices])
        if not np.isnan(val):
            boot_values.append(val)

    if len(boot_values) < 10:
        return point_est, float("nan"), float("nan")

    alpha = 1.0 - confidence
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_values, lo_pct))
    ci_upper = float(np.percentile(boot_values, hi_pct))

    return point_est, ci_lower, ci_upper
