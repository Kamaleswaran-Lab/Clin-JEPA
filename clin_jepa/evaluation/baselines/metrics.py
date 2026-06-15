"""Metric utilities for the baseline evaluation.

The MASE denominator uses the last-observed value (at hour C-1), not
``y_true[:, 0]`` (the h=1 future target), to avoid leaking the future into the
copy-forward reference.
"""
from __future__ import annotations

import numpy as np


def stratified_bootstrap_mae(
    preds: np.ndarray, y_true: np.ndarray, stay_ids: np.ndarray,
    n_boot: int = 500, seed: int = 42,
) -> tuple[float, tuple[float, float], float]:
    """Stay-clustered bootstrap MAE 95% CI."""
    rng = np.random.RandomState(seed)
    valid = np.isfinite(preds) & np.isfinite(y_true)
    if valid.sum() < 30:
        return float('nan'), (float('nan'), float('nan')), float('nan')
    p = preds[valid]; y = y_true[valid]; s = stay_ids[valid]
    point = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    unique_stays = np.unique(s)
    boot = []
    stay_to_idx = {st: np.where(s == st)[0] for st in unique_stays}
    for _ in range(n_boot):
        ss = rng.choice(unique_stays, size=len(unique_stays), replace=True)
        idx = np.concatenate([stay_to_idx[st] for st in ss])
        boot.append(float(np.mean(np.abs(p[idx] - y[idx]))))
    return point, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), rmse


def stratified_bootstrap_auroc(
    scores: np.ndarray, y_true: np.ndarray, stay_ids: np.ndarray,
    n_boot: int = 500, seed: int = 42,
) -> tuple[float, tuple[float, float], float]:
    """Stay-clustered bootstrap AUROC 95% CI + AUPRC."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.RandomState(seed)
    valid = np.isfinite(scores) & np.isfinite(y_true)
    if valid.sum() < 30 or len(np.unique(y_true[valid])) < 2:
        return float('nan'), (float('nan'), float('nan')), float('nan')
    p = scores[valid]; y = y_true[valid]; s = stay_ids[valid]
    point = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    unique_stays = np.unique(s)
    boot = []
    stay_to_idx = {st: np.where(s == st)[0] for st in unique_stays}
    for _ in range(n_boot):
        ss = rng.choice(unique_stays, size=len(unique_stays), replace=True)
        idx = np.concatenate([stay_to_idx[st] for st in ss])
        if y[idx].sum() < 5 or (y[idx] == 0).sum() < 5:
            continue
        try:
            boot.append(float(roc_auc_score(y[idx], p[idx])))
        except ValueError:
            continue
    if len(boot) < 50:
        return point, (float('nan'), float('nan')), auprc
    return point, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), auprc


def compute_mase(
    preds: np.ndarray, y_true: np.ndarray, y_last_observed: np.ndarray,
) -> float:
    """MASE = MAE_model / MAE_naive_lastobs.

    The naive baseline is the last-observed value (at context-end), not
    ``y_true[:, 0]`` which would leak the h=1 future ground truth.
    """
    valid = np.isfinite(preds) & np.isfinite(y_true) & np.isfinite(y_last_observed)
    if valid.sum() < 30:
        return float('nan')
    p = preds[valid]; y = y_true[valid]; y_lo = y_last_observed[valid]
    mae_model = float(np.mean(np.abs(p - y)))
    mae_naive = float(np.mean(np.abs(y_lo - y)))
    if mae_naive < 1e-9:
        return float('nan')
    return mae_model / mae_naive
