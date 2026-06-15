"""Classical baselines: Ridge regression and LightGBM.

Continuous targets: one Ridge and one LightGBM regressor per
(variable, horizon, aggregator). Binary outcomes: one logistic-regression
and one LGBMClassifier per outcome.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from clin_jepa.evaluation.baselines.features import (
    BaselineDataset, AGGREGATORS, REPORTED_HORIZONS, impute_and_normalize,
    flatten_features_for_classical,
)

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not available — only Ridge baseline will work")

try:
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============= Ridge regression baseline =============

def _fit_ridge_one(X_tr, y_tr, X_vl, y_vl, alphas=(1.0,)) -> tuple:
    """Fit closed-form Ridge: (X^T X + αI) w = X^T y with X centered. Returns
    (model, alpha, val_mse). With multiple alphas, picks the best by val MSE.
    """
    valid_tr = np.isfinite(y_tr)
    valid_vl = np.isfinite(y_vl)
    if valid_tr.sum() < 30 or valid_vl.sum() < 10:
        return None, None, None

    X_tr_v = X_tr[valid_tr]; y_tr_v = y_tr[valid_tr]
    X_vl_v = X_vl[valid_vl]; y_vl_v = y_vl[valid_vl]

    # Normalize y
    y_mean = float(y_tr_v.mean()); y_std = float(y_tr_v.std() + 1e-6)
    y_tr_n = (y_tr_v - y_mean) / y_std

    # Center X
    x_mean = X_tr_v.mean(axis=0)
    Xc = X_tr_v - x_mean[None, :]

    best_alpha = alphas[0]; best_val_mse = float('inf'); best_w = None
    XtX = Xc.T @ Xc
    Xty = Xc.T @ y_tr_n
    n_features = Xc.shape[1]

    for alpha in alphas:
        try:
            w = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
        except np.linalg.LinAlgError:
            continue
        # Validate
        pred_val_n = (X_vl_v - x_mean[None, :]) @ w
        pred_val = pred_val_n * y_std + y_mean
        val_mse = float(np.mean((pred_val - y_vl_v) ** 2))
        if val_mse < best_val_mse:
            best_alpha = alpha
            best_val_mse = val_mse
            best_w = w

    if best_w is None:
        return None, None, None

    model_dict = {
        'w': best_w,
        'x_mean': x_mean,
        'y_mean': y_mean,
        'y_std': y_std,
        'alpha': best_alpha,
    }
    return model_dict, best_alpha, best_val_mse


def _predict_ridge_one(X_test, model_dict):
    if model_dict is None:
        return np.full(X_test.shape[0], np.nan, dtype=np.float32)
    pred_n = (X_test - model_dict['x_mean'][None, :]) @ model_dict['w']
    return (pred_n * model_dict['y_std'] + model_dict['y_mean']).astype(np.float32)


def train_ridge_continuous(
    train_ds: BaselineDataset,
    val_ds: BaselineDataset,
    test_ds: BaselineDataset,
    target_vars: list[str],
    horizons: list[int] = None,
    aggregators: list[str] = None,
) -> dict:
    """Train Ridge per (var, h, agg). Returns dict of predictions per (var, h, agg)."""
    horizons = horizons or REPORTED_HORIZONS
    aggregators = aggregators or AGGREGATORS

    X_seq_tr, X_static_tr = impute_and_normalize(train_ds, normalize_dyn=False, normalize_stat=False)
    X_seq_vl, X_static_vl = impute_and_normalize(val_ds,   normalize_dyn=False, normalize_stat=False)
    X_seq_te, X_static_te = impute_and_normalize(test_ds,  normalize_dyn=False, normalize_stat=False)

    # Statics: normalize via train stats
    X_static_tr_n = (X_static_tr - train_ds.var_mean_stat[None, :]) / train_ds.var_std_stat[None, :]
    X_static_vl_n = (X_static_vl - train_ds.var_mean_stat[None, :]) / train_ds.var_std_stat[None, :]
    X_static_te_n = (X_static_te - train_ds.var_mean_stat[None, :]) / train_ds.var_std_stat[None, :]

    X_tr, _ = flatten_features_for_classical(X_seq_tr, X_static_tr_n, feature_mode='summary')
    X_vl, _ = flatten_features_for_classical(X_seq_vl, X_static_vl_n, feature_mode='summary')
    X_te, _ = flatten_features_for_classical(X_seq_te, X_static_te_n, feature_mode='summary')

    logger.info(f"  Ridge feature dim: {X_tr.shape[1]} (summary mode)")

    predictions = {}  # (var, h, agg) -> (N_test,) prediction in original units
    val_mse_log = {}
    alphas_log = {}

    n_total = 0
    n_skipped = 0
    for var in target_vars:
        for h in horizons:
            for agg in aggregators:
                if agg in ('mean', 'std') and h == 1:
                    continue
                from clin_jepa.evaluation.baselines.features import SOFA_TOTAL_ONLY_H24, SPARSE_VARS_SKIP_H1
                if var in SOFA_TOTAL_ONLY_H24 and h != 24:
                    continue
                if var in SPARSE_VARS_SKIP_H1 and h == 1:
                    continue

                y_tr = train_ds.y_continuous[var][agg][:, h - 1]
                y_vl = val_ds.y_continuous[var][agg][:, h - 1]

                model, alpha, val_mse = _fit_ridge_one(X_tr, y_tr, X_vl, y_vl)
                pred_test = _predict_ridge_one(X_te, model) if model is not None else np.full(
                    X_te.shape[0], np.nan, dtype=np.float32)
                predictions[(var, h, agg)] = pred_test
                val_mse_log[(var, h, agg)] = val_mse
                alphas_log[(var, h, agg)] = alpha
                if model is None:
                    n_skipped += 1
                n_total += 1

    logger.info(f"  Ridge done: {n_total - n_skipped}/{n_total} probes trained")
    return {
        'predictions': predictions,
        'val_mse_log': val_mse_log,
        'alphas_log': alphas_log,
        'n_train': X_tr.shape[0], 'n_val': X_vl.shape[0], 'n_test': X_te.shape[0],
    }


def train_ridge_binary(
    train_ds: BaselineDataset,
    val_ds: BaselineDataset,
    test_ds: BaselineDataset,
    outcomes: list[str],
) -> dict:
    """Train Logistic Regression per binary outcome (Ridge classifier in scikit-learn = LR with penalty='l2')."""
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn required for binary Ridge")

    X_seq_tr, X_static_tr = impute_and_normalize(train_ds, normalize_dyn=False, normalize_stat=False)
    X_seq_vl, X_static_vl = impute_and_normalize(val_ds, normalize_dyn=False, normalize_stat=False)
    X_seq_te, X_static_te = impute_and_normalize(test_ds, normalize_dyn=False, normalize_stat=False)

    X_static_tr_n = (X_static_tr - train_ds.var_mean_stat[None, :]) / train_ds.var_std_stat[None, :]
    X_static_vl_n = (X_static_vl - train_ds.var_mean_stat[None, :]) / train_ds.var_std_stat[None, :]
    X_static_te_n = (X_static_te - train_ds.var_mean_stat[None, :]) / train_ds.var_std_stat[None, :]

    X_tr, _ = flatten_features_for_classical(X_seq_tr, X_static_tr_n, feature_mode='summary')
    X_vl, _ = flatten_features_for_classical(X_seq_vl, X_static_vl_n, feature_mode='summary')
    X_te, _ = flatten_features_for_classical(X_seq_te, X_static_te_n, feature_mode='summary')

    predictions = {}
    metadata = {}
    for outcome in outcomes:
        y_tr_full = train_ds.y_binary[outcome]
        y_vl_full = val_ds.y_binary[outcome]
        valid_tr = np.isfinite(y_tr_full)
        valid_vl = np.isfinite(y_vl_full)
        if valid_tr.sum() < 30 or len(np.unique(y_tr_full[valid_tr])) < 2:
            predictions[outcome] = np.full(X_te.shape[0], np.nan, dtype=np.float32)
            continue
        X_tr_v = X_tr[valid_tr]; y_tr_v = y_tr_full[valid_tr].astype(int)

        best_C = 1.0; best_auc = -1.0; best_clf = None
        for C in [1.0]:
            clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=500,
                                     class_weight='balanced')
            try:
                clf.fit(X_tr_v, y_tr_v)
            except Exception:
                continue
            if valid_vl.sum() < 30:
                continue
            X_vl_v = X_vl[valid_vl]; y_vl_v = y_vl_full[valid_vl]
            try:
                auc = roc_auc_score(y_vl_v, clf.predict_proba(X_vl_v)[:, 1])
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc; best_C = C; best_clf = clf

        if best_clf is None:
            predictions[outcome] = np.full(X_te.shape[0], np.nan, dtype=np.float32)
            continue
        scores = best_clf.predict_proba(X_te)[:, 1].astype(np.float32)
        predictions[outcome] = scores
        metadata[outcome] = {'best_C': best_C, 'best_val_auroc': best_auc}

    return {'predictions': predictions, 'metadata': metadata,
            'n_train': X_tr.shape[0], 'n_val': X_vl.shape[0], 'n_test': X_te.shape[0]}


# ============= LightGBM baseline =============

def _fit_lgbm_regressor(X_tr, y_tr, X_vl, y_vl, n_estimators=300, lr=0.05,
                          num_leaves=31, max_depth=5, min_data_in_leaf=500,
                          feature_fraction=0.7, bagging_fraction=0.7,
                          bagging_freq=5, reg_lambda=1.0, early_stopping=20):
    """Fit an LGBMRegressor with validation early-stopping."""
    if not HAS_LGB:
        return None
    valid_tr = np.isfinite(y_tr)
    valid_vl = np.isfinite(y_vl)
    if valid_tr.sum() < 30 or valid_vl.sum() < 10:
        return None
    X_tr_v = X_tr[valid_tr]; y_tr_v = y_tr[valid_tr]
    X_vl_v = X_vl[valid_vl]; y_vl_v = y_vl[valid_vl]
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=lr,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        reg_lambda=reg_lambda,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
    model.fit(X_tr_v, y_tr_v, eval_set=[(X_vl_v, y_vl_v)], callbacks=callbacks)
    return model


def train_lightgbm_continuous(
    train_ds: BaselineDataset,
    val_ds: BaselineDataset,
    test_ds: BaselineDataset,
    target_vars: list[str],
    horizons: list[int] = None,
    aggregators: list[str] = None,
) -> dict:
    """Train LightGBM per (var, h, agg) on raw past-C x V_dyn features + statics."""
    if not HAS_LGB:
        raise RuntimeError("lightgbm required")
    horizons = horizons or REPORTED_HORIZONS
    aggregators = aggregators or AGGREGATORS

    # Pass raw NaN (no imputation), no z-score.
    X_seq_tr, X_static_tr = impute_and_normalize(
        train_ds, impute_dyn=False, normalize_dyn=False,
        impute_stat=True, normalize_stat=False)  # statics: median-impute, no normalize
    X_seq_vl, X_static_vl = impute_and_normalize(
        val_ds, impute_dyn=False, normalize_dyn=False,
        impute_stat=True, normalize_stat=False)
    X_seq_te, X_static_te = impute_and_normalize(
        test_ds, impute_dyn=False, normalize_dyn=False,
        impute_stat=True, normalize_stat=False)

    X_tr, _ = flatten_features_for_classical(X_seq_tr, X_static_tr, feature_mode='raw')
    X_vl, _ = flatten_features_for_classical(X_seq_vl, X_static_vl, feature_mode='raw')
    X_te, _ = flatten_features_for_classical(X_seq_te, X_static_te, feature_mode='raw')

    logger.info(f"  LightGBM feature dim: {X_tr.shape[1]} (raw mode + NaN native handling)")

    predictions = {}
    n_total = 0; n_skipped = 0
    t0 = time.time()
    for var in target_vars:
        for h in horizons:
            for agg in aggregators:
                if agg in ('mean', 'std') and h == 1:
                    continue
                from clin_jepa.evaluation.baselines.features import SOFA_TOTAL_ONLY_H24, SPARSE_VARS_SKIP_H1
                if var in SOFA_TOTAL_ONLY_H24 and h != 24:
                    continue
                if var in SPARSE_VARS_SKIP_H1 and h == 1:
                    continue

                y_tr = train_ds.y_continuous[var][agg][:, h - 1]
                y_vl = val_ds.y_continuous[var][agg][:, h - 1]
                model = _fit_lgbm_regressor(X_tr, y_tr, X_vl, y_vl)
                if model is None:
                    predictions[(var, h, agg)] = np.full(X_te.shape[0], np.nan, dtype=np.float32)
                    n_skipped += 1
                else:
                    predictions[(var, h, agg)] = model.predict(X_te).astype(np.float32)
                n_total += 1

    logger.info(f"  LightGBM done: {n_total - n_skipped}/{n_total} probes trained in {time.time()-t0:.1f}s")
    return {
        'predictions': predictions,
        'n_train': X_tr.shape[0], 'n_val': X_vl.shape[0], 'n_test': X_te.shape[0],
    }


def train_lightgbm_binary(
    train_ds: BaselineDataset,
    val_ds: BaselineDataset,
    test_ds: BaselineDataset,
    outcomes: list[str],
) -> dict:
    """Train LGBMClassifier per binary outcome."""
    if not HAS_LGB:
        raise RuntimeError("lightgbm required")

    X_seq_tr, X_static_tr = impute_and_normalize(train_ds, normalize_dyn=False, normalize_stat=False)
    X_seq_vl, X_static_vl = impute_and_normalize(val_ds, normalize_dyn=False, normalize_stat=False)
    X_seq_te, X_static_te = impute_and_normalize(test_ds, normalize_dyn=False, normalize_stat=False)
    X_tr, _ = flatten_features_for_classical(X_seq_tr, X_static_tr, feature_mode='raw')
    X_vl, _ = flatten_features_for_classical(X_seq_vl, X_static_vl, feature_mode='raw')
    X_te, _ = flatten_features_for_classical(X_seq_te, X_static_te, feature_mode='raw')

    predictions = {}
    metadata = {}
    for outcome in outcomes:
        y_tr_full = train_ds.y_binary[outcome]
        y_vl_full = val_ds.y_binary[outcome]
        valid_tr = np.isfinite(y_tr_full); valid_vl = np.isfinite(y_vl_full)
        if valid_tr.sum() < 30 or len(np.unique(y_tr_full[valid_tr])) < 2:
            predictions[outcome] = np.full(X_te.shape[0], np.nan, dtype=np.float32)
            continue
        X_tr_v = X_tr[valid_tr]; y_tr_v = y_tr_full[valid_tr].astype(int)
        X_vl_v = X_vl[valid_vl]; y_vl_v = y_vl_full[valid_vl].astype(int)

        n_pos = int(y_tr_v.sum()); n_neg = int(len(y_tr_v) - n_pos)
        scale_pos_weight = max(1.0, n_neg / max(1, n_pos))

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            min_data_in_leaf=500,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            metric='auc',
            verbose=-1,
        )
        callbacks = [lgb.early_stopping(20, verbose=False)]
        model.fit(X_tr_v, y_tr_v, eval_set=[(X_vl_v, y_vl_v)], callbacks=callbacks)
        scores = model.predict_proba(X_te)[:, 1].astype(np.float32)
        predictions[outcome] = scores
        metadata[outcome] = {'n_pos_train': n_pos, 'n_neg_train': n_neg,
                              'best_iter': model.best_iteration_}

    return {'predictions': predictions, 'metadata': metadata,
            'n_train': X_tr.shape[0], 'n_val': X_vl.shape[0], 'n_test': X_te.shape[0]}
