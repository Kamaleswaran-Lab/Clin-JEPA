"""Unified training loop for the DL baselines (LSTM / Transformer / GRU-D / TCN).

Continuous training predicts (n_vars × max_h × n_aggregators) outputs per
sample; binary training predicts n_outcomes outputs per sample with BCE loss.
A cosine LR schedule is used for all four models.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clin_jepa.evaluation.baselines.dl_models import make_model
from clin_jepa.evaluation.baselines.features import (
    BaselineDataset, AGGREGATORS, REPORTED_HORIZONS, impute_and_normalize,
)

logger = logging.getLogger(__name__)


@dataclass
class DLTrainConfig:
    model_name: str = 'lstm'  # 'lstm' | 'transformer' | 'grud' | 'tcn'
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    head_hidden: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    grad_clip: float = 1.0
    device: str = 'cuda'
    seed: int = 42


# ============= Helpers for mask + delta (GRU-D) =============

def _build_mask_delta(X_seq_raw: np.ndarray, X_seq_imp: np.ndarray,
                       median_dyn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (m_seq, delta_seq) given raw (NaN-containing) and imputed sequences.

    m_seq: 1 where original was finite, 0 where NaN-imputed.
    delta_seq: hours since last observation per (sample, var, time).
    """
    N, C, V = X_seq_raw.shape
    m_seq = np.isfinite(X_seq_raw).astype(np.float32)  # (N, C, V)

    # delta_seq: for each (n, t, v), count how many steps since last m=1
    delta_seq = np.zeros((N, C, V), dtype=np.float32)
    last_obs_step = np.full((N, V), -np.inf, dtype=np.float32)  # -∞ = never observed
    for t in range(C):
        # Where m=1 at this step: delta=0; where m=0: delta = (t - last_obs_step) clipped >= 1
        observed = m_seq[:, t, :] == 1
        delta_seq[:, t, :] = np.where(
            observed, 0.0,
            np.where(np.isfinite(last_obs_step), t - last_obs_step, t + 1.0),
        )
        # Update last_obs_step where observed
        last_obs_step = np.where(observed, float(t), last_obs_step)

    return m_seq, delta_seq


# ============= Continuous training =============

def _build_continuous_target_tensor(ds: BaselineDataset, target_vars: list[str],
                                     horizons: list[int], aggregators: list[str],
                                     y_mean_in: np.ndarray = None,
                                     y_std_in: np.ndarray = None,
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack y_continuous into (N, n_vars × n_h × n_agg) tensor + mask + per-target normalizer.

    Args:
      y_mean_in, y_std_in: if provided, use these for normalization (val/test).
                          if None, FIT from this dataset (train).

    Returns:
      Y: (N, n_outputs) — z-normalized targets, NaN-filled with 0
      M: (N, n_outputs) — 1 where target valid (finite), 0 otherwise
      y_mean: (n_outputs,) per-target mean (for un-normalize)
      y_std: (n_outputs,) per-target std (for un-normalize)
    """
    N = ds.X_seq.shape[0]
    n_vars = len(target_vars)
    n_h = len(horizons)
    n_agg = len(aggregators)
    n_outputs = n_vars * n_h * n_agg

    Y_raw = np.zeros((N, n_outputs), dtype=np.float32)
    M = np.zeros((N, n_outputs), dtype=np.float32)
    out_idx = 0
    out_meta = []  # (var, h, agg) per out_idx

    for v_idx, var in enumerate(target_vars):
        for h_idx, h in enumerate(horizons):
            for a_idx, agg in enumerate(aggregators):
                y_var_agg = ds.y_continuous[var][agg]  # (N, max_h)
                if h - 1 < y_var_agg.shape[1]:
                    col = y_var_agg[:, h - 1]
                else:
                    col = np.full(N, np.nan, dtype=np.float32)
                mask = np.isfinite(col)
                Y_raw[:, out_idx] = np.where(mask, col, 0.0)
                M[:, out_idx] = mask.astype(np.float32)
                out_meta.append((var, h, agg))
                out_idx += 1

    if y_mean_in is None or y_std_in is None:
        # Fit normalizers from this dataset (train path)
        y_mean = np.zeros(n_outputs, dtype=np.float32)
        y_std = np.ones(n_outputs, dtype=np.float32)
        for j in range(n_outputs):
            col = Y_raw[:, j]
            m = M[:, j] > 0
            if m.sum() >= 10:
                vals = col[m]
                y_mean[j] = float(np.mean(vals))
                y_std[j] = float(np.std(vals) + 1e-6)
    else:
        # Reuse provided normalizers (val/test path)
        y_mean = y_mean_in.astype(np.float32)
        y_std = y_std_in.astype(np.float32)

    Y_norm = (Y_raw - y_mean[None, :]) / y_std[None, :]
    return Y_norm.astype(np.float32), M, y_mean, y_std, out_meta


def train_dl_continuous(
    train_ds: BaselineDataset,
    val_ds: BaselineDataset,
    test_ds: BaselineDataset,
    target_vars: list[str],
    horizons: list[int] = None,
    aggregators: list[str] = None,
    cfg: DLTrainConfig = None,
) -> dict:
    """Train one model on full continuous task; return predictions on test.

    Returns dict:
      predictions: (N_test, n_outputs) un-normalized predictions
      y_mean, y_std: target normalizers
      out_meta: list of (var, h, agg) per output
      metadata: training info
    """
    cfg = cfg or DLTrainConfig()
    horizons = horizons or REPORTED_HORIZONS
    aggregators = aggregators or AGGREGATORS

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Pre-compute imputed/normalized features (statics already z-scored upstream)
    X_seq_train, X_static_train = impute_and_normalize(train_ds, normalize_stat=True)
    X_seq_val,   X_static_val   = impute_and_normalize(val_ds,   normalize_stat=True)
    X_seq_test,  X_static_test  = impute_and_normalize(test_ds,  normalize_stat=True)

    # GRU-D needs mask + delta from raw sequences
    if cfg.model_name == 'grud':
        m_train, d_train = _build_mask_delta(train_ds.X_seq, X_seq_train, train_ds.var_median_dyn)
        m_val,   d_val   = _build_mask_delta(val_ds.X_seq,   X_seq_val,   train_ds.var_median_dyn)
        m_test,  d_test  = _build_mask_delta(test_ds.X_seq,  X_seq_test,  train_ds.var_median_dyn)
    else:
        m_train = d_train = m_val = d_val = m_test = d_test = None

    # Build target tensors. Train fits y_mean/y_std; val reuses train stats.
    Y_tr, M_tr, y_mean, y_std, out_meta = _build_continuous_target_tensor(
        train_ds, target_vars, horizons, aggregators)
    Y_vl, M_vl, _, _, _ = _build_continuous_target_tensor(
        val_ds, target_vars, horizons, aggregators,
        y_mean_in=y_mean, y_std_in=y_std)

    # Convert to tensors
    X_tr = torch.from_numpy(X_seq_train); S_tr = torch.from_numpy(X_static_train)
    X_vl = torch.from_numpy(X_seq_val);   S_vl = torch.from_numpy(X_static_val)
    X_te = torch.from_numpy(X_seq_test);  S_te = torch.from_numpy(X_static_test)
    Y_tr_t = torch.from_numpy(Y_tr); M_tr_t = torch.from_numpy(M_tr)
    Y_vl_t = torch.from_numpy(Y_vl); M_vl_t = torch.from_numpy(M_vl)
    if cfg.model_name == 'grud':
        Mraw_tr = torch.from_numpy(m_train); D_tr = torch.from_numpy(d_train)
        Mraw_vl = torch.from_numpy(m_val);   D_vl = torch.from_numpy(d_val)
        Mraw_te = torch.from_numpy(m_test);  D_te = torch.from_numpy(d_test)

    # Build model
    input_dim = X_tr.shape[2]
    static_dim = S_tr.shape[1]
    n_outputs = Y_tr.shape[1]
    model = make_model(
        cfg.model_name, input_dim, static_dim, n_outputs,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        dropout=cfg.dropout, head_hidden=cfg.head_hidden,
    ).to(device)
    logger.info(f"  model={cfg.model_name}, n_outputs={n_outputs}, n_train={X_tr.shape[0]}, "
                f"n_val={X_vl.shape[0]}, n_test={X_te.shape[0]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.01)

    best_val = float('inf')
    best_state = None
    patience_ctr = 0

    n_train = X_tr.shape[0]
    for epoch in range(1, cfg.max_epochs + 1):
        # === Training ===
        model.train()
        perm = torch.randperm(n_train)
        train_err_sum = 0.0
        train_mask_sum = 0.0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x = X_tr[idx].to(device); s = S_tr[idx].to(device)
            y = Y_tr_t[idx].to(device); m = M_tr_t[idx].to(device)
            if cfg.model_name == 'grud':
                pred = model(x, s, Mraw_tr[idx].to(device), D_tr[idx].to(device))
            else:
                pred = model(x, s)
            err = (pred - y) ** 2 * m
            loss = err.sum() / m.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_err_sum += float(err.sum().item())
            train_mask_sum += float(m.sum().item())
        train_loss = train_err_sum / max(train_mask_sum, 1.0)

        # === Validation ===
        model.eval()
        val_err_sum = 0.0
        val_mask_sum = 0.0
        with torch.no_grad():
            for i in range(0, X_vl.shape[0], cfg.batch_size):
                x = X_vl[i:i + cfg.batch_size].to(device); s = S_vl[i:i + cfg.batch_size].to(device)
                y = Y_vl_t[i:i + cfg.batch_size].to(device); m = M_vl_t[i:i + cfg.batch_size].to(device)
                if cfg.model_name == 'grud':
                    pred = model(x, s, Mraw_vl[i:i + cfg.batch_size].to(device),
                                 D_vl[i:i + cfg.batch_size].to(device))
                else:
                    pred = model(x, s)
                err = (pred - y) ** 2 * m
                val_err_sum += float(err.sum().item())
                val_mask_sum += float(m.sum().item())
        val_loss = val_err_sum / max(val_mask_sum, 1.0)

        scheduler.step()
        logger.info(f"  Epoch {epoch:2d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"best={best_val:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                logger.info(f"  Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # === Predict on test ===
    preds_norm = np.zeros((X_te.shape[0], n_outputs), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X_te.shape[0], cfg.batch_size):
            x = X_te[i:i + cfg.batch_size].to(device); s = S_te[i:i + cfg.batch_size].to(device)
            if cfg.model_name == 'grud':
                pred = model(x, s, Mraw_te[i:i + cfg.batch_size].to(device),
                             D_te[i:i + cfg.batch_size].to(device))
            else:
                pred = model(x, s)
            preds_norm[i:i + cfg.batch_size] = pred.cpu().numpy()

    preds_unit = preds_norm * y_std[None, :] + y_mean[None, :]

    return {
        'predictions': preds_unit,  # (N_test, n_outputs)
        'predictions_norm': preds_norm,
        'y_mean': y_mean,
        'y_std': y_std,
        'out_meta': out_meta,
        'best_val_loss': best_val,
        'epochs_trained': epoch,
        'n_train': X_tr.shape[0], 'n_val': X_vl.shape[0], 'n_test': X_te.shape[0],
    }


# ============= Binary training =============

def train_dl_binary(
    train_ds: BaselineDataset,
    val_ds: BaselineDataset,
    test_ds: BaselineDataset,
    outcomes: list[str],
    cfg: DLTrainConfig = None,
) -> dict:
    """Train one model jointly on all binary outcomes. Returns test predictions per outcome."""
    cfg = cfg or DLTrainConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_seq_train, X_static_train = impute_and_normalize(train_ds, normalize_stat=True)
    X_seq_val,   X_static_val   = impute_and_normalize(val_ds,   normalize_stat=True)
    X_seq_test,  X_static_test  = impute_and_normalize(test_ds,  normalize_stat=True)

    if cfg.model_name == 'grud':
        m_train, d_train = _build_mask_delta(train_ds.X_seq, X_seq_train, train_ds.var_median_dyn)
        m_val,   d_val   = _build_mask_delta(val_ds.X_seq,   X_seq_val,   train_ds.var_median_dyn)
        m_test,  d_test  = _build_mask_delta(test_ds.X_seq,  X_seq_test,  train_ds.var_median_dyn)

    # Targets: (N, n_outcomes) with NaN where outcome not available
    n_outcomes = len(outcomes)
    Y_tr_raw = np.stack([train_ds.y_binary[o] for o in outcomes], axis=1)  # (N, n_outcomes)
    Y_vl_raw = np.stack([val_ds.y_binary[o] for o in outcomes], axis=1)
    Y_te_raw = np.stack([test_ds.y_binary[o] for o in outcomes], axis=1)
    M_tr = np.isfinite(Y_tr_raw).astype(np.float32)
    M_vl = np.isfinite(Y_vl_raw).astype(np.float32)
    M_te = np.isfinite(Y_te_raw).astype(np.float32)
    Y_tr = np.where(np.isfinite(Y_tr_raw), Y_tr_raw, 0.0).astype(np.float32)
    Y_vl = np.where(np.isfinite(Y_vl_raw), Y_vl_raw, 0.0).astype(np.float32)

    # pos_weight per outcome (from train)
    pos_weight = np.ones(n_outcomes, dtype=np.float32)
    for j, o in enumerate(outcomes):
        valid = M_tr[:, j] > 0
        if valid.sum() > 10:
            n_pos = int(Y_tr[valid, j].sum())
            n_neg = int(valid.sum()) - n_pos
            pos_weight[j] = max(1.0, n_neg / max(1, n_pos))

    X_tr = torch.from_numpy(X_seq_train); S_tr = torch.from_numpy(X_static_train)
    X_vl = torch.from_numpy(X_seq_val);   S_vl = torch.from_numpy(X_static_val)
    X_te = torch.from_numpy(X_seq_test);  S_te = torch.from_numpy(X_static_test)
    Y_tr_t = torch.from_numpy(Y_tr); M_tr_t = torch.from_numpy(M_tr)
    Y_vl_t = torch.from_numpy(Y_vl); M_vl_t = torch.from_numpy(M_vl)
    pw_t = torch.from_numpy(pos_weight).to(device)
    if cfg.model_name == 'grud':
        Mraw_tr = torch.from_numpy(m_train); D_tr = torch.from_numpy(d_train)
        Mraw_vl = torch.from_numpy(m_val);   D_vl = torch.from_numpy(d_val)
        Mraw_te = torch.from_numpy(m_test);  D_te = torch.from_numpy(d_test)

    input_dim = X_tr.shape[2]
    static_dim = S_tr.shape[1]
    model = make_model(
        cfg.model_name, input_dim, static_dim, n_outcomes,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        dropout=cfg.dropout, head_hidden=cfg.head_hidden,
    ).to(device)
    logger.info(f"  binary model={cfg.model_name}, n_outcomes={n_outcomes}, "
                f"n_train={X_tr.shape[0]}, n_val={X_vl.shape[0]}, n_test={X_te.shape[0]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.01)

    best_val_auroc = -1.0
    best_state = None
    patience_ctr = 0
    n_train = X_tr.shape[0]

    from sklearn.metrics import roc_auc_score

    for epoch in range(1, cfg.max_epochs + 1):
        # Training
        model.train()
        perm = torch.randperm(n_train)
        train_err_sum = 0.0
        train_mask_sum = 0.0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x = X_tr[idx].to(device); s = S_tr[idx].to(device)
            y = Y_tr_t[idx].to(device); m = M_tr_t[idx].to(device)
            if cfg.model_name == 'grud':
                logits = model(x, s, Mraw_tr[idx].to(device), D_tr[idx].to(device))
            else:
                logits = model(x, s)
            # BCE with per-outcome pos_weight
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw_t, reduction='none')
            loss_per = bce * m
            loss = loss_per.sum() / m.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_err_sum += float(loss_per.sum().item())
            train_mask_sum += float(m.sum().item())
        train_loss = train_err_sum / max(train_mask_sum, 1.0)

        # Val: compute mean AUROC across outcomes
        model.eval()
        val_logits_all = []
        with torch.no_grad():
            for i in range(0, X_vl.shape[0], cfg.batch_size):
                x = X_vl[i:i + cfg.batch_size].to(device); s = S_vl[i:i + cfg.batch_size].to(device)
                if cfg.model_name == 'grud':
                    lg = model(x, s, Mraw_vl[i:i + cfg.batch_size].to(device),
                               D_vl[i:i + cfg.batch_size].to(device))
                else:
                    lg = model(x, s)
                val_logits_all.append(lg.cpu().numpy())
        val_logits = np.concatenate(val_logits_all, axis=0)
        val_aurocs = []
        for j, o in enumerate(outcomes):
            valid = M_vl[:, j] > 0
            if valid.sum() < 30 or len(np.unique(Y_vl_raw[valid, j][np.isfinite(Y_vl_raw[valid, j])])) < 2:
                continue
            try:
                a = roc_auc_score(Y_vl_raw[valid, j], val_logits[valid, j])
                val_aurocs.append(a)
            except ValueError:
                pass
        val_auroc = float(np.mean(val_aurocs)) if val_aurocs else 0.5

        scheduler.step()
        logger.info(f"  Epoch {epoch:2d} train_loss={train_loss:.4f} val_auroc={val_auroc:.4f} "
                    f"best={best_val_auroc:.4f}")

        if val_auroc > best_val_auroc + 1e-6:
            best_val_auroc = val_auroc
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                logger.info(f"  Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Test predictions (logits + sigmoid)
    test_logits = []
    with torch.no_grad():
        for i in range(0, X_te.shape[0], cfg.batch_size):
            x = X_te[i:i + cfg.batch_size].to(device); s = S_te[i:i + cfg.batch_size].to(device)
            if cfg.model_name == 'grud':
                lg = model(x, s, Mraw_te[i:i + cfg.batch_size].to(device),
                           D_te[i:i + cfg.batch_size].to(device))
            else:
                lg = model(x, s)
            test_logits.append(lg.cpu().numpy())
    test_logits = np.concatenate(test_logits, axis=0)
    test_scores = 1.0 / (1.0 + np.exp(-test_logits))

    # Per-outcome predictions dict
    preds = {o: test_scores[:, j] for j, o in enumerate(outcomes)}
    targets = {o: Y_te_raw[:, j] for j, o in enumerate(outcomes)}

    return {
        'predictions': preds,  # outcome -> (N_test,) sigmoid scores
        'targets': targets,
        'best_val_auroc': best_val_auroc,
        'epochs_trained': epoch,
        'n_train': X_tr.shape[0], 'n_val': X_vl.shape[0], 'n_test': X_te.shape[0],
    }
