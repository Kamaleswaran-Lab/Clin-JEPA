"""Feature builder shared by the Ridge, LightGBM, and deep-learning (LSTM, Transformer, GRU-D, TCN) baselines.

Builds ``(X, y)`` tensors for two cohort modes and three target aggregators:

* ``rolling`` (Track 1, continuous forecasting): ``start_step ∈ {0, 6, 12, 18}``,
  ``length ≥ C + h``.
* ``admission`` (Track 2, stay-level binary outcomes): ``start_step = 0``,
  ``length ≥ C``, one sample per stay.

For continuous targets the aggregator may be ``"point"``
(``y = labels[var][C+h-1]``), ``"mean"`` (``nanmean(labels[var][C : C+h])``),
or ``"std"`` (``nanstd(labels[var][C : C+h])``).

Data layout (all paths resolve against ``$CLIN_JEPA_DATA``):

* ``trajectories/`` — per-window trajectory shards
* ``wide_features/`` — wide feature tables (63 observation + 10 action + 8 static)
* ``labels/`` — 27 continuous targets for forecasting
* ``cohort/classical_task_labels.parquet`` — 8 stay-level binary outcomes
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

_DATA = Path(os.environ.get("CLIN_JEPA_DATA", "./data")).expanduser()
TRAJ_DIR = _DATA / "trajectories"
WIDE_FEAT_DIR = _DATA / "wide_features"
LABELS_27_DIR = _DATA / "labels"
CLASSICAL_LABELS = _DATA / "cohort" / "classical_task_labels.parquet"

TARGET_VARS_27 = [
    "label_hr", "label_map", "label_sbp", "label_dbp",
    "label_rr", "label_spo2", "label_pao2fio2_ratio", "label_temperature",
    "label_lactate", "label_creatinine", "label_glucose", "label_urine_output",
    "label_sodium", "label_potassium", "label_magnesium", "label_pH", "label_bicarbonate",
    "label_hemoglobin", "label_platelet", "label_wbc", "label_inr",
    "label_total_bilirubin", "label_albumin", "label_ast", "label_alt",
    "label_gcs_total", "label_sofa_total",
]

# 8 binary outcomes (stay-level, post-pruning)
BINARY_OUTCOMES_8 = [
    "hospital_mortality", "true_icu_mortality",
    "mortality_7d", "mortality_14d", "mortality_30d", "mortality_90d",
    "prolonged_stay_7d", "sepsis_ever",
]

REPORTED_HORIZONS = [1, 6, 12, 24, 48]
AGGREGATORS = ["point", "mean", "std"]
TRAJ_MIN_VALID_MEAN = 3
TRAJ_MIN_VALID_STD = 2

SOFA_TOTAL_ONLY_H24 = {"label_sofa_total"}
SPARSE_VARS_SKIP_H1 = {
    "label_magnesium", "label_albumin", "label_ast", "label_alt",
    "label_total_bilirubin", "label_pao2fio2_ratio",
}
LOG_TRANSFORM_VARS = {
    "label_creatinine", "label_urine_output", "label_platelet",
    "label_magnesium", "label_total_bilirubin", "label_ast", "label_alt",
}


@dataclass
class BaselineDataset:
    """Output of build_dataset."""
    # Inputs
    X_seq: np.ndarray           # (N, C, V_dyn=73)  past-C hourly values [obs+act], NaN-imputed
    X_static: np.ndarray        # (N, V_stat=8)  static features, age-normalized
    feature_names_dyn: list[str]   # 73 names: obs + act
    feature_names_stat: list[str]  # 8 names

    # Targets
    y_continuous: dict          # var -> agg -> (N, max_h)  np.float32 with NaN missing
    y_binary: dict              # outcome -> (N,) np.float32 with NaN missing
    y_last_observed: dict       # var -> (N,) value at hour C-1 (for MASE denominator)

    # Metadata
    stay_ids: np.ndarray        # (N,)
    start_steps: np.ndarray     # (N,)
    lengths: np.ndarray         # (N,)  full ICU stay length
    context_length: int
    max_rollout: int
    cohort_mode: str            # 'rolling' or 'admission'

    # Normalization stats (computed from train, used by val/test)
    var_mean_dyn: Optional[np.ndarray] = None  # (V_dyn,) per-var mean for z-score
    var_std_dyn: Optional[np.ndarray] = None
    var_median_dyn: Optional[np.ndarray] = None  # (V_dyn,) median for NaN imputation
    var_mean_stat: Optional[np.ndarray] = None  # (V_stat,)
    var_std_stat: Optional[np.ndarray] = None
    var_median_stat: Optional[np.ndarray] = None


def _aggregate_target(label_arr: np.ndarray, ctx: int, h: int, agg: str) -> float:
    """Compute target value at horizon h for given aggregator.

    label_arr: (n_steps_for_stay,) array of label values per hour.
    Returns NaN if insufficient valid data.
    """
    if agg == "point":
        idx = ctx + h - 1
        if idx >= len(label_arr):
            return np.nan
        v = label_arr[idx]
        return float(v) if np.isfinite(v) else np.nan
    elif agg in ("mean", "std"):
        end = ctx + h
        if end > len(label_arr):
            return np.nan
        window = label_arr[ctx:end]
        mask = np.isfinite(window)
        n_valid = int(mask.sum())
        if agg == "mean":
            if n_valid < TRAJ_MIN_VALID_MEAN:
                return np.nan
            return float(np.mean(window[mask]))
        else:  # std
            if n_valid < TRAJ_MIN_VALID_STD:
                return np.nan
            return float(np.std(window[mask], ddof=0))
    else:
        raise ValueError(f"Unknown aggregator: {agg}")


def _load_classical_labels() -> pd.DataFrame:
    df = pd.read_parquet(CLASSICAL_LABELS).set_index('stay_id')
    return df


def build_dataset(
    split: str,
    context_length: int,
    max_rollout: int = 60,
    cohort_mode: str = 'rolling',  # 'rolling' or 'admission'
    rolling_step_strides: list[int] = None,  # rolling mode; None = all emitted start_steps
    max_windows: Optional[int] = None,
    fit_normalizers_on: Optional['BaselineDataset'] = None,
    test_alignment_path: Optional[Path] = None,  # for test split: enforce ordering
    target_outcomes: list[str] = None,  # binary outcomes to extract
    target_vars: list[str] = None,  # continuous vars to extract
) -> BaselineDataset:
    """Build baseline dataset.

    Args:
        split: 'train' / 'val' / 'test'
        context_length: C (e.g., 24)
        max_rollout: max horizon to predict (60 = full predictor capacity)
        cohort_mode: 'rolling' (Track 1) or 'admission' (Track 2)
        rolling_step_strides: which start_steps to include for rolling mode. None = all emitted
                              start_steps (no whitelist filter). The trajectory builder emits stride=12, so actual
                              start_steps are {0, 12, 24, 36, ...}. Setting e.g. [0, 12] would silently subset.
        fit_normalizers_on: if given, reuse this dataset's normalizers (for val/test);
                            if None, fit normalizers on this dataset (for train).
        test_alignment_path: optional the rollout alignment sidecar for test ordering
        target_outcomes: subset of binary outcomes (default all 8)
        target_vars: subset of continuous vars (default all 27)
    """
    if cohort_mode not in ('rolling', 'admission'):
        raise ValueError(f"cohort_mode must be 'rolling' or 'admission', got {cohort_mode}")
    if target_outcomes is None:
        target_outcomes = BINARY_OUTCOMES_8
    if target_vars is None:
        target_vars = TARGET_VARS_27

    traj_split = TRAJ_DIR / split
    feat_split = WIDE_FEAT_DIR / split
    label_split = LABELS_27_DIR / split

    traj_shards = sorted(traj_split.glob('trajectories_c*.pt'))
    if not traj_shards:
        raise FileNotFoundError(f"No trajectory shards in {traj_split}")

    classical_df = _load_classical_labels()

    # Load test alignment if requested
    forced_order = None
    if split == 'test' and test_alignment_path is not None and test_alignment_path.exists():
        align = torch.load(test_alignment_path, map_location='cpu', weights_only=False)
        forced_order = list(zip(
            [int(x) for x in align['shard_idx']],
            [int(x) for x in align['window_idx_in_shard']],
        ))
        forced_stay_ids_check = np.asarray(align['stay_ids'])
        logger.info(f"  Using test alignment: N={len(forced_order)}")
    else:
        forced_stay_ids_check = None

    # Accumulators
    X_seq_list = []      # list of (C, V_dyn) per window
    X_static_list = []   # list of (V_stat,) per window
    y_cont_lists = {var: {agg: [] for agg in AGGREGATORS} for var in target_vars}
    y_last_observed_lists = {var: [] for var in target_vars}
    y_bin_lists = {o: [] for o in target_outcomes}
    stay_ids_list = []
    start_steps_list = []
    lengths_list = []

    n_total_windows = 0
    n_eligible_windows = 0

    def _process_window(traj, feat_shard, label_shard, w_idx, s_idx_force=None, w_idx_force=None):
        """Process one window: extract features + targets, append to lists."""
        nonlocal n_total_windows, n_eligible_windows
        windows = traj['windows']
        per_stay = traj['per_stay']
        stay_ids_shard = np.asarray(per_stay['stay_ids'])
        w = windows[w_idx]
        n_total_windows += 1

        stay_index = int(w['stay_index'])
        start_step = int(w['start_step'])
        length = int(w['length'])
        stay_id = int(stay_ids_shard[stay_index])

        # Cohort filter
        if cohort_mode == 'admission':
            if start_step != 0 or length < context_length:
                return
        else:  # rolling
            # If rolling_step_strides explicitly specified, filter; else accept all start_steps.
            if rolling_step_strides is not None and start_step not in rolling_step_strides:
                return
            # For continuous: need length ≥ C+h for any horizon h. Keep if length ≥ C+1 (h=1 valid).
            if length < context_length + 1:
                return

        n_eligible_windows += 1

        # === Build X_seq (past-C hourly values per dyn var) ===
        feat_obs = feat_shard['obs_data']  # var -> list of (n_steps,) per stay
        feat_act = feat_shard['act_data']
        obs_names = feat_shard['feature_names']['obs']
        act_names = feat_shard['feature_names']['act']

        x_seq = np.zeros((context_length, len(obs_names) + len(act_names)), dtype=np.float32)

        col_idx = 0
        for var in obs_names:
            full = feat_obs[var][stay_index]  # (n_steps_stay,)
            window = np.full(context_length, np.nan, dtype=np.float32)
            n_avail = min(context_length, max(0, len(full) - start_step))
            if n_avail > 0:
                window[:n_avail] = full[start_step:start_step + n_avail]
            x_seq[:, col_idx] = window
            col_idx += 1
        for var in act_names:
            full = feat_act[var][stay_index]
            window = np.full(context_length, np.nan, dtype=np.float32)
            n_avail = min(context_length, max(0, len(full) - start_step))
            if n_avail > 0:
                window[:n_avail] = full[start_step:start_step + n_avail]
            x_seq[:, col_idx] = window
            col_idx += 1

        X_seq_list.append(x_seq)

        # === Build X_static ===
        statics = feat_shard['statics'][stay_index]  # (V_stat,)
        X_static_list.append(statics.astype(np.float32))

        for var in target_vars:
            if label_shard is None or var not in label_shard:
                for agg in AGGREGATORS:
                    y_cont_lists[var][agg].append(np.full(max_rollout, np.nan, dtype=np.float32))
                y_last_observed_lists[var].append(np.nan)
                continue
            try:
                label_arr_full = np.asarray(label_shard[var][w_idx], dtype=np.float32)
            except (IndexError, TypeError):
                for agg in AGGREGATORS:
                    y_cont_lists[var][agg].append(np.full(max_rollout, np.nan, dtype=np.float32))
                y_last_observed_lists[var].append(np.nan)
                continue
            if len(label_arr_full) == 0:
                for agg in AGGREGATORS:
                    y_cont_lists[var][agg].append(np.full(max_rollout, np.nan, dtype=np.float32))
                y_last_observed_lists[var].append(np.nan)
                continue

            for agg in AGGREGATORS:
                y_h = np.full(max_rollout, np.nan, dtype=np.float32)
                # Skip rules
                for h in range(1, max_rollout + 1):
                    if agg in ('mean', 'std') and h == 1:
                        continue
                    if var in SOFA_TOTAL_ONLY_H24 and h != 24:
                        continue
                    if var in SPARSE_VARS_SKIP_H1 and h == 1:
                        continue
                    # Labels are already pre-sliced per-window: label_arr_full[0] = window's first hour
                    y_h[h - 1] = _aggregate_target(label_arr_full, context_length, h, agg)
                y_cont_lists[var][agg].append(y_h)

            # Last observed value (at hour C-1 within window) — for MASE
            last_idx = context_length - 1
            if last_idx < len(label_arr_full):
                v = label_arr_full[last_idx]
                y_last_observed_lists[var].append(float(v) if np.isfinite(v) else np.nan)
            else:
                y_last_observed_lists[var].append(np.nan)

        # === Build y_binary (admission cohort or stay-level even in rolling) ===
        for outcome in target_outcomes:
            v = classical_df.loc[stay_id, outcome] if stay_id in classical_df.index else np.nan
            if pd.isna(v):
                y_bin_lists[outcome].append(np.nan)
            else:
                y_bin_lists[outcome].append(float(bool(v)))

        stay_ids_list.append(stay_id)
        start_steps_list.append(start_step)
        lengths_list.append(length)

    # === Iterate shards ===
    if forced_order is not None:
        from collections import defaultdict
        order_by_shard = defaultdict(list)
        for s_idx, w_idx in forced_order:
            order_by_shard[s_idx].append(w_idx)

        for s_idx in sorted(order_by_shard.keys()):
            traj_path = traj_shards[s_idx]
            feat_path = feat_split / traj_path.name.replace('trajectories_', 'features_')
            label_path = label_split / traj_path.name.replace('trajectories_', 'embeddings_')

            traj = torch.load(traj_path, map_location='cpu', weights_only=False)
            feat_shard = torch.load(feat_path, map_location='cpu', weights_only=False) if feat_path.exists() else None
            label_shard = torch.load(label_path, map_location='cpu', weights_only=False) if label_path.exists() else None
            if feat_shard is None:
                raise FileNotFoundError(f"Missing wide-feature shard: {feat_path}")

            for w_idx in order_by_shard[s_idx]:
                _process_window(traj, feat_shard, label_shard, w_idx)
                if max_windows is not None and len(stay_ids_list) >= max_windows:
                    break
            if max_windows is not None and len(stay_ids_list) >= max_windows:
                break
    else:
        # Train/val: iterate shards sequentially
        for traj_path in traj_shards:
            feat_path = feat_split / traj_path.name.replace('trajectories_', 'features_')
            label_path = label_split / traj_path.name.replace('trajectories_', 'embeddings_')

            traj = torch.load(traj_path, map_location='cpu', weights_only=False)
            feat_shard = torch.load(feat_path, map_location='cpu', weights_only=False) if feat_path.exists() else None
            label_shard = torch.load(label_path, map_location='cpu', weights_only=False) if label_path.exists() else None
            if feat_shard is None:
                raise FileNotFoundError(f"Missing wide-feature shard: {feat_path}")

            n_windows_in_shard = len(traj['windows'])
            for w_idx in range(n_windows_in_shard):
                _process_window(traj, feat_shard, label_shard, w_idx)
                if max_windows is not None and len(stay_ids_list) >= max_windows:
                    break
            if max_windows is not None and len(stay_ids_list) >= max_windows:
                break

    if not stay_ids_list:
        raise RuntimeError(f"No eligible windows for {split} cohort_mode={cohort_mode}")

    logger.info(f"  Processed {n_total_windows} total windows, {n_eligible_windows} eligible "
                f"({len(stay_ids_list)} accepted)")

    # === Stack arrays ===
    X_seq = np.stack(X_seq_list, axis=0).astype(np.float32)            # (N, C, V_dyn)
    X_static = np.stack(X_static_list, axis=0).astype(np.float32)      # (N, V_stat)
    stay_ids = np.array(stay_ids_list, dtype=np.int64)
    start_steps = np.array(start_steps_list, dtype=np.int64)
    lengths = np.array(lengths_list, dtype=np.int64)

    y_continuous = {var: {agg: np.stack(y_cont_lists[var][agg], axis=0).astype(np.float32)
                          for agg in AGGREGATORS}
                    for var in target_vars}
    y_last_observed = {var: np.array(y_last_observed_lists[var], dtype=np.float32)
                       for var in target_vars}
    y_binary = {o: np.array(y_bin_lists[o], dtype=np.float32) for o in target_outcomes}

    # Sanity: sample order check vs alignment if applicable
    if forced_stay_ids_check is not None:
        if not np.array_equal(stay_ids[:len(forced_stay_ids_check)], forced_stay_ids_check):
            logger.warning("Test alignment stay_id mismatch — first 5: "
                           f"got {stay_ids[:5]}, expected {forced_stay_ids_check[:5]}")

    # Feature names
    feat0 = torch.load(WIDE_FEAT_DIR / split / sorted((WIDE_FEAT_DIR / split).glob('features_*.pt'))[0].name,
                       map_location='cpu', weights_only=False)
    feature_names_dyn = list(feat0['feature_names']['obs']) + list(feat0['feature_names']['act'])
    feature_names_stat = list(feat0['feature_names']['statics'])
    n_dyn = X_seq.shape[2]
    n_stat = X_static.shape[1]
    assert n_dyn == len(feature_names_dyn), f"dyn feat dim mismatch: {n_dyn} vs {len(feature_names_dyn)}"
    assert n_stat == len(feature_names_stat), f"stat feat dim mismatch: {n_stat} vs {len(feature_names_stat)}"

    # === Compute or reuse normalizers ===
    if fit_normalizers_on is None:
        # TRAIN: fit normalizers from this dataset
        var_median_dyn = np.zeros(n_dyn, dtype=np.float32)
        var_mean_dyn = np.zeros(n_dyn, dtype=np.float32)
        var_std_dyn = np.ones(n_dyn, dtype=np.float32)
        for v in range(n_dyn):
            col = X_seq[:, :, v].ravel()
            valid = col[np.isfinite(col)]
            if len(valid) > 10:
                var_median_dyn[v] = float(np.median(valid))
                var_mean_dyn[v] = float(np.mean(valid))
                var_std_dyn[v] = float(np.std(valid) + 1e-6)

        var_median_stat = np.zeros(n_stat, dtype=np.float32)
        var_mean_stat = np.zeros(n_stat, dtype=np.float32)
        var_std_stat = np.ones(n_stat, dtype=np.float32)
        for v in range(n_stat):
            col = X_static[:, v]
            valid = col[np.isfinite(col)]
            if len(valid) > 10:
                var_median_stat[v] = float(np.median(valid))
                var_mean_stat[v] = float(np.mean(valid))
                var_std_stat[v] = float(np.std(valid) + 1e-6)
    else:
        # VAL/TEST: reuse normalizers from train
        var_median_dyn = fit_normalizers_on.var_median_dyn
        var_mean_dyn = fit_normalizers_on.var_mean_dyn
        var_std_dyn = fit_normalizers_on.var_std_dyn
        var_median_stat = fit_normalizers_on.var_median_stat
        var_mean_stat = fit_normalizers_on.var_mean_stat
        var_std_stat = fit_normalizers_on.var_std_stat

    return BaselineDataset(
        X_seq=X_seq,
        X_static=X_static,
        feature_names_dyn=feature_names_dyn,
        feature_names_stat=feature_names_stat,
        y_continuous=y_continuous,
        y_binary=y_binary,
        y_last_observed=y_last_observed,
        stay_ids=stay_ids,
        start_steps=start_steps,
        lengths=lengths,
        context_length=context_length,
        max_rollout=max_rollout,
        cohort_mode=cohort_mode,
        var_mean_dyn=var_mean_dyn,
        var_std_dyn=var_std_dyn,
        var_median_dyn=var_median_dyn,
        var_mean_stat=var_mean_stat,
        var_std_stat=var_std_stat,
        var_median_stat=var_median_stat,
    )


def impute_and_normalize(
    ds: BaselineDataset,
    impute_dyn: bool = True,
    normalize_dyn: bool = True,
    impute_stat: bool = True,
    normalize_stat: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply median imputation + z-score using the dataset's stored normalizers.

    Returns (X_seq_processed, X_static_processed) — both with NaN replaced + z-scored.

    For LightGBM / Linear-stat: impute=True, normalize=True for X_static;
    impute=True, normalize=False for X_seq.

    For DL baselines (LSTM/Transformer/GRU-D/TCN): impute=True, normalize=True both.
    """
    X_seq = ds.X_seq.copy()  # (N, C, V_dyn)
    X_static = ds.X_static.copy()  # (N, V_stat)

    if impute_dyn:
        for v in range(X_seq.shape[2]):
            col = X_seq[:, :, v]
            mask = ~np.isfinite(col)
            col[mask] = ds.var_median_dyn[v]
            X_seq[:, :, v] = col
    if normalize_dyn:
        X_seq = (X_seq - ds.var_mean_dyn[None, None, :]) / ds.var_std_dyn[None, None, :]

    if impute_stat:
        for v in range(X_static.shape[1]):
            col = X_static[:, v]
            mask = ~np.isfinite(col)
            col[mask] = ds.var_median_stat[v]
            X_static[:, v] = col
    if normalize_stat:
        X_static = (X_static - ds.var_mean_stat[None, :]) / ds.var_std_stat[None, :]

    return X_seq.astype(np.float32), X_static.astype(np.float32)


def flatten_features_for_classical(
    X_seq: np.ndarray,
    X_static: np.ndarray,
    feature_mode: str = 'raw',
) -> tuple[np.ndarray, list[str]]:
    """Flatten (N, C, V_dyn) + (N, V_stat) into (N, D) for classical baselines.

    feature_mode:
      'raw': concat past-C × V_dyn hourly values + V_stat statics → D = C*V_dyn + V_stat
      'summary': 6 stats (mean/last/min/max/std/trend) per V_dyn + V_stat → D = 6*V_dyn + V_stat
    """
    N, C, V_dyn = X_seq.shape
    V_stat = X_static.shape[1]

    if feature_mode == 'raw':
        # Layout: [stay × time × var] flatten as [var1*C, var2*C, ...]
        x_flat = np.transpose(X_seq, (0, 2, 1)).reshape(N, V_dyn * C)
        X = np.concatenate([x_flat, X_static], axis=1)
        names = [f"{i}_h{h}" for i in range(V_dyn) for h in range(C)] + [f"stat_{j}" for j in range(V_stat)]
    elif feature_mode == 'summary':
        # 6 stats per var
        feats = []
        for v in range(V_dyn):
            col = X_seq[:, :, v]  # (N, C)
            mean = np.nanmean(col, axis=1, keepdims=False)
            last = col[:, -1]
            vmin = np.nanmin(col, axis=1)
            vmax = np.nanmax(col, axis=1)
            std = np.nanstd(col, axis=1)
            trend = col[:, -1] - col[:, 0]
            feats.append(np.stack([mean, last, vmin, vmax, std, trend], axis=1))  # (N, 6)
        x_summary = np.concatenate(feats, axis=1)  # (N, V_dyn*6)
        X = np.concatenate([x_summary, X_static], axis=1)
        names = [f"v{i}_{stat}" for i in range(V_dyn) for stat in ['mean', 'last', 'min', 'max', 'std', 'trend']]
        names += [f"stat_{j}" for j in range(V_stat)]
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    return X.astype(np.float32), names
