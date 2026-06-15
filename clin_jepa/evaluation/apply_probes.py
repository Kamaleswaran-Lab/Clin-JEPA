"""Apply trained MLP probes to rollout outputs to get per-step decoded MAE.

For each context length ``C ∈ {6, 12, 24, 48}``:

1. Load rollout outputs: ``predicted_z (N, max_rollout, 4096)``,
   ``valid_mask``, ``labels_at_step``, ``stay_ids``.
2. Load the alignment sidecar produced by
   :mod:`clin_jepa.evaluation.build_alignment` for true-copy-forward MASE
   computation and supplementary-label alignment.
3. Build per-variable ``y_true`` matrices for every clinical-target
   variable in scope:

   * variables already present in the rollout-output file are read directly
     from ``labels_at_step``;
   * supplementary / extended variables are aligned via the sidecar and
     joined against the per-stay label tables.
4. Apply the trained MLP probe to predict in the original clinical units.
5. Compute MAE / RMSE / MASE / Spearman with clustered bootstrap CIs.
6. Save per-context metrics under the configured ``metrics_dir``.

Usage::

    python -m clin_jepa.evaluation.apply_probes --plan clin_jepa
    python -m clin_jepa.evaluation.apply_probes --plan clin_jepa --contexts 12
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from clin_jepa.evaluation.probes import apply_mlp_probe, load_mlp_probe
from clin_jepa.evaluation.metrics import (
    bootstrap_ci_mae_clustered,
    bootstrap_ci_mase_clustered,
    compute_mae,
    compute_mase,
    compute_spearman,
)

logger = logging.getLogger(__name__)

EXISTING_11 = [
    "label_hr", "label_map", "label_sbp", "label_rr",
    "label_sofa_total", "label_sofa_resp", "label_sofa_cardio",
    "label_sofa_renal", "label_sofa_hepatic", "label_sofa_coag",
    "label_sofa_neuro",
]
SUPPLEMENTARY_4 = [
    "label_temperature", "label_spo2", "label_lactate", "label_urine_output",
]
EXTENDED_9 = [
    "label_dbp", "label_gcs_total", "label_glucose",
    "label_creatinine", "label_bun", "label_sodium",
    "label_potassium", "label_hemoglobin", "label_platelet",
]
ALL_24 = EXISTING_11 + SUPPLEMENTARY_4 + EXTENDED_9
DEFAULT_CI_HORIZONS = [1, 6, 12, 24, 48]


def build_aligned_labels(
    alignment: dict,
    labels_dir: Path,
    var_names: list[str],
    context_length: int,
    max_rollout: int,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Build per-var y_true (N, max_rollout) by aligning sidecar labels.

    Works for both supplementary and extended label dirs (same per-window
    layout, NaN = missing).
    """
    shard_idx_arr = alignment["shard_idx"]
    window_idx_arr = alignment["window_idx_in_shard"]
    shard_names = alignment["shard_names"]
    k = context_length - 1

    # Preload all label files for the test split
    by_shard: dict[int, dict] = {}
    for s_idx, shard_name in enumerate(shard_names):
        path = labels_dir / "test" / shard_name
        if not path.exists():
            raise FileNotFoundError(f"Label file missing: {path}")
        by_shard[s_idx] = torch.load(path, map_location="cpu", weights_only=False)

    # Filter var_names to those present in the loaded shards
    available = [v for v in var_names if v in by_shard[0]]
    aligned = {
        var: np.full((n_samples, max_rollout), np.nan, dtype=np.float32)
        for var in available
    }

    for i in range(n_samples):
        s_idx = int(shard_idx_arr[i])
        w_in_shard = int(window_idx_arr[i])
        shard = by_shard[s_idx]
        for var in available:
            arr = shard[var][w_in_shard]
            L_arr = len(arr)
            for step in range(max_rollout):
                target_idx = k + step + 1
                if target_idx < L_arr:
                    val = arr[target_idx]
                    if not np.isnan(val):
                        aligned[var][i, step] = float(val)

    return aligned


def apply_for_context(
    context_length: int,
    plan: str,
    cfg: dict,
    probes_dir: Path,
    out_dir: Path,
    device: str,
    ci_horizons: list[int],
    target_vars: list[str],
) -> Path:
    logger.info("=" * 70)
    logger.info("Plan %s | C=%d | applying %d MLP probes",
                plan.upper(), context_length, len(target_vars))
    logger.info("=" * 70)
    t_run = time.time()

    plan_cfg = cfg["plans"][plan]
    embedding_dir = Path(plan_cfg["embedding_dir"])
    supplementary_dir = Path(plan_cfg["supplementary_dir"])
    extended_dir = embedding_dir / "extended_labels"
    pred_root = Path(plan_cfg["predictions_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rollout outputs
    pred_path = pred_root / f"context_{context_length:02d}" / "predictions_absolute.pt"
    if not pred_path.exists():
        raise FileNotFoundError(f"rollout prediction file not found: {pred_path}")
    logger.info("Loading rollout outputs: %s", pred_path)
    t0 = time.time()
    rollout_out = torch.load(pred_path, map_location="cpu", weights_only=False)
    logger.info("  loaded in %.1f s", time.time() - t0)

    predicted_z = torch.as_tensor(rollout_out["predicted_z"])
    valid_mask = np.asarray(rollout_out["valid_mask"])
    stay_ids = np.asarray(rollout_out["stay_ids"])
    lengths = np.asarray(rollout_out["lengths"])
    labels_existing: dict = rollout_out["labels_at_step"]
    N, max_rollout, D = predicted_z.shape
    logger.info("rollout shape: N=%d, max_rollout=%d, D=%d", N, max_rollout, D)

    # Load alignment sidecar
    align_path = pred_root / f"context_{context_length:02d}" / "rollout_alignment.pt"
    if not align_path.exists():
        raise FileNotFoundError(f"Alignment sidecar missing: {align_path}")
    alignment = torch.load(align_path, map_location="cpu", weights_only=False)
    if alignment["n_samples"] != N:
        raise AssertionError(f"alignment.n_samples={alignment['n_samples']} != N={N}")
    if not np.array_equal(np.asarray(alignment["stay_ids"], dtype=stay_ids.dtype), stay_ids):
        raise AssertionError("alignment.stay_ids != rollout.stay_ids")
    if not np.array_equal(np.asarray(alignment["lengths"], dtype=lengths.dtype), lengths):
        raise AssertionError("alignment.lengths != rollout.lengths")

    # Build aligned label arrays for SUPP_4 + EXTENDED_9 (existing_11 already in labels_existing)
    supp_targets = [v for v in target_vars if v in SUPPLEMENTARY_4]
    ext_targets = [v for v in target_vars if v in EXTENDED_9]

    supp_aligned: dict = {}
    if supp_targets:
        logger.info("Aligning %d supplementary labels...", len(supp_targets))
        supp_aligned = build_aligned_labels(
            alignment, supplementary_dir,
            supp_targets, context_length, max_rollout, N,
        )

    ext_aligned: dict = {}
    if ext_targets:
        logger.info("Aligning %d extended labels...", len(ext_targets))
        ext_aligned = build_aligned_labels(
            alignment, extended_dir,
            ext_targets, context_length, max_rollout, N,
        )

    # Load all MLP probes
    logger.info("Loading %d MLP probes from %s", len(target_vars), probes_dir)
    probes: dict = {}
    for var in target_vars:
        p = probes_dir / f"probe_{var}.pt"
        if not p.exists():
            logger.warning("  Missing MLP probe: %s", p)
            continue
        probe, normalizer, metadata = load_mlp_probe(p, device=device)
        probes[var] = (probe, normalizer, metadata)
    logger.info("Loaded %d/%d probes", len(probes), len(target_vars))

    # Flatten predicted_z for batched probe inference
    predicted_flat = predicted_z.reshape(N * max_rollout, D)
    z_C = alignment["z_C"]  # (N, 4096)

    results: dict = {
        "plan": plan,
        "context_length": int(context_length),
        "model_name": "absolute",
        "probe_type": "mlp_4096_512_1_relu_dropout_0.1",
        "max_rollout": int(max_rollout),
        "n_samples": int(N),
        "ci_horizons": list(ci_horizons),
        "per_variable": {},
    }
    ci_step_indices = {h: h - 1 for h in ci_horizons if h - 1 < max_rollout}

    for var in target_vars:
        if var not in probes:
            continue
        probe, normalizer, meta = probes[var]
        ceiling_mae = meta.get("ceiling", {}).get("mae")
        log_xform = bool(normalizer.get("log_transform", False))

        t_var = time.time()

        # Apply MLP probe to predicted_z
        y_model_flat = apply_mlp_probe(
            probe, predicted_flat, normalizer,
            device=device, batch_size=65536,
        )
        y_model = y_model_flat.reshape(N, max_rollout)

        # Apply to z_C → decoded copy-forward baseline
        y_dec_cf = apply_mlp_probe(
            probe, z_C, normalizer,
            device=device, batch_size=65536,
        )

        # Resolve y_true source
        if var in labels_existing:
            y_true_full = np.asarray(labels_existing[var], dtype=np.float64)
        elif var in supp_aligned:
            y_true_full = supp_aligned[var].astype(np.float64)
        elif var in ext_aligned:
            y_true_full = ext_aligned[var].astype(np.float64)
        else:
            logger.warning("No y_true source for %s; skip", var)
            continue

        y_true_cf = y_true_full[:, 0].astype(np.float64)

        var_res = {
            "n_valid_per_step": [None] * max_rollout,
            "mae_per_step": [None] * max_rollout,
            "rmse_per_step": [None] * max_rollout,
            "mae_decoded_cf_per_step": [None] * max_rollout,
            "mae_true_cf_per_step": [None] * max_rollout,
            "mase_decoded_cf_per_step": [None] * max_rollout,
            "mase_true_cf_per_step": [None] * max_rollout,
            "mase_per_step": [None] * max_rollout,  # alias for decoded_cf
            "spearman_per_step": [None] * max_rollout,
            "mae_ci_at_horizons": {},
            "mase_decoded_cf_ci_at_horizons": {},
            "mase_true_cf_ci_at_horizons": {},
            "ceiling_mae": float(ceiling_mae) if ceiling_mae is not None else None,
            "log_transform": log_xform,
            "probe_arch": meta.get("arch"),
        }

        for step in range(max_rollout):
            y_m = y_model[:, step].astype(np.float64)
            y_t = y_true_full[:, step]
            y_bdec = y_dec_cf.astype(np.float64)
            y_btrue = y_true_cf
            valid_step = valid_mask[:, step] if step < valid_mask.shape[1] else np.ones(N, dtype=bool)
            valid = (
                ~np.isnan(y_m)
                & ~np.isnan(y_t)
                & ~np.isnan(y_bdec)
                & ~np.isnan(y_btrue)
                & valid_step
            )
            n_valid = int(valid.sum())
            var_res["n_valid_per_step"][step] = n_valid
            if n_valid < 10:
                continue

            y_m_v = y_m[valid]
            y_t_v = y_t[valid]
            y_bdec_v = y_bdec[valid]
            y_btrue_v = y_btrue[valid]
            stay_v = stay_ids[valid]

            mae_model = compute_mae(y_m_v, y_t_v)
            rmse_model = float(np.sqrt(np.mean((y_m_v - y_t_v) ** 2)))
            mae_decoded_cf = compute_mae(y_bdec_v, y_t_v)
            mae_true_cf = compute_mae(y_btrue_v, y_t_v)
            mase_decoded_cf = compute_mase(y_m_v, y_t_v, y_bdec_v)

            if step == 0 or not np.isfinite(mae_true_cf) or mae_true_cf < 1e-9:
                mase_true_cf = None
            else:
                mase_true_cf = float(mae_model / mae_true_cf)

            spearman = compute_spearman(y_m_v, y_t_v)

            var_res["mae_per_step"][step] = float(mae_model)
            var_res["rmse_per_step"][step] = rmse_model
            var_res["mae_decoded_cf_per_step"][step] = float(mae_decoded_cf)
            var_res["mae_true_cf_per_step"][step] = float(mae_true_cf)
            var_res["mase_decoded_cf_per_step"][step] = (
                float(mase_decoded_cf) if not np.isnan(mase_decoded_cf) else None
            )
            var_res["mase_true_cf_per_step"][step] = mase_true_cf
            var_res["mase_per_step"][step] = var_res["mase_decoded_cf_per_step"][step]
            var_res["spearman_per_step"][step] = float(spearman) if not np.isnan(spearman) else None

            if step in ci_step_indices.values():
                horizon_h = step + 1
                _, mae_lo, mae_hi = bootstrap_ci_mae_clustered(
                    y_m_v, y_t_v, stay_v,
                    n_bootstrap=cfg["metrics"]["bootstrap_iterations"],
                    confidence=cfg["metrics"]["bootstrap_confidence"],
                )
                _, mase_dec_lo, mase_dec_hi = bootstrap_ci_mase_clustered(
                    y_m_v, y_t_v, y_bdec_v, stay_v,
                    n_bootstrap=cfg["metrics"]["bootstrap_iterations"],
                    confidence=cfg["metrics"]["bootstrap_confidence"],
                )
                var_res["mae_ci_at_horizons"][horizon_h] = [
                    float(mae_lo) if not np.isnan(mae_lo) else None,
                    float(mae_hi) if not np.isnan(mae_hi) else None,
                ]
                var_res["mase_decoded_cf_ci_at_horizons"][horizon_h] = [
                    float(mase_dec_lo) if not np.isnan(mase_dec_lo) else None,
                    float(mase_dec_hi) if not np.isnan(mase_dec_hi) else None,
                ]
                if mase_true_cf is not None:
                    _, mase_true_lo, mase_true_hi = bootstrap_ci_mase_clustered(
                        y_m_v, y_t_v, y_btrue_v, stay_v,
                        n_bootstrap=cfg["metrics"]["bootstrap_iterations"],
                        confidence=cfg["metrics"]["bootstrap_confidence"],
                    )
                    var_res["mase_true_cf_ci_at_horizons"][horizon_h] = [
                        float(mase_true_lo) if not np.isnan(mase_true_lo) else None,
                        float(mase_true_hi) if not np.isnan(mase_true_hi) else None,
                    ]

        results["per_variable"][var] = var_res
        h1 = 0
        logger.info(
            "  %-22s done in %5.1fs | h=1 MAE=%.3f n=%d | ceiling_MAE=%.3f",
            var, time.time() - t_var,
            (var_res["mae_per_step"][h1] or float("nan")),
            (var_res["n_valid_per_step"][h1] or 0),
            (var_res["ceiling_mae"] or float("nan")),
        )

    out_path = out_dir / f"e2_decoded_C{context_length:02d}_absolute.pt"
    torch.save(results, out_path, pickle_protocol=4)
    logger.info("Saved: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    logger.info("Run total: %.1f s", time.time() - t_run)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply MLP probes (24 vars) to rollout outputs")
    parser.add_argument(
        "--plan",
        choices=["clin_jepa", "vjepa2ac", "sft_baseline"],
        required=True,
        help="Which paradigm's rollout outputs to apply probes to.",
    )
    parser.add_argument(
        "--config",
        default="configs/eval/probes.yaml",
        help="Path to probes.yaml (per-plan paths are read from cfg['plans'][plan]).",
    )
    parser.add_argument(
        "--probes_dir",
        default=None,
        help="Override the trained-probes directory (default: cfg['plans'][plan]['probes_dir']).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Override the metrics output directory (default: cfg['plans'][plan]['metrics_dir']).",
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[6, 12, 24, 48])
    parser.add_argument("--variables", nargs="+", default=None,
                        help="Subset of 24 vars (default: all)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--ci_horizons", nargs="+", type=int, default=DEFAULT_CI_HORIZONS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        logger.info("Using GPU %d: %s", args.gpu_id, torch.cuda.get_device_name(args.gpu_id))
    else:
        device = "cpu"
        logger.warning("No GPU — using CPU")

    plan = args.plan
    plan_cfg = cfg["plans"][plan]
    probes_dir = Path(args.probes_dir or plan_cfg.get("probes_dir") or f"outputs/probes/{plan}")
    out_dir = Path(args.out_dir or plan_cfg.get("metrics_dir") or f"outputs/metrics/{plan}")

    target_vars = args.variables or ALL_24
    unknown = set(target_vars) - set(ALL_24)
    if unknown:
        raise ValueError(f"Unknown variables: {unknown}")

    t0 = time.time()
    for C in args.contexts:
        try:
            apply_for_context(
                context_length=C, plan=plan, cfg=cfg,
                probes_dir=probes_dir, out_dir=out_dir,
                device=device, ci_horizons=args.ci_horizons,
                target_vars=target_vars,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed for Plan %s C=%d: %s", plan, C, e)
            raise

    logger.info("All contexts complete in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
