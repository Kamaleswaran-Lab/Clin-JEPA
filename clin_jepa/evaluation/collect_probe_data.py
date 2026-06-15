"""Collect probe-training data — pool encoder embeddings and clinical labels.

Walks the embedding shards produced by
:mod:`clin_jepa.evaluation.precompute_embeddings` and, for each requested
clinical variable, gathers ``(embedding, label)`` pairs into a single
per-variable file. A memory-efficient reservoir-subsampling strategy caps
each variable at ``--max_per_var`` samples to keep probes fast to train.

Usage::

    # Collect train + val for one paradigm:
    python -m clin_jepa.evaluation.collect_probe_data --plan clin_jepa

    # Single variable (for smoke):
    python -m clin_jepa.evaluation.collect_probe_data \\
        --plan clin_jepa --variables label_hr

    # Smaller cap for smoke:
    python -m clin_jepa.evaluation.collect_probe_data \\
        --plan clin_jepa --variables label_hr --max_per_var 10000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# Canonical continuous-label list (clinical variables decoded by the probes).
CONTINUOUS_LABELS = [
    # 11 existing labels in shard
    "label_sofa_total", "label_sofa_resp", "label_sofa_cardio",
    "label_sofa_renal", "label_sofa_hepatic", "label_sofa_coag",
    "label_sofa_neuro", "label_hr", "label_map", "label_sbp", "label_rr",
    # 4 supplementary labels (read from per-shard supplementary files)
    "label_temperature", "label_spo2", "label_lactate", "label_urine_output",
]
EXISTING_11 = set(CONTINUOUS_LABELS[:11])
NEW_4 = set(CONTINUOUS_LABELS[11:])


def collect_one_variable(
    var_name: str,
    is_existing_label: bool,
    embedding_dir: Path,
    supplementary_dir: Path,
    split: str,
    max_per_var: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Stream embedding shards to collect ``(z_state, y)`` pairs for one variable.

    Args:
        var_name: Label name (e.g. ``"label_hr"``).
        is_existing_label: ``True`` if the label is already stored in
            ``shard["windows"][i]["labels"]``; ``False`` if it lives in a
            supplementary file.
        embedding_dir: Root directory of the embedding cache for one
            paradigm (contains ``train/``, ``val/``, ``test/``).
        supplementary_dir: Root of the supplementary-label directory
            (same paradigm).
        split: ``"train"`` or ``"val"``.
        max_per_var: Cap on the number of samples per variable.
        seed: RNG seed for the reservoir subsampling.

    Returns:
        ``(z_tensor, y_tensor, n_total_seen)``.
    """
    split_dir = embedding_dir / split
    shard_files = sorted(split_dir.glob("embeddings_c*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No embedding shards in {split_dir}")

    z_buffer_list: list[torch.Tensor] = []
    y_buffer_list: list[torch.Tensor] = []
    buffer_size = 0
    n_total = 0
    rng = np.random.default_rng(seed + hash(var_name) % 10000)

    buffer_cap = int(max_per_var * 1.5)

    for shard_idx, shard_path in enumerate(shard_files):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        z_emb = shard["z_state_embeddings"]

        if not is_existing_label:
            supp_path = supplementary_dir / split / shard_path.name
            if not supp_path.exists():
                raise FileNotFoundError(
                    f"Supplementary labels missing: {supp_path}. "
                    "Run the supplementary-label extraction step first."
                )
            supp = torch.load(supp_path, map_location="cpu", weights_only=False)
        else:
            supp = None

        shard_zs = []
        shard_ys = []

        for win_idx, w in enumerate(shard["windows"]):
            emb_idx = w["emb_indices_state"]
            L = int(w["length"])

            if is_existing_label:
                y_raw = w["labels"][var_name]
            else:
                y_raw = supp[var_name][win_idx]

            y_t = torch.as_tensor(y_raw, dtype=torch.float32)
            if y_t.shape[0] != L:
                raise ValueError(
                    f"Label length mismatch for {var_name}: {y_t.shape[0]} vs {L}"
                )

            valid = ~torch.isnan(y_t)
            if not valid.any():
                continue

            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            # Gather only valid embeddings from z_state_embeddings
            z_valid = z_emb[emb_idx[valid_idx]]
            y_valid = y_t[valid_idx]

            shard_zs.append(z_valid)
            shard_ys.append(y_valid)
            n_total += len(z_valid)

        if shard_zs:
            shard_z = torch.cat(shard_zs, dim=0)
            shard_y = torch.cat(shard_ys, dim=0)
            z_buffer_list.append(shard_z)
            y_buffer_list.append(shard_y)
            buffer_size += len(shard_z)

        del shard, z_emb, supp, shard_zs, shard_ys
        if 'shard_z' in locals():
            del shard_z, shard_y
        gc.collect()

        if buffer_size > buffer_cap:
            Z = torch.cat(z_buffer_list, dim=0)
            Y = torch.cat(y_buffer_list, dim=0)
            del z_buffer_list, y_buffer_list
            gc.collect()

            idx = rng.choice(len(Z), size=max_per_var, replace=False)
            idx = torch.as_tensor(np.sort(idx), dtype=torch.long)
            Z = Z[idx].contiguous()
            Y = Y[idx].contiguous()

            z_buffer_list = [Z]
            y_buffer_list = [Y]
            buffer_size = len(Z)

            logger.info(
                "  [subsampled at shard %d] buffer=%d, n_total_seen=%d",
                shard_idx + 1, buffer_size, n_total,
            )
            del Z, Y
            gc.collect()

    if not z_buffer_list:
        return torch.empty(0, 4096, dtype=torch.float16), torch.empty(0, dtype=torch.float32), 0

    Z = torch.cat(z_buffer_list, dim=0)
    Y = torch.cat(y_buffer_list, dim=0)
    del z_buffer_list, y_buffer_list
    gc.collect()

    if len(Z) > max_per_var:
        idx = rng.choice(len(Z), size=max_per_var, replace=False)
        idx = torch.as_tensor(np.sort(idx), dtype=torch.long)
        Z = Z[idx].contiguous()
        Y = Y[idx].contiguous()

    return Z, Y, n_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect probe-training data")
    parser.add_argument(
        "--plan",
        choices=["clin_jepa", "vjepa2ac", "sft_baseline"],
        required=True,
        help="Which paradigm's embedding shards + output dir.",
    )
    parser.add_argument(
        "--config",
        default="configs/eval/probes.yaml",
        help="Path to probes.yaml (per-plan paths are read from cfg['plans'][plan]).",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--max_per_var",
        type=int,
        default=1_000_000,
        help="Cap samples per variable.",
    )
    parser.add_argument("--variables", nargs="+", default=None, help="Subset for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    plan_cfg = cfg["plans"][args.plan]
    embedding_dir = Path(plan_cfg["embedding_dir"])
    supplementary_dir = Path(plan_cfg.get("supplementary_dir", embedding_dir / "supplementary_labels"))
    output_dir = Path(plan_cfg["probe_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_vars = list(CONTINUOUS_LABELS)
    if args.variables:
        all_vars = [v for v in args.variables if v in CONTINUOUS_LABELS]
        if not all_vars:
            raise ValueError(
                f"No valid variables from --variables={args.variables}. "
                f"Known: {CONTINUOUS_LABELS}"
            )

    logger.info("Probe training data collection")
    logger.info("Plan:              %s", args.plan)
    logger.info("Embedding dir:     %s", embedding_dir)
    logger.info("Supplementary dir: %s", supplementary_dir)
    logger.info("Output dir:        %s", output_dir)
    logger.info("max_per_var:       %d", args.max_per_var)
    logger.info("Variables:         %d", len(all_vars))

    all_stats: dict = {}
    t_total_start = time.time()

    for split in args.splits:
        logger.info("=" * 60)
        logger.info("Plan %s — Split: %s", args.plan, split)
        logger.info("=" * 60)
        all_stats[split] = {}

        for var in all_vars:
            t0 = time.time()
            is_existing = var in EXISTING_11
            logger.info("[%s] Collecting %s (is_existing=%s)...", split, var, is_existing)

            Z, Y, n_total = collect_one_variable(
                var_name=var,
                is_existing_label=is_existing,
                embedding_dir=embedding_dir,
                supplementary_dir=supplementary_dir,
                split=split,
                max_per_var=args.max_per_var,
                seed=args.seed,
            )

            n_saved = len(Z)
            output_path = output_dir / f"{split}_{var}.pt"
            torch.save({"z": Z, "y": Y}, output_path)

            logger.info(
                "[%s] %s: n_total=%d, n_saved=%d, elapsed=%.1fs, file=%.1f MB",
                split, var, n_total, n_saved, time.time() - t0,
                output_path.stat().st_size / 1e6,
            )

            all_stats[split][var] = {
                "n_total": int(n_total),
                "n_saved": int(n_saved),
                "elapsed_sec": round(time.time() - t0, 1),
            }

            del Z, Y
            gc.collect()

    summary_path = output_dir / "collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"plan": args.plan, "stats": all_stats}, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    logger.info("=" * 60)
    logger.info("Plan %s total time: %.1f min", args.plan, (time.time() - t_total_start) / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
