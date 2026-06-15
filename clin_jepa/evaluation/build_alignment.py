"""Build the rollout-output alignment sidecar.

Replays the deterministic batching order used by
:mod:`clin_jepa.evaluation.rollout`, builds a per-sample bijection back to
the source trajectory shard, and gathers ``z_C`` (the last context-hour
state embedding) for later use as the decoded copy-forward MASE baseline
denominator.

Output payload schema::

    {
        "context_length":     int,
        "n_samples":          int,
        "plan":               str,
        "stay_ids":           (N,) int64,
        "lengths":            (N,) int32,
        "start_steps":        (N,) int32,
        "shard_idx":          (N,) int32,
        "window_idx_in_shard":(N,) int32,
        "shard_names":        [str, ...],
        "z_C":                (N, 4096) float16,
    }

Usage::

    # Build all four contexts for one paradigm:
    python -m clin_jepa.evaluation.build_alignment --plan clin_jepa

    # Single context only:
    python -m clin_jepa.evaluation.build_alignment --plan clin_jepa --contexts 12
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from clin_jepa.evaluation.rollout import create_batches, load_test_data

logger = logging.getLogger(__name__)


Z_C_NORM_EXPECTED_RANGE = {
    "vjepa2ac":     (5.0, 25.0),
    "clin_jepa":    (25.0, 80.0),
    "sft_baseline": (100.0, 400.0),
}


def build_alignment_for_context(
    context_length: int,
    plan: str,
    embedding_dir: Path,
    predictions_root: Path,
    batch_size: int,
    cross_check_model: str = "absolute",
    max_windows: int | None = None,
) -> None:
    """Build alignment sidecar for one (plan, context) combo and save it.

    For smoke testing: pass `max_windows` matching the value used in the
    corresponding rollout call so the cross-check against rollout's stay_ids
    + lengths passes.
    """
    logger.info("=" * 70)
    logger.info("Plan %s — Context length C=%d", plan.upper(), context_length)
    if max_windows:
        logger.info("max_windows=%d (smoke mode)", max_windows)
    logger.info("=" * 70)

    t0 = time.time()

    # ----------------------------------------------------------------------
    # Step 1: Reproduce rollout's data loading + batching
    # ----------------------------------------------------------------------
    shards, windows = load_test_data(
        embedding_dir=embedding_dir,
        split="test",
        max_windows=max_windows,
    )
    logger.info("Loaded %d test shards, %d total windows", len(shards), len(windows))

    global_to_within_shard: list[int] = []
    shard_counter: dict[int, int] = {}
    for w in windows:
        s_idx = int(w["_shard_idx"])
        w_in_shard = shard_counter.get(s_idx, 0)
        shard_counter[s_idx] = w_in_shard + 1
        global_to_within_shard.append(w_in_shard)
    assert len(global_to_within_shard) == len(windows), (
        f"global_to_within_shard length {len(global_to_within_shard)} "
        f"vs windows {len(windows)}"
    )

    # Replay rollout's exact batch creator
    batches, n_skipped = create_batches(
        windows=windows,
        context_length=context_length,
        batch_size=batch_size,
    )
    n_valid = sum(len(b) for b in batches)
    logger.info(
        "create_batches → %d batches, %d valid windows (%d skipped)",
        len(batches), n_valid, n_skipped,
    )

    # Walk batches in order → rollout sample indices
    aligned_global_indices: list[int] = []
    for batch in batches:
        aligned_global_indices.extend(batch)
    N = len(aligned_global_indices)
    logger.info("Total aligned rollout samples: N=%d", N)

    # ----------------------------------------------------------------------
    # Step 2: Build alignment arrays + gather z_C from z_state_embeddings
    # ----------------------------------------------------------------------
    stay_ids       = np.zeros(N, dtype=np.int64)
    lengths        = np.zeros(N, dtype=np.int32)
    start_steps    = np.zeros(N, dtype=np.int32)
    shard_idx_arr  = np.zeros(N, dtype=np.int32)
    window_idx_arr = np.zeros(N, dtype=np.int32)
    z_C            = torch.zeros(N, 4096, dtype=torch.float16)

    k = context_length - 1  # 0-indexed last-context-hour

    for i, g_idx in enumerate(aligned_global_indices):
        w = windows[g_idx]
        s_idx = int(w["_shard_idx"])
        w_in_shard = int(global_to_within_shard[g_idx])

        stay_ids[i]       = int(w["stay_id"])
        lengths[i]        = int(w["length"])
        start_steps[i]    = int(w["start_step"])
        shard_idx_arr[i]  = s_idx
        window_idx_arr[i] = w_in_shard

        # Gather state embeddings via the two-table shard schema.
        emb_idx = int(w["emb_indices_state"][k])
        z_C[i] = shards[s_idx]["z_state_embeddings"][emb_idx]

    elapsed_build = time.time() - t0
    logger.info("Built alignment arrays in %.1f s", elapsed_build)

    # ----------------------------------------------------------------------
    # Step 3: CROSS-CHECK against saved rollout prediction file
    # ----------------------------------------------------------------------
    pred_path = (
        predictions_root
        / f"context_{context_length:02d}"
        / f"predictions_{cross_check_model}.pt"
    )
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Cannot cross-check: rollout prediction file missing: {pred_path}\n"
            f"Make sure `python -m clin_jepa.evaluation.rollout` has completed for plan={plan} C={context_length}."
        )
    logger.info("Cross-checking against %s", pred_path)
    rollout_out = torch.load(pred_path, map_location="cpu", weights_only=False)
    rollout_stay_ids = np.asarray(rollout_out["stay_ids"], dtype=np.int64)
    rollout_lengths  = np.asarray(rollout_out["lengths"],  dtype=np.int32)

    if len(rollout_stay_ids) != N:
        raise AssertionError(
            f"Plan {plan} C={context_length}: N mismatch — "
            f"rollout has {len(rollout_stay_ids)} samples, alignment has {N}"
        )
    if not np.array_equal(stay_ids, rollout_stay_ids):
        n_mismatch = int((stay_ids != rollout_stay_ids).sum())
        first_bad = int(np.argmax(stay_ids != rollout_stay_ids))
        raise AssertionError(
            f"Plan {plan} C={context_length}: stay_ids mismatch! "
            f"{n_mismatch} positions differ. First bad idx {first_bad}: "
            f"aligned={stay_ids[first_bad]} rollout={rollout_stay_ids[first_bad]}"
        )
    if not np.array_equal(lengths, rollout_lengths):
        n_mismatch = int((lengths != rollout_lengths).sum())
        first_bad = int(np.argmax(lengths != rollout_lengths))
        raise AssertionError(
            f"Plan {plan} C={context_length}: lengths mismatch! "
            f"{n_mismatch} positions differ. First bad idx {first_bad}: "
            f"aligned={lengths[first_bad]} rollout={rollout_lengths[first_bad]}"
        )
    logger.info(
        "Cross-check passed: stay_ids + lengths match rollout exactly (N=%d)", N
    )

    del rollout_out, rollout_stay_ids, rollout_lengths

    # ----------------------------------------------------------------------
    # Step 4: Save sidecar
    # ----------------------------------------------------------------------
    shard_files = sorted((embedding_dir / "test").glob("embeddings_c*.pt"))
    shard_names = [p.name for p in shard_files]

    out_path = (
        predictions_root
        / f"context_{context_length:02d}"
        / "rollout_alignment.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "context_length": int(context_length),
        "n_samples": int(N),
        "plan": plan,
        "stay_ids": stay_ids,
        "lengths": lengths,
        "start_steps": start_steps,
        "shard_idx": shard_idx_arr,
        "window_idx_in_shard": window_idx_arr,
        "shard_names": shard_names,
        "z_C": z_C,
    }
    torch.save(payload, out_path, pickle_protocol=4)
    file_size_mb = out_path.stat().st_size / 1e6
    logger.info("Saved %s (%.1f MB)", out_path, file_size_mb)

    # ----------------------------------------------------------------------
    # z_C norm sanity check — plan-aware
    # ----------------------------------------------------------------------
    z_C_norms = z_C.float().norm(dim=-1)
    mean_norm = float(z_C_norms.mean())
    logger.info(
        "z_C norm stats: mean=%.2f std=%.2f min=%.2f max=%.2f",
        mean_norm, float(z_C_norms.std()),
        float(z_C_norms.min()), float(z_C_norms.max()),
    )
    expected_lo, expected_hi = Z_C_NORM_EXPECTED_RANGE.get(plan, (None, None))
    if expected_lo is not None and not (expected_lo <= mean_norm <= expected_hi):
        logger.warning(
            "z_C norm mean %.2f for %s is outside the expected range [%s, %s]",
            mean_norm, plan, expected_lo, expected_hi,
        )
    else:
        logger.info(
            "z_C norm mean %.2f for %s is within the expected range [%s, %s]",
            mean_norm, plan, expected_lo, expected_hi,
        )

    del shards, windows, batches, aligned_global_indices, z_C
    gc.collect()

    logger.info(
        "Plan %s C=%d alignment complete in %.1f s total",
        plan, context_length, time.time() - t0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build rollout alignment sidecars",
    )
    parser.add_argument(
        "--plan",
        choices=["clin_jepa", "vjepa2ac", "sft_baseline"],
        required=True,
        help="Which paradigm's rollout output to align.",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[6, 12, 24, 48],
        help="Context lengths to build alignment for.",
    )
    parser.add_argument(
        "--config",
        default="configs/eval/rollout.yaml",
        help="Path to rollout.yaml (per-plan embedding_dir + batch_size; must match the rollout run).",
    )
    parser.add_argument(
        "--cross_check_model",
        default="absolute",
        help="Which rollout prediction file to cross-check against",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Cap on windows loaded (smoke mode — must match the rollout step's --max_windows)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    plan_cfg = cfg["plans"][args.plan]
    embedding_dir = Path(plan_cfg["embedding_dir"])
    predictions_root = Path(plan_cfg["output_dir"])
    batch_size = int(cfg["rollout"]["batch_size"])

    logger.info("Build rollout alignment sidecars")
    logger.info("plan:                %s", args.plan)
    logger.info("embedding_dir:       %s", embedding_dir)
    logger.info("predictions_root: %s", predictions_root)
    logger.info("batch_size:          %d", batch_size)
    logger.info("contexts:            %s", args.contexts)
    logger.info("cross_check_model:   %s", args.cross_check_model)

    t_start_all = time.time()
    for C in args.contexts:
        build_alignment_for_context(
            context_length=C,
            plan=args.plan,
            embedding_dir=embedding_dir,
            predictions_root=predictions_root,
            batch_size=batch_size,
            cross_check_model=args.cross_check_model,
            max_windows=args.max_windows,
        )

    logger.info("=" * 70)
    logger.info(
        "ALL ALIGNMENTS BUILT for plan %s in %.1f s",
        args.plan, time.time() - t_start_all,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
