"""Precompute encoder embeddings for one of the three training paradigms.

Loads the selected paradigm's encoder LoRA (Clin-JEPA, V-JEPA 2-AC, or the
SFT baseline) and encodes every (stay_id, hour) state text, every unique
action text, and every demographics text into a 4096-D last-token hidden
state. Output shards use the two-table schema consumed by the rollout and
probe pipelines.

Schema per output shard:

    {
        "z_state_embeddings":  Tensor(N_state_unique,  4096) float16,
        "z_action_embeddings": Tensor(N_action_unique, 4096) float16,
        "z_statics":           Tensor(N_stays,         4096) float16,
        "windows": list[{
            "stay_id": int,
            "start_step": int,
            "length": int,
            "emb_indices_state":  Tensor(length,) int32,  -> z_state_embeddings
            "emb_indices_action": Tensor(length,) int32,  -> z_action_embeddings
            "static_idx": int,                             -> z_statics
            "labels": dict[str, Tensor(length,) float32], # 11 clinical labels
            "binary_labels": dict[str, bool],             # 4 outcome labels
        }],
        "schema_version": "clin_jepa_embeddings_v1",
    }

Usage::

    # All shards in one split (single GPU):
    python -m clin_jepa.evaluation.precompute_embeddings \\
        --config configs/eval/precompute_embeddings.yaml \\
        --plan clin_jepa --split train --gpu_id 0

    # Specific shard indices (for parallel multi-GPU):
    python -m clin_jepa.evaluation.precompute_embeddings \\
        --config configs/eval/precompute_embeddings.yaml \\
        --plan clin_jepa --split train --shard_indices 0,1,2,3 --gpu_id 0

    # Stride mode (4 GPUs each take every 4th shard):
    python -m clin_jepa.evaluation.precompute_embeddings \\
        --config configs/eval/precompute_embeddings.yaml \\
        --plan clin_jepa --split train --shard_stride 4 --shard_offset 0 \\
        --gpu_id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

_HF_ENV_APPLIED = False


def _preconfigure_hf_env() -> None:
    """Set HF_HOME / offline env vars BEFORE any HF-adjacent import.

    Reads --config from sys.argv if present; otherwise a no-op (falls back to
    SLURM-exported HF_HOME). Safe to call multiple times.
    """
    global _HF_ENV_APPLIED
    if _HF_ENV_APPLIED:
        return
    _HF_ENV_APPLIED = True

    import sys as _sys
    cfg_path: Optional[str] = None
    args_iter = iter(_sys.argv[1:])
    for tok in args_iter:
        if tok == "--config":
            cfg_path = next(args_iter, None)
            break
        if tok.startswith("--config="):
            cfg_path = tok.split("=", 1)[1]
            break
    if cfg_path and os.path.exists(cfg_path):
        try:
            import yaml as _yaml
            with open(cfg_path) as _f:
                _cfg = _yaml.safe_load(_f)
            _hf = (_cfg or {}).get("hf", {}) or {}
            if _hf.get("cache_dir") and not os.environ.get("HF_HOME"):
                os.environ["HF_HOME"] = _hf["cache_dir"]
            if _hf.get("offline"):
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        except Exception:
            # If pre-parse fails, let SLURM env handle it.
            pass


_preconfigure_hf_env()

import torch
import yaml

from clin_jepa.training.encoder_ops import (
    PROJECT_ROOT,
    atomic_torch_save,
    encode_texts_batched,
    load_sft_initialized_encoder,
)
from clin_jepa.training.trajectory_dataset import _build_demo_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def should_skip_shard(shard_path: Path, output_dir: Path) -> bool:
    """Resume support: skip if output already exists."""
    output_name = shard_path.stem.replace("trajectories_", "embeddings_") + ".pt"
    return (output_dir / output_name).exists()


def process_shard(
    shard_path: Path,
    model,
    tokenizer,
    config: dict,
    device: torch.device,
    output_dir: Path,
    max_windows: Optional[int] = None,
) -> dict:
    """Process one trajectory shard into one embedding shard.

    Pipeline:
      1. Load shard (per_hour state/action texts + per_stay demographics + windows)
      2. Encode unique state texts (per_hour is already deduped)
      3. Dedup + encode unique action texts (string hash dedup)
      4. Encode demographics texts (one per stay)
      5. Assemble per-window output with index arrays
      6. Save deduped .pt
    """
    t0 = time.time()
    inf_cfg = config["inference"]
    max_seq_len = config["data"]["max_seq_len"]
    emb_dtype = getattr(torch, inf_cfg.get("embedding_dtype", "float16"))
    schema_version = config.get("output", {}).get("schema_version", "clin_jepa_embeddings_v1")

    # Step 1: Load shard
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    ph = shard["per_hour"]
    phl = shard["per_hour_labels"]
    ps = shard["per_stay"]
    windows = shard["windows"]

    if max_windows is not None and len(windows) > max_windows:
        windows = windows[:max_windows]
        # Collect referenced hour and stay indices
        referenced_hours = sorted({int(h) for w in windows for h in w["hour_indices"].tolist()})
        referenced_stays = sorted({int(w["stay_index"]) for w in windows})
        old_to_new_hour = {old: new for new, old in enumerate(referenced_hours)}
        old_to_new_stay = {old: new for new, old in enumerate(referenced_stays)}
        # Re-project per_hour tables
        import numpy as _np
        ph = {
            "stay_ids":    ph["stay_ids"][referenced_hours],
            "hours":       ph["hours"][referenced_hours],
            "state_texts":  [ph["state_texts"][i] for i in referenced_hours],
            "action_texts": [ph["action_texts"][i] for i in referenced_hours],
        }
        phl = {k: v[referenced_hours] for k, v in phl.items()}
        # Re-project per_stay tables
        ps = {
            "stay_ids":            ps["stay_ids"][referenced_stays],
            "subject_ids":         ps["subject_ids"][referenced_stays] if "subject_ids" in ps else None,
            "ages":                ps["ages"][referenced_stays],
            "genders":             [ps["genders"][i] for i in referenced_stays],
            "races":               [ps["races"][i] for i in referenced_stays],
            "charlson_cis":        ps["charlson_cis"][referenced_stays] if "charlson_cis" in ps else None,
            "icu_mortalities":     ps["icu_mortalities"][referenced_stays],
            "hospital_mortalities":ps["hospital_mortalities"][referenced_stays],
            "prolonged_stays":     ps["prolonged_stays"][referenced_stays],
            "sepsis3s":            ps["sepsis3s"][referenced_stays],
        }
        # Remap window hour_indices + stay_index to the new compact tables
        windows = [
            {
                **{k: v for k, v in w.items() if k not in ("hour_indices", "stay_index")},
                "hour_indices": _np.asarray([old_to_new_hour[int(h)] for h in w["hour_indices"].tolist()], dtype=_np.int32),
                "stay_index":   old_to_new_stay[int(w["stay_index"])],
            }
            for w in windows
        ]
        logger.info("  SMOKE mode: truncated to %d windows, %d unique hours, %d unique stays",
                    len(windows), len(ph["state_texts"]), len(ps["stay_ids"]))

    n_unique_hours = len(ph["state_texts"])
    n_unique_stays = len(ps["stay_ids"])

    logger.info(
        "  Loaded %s: %d windows, %d unique hours, %d unique stays",
        shard_path.name, len(windows), n_unique_hours, n_unique_stays,
    )

    # Step 2: Encode state texts (already deduped)
    state_texts = ph["state_texts"]  # list[str]
    logger.info("  Encoding %d unique state texts...", len(state_texts))
    z_state = encode_texts_batched(
        model, tokenizer, state_texts, device=device,
        max_seq_len=max_seq_len, train_mode=False, output_dtype=emb_dtype,
    )  # (N_unique_hours, 4096)

    # Step 3: Dedup action texts by string hash, then encode unique set
    unique_action_texts: list[str] = []
    action_str_to_idx: dict[str, int] = {}
    action_idx_per_hour: list[int] = []  # length N_unique_hours

    for at in ph["action_texts"]:
        if at not in action_str_to_idx:
            action_str_to_idx[at] = len(unique_action_texts)
            unique_action_texts.append(at)
        action_idx_per_hour.append(action_str_to_idx[at])

    n_unique_action = len(unique_action_texts)
    dedup_ratio = len(ph["action_texts"]) / max(n_unique_action, 1)
    logger.info(
        "  Action dedup: %d -> %d unique (%.1fx ratio)",
        len(ph["action_texts"]), n_unique_action, dedup_ratio,
    )

    logger.info("  Encoding %d unique action texts...", n_unique_action)
    z_action = encode_texts_batched(
        model, tokenizer, unique_action_texts, device=device,
        max_seq_len=max_seq_len, train_mode=False, output_dtype=emb_dtype,
    )  # (N_unique_action, 4096)

    # Step 4: Encode demographics (one text per stay)
    demo_texts = [
        _build_demo_text(
            age=int(round(float(ps["ages"][i]))),
            gender=ps["genders"][i],
            race=ps["races"][i],
        )
        for i in range(n_unique_stays)
    ]
    logger.info("  Encoding %d demographics texts...", n_unique_stays)
    z_static = encode_texts_batched(
        model, tokenizer, demo_texts, device=device,
        max_seq_len=max_seq_len, train_mode=False, output_dtype=emb_dtype,
    )  # (N_unique_stays, 4096)

    # Step 5: Assemble per-window output
    windows_out = []
    for w in windows:
        hi_arr = w["hour_indices"]  # numpy int32
        hi = hi_arr.tolist()
        si = int(w["stay_index"])
        L = int(w["length"])

        # State embeddings: index via per-hour position
        emb_indices_state = torch.tensor(hi, dtype=torch.int32)

        # Action embeddings: route through action_idx_per_hour mapping
        emb_indices_action = torch.tensor(
            [action_idx_per_hour[h] for h in hi],
            dtype=torch.int32,
        )

        # Per-hour labels (gather from per_hour_labels arrays)
        labels = {
            k: torch.from_numpy(phl[k][hi].copy())  # (L,) float32
            for k in phl
        }

        # Per-stay binary outcomes
        binary_labels = {
            "icu_mortality":      bool(ps["icu_mortalities"][si]),
            "hospital_mortality": bool(ps["hospital_mortalities"][si]),
            "prolonged_stay":     bool(ps["prolonged_stays"][si]),
            "sepsis3":            bool(ps["sepsis3s"][si]),
        }

        windows_out.append({
            "stay_id":            int(ps["stay_ids"][si]),
            "start_step":         int(w["start_step"]),
            "length":             L,
            "emb_indices_state":  emb_indices_state,
            "emb_indices_action": emb_indices_action,
            "static_idx":         si,
            "labels":             labels,
            "binary_labels":      binary_labels,
        })

    out_name = shard_path.stem.replace("trajectories_", "embeddings_") + ".pt"
    output_path = output_dir / out_name
    atomic_torch_save({
        "z_state_embeddings":  z_state,
        "z_action_embeddings": z_action,
        "z_statics":           z_static,
        "windows":             windows_out,
        "schema_version":      schema_version,
    }, output_path)

    elapsed = time.time() - t0
    file_mb = output_path.stat().st_size / 1e6
    stats = {
        "shard": shard_path.name,
        "n_windows": len(windows_out),
        "n_unique_state": n_unique_hours,
        "n_unique_action": n_unique_action,
        "n_unique_stays": n_unique_stays,
        "time_seconds": elapsed,
        "output_mb": file_mb,
    }
    logger.info(
        "  Saved %s (%.0f MB, %d windows, %.0fs)",
        out_name, file_mb, len(windows_out), elapsed,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute encoder embeddings for downstream evaluation.",
    )
    parser.add_argument(
        "--config",
        default="configs/eval/precompute_embeddings.yaml",
        help="Path to precompute_embeddings.yaml.",
    )
    parser.add_argument(
        "--plan",
        choices=["clin_jepa", "vjepa2ac", "sft_baseline"],
        required=True,
        help="Which paradigm's encoder to use (selects checkpoint_dir + output_dir "
             "from cfg['plans'][plan]).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Data split to encode.",
    )
    parser.add_argument(
        "--shard_indices",
        type=str,
        default=None,
        help="Comma-separated shard indices to process. None means all.",
    )
    parser.add_argument(
        "--shard_stride",
        type=int,
        default=None,
        help="Process every Nth shard (parallel mode across GPUs).",
    )
    parser.add_argument(
        "--shard_offset",
        type=int,
        default=0,
        help="Offset into the shard-stride sweep.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Smoke-test option: truncate each shard to the first N windows. "
             "Do not use for full runs.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Merge plan-specific overrides into top-level config.
    plan_cfg = config.get("plans", {}).get(args.plan)
    if plan_cfg is None:
        raise KeyError(
            f"--plan {args.plan} not found in config['plans']. "
            f"Available: {list(config.get('plans', {}).keys())}"
        )
    config.setdefault("model", {})["checkpoint_dir"] = plan_cfg["checkpoint_dir"]
    config.setdefault("data", {})["output_dir"] = plan_cfg["output_dir"]


    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Precompute embeddings: plan=%s split=%s device=%s", args.plan, args.split, device)

    # Load refined encoder
    model, tokenizer = load_sft_initialized_encoder(config, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Discover input shards
    trajectory_dir = PROJECT_ROOT / config["data"]["trajectory_dir"] / args.split
    all_shards = sorted(trajectory_dir.glob("trajectories_*.pt"))
    logger.info("Found %d shards in %s", len(all_shards), trajectory_dir)

    if args.shard_indices:
        indices = [int(x) for x in args.shard_indices.split(",")]
        shards = [all_shards[i] for i in indices if i < len(all_shards)]
    elif args.shard_stride:
        shards = all_shards[args.shard_offset::args.shard_stride]
    else:
        shards = all_shards

    output_dir = PROJECT_ROOT / config["data"]["output_dir"] / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    for shard_path in shards:
        if should_skip_shard(shard_path, output_dir):
            logger.info("SKIP (exists): %s", shard_path.name)
            continue
        try:
            stats = process_shard(
                shard_path, model, tokenizer, config, device, output_dir,
                max_windows=args.max_windows,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error("FAILED on %s: %s", shard_path.name, e)
            raise

    if all_stats:
        total_windows = sum(s["n_windows"] for s in all_stats)
        total_state = sum(s["n_unique_state"] for s in all_stats)
        total_action = sum(s["n_unique_action"] for s in all_stats)
        total_time = sum(s["time_seconds"] for s in all_stats)
        total_mb = sum(s["output_mb"] for s in all_stats)
        logger.info(
            "COMPLETE: %d shards, %d windows, %d state texts, %d action texts, %.0f MB, %.1f min",
            len(all_stats), total_windows, total_state, total_action, total_mb, total_time / 60,
        )

        # Save summary stats
        summary = {
            "split": args.split,
            "n_shards_processed": len(all_stats),
            "total_windows": total_windows,
            "total_unique_state": total_state,
            "total_unique_action": total_action,
            "total_seconds": total_time,
            "total_output_mb": total_mb,
            "per_shard": all_stats,
        }
        summary_path = output_dir / f"_summary_gpu{args.gpu_id}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        logger.info("No shards processed (all already exist).")


if __name__ == "__main__":
    main()
