"""Trajectory extraction — turn discretized events into per-hour trajectory shards.

Reads the per-stay discretized pickles produced by
:mod:`clin_jepa.data.step04_discretization` and emits deduplicated per-hour
trajectory shards. Each shard contains:

* a ``state_text`` per (stay, hour) — the encoder-ready textual description
  of all observations recorded during that hour;
* an ``action_text`` per (stay, hour) — the textual description of clinical
  interventions during that hour, built after a two-stage noise filter
  (see :mod:`clin_jepa.data.helpers.action_filter`);
* per-hour continuous labels (clinical-target variables);
* per-stay metadata and thin window references that index back into the
  per-hour text/label tables.

Output is one ``.pt`` shard per group of stays (via :func:`torch.save`).
See paper §3.1 for the encoder-input text design.

Usage::

    python -m clin_jepa.data.step06_trajectories \\
        --config configs/data/trajectories.yaml

    # Smoke test on 5 stays, single worker:
    python -m clin_jepa.data.step06_trajectories \\
        --config configs/data/trajectories.yaml --n_workers 1 --max_stays 5
"""

import argparse
import json
import logging
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from clin_jepa.data.helpers.action_filter import filter_act_events
from clin_jepa.data.helpers.event_ordering import (
    _has_value,
    _sort_events,
)
from clin_jepa.data.helpers.text_templates import format_event
from clin_jepa.utils import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State + action text generation (keep-all, no intra-hour dedup)
# ---------------------------------------------------------------------------


def _format_state_text_keep_all(
    obs_events_per_step: pd.DataFrame,
    step_idx: int,
) -> str:
    """Format one step's observations preserving every reading.

    Stable sort keeps same-variable rows in chronological order.
    """
    if len(obs_events_per_step) == 0:
        return f"t={step_idx}h | No observations recorded."

    sorted_events = _sort_events(obs_events_per_step)

    fragments = []
    for _, row in sorted_events.iterrows():
        if not _has_value(row.get("numeric_value"), row.get("text_value")):
            continue
        fragment = format_event(
            row["source_table"],
            row["variable_name"],
            row.get("numeric_value"),
            row.get("text_value"),
            row.get("unit", ""),
        )
        if fragment is not None:
            fragments.append(fragment)

    if not fragments:
        return f"t={step_idx}h | No observations recorded."

    body = ". ".join(fragments)
    return f"t={step_idx}h | {body}"


def _format_action_text_keep_all(
    act_events_per_step: pd.DataFrame,
    step_idx: int,
) -> str:
    """Format one step's actions preserving every event.

    Caller must have already applied filter_act_events(). Intra-hour
    duplicates are preserved in chronological order.
    """
    if len(act_events_per_step) == 0:
        return f"t={step_idx}h | No active interventions."

    sorted_events = _sort_events(act_events_per_step)

    fragments = []
    for _, row in sorted_events.iterrows():
        if not _has_value(row.get("numeric_value"), row.get("text_value")):
            continue
        fragment = format_event(
            row["source_table"],
            row["variable_name"],
            row.get("numeric_value"),
            row.get("text_value"),
            row.get("unit", ""),
        )
        if fragment is not None:
            fragments.append(fragment)

    if not fragments:
        return f"t={step_idx}h | No active interventions."

    body = ". ".join(fragments)
    return f"t={step_idx}h | {body}"


def build_state_text(
    obs_events_per_step: pd.DataFrame,
    step_idx: int,
) -> str:
    """Format one step's observations as a state-text line (keep-all).

    Args:
        obs_events_per_step: slice of obs_events for a single step_idx, AFTER
            label-leakage exclusions are applied upstream.
        step_idx: integer hour index within the stay.

    Returns:
        A single string like ``"t=12h | Heart rate: 88 bpm. MAP: 56 mmHg. ..."``
        or the empty-sentinel ``"t=12h | No observations recorded."``.
    """
    return _format_state_text_keep_all(obs_events_per_step, step_idx)


def build_action_text(
    act_events_per_step: pd.DataFrame,
    step_idx: int,
) -> str:
    """Format one step's actions as an action-text line (keep-all).

    Args:
        act_events_per_step: slice of act_events for a single step_idx, AFTER
            action_filter.filter_act_events() has already been applied
            upstream (noise drops + prescriptions continuous-fan-out collapse).
        step_idx: integer hour index within the stay.

    Returns:
        A single string like ``"t=12h | Norepinephrine: 0.08 mcg/kg/min. ..."``
        or the empty-sentinel ``"t=12h | No active interventions."``.
    """
    return _format_action_text_keep_all(act_events_per_step, step_idx)


def build_state_texts(
    obs_events: pd.DataFrame,
    n_steps: int,
) -> list[str]:
    """Build all per-hour state texts for one stay.

    Groups obs_events by step_idx once, then iterates to avoid O(n^2)
    filtering. Caller is responsible for having applied label-leakage
    exclusions to obs_events before this function.
    """
    if len(obs_events) == 0:
        return [f"t={s}h | No observations recorded." for s in range(n_steps)]
    grouped = dict(list(obs_events.groupby("step_idx")))
    empty = pd.DataFrame(columns=obs_events.columns)
    return [
        build_state_text(grouped.get(step, empty), step)
        for step in range(n_steps)
    ]


def build_action_texts(
    act_events: pd.DataFrame,
    n_steps: int,
) -> list[str]:
    """Build all per-hour action texts for one stay.

    Groups act_events by step_idx once, then iterates. Caller is responsible
    for having applied ``filter_act_events()`` to act_events before this
    function.
    """
    if len(act_events) == 0:
        return [f"t={s}h | No active interventions." for s in range(n_steps)]
    grouped = dict(list(act_events.groupby("step_idx")))
    empty = pd.DataFrame(columns=act_events.columns)
    return [
        build_action_text(grouped.get(step, empty), step)
        for step in range(n_steps)
    ]


# ---------------------------------------------------------------------------
# Per-step label extraction
# ---------------------------------------------------------------------------

def extract_labels(
    obs_events: pd.DataFrame,
    n_steps: int,
    config: dict,
) -> dict[str, np.ndarray]:
    """Extract per-step label arrays for linear probe targets.

    Returns:
        Dict of label_name -> (n_steps,) float32 arrays. NaN where no measurement.
    """
    label_targets = config["label_targets"]
    labels = {}

    for target in label_targets:
        arr = np.full(n_steps, np.nan, dtype=np.float32)
        mask = (obs_events["variable_name"] == target["variable_name"]) & (
            obs_events["source_table"] == target["source_table"]
        )
        rows = obs_events.loc[mask]
        if len(rows) > 0:
            if target["agg"] == "mean":
                grouped = rows.groupby("step_idx")["numeric_value"].mean()
            else:  # first
                grouped = rows.groupby("step_idx")["numeric_value"].first()
            for step_idx, val in grouped.items():
                if 0 <= step_idx < n_steps and pd.notna(val):
                    arr[int(step_idx)] = val
        labels[target["name"]] = arr

    return labels


# ---------------------------------------------------------------------------
# Per-stay bundle + window slicing
# ---------------------------------------------------------------------------


def slice_windows(stay_data: dict, config: dict) -> list[tuple[int, int]]:
    """Return list of (start_step, length) window positions for this stay.

    Asymmetric window rule:
        if n_steps >= t_max: sliding windows of length t_max with stride
        else:                one window of length n_steps (if >= min_len)
    """
    t_max = config["t_max"]
    min_len = config["min_len"]
    stride = config["stride"]
    n_steps = stay_data["n_steps"]

    if n_steps < min_len:
        return []

    if n_steps >= t_max:
        starts = range(0, n_steps - t_max + 1, stride)
    else:
        starts = [0]

    out: list[tuple[int, int]] = []
    for start in starts:
        length = min(t_max, n_steps - start)
        if length < min_len:
            continue
        out.append((int(start), int(length)))
    return out


def build_per_stay_metadata(stay_data: dict) -> dict:
    """Extract per-stay metadata fields that go into the per_stay block of
    the output shard. One entry per (stay_id).

    ``hospital_mortality`` is an alias for ``icu_mortality`` (same underlying
    ``mortality`` field).
    """
    metadata = stay_data["metadata"]
    icu_mortality = bool(metadata.get("mortality", False))
    return {
        "stay_id": int(stay_data["stay_id"]),
        "subject_id": int(metadata.get("subject_id", 0)),
        "age": float(metadata.get("age", 0.0)),
        "gender": str(metadata.get("gender", "")),
        "race": str(metadata.get("race", "")),
        "charlson_ci": int(metadata.get("charlson_ci", 0)),
        "icu_mortality": icu_mortality,
        "hospital_mortality": icu_mortality,  # alias
        "prolonged_stay": float(metadata.get("icu_los_hours", 0)) > 168.0,
        "sepsis3": bool(metadata.get("sepsis3", False)),
    }


# ---------------------------------------------------------------------------
# Per-stay processing
# ---------------------------------------------------------------------------

def process_stay(args: tuple) -> dict | None:
    """Process a single Step 04 pickle file into a per-stay bundle.

    Args:
        args: (pkl_path, config) tuple for multiprocessing compatibility.

    Returns:
        Per-stay bundle dict:
          {
            "stay_id":      int,
            "n_steps":      int,
            "state_texts":  list[str] of length n_steps,
            "action_texts": list[str] of length n_steps,
            "labels":       dict[str, ndarray(n_steps,)],   # 15 labels
            "per_stay":     dict (age/gender/race/mortality/...),
            "window_positions": list[(start_step, length)],
          }
        Or None if the stay has no windows (shorter than min_len).
    """
    pkl_path, config = args
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        n_steps = data["n_steps"]
        obs_events = data["obs_events"]
        act_events = data["act_events"]

        # 1. Apply label-leakage exclusions to observations (read list from config)
        exclude_cats = set(config["state_text"]["exclude_categories"])
        exclude_vars = set(config["state_text"]["exclude_variables"])
        if len(obs_events) > 0:
            obs_events = obs_events.loc[
                ~obs_events["category"].isin(exclude_cats)
                & ~obs_events["variable_name"].isin(exclude_vars)
            ].reset_index(drop=True)

        # 2. Apply action noise filter + prescriptions collapse
        act_events = filter_act_events(act_events)

        # 3. Generate state + action texts
        state_texts = build_state_texts(obs_events, n_steps)
        action_texts = build_action_texts(act_events, n_steps)

        # 4. Extract 15 per-hour labels
        labels = extract_labels(obs_events, n_steps, config)

        # 5. Compute window positions (asymmetric rule, see slice_windows)
        window_positions = slice_windows(data, config)
        if len(window_positions) == 0:
            return None

        return {
            "stay_id": int(data["stay_id"]),
            "n_steps": int(n_steps),
            "state_texts": state_texts,
            "action_texts": action_texts,
            "labels": labels,
            "per_stay": build_per_stay_metadata(data),
            "window_positions": window_positions,
        }

    except Exception as e:
        logger.error("Failed to process %s: %s", pkl_path, e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Shard assembly (per-shard dedup at (stay_id, hour) level)
# ---------------------------------------------------------------------------


def _make_empty_shard(shard_idx: int, chunk_idx: int | None, label_names: list[str]) -> dict:
    """Initialize an empty per-shard accumulator."""
    return {
        "per_hour_key_to_index": {},  # (stay_id, hour) -> int (working only)
        "per_hour_stay_ids": [],
        "per_hour_hours": [],
        "per_hour_state_texts": [],
        "per_hour_action_texts": [],
        "per_hour_labels": {name: [] for name in label_names},
        "per_stay_key_to_index": {},  # stay_id -> int (working only)
        "per_stay_rows": [],           # list of per-stay metadata dicts
        "windows": [],                  # list of {stay_index, start_step, length, hour_indices}
        "schema_version": "clin_jepa_trajectories_v1",
        "shard_idx": int(shard_idx),
        "chunk_idx": int(chunk_idx) if chunk_idx is not None else -1,
    }


def _add_stay_to_shard(shard: dict, bundle: dict) -> None:
    """Append one stay's bundle (all its windows) into the per-shard accumulator.

    A stay is appended atomically and never spans shards; caller starts a
    new shard before exceeding ``shard_size``.
    """
    stay_id = bundle["stay_id"]

    # Register the stay in per_stay block
    if stay_id not in shard["per_stay_key_to_index"]:
        shard["per_stay_key_to_index"][stay_id] = len(shard["per_stay_rows"])
        shard["per_stay_rows"].append(bundle["per_stay"])
    stay_index = shard["per_stay_key_to_index"][stay_id]

    for start_step, length in bundle["window_positions"]:
        hour_indices = np.empty(length, dtype=np.int32)
        for i in range(length):
            abs_hour = start_step + i
            key = (stay_id, abs_hour)
            idx = shard["per_hour_key_to_index"].get(key)
            if idx is None:
                idx = len(shard["per_hour_state_texts"])
                shard["per_hour_key_to_index"][key] = idx
                shard["per_hour_stay_ids"].append(stay_id)
                shard["per_hour_hours"].append(abs_hour)
                shard["per_hour_state_texts"].append(bundle["state_texts"][abs_hour])
                shard["per_hour_action_texts"].append(bundle["action_texts"][abs_hour])
                for name, arr in bundle["labels"].items():
                    shard["per_hour_labels"][name].append(float(arr[abs_hour]))
            hour_indices[i] = idx

        shard["windows"].append({
            "stay_index": int(stay_index),
            "start_step": int(start_step),
            "length": int(length),
            "hour_indices": hour_indices,
        })


def _finalize_shard(shard: dict, label_names: list[str]) -> dict:
    """Convert per-shard working lists into packed ndarrays + nested dict
    matching the §5.2 schema.
    """
    n_unique_hours = len(shard["per_hour_state_texts"])
    n_unique_stays = len(shard["per_stay_rows"])

    per_hour = {
        "stay_ids": np.asarray(shard["per_hour_stay_ids"], dtype=np.int64),
        "hours":    np.asarray(shard["per_hour_hours"], dtype=np.int32),
        "state_texts":  list(shard["per_hour_state_texts"]),
        "action_texts": list(shard["per_hour_action_texts"]),
    }
    per_hour_labels = {
        name: np.asarray(shard["per_hour_labels"][name], dtype=np.float32)
        for name in label_names
    }

    # Pack per_stay columns
    ps_rows = shard["per_stay_rows"]
    per_stay = {
        "stay_ids":             np.asarray([r["stay_id"]       for r in ps_rows], dtype=np.int64),
        "subject_ids":          np.asarray([r["subject_id"]    for r in ps_rows], dtype=np.int64),
        "ages":                 np.asarray([r["age"]           for r in ps_rows], dtype=np.float64),
        "genders":              [r["gender"] for r in ps_rows],
        "races":                [r["race"]   for r in ps_rows],
        "charlson_cis":         np.asarray([r["charlson_ci"]   for r in ps_rows], dtype=np.int32),
        "icu_mortalities":      np.asarray([r["icu_mortality"] for r in ps_rows], dtype=np.bool_),
        "hospital_mortalities": np.asarray([r["hospital_mortality"] for r in ps_rows], dtype=np.bool_),
        "prolonged_stays":      np.asarray([r["prolonged_stay"] for r in ps_rows], dtype=np.bool_),
        "sepsis3s":             np.asarray([r["sepsis3"]       for r in ps_rows], dtype=np.bool_),
    }

    return {
        "per_hour":         per_hour,
        "per_hour_labels":  per_hour_labels,
        "per_stay":         per_stay,
        "windows":          shard["windows"],  # list of dicts with ndarray hour_indices
        "schema_version":   shard["schema_version"],
        "shard_idx":        shard["shard_idx"],
        "chunk_idx":        shard["chunk_idx"],
        "n_windows":        len(shard["windows"]),
        "n_unique_hours":   n_unique_hours,
        "n_unique_stays":   n_unique_stays,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract trajectory shards with per-hour state + action text"
    )
    parser.add_argument("--config", required=True, help="Path to configs/data/trajectories.yaml")
    parser.add_argument("--n_workers", type=int, default=None, help="Override n_workers")
    parser.add_argument("--max_stays", type=int, default=None, help="Limit stays (for testing)")
    parser.add_argument("--chunk_idx", type=int, default=None, help="SLURM array chunk index (0-based)")
    parser.add_argument("--total_chunks", type=int, default=None, help="Total number of SLURM array chunks")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_workers = args.n_workers or config.get("n_workers", 10)
    shard_size = config.get("shard_size", 10000)
    label_names = [t["name"] for t in config["label_targets"]]

    # --- Load splits ---
    project_root = Path(__file__).resolve().parents[2]
    splits_path = project_root / config["splits_path"]
    splits_df = pd.read_csv(splits_path)
    stay_to_split = dict(zip(splits_df["stay_id"], splits_df["split"]))
    logger.info("Loaded splits: %d stays", len(stay_to_split))

    # --- Collect pickle files ---
    input_dir = project_root / config["input_dir"]
    pkl_files = sorted(input_dir.glob("*.pkl"))
    if args.max_stays:
        pkl_files = pkl_files[: args.max_stays]

    # SLURM array chunking: each chunk processes a subset of stays
    if args.chunk_idx is not None and args.total_chunks is not None:
        chunk_size = (len(pkl_files) + args.total_chunks - 1) // args.total_chunks
        start = args.chunk_idx * chunk_size
        end = min(start + chunk_size, len(pkl_files))
        pkl_files = pkl_files[start:end]
        logger.info(
            "Chunk %d/%d: processing stays %d-%d (%d files)",
            args.chunk_idx, args.total_chunks, start, end - 1, len(pkl_files),
        )

    logger.info("Processing %d pickle files from %s", len(pkl_files), input_dir)

    # --- Setup output dirs ---
    output_dir = project_root / config["output_dir"]
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # --- Process stays into bundles ---
    t0 = time.time()
    work_args = [(pkl, config) for pkl in pkl_files]

    bundles: list[dict] = []
    if n_workers <= 1:
        for i, wa in enumerate(work_args):
            bundle = process_stay(wa)
            if bundle is not None:
                bundles.append(bundle)
            if (i + 1) % 100 == 0:
                logger.info(
                    "Processed %d/%d stays, %d bundles so far",
                    i + 1, len(work_args), len(bundles),
                )
    else:
        with Pool(n_workers) as pool:
            for i, bundle in enumerate(
                pool.imap_unordered(process_stay, work_args, chunksize=50)
            ):
                if bundle is not None:
                    bundles.append(bundle)
                if (i + 1) % 1000 == 0:
                    logger.info(
                        "Processed %d/%d stays, %d bundles so far",
                        i + 1, len(work_args), len(bundles),
                    )

    elapsed_process = time.time() - t0
    total_windows = sum(len(b["window_positions"]) for b in bundles)
    logger.info(
        "Processed %d stays → %d bundles → %d windows in %.1f sec",
        len(pkl_files), len(bundles), total_windows, elapsed_process,
    )

    # --- Assign each bundle to a split ---
    split_bundles: dict[str, list[dict]] = defaultdict(list)
    for b in bundles:
        split_name = stay_to_split.get(b["stay_id"], "train")
        split_bundles[split_name].append(b)

    # --- Assemble shards (per-split, same-stay-stays-together packing) ---
    t1 = time.time()
    per_split_stats: dict[str, dict] = {}

    for split_name in ("train", "val", "test"):
        split_dir = output_dir / split_name
        split_bundles_list = split_bundles.get(split_name, [])

        # Sort bundles by stay_id for deterministic order
        split_bundles_list.sort(key=lambda b: b["stay_id"])

        shard_idx = 0
        shard = _make_empty_shard(shard_idx, args.chunk_idx, label_names)
        split_total_windows = 0
        split_n_shards = 0

        def flush(current_shard: dict, current_shard_idx: int) -> None:
            nonlocal split_n_shards
            if len(current_shard["windows"]) == 0:
                return
            finalized = _finalize_shard(current_shard, label_names)
            chunk_prefix = (
                f"c{args.chunk_idx:02d}_" if args.chunk_idx is not None else ""
            )
            out_path = split_dir / f"trajectories_{chunk_prefix}{current_shard_idx:03d}.pt"
            torch.save(finalized, out_path)
            split_n_shards += 1
            logger.info(
                "  [%s] shard %03d: %d windows, %d unique hours, %d unique stays",
                split_name, current_shard_idx,
                finalized["n_windows"],
                finalized["n_unique_hours"],
                finalized["n_unique_stays"],
            )

        for bundle in split_bundles_list:
            bundle_n_windows = len(bundle["window_positions"])

            if (
                len(shard["windows"]) > 0
                and len(shard["windows"]) + bundle_n_windows > shard_size
            ):
                flush(shard, shard_idx)
                shard_idx += 1
                shard = _make_empty_shard(shard_idx, args.chunk_idx, label_names)

            _add_stay_to_shard(shard, bundle)
            split_total_windows += bundle_n_windows

        # Flush final shard
        flush(shard, shard_idx)

        per_split_stats[split_name] = {
            "n_windows": split_total_windows,
            "n_stays": len(split_bundles_list),
            "n_shards": split_n_shards,
        }
        logger.info(
            "Wrote %s: %d stays, %d windows, %d shards",
            split_name, len(split_bundles_list), split_total_windows, split_n_shards,
        )

    elapsed_shard = time.time() - t1
    logger.info(
        "Shard assembly done in %.1f sec (process: %.1f sec, total: %.1f sec)",
        elapsed_shard, elapsed_process, elapsed_process + elapsed_shard,
    )

    # --- Write metadata.json ---
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": "clin_jepa_trajectories_v1",
        "params": {
            "t_max": config["t_max"],
            "min_len": config["min_len"],
            "stride": config["stride"],
            "shard_size": shard_size,
        },
        "counts": {
            "total_stays": len(pkl_files),
            "total_bundles": len(bundles),
            "total_windows": total_windows,
        },
        "per_split": per_split_stats,
        "label_targets": label_names,
    }

    # Window length distribution
    lengths = [length for b in bundles for (_, length) in b["window_positions"]]
    if lengths:
        meta["length_distribution"] = {
            "mean":   float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "min":    int(np.min(lengths)),
            "max":    int(np.max(lengths)),
            "p10":    float(np.percentile(lengths, 10)),
            "p90":    float(np.percentile(lengths, 90)),
        }

    meta_suffix = f"_chunk{args.chunk_idx:02d}" if args.chunk_idx is not None else ""
    meta_path = output_dir / f"metadata{meta_suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata to %s", meta_path)
    logger.info("Trajectory extraction complete. Total windows: %d", total_windows)


if __name__ == "__main__":
    main()
