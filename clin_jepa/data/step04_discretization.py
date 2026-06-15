"""Step 04: Temporal Discretization.

Merges per-stay observation and action parquet files, deduplicates, and
bins events into hourly time steps, producing the discrete per-hour
representation consumed by the encoder.

Key design choices:
- Exact duplicate rows present in MIMIC-IV source data are dropped during
  the merge.
- Continuous events (infusions, ventilation, procedures) are expanded into
  every hourly bin they overlap.
- delta_t = hours until the next step with >= 1 observation; NaN after the
  last observation.
- is_no_action = True for steps with zero action events.
- Maximum 336 steps (14 days), matching features.yaml temporal.max_steps.

Input:
  - $CLIN_JEPA_DATA/observations/{stay_id}.parquet  (Step 02)
  - $CLIN_JEPA_DATA/actions/{stay_id}.parquet       (Step 03)
  - $CLIN_JEPA_DATA/cohort/cohort.csv               (Step 01)

Output:
  - $CLIN_JEPA_DATA/discretized/{stay_id}.pkl
  - $CLIN_JEPA_DATA/discretized/discretize_stats.json  (single-chunk)

Usage:
    python -m clin_jepa.data.step04_discretization
    python -m clin_jepa.data.step04_discretization --chunk_idx 0 --total_chunks 5
"""

import argparse
import json
import logging
import math
import pickle  # noqa: S403
from pathlib import Path

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.io_utils import chunk_stay_ids, resolve_table_path
from clin_jepa.data.helpers.schema import validate_schema
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

# Columns kept in discretized obs/act event DataFrames
OBS_COLS = [
    "step_idx", "variable_name", "numeric_value",
    "text_value", "unit", "category", "source_table",
]
ACT_COLS = [
    "step_idx", "variable_name", "numeric_value",
    "text_value", "unit", "category", "source_table",
    "is_continuous",
]


# ── Timeline merge ────────────────────────────────────────────────────


def build_stay_timeline(
    stay_id: int,
    obs_dir: Path,
    act_dir: Path,
) -> tuple[pd.DataFrame | None, int]:
    """Load and merge observation + action events for a single stay.

    Drops exact duplicate rows present in MIMIC-IV source data.

    Args:
        stay_id: ICU stay identifier.
        obs_dir: Path to observations parquet directory.
        act_dir: Path to actions parquet directory.

    Returns:
        Tuple of (merged DataFrame sorted by timestamp_hours, number of
        duplicate rows dropped). Returns (None, 0) if neither file exists.
    """
    obs_path = obs_dir / f"{stay_id}.parquet"
    act_path = act_dir / f"{stay_id}.parquet"

    frames = []

    if obs_path.exists():
        frames.append(pd.read_parquet(obs_path))

    if act_path.exists():
        frames.append(pd.read_parquet(act_path))

    if not frames:
        return None, 0

    if len(frames) == 1:
        timeline = frames[0]
    else:
        timeline = pd.concat(frames, ignore_index=True)

    # Drop exact duplicate rows (from MIMIC-IV source data)
    n_before = len(timeline)
    timeline = timeline.drop_duplicates(ignore_index=True)
    n_dropped = n_before - len(timeline)

    # Sort chronologically; use timestamp as tiebreaker for determinism
    timeline = timeline.sort_values(
        ["timestamp_hours", "timestamp"],
        ignore_index=True,
    )

    return timeline, n_dropped


# ── Metadata extraction ──────────────────────────────────────────────


def extract_metadata(cohort_row: pd.Series) -> dict:
    """Extract stay metadata from a cohort DataFrame row.

    Casts all values to native Python types for clean pickle serialization.

    Args:
        cohort_row: A single row from cohort.csv.

    Returns:
        Metadata dict with subject_id, hadm_id, age, gender, race,
        charlson_ci, sepsis3, mortality, icu_los_hours.
    """
    return {
        "subject_id": int(cohort_row["subject_id"]),
        "hadm_id": int(cohort_row["hadm_id"]),
        "age": float(cohort_row["admission_age"]),
        "gender": str(cohort_row["gender"]),
        "race": str(cohort_row["race"]),
        "charlson_ci": int(cohort_row["charlson_comorbidity_index"]),
        "sepsis3": bool(cohort_row["sepsis3"]),
        "mortality": bool(cohort_row["hospital_expire_flag"]),
        "icu_los_hours": float(cohort_row["icu_los_hours"]),
    }


# ── Core discretization ─────────────────────────────────────────────


def discretize_stay(
    timeline_df: pd.DataFrame,
    stay_id: int,
    metadata: dict,
    step_size: float,
    max_steps: int,
    min_obs: int,
) -> dict | None:
    """Bin a stay's timeline events into hourly steps.

    Args:
        timeline_df: Timeline DataFrame with the 13-column schema.
        stay_id: ICU stay identifier.
        metadata: Stay metadata dict from extract_metadata().
        step_size: Hours per bin (typically 1.0).
        max_steps: Maximum number of time steps (typically 336).
        min_obs: Minimum observation events required to keep a stay.

    Returns:
        Discretized stay dict, or None if below min_obs threshold.
    """
    # ── 1. Separate domains ──────────────────────────────────────────
    obs_df = timeline_df[timeline_df["domain"] == "observation"]
    act_df = timeline_df[timeline_df["domain"] == "action"]

    # ── 2. Min obs check ─────────────────────────────────────────────
    if len(obs_df) < min_obs:
        return None

    # ── 3. Compute n_steps from ICU LOS ──────────────────────────────
    icu_los_hours = metadata["icu_los_hours"]
    n_steps = min(max_steps, max(1, math.ceil(icu_los_hours / step_size)))

    # ── 4. Truncate events beyond the step window ────────────────────
    window_end = n_steps * step_size
    obs_df = obs_df[obs_df["timestamp_hours"] < window_end].copy()
    act_df = act_df[act_df["timestamp_hours"] < window_end].copy()

    # Clamp continuous end timestamps to window boundary
    if len(act_df) > 0 and "end_timestamp_hours" in act_df.columns:
        mask = act_df["end_timestamp_hours"].notna()
        act_df.loc[mask, "end_timestamp_hours"] = act_df.loc[
            mask, "end_timestamp_hours"
        ].clip(upper=window_end)

    # ── 5. Bin observation events (point events) ─────────────────────
    if len(obs_df) > 0:
        obs_bins = np.clip(
            np.floor(np.round(obs_df["timestamp_hours"].values, 6) / step_size).astype(int),
            0, n_steps - 1,
        )
        obs_events = obs_df.assign(step_idx=obs_bins)[OBS_COLS].reset_index(drop=True)
    else:
        obs_events = pd.DataFrame(columns=OBS_COLS)

    # ── 6. Bin action events (point + continuous expansion) ──────────
    point_acts = act_df[~act_df["is_continuous"]].copy()
    cont_acts = act_df[act_df["is_continuous"]].copy()

    act_frames = []
    n_expanded = 0  # track expansion count for stats

    # Point actions: simple binning
    if len(point_acts) > 0:
        point_bins = np.clip(
            np.floor(np.round(point_acts["timestamp_hours"].values, 6) / step_size).astype(int),
            0, n_steps - 1,
        )
        point_acts = point_acts.assign(step_idx=point_bins)
        act_frames.append(point_acts[ACT_COLS])

    # Continuous actions: expand into all overlapping bins (vectorized)
    if len(cont_acts) > 0:
        start_h = cont_acts["timestamp_hours"].values
        end_h = cont_acts["end_timestamp_hours"].values

        start_bins = np.clip(
            np.floor(start_h / step_size).astype(int), 0, n_steps - 1,
        )

        # Compute end bins; treat NaN/invalid as point event (end_bin = start_bin)
        end_bins = start_bins.copy()
        valid_end = ~np.isnan(end_h) & (end_h > start_h)
        end_bins[valid_end] = np.clip(
            np.ceil(end_h[valid_end] / step_size).astype(int) - 1,
            start_bins[valid_end], n_steps - 1,
        )

        # Compute how many bins each event spans
        spans = end_bins - start_bins + 1  # at least 1
        total_expanded_rows = int(spans.sum())
        n_expanded = total_expanded_rows - len(cont_acts)

        # Build expanded step_idx array and row indices
        row_indices = np.repeat(np.arange(len(cont_acts)), spans)
        offsets = np.arange(total_expanded_rows) - np.repeat(
            np.concatenate([[0], np.cumsum(spans[:-1])]), spans,
        )
        expanded_step_idx = np.repeat(start_bins, spans) + offsets

        # Build expanded DataFrame by indexing into cont_acts
        keep_cols = ["variable_name", "numeric_value", "text_value",
                     "unit", "category", "source_table"]
        expanded_df = cont_acts.iloc[row_indices][keep_cols].reset_index(drop=True)
        expanded_df["step_idx"] = expanded_step_idx
        expanded_df["is_continuous"] = True
        act_frames.append(expanded_df)

    if act_frames:
        act_events = pd.concat(act_frames, ignore_index=True)[ACT_COLS]
    else:
        act_events = pd.DataFrame(columns=ACT_COLS)

    # ── 7. Compute per-step flags ────────────────────────────────────
    has_obs = np.zeros(n_steps, dtype=bool)
    if len(obs_events) > 0:
        obs_step_set = set(obs_events["step_idx"].values)
        for s in obs_step_set:
            has_obs[s] = True

    is_no_action = np.ones(n_steps, dtype=bool)
    if len(act_events) > 0:
        act_step_set = set(act_events["step_idx"].values)
        for s in act_step_set:
            is_no_action[s] = False

    # ── 8. Compute delta_t (vectorized) ──────────────────────────────
    obs_step_indices = np.where(has_obs)[0]
    delta_t = np.full(n_steps, np.nan, dtype=np.float64)

    if len(obs_step_indices) > 0:
        all_steps = np.arange(n_steps)
        insert_pos = np.searchsorted(obs_step_indices, all_steps, side="right")
        valid = insert_pos < len(obs_step_indices)
        delta_t[valid] = (
            obs_step_indices[insert_pos[valid]] - all_steps[valid]
        ) * step_size

    # ── 9. Assemble output dict ──────────────────────────────────────
    return {
        "stay_id": stay_id,
        "metadata": metadata,
        "n_steps": n_steps,
        "delta_t": delta_t,
        "has_obs": has_obs,
        "is_no_action": is_no_action,
        "obs_events": obs_events,
        "act_events": act_events,
        "_n_expanded": n_expanded,  # internal stat, not part of output spec
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for temporal discretization."""
    parser = argparse.ArgumentParser(
        description="Step 04: Temporal Discretization",
    )
    parser.add_argument(
        "--paths_config", default="configs/data/mimic_paths.yaml",
        help="Path to mimic_paths.yaml",
    )
    parser.add_argument(
        "--features_config", default="configs/data/features.yaml",
        help="Path to features.yaml",
    )
    parser.add_argument(
        "--chunk_idx", type=int, default=0,
        help="This worker's chunk index (0-based)",
    )
    parser.add_argument(
        "--total_chunks", type=int, default=1,
        help="Total number of chunks/workers",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info(
        "Step 04: Temporal Discretization (chunk %d/%d)",
        args.chunk_idx, args.total_chunks,
    )
    logger.info("=" * 60)

    # Load configs
    paths_config = load_config(args.paths_config)
    features_config = load_config(args.features_config)

    temporal = features_config["temporal"]
    step_size = float(temporal["step_size_hours"])
    max_steps = int(temporal["max_steps"])
    min_obs = int(temporal["min_obs_per_stay"])
    logger.info(
        "Temporal params: step_size=%.1fh, max_steps=%d, min_obs=%d",
        step_size, max_steps, min_obs,
    )

    # Resolve directories
    cohort_dir = resolve_table_path(paths_config, "output.cohort")
    obs_dir = resolve_table_path(paths_config, "output.observations")
    act_dir = resolve_table_path(paths_config, "output.actions")
    output_dir = ensure_dir(resolve_table_path(paths_config, "output.discretized"))

    # Load cohort (full, for metadata lookup)
    cohort_path = cohort_dir / "cohort.csv"
    cohort_df = pd.read_csv(cohort_path)
    cohort_df = cohort_df.set_index("stay_id")
    all_stay_ids = np.sort(cohort_df.index.values)
    logger.info("Loaded cohort: %d stays", len(all_stay_ids))

    # Get this chunk's stay_ids
    my_stay_ids = chunk_stay_ids(all_stay_ids, args.chunk_idx, args.total_chunks)

    # ── Process each stay ────────────────────────────────────────────
    n_written = 0
    n_no_data = 0
    n_too_few_obs = 0
    n_truncated = 0
    total_duplicates = 0
    total_steps = 0
    total_obs_events = 0
    total_act_events = 0
    total_expanded = 0
    total_no_action_steps = 0
    total_no_obs_steps = 0
    n_obs_only = 0
    n_act_only = 0
    n_both = 0
    steps_per_stay: list[int] = []
    delta_t_values: list[float] = []
    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for i, stay_id in enumerate(my_stay_ids):
        stay_id_int = int(stay_id)

        # ── Merge obs + act (inline build_stay_timeline) ─────────────
        timeline, n_dropped = build_stay_timeline(stay_id_int, obs_dir, act_dir)

        if timeline is None:
            n_no_data += 1
            continue

        total_duplicates += n_dropped

        # Track domain presence
        has_obs_domain = (timeline["domain"] == "observation").any()
        has_act_domain = (timeline["domain"] == "action").any()
        if has_obs_domain and has_act_domain:
            n_both += 1
        elif has_obs_domain:
            n_obs_only += 1
        elif has_act_domain:
            n_act_only += 1

        # Validate merged schema
        validate_schema(timeline)

        # Per-source counts (from merged timeline)
        for src, cnt in timeline["source_table"].value_counts().items():
            source_counts[src] = source_counts.get(src, 0) + int(cnt)

        # ── Discretize ───────────────────────────────────────────────
        cohort_row = cohort_df.loc[stay_id_int]
        metadata = extract_metadata(cohort_row)

        result = discretize_stay(
            timeline, stay_id_int, metadata,
            step_size, max_steps, min_obs,
        )

        if result is None:
            n_too_few_obs += 1
            continue

        # Check if truncated
        icu_los_steps = math.ceil(metadata["icu_los_hours"] / step_size)
        if icu_los_steps > max_steps:
            n_truncated += 1

        # Pop internal stat before writing
        n_expanded = result.pop("_n_expanded")

        # Write pickle
        out_path = output_dir / f"{stay_id_int}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301

        # ── Accumulate statistics ────────────────────────────────────
        n_written += 1
        ns = result["n_steps"]
        total_steps += ns
        steps_per_stay.append(ns)

        n_obs = len(result["obs_events"])
        n_act = len(result["act_events"])
        total_obs_events += n_obs
        total_act_events += n_act
        total_expanded += n_expanded

        total_no_action_steps += int(result["is_no_action"].sum())
        total_no_obs_steps += int((~result["has_obs"]).sum())

        # Collect non-NaN delta_t for stats
        valid_dt = result["delta_t"][~np.isnan(result["delta_t"])]
        if len(valid_dt) > 0:
            delta_t_values.extend(valid_dt.tolist())

        # Per-category event counts (obs + act combined)
        for df_events in [result["obs_events"], result["act_events"]]:
            if len(df_events) > 0:
                for cat, cnt in df_events["category"].value_counts().items():
                    category_counts[cat] = category_counts.get(cat, 0) + int(cnt)

        # Progress logging
        if (i + 1) % 10_000 == 0:
            logger.info(
                "  Progress: %d / %d stays processed (%d written)",
                i + 1, len(my_stay_ids), n_written,
            )

    # ── Summary statistics ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "TEMPORAL DISCRETIZATION SUMMARY (chunk %d/%d)",
        args.chunk_idx, args.total_chunks,
    )
    logger.info("=" * 60)
    logger.info("  Stays processed:       %d", len(my_stay_ids))
    logger.info("  Files written:         %d", n_written)
    logger.info("  Both obs + actions:    %d", n_both)
    logger.info("  Observations only:     %d", n_obs_only)
    logger.info("  Actions only:          %d", n_act_only)
    logger.info("  Skipped (no data):     %d", n_no_data)
    logger.info("  Skipped (too few obs): %d", n_too_few_obs)
    logger.info("  Truncated (>%d steps): %d", max_steps, n_truncated)
    logger.info("  Duplicates dropped:    %d", total_duplicates)
    logger.info("  Total hourly steps:    %d", total_steps)

    if n_written > 0:
        sps = np.array(steps_per_stay)
        logger.info("  Steps/stay median:     %.0f", np.median(sps))
        logger.info("  Steps/stay P5:         %.0f", np.percentile(sps, 5))
        logger.info("  Steps/stay P95:        %.0f", np.percentile(sps, 95))
        logger.info("  Steps/stay max:        %d", sps.max())
        logger.info("  Total obs events:      %d", total_obs_events)
        logger.info("  Total act events:      %d (incl. expanded)", total_act_events)
        logger.info("  Continuous expansion:  %d extra event-bins", total_expanded)
        logger.info(
            "  No-action steps:       %d / %d (%.1f%%)",
            total_no_action_steps, total_steps,
            100.0 * total_no_action_steps / total_steps if total_steps > 0 else 0,
        )
        logger.info(
            "  No-obs steps:          %d / %d (%.1f%%)",
            total_no_obs_steps, total_steps,
            100.0 * total_no_obs_steps / total_steps if total_steps > 0 else 0,
        )

        if delta_t_values:
            dt_arr = np.array(delta_t_values)
            logger.info("  Delta_t mean:          %.2f h", dt_arr.mean())
            logger.info("  Delta_t median:        %.1f h", np.median(dt_arr))
            logger.info("  Delta_t min:           %.1f h", dt_arr.min())

    logger.info("")
    logger.info("  Per-category event counts:")
    for cat, count in sorted(category_counts.items()):
        logger.info("    %-30s %d", cat, count)
    logger.info("")
    logger.info("  Per-source event counts:")
    for src, count in sorted(source_counts.items()):
        logger.info("    %-40s %d", src, count)
    logger.info("  Output dir: %s", output_dir)
    logger.info("=" * 60)

    # ── Write stats JSON (single-chunk mode only) ────────────────────
    if args.total_chunks == 1 and n_written > 0:
        sps = np.array(steps_per_stay)
        dt_arr = np.array(delta_t_values) if delta_t_values else np.array([])
        stats = {
            "total_stays_processed": int(len(my_stay_ids)),
            "files_written": int(n_written),
            "stays_both_domains": int(n_both),
            "stays_obs_only": int(n_obs_only),
            "stays_act_only": int(n_act_only),
            "stays_no_data": int(n_no_data),
            "skipped_too_few_obs": int(n_too_few_obs),
            "truncated": int(n_truncated),
            "duplicates_dropped": int(total_duplicates),
            "total_steps": int(total_steps),
            "steps_per_stay_median": float(np.median(sps)),
            "steps_per_stay_p5": float(np.percentile(sps, 5)),
            "steps_per_stay_p95": float(np.percentile(sps, 95)),
            "steps_per_stay_min": int(sps.min()),
            "steps_per_stay_max": int(sps.max()),
            "total_obs_events": int(total_obs_events),
            "total_act_events": int(total_act_events),
            "continuous_expansion_extra": int(total_expanded),
            "no_action_steps": int(total_no_action_steps),
            "no_action_fraction": round(
                total_no_action_steps / total_steps if total_steps > 0 else 0, 4,
            ),
            "no_obs_steps": int(total_no_obs_steps),
            "no_obs_fraction": round(
                total_no_obs_steps / total_steps if total_steps > 0 else 0, 4,
            ),
            "delta_t_mean": round(float(dt_arr.mean()), 3) if len(dt_arr) > 0 else None,
            "delta_t_median": round(float(np.median(dt_arr)), 3) if len(dt_arr) > 0 else None,
            "category_counts": {k: int(v) for k, v in sorted(category_counts.items())},
            "source_counts": {k: int(v) for k, v in sorted(source_counts.items())},
        }
        stats_path = output_dir / "discretize_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Wrote stats to %s", stats_path)


if __name__ == "__main__":
    main()
