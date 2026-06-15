"""Step 02: Extract Observations.

Extracts observation events from 15 MIMIC-IV concepts tables, converts
to long format using the unified output schema, and writes per-stay
parquet files to $CLIN_JEPA_DATA/observations/.

Processing strategy:
  - Per-table processing with per-stay accumulation
  - Tables >300MB use chunked reading (1M rows at a time)
  - Tables joining on subject_id get ICU window matching to assign stay_id
  - Wide tables are melted to long format
  - NaN measurements are dropped

Usage:
    python -m clin_jepa.data.step02_observations
    python -m clin_jepa.data.step02_observations --chunk_idx 0 --total_chunks 10
"""

import argparse
import gc
import logging

import numpy as np
import pandas as pd

from clin_jepa.data.helpers.io_utils import (
    chunk_stay_ids,
    load_and_filter_concepts_table,
    load_and_filter_raw_table,
    resolve_table_path,
)
from clin_jepa.data.helpers.schema import OUTPUT_COLUMNS, empty_events_df, validate_schema
from clin_jepa.data.helpers.temporal_utils import (
    assign_stay_ids_vectorized,
    build_icu_windows,
    build_intime_map,
    clip_to_icu_window,
    compute_hours_since_admission,
)
from clin_jepa.data.helpers.units import get_unit
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

# Columns that are never value columns (excluded when value_cols == "all")
EXCLUDE_COLS = {
    "subject_id", "hadm_id", "stay_id", "specimen_id", "specimen",
    "charttime", "starttime", "endtime", "stoptime", "hr",
    "icu_intime", "icu_outtime",
}


def resolve_value_columns(
    df: pd.DataFrame, config_value_cols: list[str] | str,
) -> tuple[list[str], list[str]]:
    """Resolve value column specification to (numeric_cols, categorical_cols).

    Args:
        df: Loaded DataFrame to inspect dtypes.
        config_value_cols: Either a list of column names or the string "all".

    Returns:
        Tuple of (numeric_cols, categorical_cols).
    """
    if config_value_cols == "all":
        candidates = [c for c in df.columns if c not in EXCLUDE_COLS]
    else:
        candidates = [c for c in config_value_cols if c in df.columns]

    numeric_cols = []
    categorical_cols = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols


def melt_to_long(
    df: pd.DataFrame,
    source_name: str,
    category: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_col: str,
    intime_map: dict,
) -> pd.DataFrame:
    """Melt wide-format table to long-format event rows.

    Args:
        df: DataFrame with stay_id and timestamp columns.
        source_name: Config key name (e.g. "vitalsign").
        category: High-level category (e.g. "vitals").
        numeric_cols: Columns with numeric values to melt.
        categorical_cols: Columns with categorical values to melt.
        timestamp_col: Name of the timestamp column.
        intime_map: Dict mapping stay_id → icu_intime (pd.Timestamp).

    Returns:
        DataFrame conforming to the unified output schema.
    """
    frames = []

    # Numeric columns: melt
    if numeric_cols:
        melted = df.melt(
            id_vars=["stay_id", timestamp_col],
            value_vars=numeric_cols,
            var_name="variable_name",
            value_name="numeric_value",
        )
        melted = melted.dropna(subset=["numeric_value"])
        melted["text_value"] = ""
        frames.append(melted)

    # Categorical columns: one at a time
    for col in categorical_cols:
        subset = df[["stay_id", timestamp_col, col]].copy()
        subset = subset.dropna(subset=[col])
        subset = subset[subset[col].astype(str).str.strip() != ""]
        subset = subset.rename(columns={col: "text_value"})
        subset["variable_name"] = col
        subset["numeric_value"] = np.nan
        subset["text_value"] = subset["text_value"].astype(str)
        frames.append(subset)

    if not frames:
        return empty_events_df()

    result = pd.concat(frames, ignore_index=True)

    # Parse timestamp
    result["timestamp"] = pd.to_datetime(result[timestamp_col])
    if timestamp_col != "timestamp":
        result = result.drop(columns=[timestamp_col])

    # Compute hours since admission (vectorized)
    result["timestamp_hours"] = compute_hours_since_admission(
        result["timestamp"], result["stay_id"], intime_map
    )

    # Add constant columns
    result["domain"] = "observation"
    result["source_table"] = source_name
    result["category"] = category
    result["end_timestamp"] = pd.NaT
    result["end_timestamp_hours"] = np.nan
    result["is_continuous"] = False

    # Add units (vectorized via map)
    result["unit"] = result["variable_name"].map(
        lambda vn: get_unit(source_name, vn)
    )

    return result[OUTPUT_COLUMNS]


def process_observation_source(
    source_name: str,
    source_config: dict,
    paths_config: dict,
    chunk_stay_ids_set: set,
    chunk_subject_ids_set: set,
    windows_df: pd.DataFrame,
    intime_map: dict,
) -> pd.DataFrame:
    """Process a single observation source table.

    Args:
        source_name: Config key for this source.
        source_config: Source configuration dict from features.yaml.
        paths_config: Loaded mimic_paths.yaml config.
        chunk_stay_ids_set: Set of stay_ids in this chunk.
        chunk_subject_ids_set: Set of subject_ids in this chunk.
        windows_df: ICU windows DataFrame for subject_id matching.
        intime_map: Dict mapping stay_id → icu_intime.

    Returns:
        DataFrame in unified output schema.
    """
    join_col = source_config["join_col"]
    timestamp_col = source_config["timestamp_col"]
    category = source_config["category"]
    source_dotpath = source_config["source"]

    # Determine which cohort IDs to filter on
    if join_col == "stay_id":
        cohort_ids = chunk_stay_ids_set
    elif join_col == "subject_id":
        cohort_ids = chunk_subject_ids_set
    else:
        raise ValueError(f"Unknown join_col: {join_col} for source {source_name}")

    logger.info(
        "Processing %s (join=%s, category=%s)", source_name, join_col, category
    )

    # Load and filter
    df = load_and_filter_concepts_table(
        paths_config, source_dotpath, join_col, cohort_ids
    )

    if len(df) == 0:
        logger.warning("  No data found for %s", source_name)
        return empty_events_df()

    if join_col == "subject_id":
        df = assign_stay_ids_vectorized(df, windows_df, timestamp_col)
        if len(df) == 0:
            logger.warning("  No rows matched ICU windows for %s", source_name)
            return empty_events_df()
        logger.info("  After ICU window matching: %d rows", len(df))
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = clip_to_icu_window(df, windows_df, timestamp_col)
        if len(df) == 0:
            logger.warning("  No rows within ICU window for %s", source_name)
            return empty_events_df()

    # Apply row-level filters (e.g., blood gas arterial only)
    row_filter = source_config.get("row_filter")
    if row_filter and row_filter["column"] in df.columns:
        allowed = set(row_filter["values"])
        before = len(df)
        df = df[df[row_filter["column"]].isin(allowed)]
        logger.info(
            "  Row filter %s in %s: %d → %d rows",
            row_filter["column"], allowed, before, len(df),
        )
        if len(df) == 0:
            return empty_events_df()

    # Resolve value columns
    config_value_cols = source_config["value_cols"]
    numeric_cols, categorical_cols = resolve_value_columns(df, config_value_cols)
    logger.info(
        "  Value columns: %d numeric, %d categorical",
        len(numeric_cols), len(categorical_cols),
    )

    # Melt to long format
    events = melt_to_long(
        df, source_name, category, numeric_cols, categorical_cols,
        timestamp_col, intime_map,
    )

    logger.info("  Produced %d event rows", len(events))

    del df
    gc.collect()

    return events


def process_diagnoses(
    paths_config: dict,
    cohort_df: pd.DataFrame,
    chunk_stay_ids_set: set,
    intime_map: dict,
) -> pd.DataFrame:
    """Process ICD diagnosis codes as admission-level observation events.

    ICD codes are admission-level; assigned to timestamp_hours = 0.0.
    One event row per ICD code.

    Args:
        paths_config: Loaded mimic_paths.yaml config.
        cohort_df: Full cohort DataFrame (filtered to this chunk).
        chunk_stay_ids_set: Set of stay_ids in this chunk.
        intime_map: Dict mapping stay_id -> icu_intime.

    Returns:
        DataFrame in unified output schema.
    """
    logger.info("Processing diagnoses (ICD codes, admission-level)")

    # Build hadm_id -> stay_id mapping for this chunk
    hadm_to_stay = dict(zip(cohort_df["hadm_id"], cohort_df["stay_id"]))
    chunk_hadm_ids = set(cohort_df["hadm_id"].tolist())

    # Load diagnoses_icd.csv.gz filtered to chunk hadm_ids
    diag_df = load_and_filter_raw_table(
        paths_config, "hosp.diagnoses_icd",
        join_col="hadm_id", cohort_ids=chunk_hadm_ids,
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )

    if len(diag_df) == 0:
        logger.warning("  No diagnoses found for this chunk")
        return empty_events_df()

    # Map hadm_id -> stay_id
    diag_df["stay_id"] = diag_df["hadm_id"].map(hadm_to_stay)
    diag_df = diag_df.dropna(subset=["stay_id"])
    diag_df["stay_id"] = diag_df["stay_id"].astype(int)

    # Filter to chunk stay_ids (in case multiple stays per hadm_id)
    diag_df = diag_df[diag_df["stay_id"].isin(chunk_stay_ids_set)]
    logger.info("  Diagnoses after stay_id mapping: %d rows", len(diag_df))

    if len(diag_df) == 0:
        return empty_events_df()

    # Load d_icd_diagnoses for long_title descriptions
    desc_df = load_and_filter_raw_table(
        paths_config, "hosp.d_icd_diagnoses",
        join_col="icd_code",
        cohort_ids=set(diag_df["icd_code"].tolist()),
        usecols=["icd_code", "icd_version", "long_title"],
    )

    # Join descriptions (on both icd_code and icd_version for accuracy)
    diag_df["icd_code"] = diag_df["icd_code"].astype(str)
    desc_df["icd_code"] = desc_df["icd_code"].astype(str)
    diag_df = diag_df.merge(
        desc_df, on=["icd_code", "icd_version"], how="left",
    )

    # Build output in unified schema
    result = pd.DataFrame({
        "stay_id": diag_df["stay_id"],
        "timestamp": diag_df["stay_id"].map(intime_map),
        "timestamp_hours": 0.0,
        "domain": "observation",
        "source_table": "diagnoses",
        "category": "diagnosis",
        "variable_name": diag_df["icd_code"],
        "numeric_value": np.nan,
        "text_value": diag_df["long_title"].fillna(""),
        "unit": "",
        "end_timestamp": pd.NaT,
        "end_timestamp_hours": np.nan,
        "is_continuous": False,
    })

    result = result[OUTPUT_COLUMNS]
    logger.info("  Produced %d diagnosis event rows", len(result))

    del diag_df, desc_df
    gc.collect()

    return result


def main() -> None:
    """Main entry point for observation extraction."""
    parser = argparse.ArgumentParser(description="Step 02: Extract Observations")
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
        "Step 02: Extract Observations (chunk %d/%d)",
        args.chunk_idx, args.total_chunks,
    )
    logger.info("=" * 60)

    # Load configs
    paths_config = load_config(args.paths_config)
    features_config = load_config(args.features_config)

    # Load cohort
    cohort_dir = resolve_table_path(paths_config, "output.cohort")
    cohort_path = cohort_dir / "cohort.csv"
    cohort_df = pd.read_csv(cohort_path)
    cohort_df["icu_intime"] = pd.to_datetime(cohort_df["icu_intime"])
    cohort_df["icu_outtime"] = pd.to_datetime(cohort_df["icu_outtime"])
    logger.info("Loaded cohort: %d stays", len(cohort_df))

    # Get this chunk's stay_ids
    all_stay_ids = np.sort(cohort_df["stay_id"].values)
    my_stay_ids = chunk_stay_ids(all_stay_ids, args.chunk_idx, args.total_chunks)
    my_stay_ids_set = set(my_stay_ids.tolist())

    # Filter cohort to chunk
    chunk_cohort = cohort_df[cohort_df["stay_id"].isin(my_stay_ids_set)].copy()
    my_subject_ids_set = set(chunk_cohort["subject_id"].tolist())

    # Build lookup structures
    intime_map = build_intime_map(chunk_cohort)
    windows_df = build_icu_windows(chunk_cohort)

    del cohort_df
    gc.collect()

    # Prepare output directory
    output_dir = ensure_dir(resolve_table_path(paths_config, "output.observations"))

    # Process all observation sources
    obs_config = features_config["observations"]
    all_events: dict[int, list[pd.DataFrame]] = {}
    source_counts: dict[str, int] = {}

    for source_name, source_config in obs_config.items():
        events = process_observation_source(
            source_name, source_config, paths_config,
            my_stay_ids_set, my_subject_ids_set,
            windows_df, intime_map,
        )

        n_events = len(events)
        source_counts[source_name] = n_events

        if n_events > 0:
            for stay_id, group in events.groupby("stay_id"):
                if stay_id not in all_events:
                    all_events[stay_id] = []
                all_events[stay_id].append(group)

        del events
        gc.collect()

    # Process ICD diagnosis codes (admission-level, specialized handler)
    diag_events = process_diagnoses(
        paths_config, chunk_cohort, my_stay_ids_set, intime_map,
    )
    n_diag = len(diag_events)
    source_counts["diagnoses"] = n_diag
    if n_diag > 0:
        for stay_id, group in diag_events.groupby("stay_id"):
            if stay_id not in all_events:
                all_events[stay_id] = []
            all_events[stay_id].append(group)
    del diag_events
    gc.collect()

    # Write per-stay parquet files
    n_written = 0
    n_empty = 0
    total_events = 0

    for stay_id in my_stay_ids:
        stay_id_int = int(stay_id)
        if stay_id_int in all_events:
            stay_df = pd.concat(all_events[stay_id_int], ignore_index=True)
            stay_df = stay_df.sort_values("timestamp_hours").reset_index(drop=True)
            validate_schema(stay_df)

            outpath = output_dir / f"{stay_id_int}.parquet"
            stay_df.to_parquet(outpath, index=False, engine="pyarrow")

            n_written += 1
            total_events += len(stay_df)
        else:
            n_empty += 1

    # Summary
    logger.info("=" * 60)
    logger.info(
        "EXTRACTION SUMMARY (chunk %d/%d)",
        args.chunk_idx, args.total_chunks,
    )
    logger.info("=" * 60)
    logger.info("  Stays processed:    %d", len(my_stay_ids))
    logger.info("  Stays with data:    %d", n_written)
    logger.info("  Stays without data: %d", n_empty)
    logger.info("  Total event rows:   %d", total_events)
    if n_written > 0:
        logger.info("  Avg events/stay:    %.1f", total_events / n_written)
    logger.info("  Output dir:         %s", output_dir)
    logger.info("")
    logger.info("  Per-source event counts:")
    for source_name, count in sorted(source_counts.items()):
        logger.info("    %-30s %d", source_name, count)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
