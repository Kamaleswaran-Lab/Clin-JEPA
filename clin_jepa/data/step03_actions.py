"""Step 03: Extract Actions.

Extracts clinical intervention/action events from 12 MIMIC-IV source tables
(8 concepts tables + 4 raw .csv.gz tables), converts to long format using
the unified output schema, and writes per-stay parquet files to
$CLIN_JEPA_DATA/actions/.

Processing strategy:
  - 8 concepts tables (plain CSV, join on stay_id)
  - 4 raw tables (.csv.gz, chunked reading):
    - inputevents, procedureevents: join on stay_id, d_items resolution
    - prescriptions, emar: join on hadm_id, ICU window matching
  - Continuous events populate end_timestamp and end_timestamp_hours
  - End timestamps are clamped to icu_outtime

Usage:
    python -m clin_jepa.data.step03_actions
    python -m clin_jepa.data.step03_actions --chunk_idx 0 --total_chunks 10
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
    load_raw_table,
    resolve_table_path,
)
from clin_jepa.data.helpers.schema import OUTPUT_COLUMNS, empty_events_df, validate_schema
from clin_jepa.data.helpers.temporal_utils import (
    assign_stay_ids_from_hadm,
    build_hadm_windows,
    build_icu_windows,
    build_intime_map,
    build_outtime_map,
    clip_to_icu_window,
    compute_hours_since_admission,
)
from clin_jepa.data.helpers.units import get_unit
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

# Columns excluded when value_cols == "all" (same set as step02, extended)
EXCLUDE_COLS = {
    "subject_id", "hadm_id", "stay_id", "specimen_id", "specimen",
    "charttime", "starttime", "endtime", "stoptime", "hr",
    "icu_intime", "icu_outtime",
}

# Valid statusdescription values for inputevents/procedureevents
VALID_INPUT_STATUS = {"FinishedRunning", "Paused", "Stopped"}

# emar event_txt values that represent actual medication delivery
VALID_EMAR_EVENTS = {"Administered", "Applied"}


# ── Column resolution ────────────────────────────────────────────────────


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


# ── Melt to long (action-specific) ──────────────────────────────────────


def melt_to_long_action(
    df: pd.DataFrame,
    source_name: str,
    category: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_col: str,
    intime_map: dict,
    outtime_map: dict,
    end_col: str | None,
    is_continuous: bool,
) -> pd.DataFrame:
    """Melt wide-format action table to long-format event rows.

    Args:
        df: DataFrame with stay_id and timestamp columns.
        source_name: Config key name (e.g. "vasoactive_agent").
        category: High-level category (e.g. "vasopressor").
        numeric_cols: Columns with numeric values to melt.
        categorical_cols: Columns with categorical values to melt.
        timestamp_col: Name of the start timestamp column.
        intime_map: Dict mapping stay_id -> icu_intime.
        outtime_map: Dict mapping stay_id -> icu_outtime.
        end_col: Name of end timestamp column (e.g. "endtime"), or None.
        is_continuous: Whether events are continuous (have duration).

    Returns:
        DataFrame conforming to the unified output schema.
    """
    frames = []

    # Determine id_vars for melt (carry end_col through if present)
    has_end = end_col is not None and end_col in df.columns
    id_vars = ["stay_id", timestamp_col]
    if has_end:
        id_vars.append(end_col)

    # Numeric columns: melt
    if numeric_cols:
        melted = df.melt(
            id_vars=id_vars,
            value_vars=numeric_cols,
            var_name="variable_name",
            value_name="numeric_value",
        )
        melted = melted.dropna(subset=["numeric_value"])
        melted["text_value"] = ""
        frames.append(melted)

    # Categorical columns: one at a time
    for col in categorical_cols:
        keep_cols = ["stay_id", timestamp_col]
        if has_end:
            keep_cols.append(end_col)
        keep_cols.append(col)
        subset = df[keep_cols].copy()
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

    # Parse start timestamp
    result["timestamp"] = pd.to_datetime(result[timestamp_col])
    if timestamp_col != "timestamp":
        result = result.drop(columns=[timestamp_col])

    # Compute hours since admission for start time
    result["timestamp_hours"] = compute_hours_since_admission(
        result["timestamp"], result["stay_id"], intime_map,
    )

    # Handle end timestamp
    if has_end:
        result["end_timestamp"] = pd.to_datetime(result[end_col])
        # Clamp end_timestamp to icu_outtime
        outtimes = result["stay_id"].map(outtime_map)
        exceed_mask = result["end_timestamp"] > outtimes
        if exceed_mask.any():
            logger.info(
                "  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum(),
            )
            result.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
        result["end_timestamp_hours"] = compute_hours_since_admission(
            result["end_timestamp"], result["stay_id"], intime_map,
        )
        if end_col != "end_timestamp":
            result = result.drop(columns=[end_col])
    else:
        result["end_timestamp"] = pd.NaT
        result["end_timestamp_hours"] = np.nan

    # Add constant columns
    result["domain"] = "action"
    result["source_table"] = source_name
    result["category"] = category
    result["is_continuous"] = is_continuous

    # Add units (vectorized via map)
    result["unit"] = result["variable_name"].map(
        lambda vn: get_unit(source_name, vn)
    )

    return result[OUTPUT_COLUMNS]


# ── Concepts action source processor ────────────────────────────────────


def process_concepts_action_source(
    source_name: str,
    source_config: dict,
    paths_config: dict,
    chunk_stay_ids_set: set,
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process a single concepts-table action source.

    Args:
        source_name: Config key for this source.
        source_config: Source configuration dict from features.yaml.
        paths_config: Loaded mimic_paths.yaml config.
        chunk_stay_ids_set: Set of stay_ids in this chunk.
        windows_df: ICU windows DataFrame (with stay_id, icu_intime, icu_outtime).
        intime_map: Dict mapping stay_id -> icu_intime.
        outtime_map: Dict mapping stay_id -> icu_outtime.

    Returns:
        DataFrame in unified output schema.
    """
    join_col = source_config["join_col"]
    timestamp_col = source_config["timestamp_col"]
    category = source_config["category"]
    source_dotpath = source_config["source"]
    end_col = source_config.get("end_col")
    is_continuous = source_config.get("is_continuous", False)

    logger.info(
        "Processing %s (join=%s, category=%s, end_col=%s, continuous=%s)",
        source_name, join_col, category, end_col, is_continuous,
    )

    # Load and filter (all concepts action tables join on stay_id)
    df = load_and_filter_concepts_table(
        paths_config, source_dotpath, join_col, chunk_stay_ids_set,
    )

    if len(df) == 0:
        logger.warning("  No data found for %s", source_name)
        return empty_events_df()

    # Parse timestamps and clip to ICU window
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if end_col and end_col in df.columns:
        df[end_col] = pd.to_datetime(df[end_col])
    df = clip_to_icu_window(df, windows_df, timestamp_col)
    if len(df) == 0:
        logger.warning("  No rows within ICU window for %s", source_name)
        return empty_events_df()

    # Resolve value columns
    config_value_cols = source_config.get("value_cols", "all")
    numeric_cols, categorical_cols = resolve_value_columns(df, config_value_cols)
    logger.info(
        "  Value columns: %d numeric, %d categorical",
        len(numeric_cols), len(categorical_cols),
    )

    # Melt to long format
    events = melt_to_long_action(
        df, source_name, category, numeric_cols, categorical_cols,
        timestamp_col, intime_map, outtime_map, end_col, is_continuous,
    )

    logger.info("  Produced %d event rows", len(events))

    del df
    gc.collect()

    return events


# ── Raw table processors ────────────────────────────────────────────────


def load_d_items(paths_config: dict) -> dict[int, str]:
    """Load d_items.csv.gz and return itemid -> label mapping.

    Args:
        paths_config: Loaded mimic_paths.yaml config.

    Returns:
        Dict mapping itemid (int) to label (str).
    """
    df = load_raw_table(
        paths_config, "icu", "d_items",
        usecols=["itemid", "label"],
    )
    mapping = dict(zip(df["itemid"].astype(int), df["label"]))
    logger.info("Loaded d_items: %d item mappings", len(mapping))
    return mapping


def process_inputevents(
    paths_config: dict,
    chunk_stay_ids_set: set,
    d_items_map: dict[int, str],
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process inputevents (IV fluids, medications, blood products).

    Args:
        paths_config: Loaded mimic_paths.yaml config.
        chunk_stay_ids_set: Set of stay_ids in this chunk.
        d_items_map: Dict mapping itemid -> label.
        windows_df: ICU windows DataFrame.
        intime_map: Dict mapping stay_id -> icu_intime.
        outtime_map: Dict mapping stay_id -> icu_outtime.

    Returns:
        DataFrame in unified output schema.
    """
    logger.info("Processing inputevents (raw, stay_id, continuous)")

    usecols = [
        "stay_id", "starttime", "endtime", "itemid",
        "amount", "amountuom", "rate", "rateuom",
        "ordercategorydescription", "statusdescription",
    ]

    df = load_and_filter_raw_table(
        paths_config, "icu.inputevents", "stay_id",
        chunk_stay_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No inputevents data found")
        return empty_events_df()

    # Filter to valid status (exclude Rewritten, ChangeDose/Rate)
    n_before = len(df)
    df = df[df["statusdescription"].isin(VALID_INPUT_STATUS)].reset_index(drop=True)
    logger.info(
        "  Status filter: %d -> %d rows (dropped %d)",
        n_before, len(df), n_before - len(df),
    )

    if len(df) == 0:
        return empty_events_df()

    # Parse timestamps and clip to ICU window
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["endtime"] = pd.to_datetime(df["endtime"])
    df = clip_to_icu_window(df, windows_df, "starttime")

    if len(df) == 0:
        return empty_events_df()

    # Resolve itemid to human-readable label
    df["variable_name"] = df["itemid"].map(d_items_map)
    unmapped = df["variable_name"].isna()
    if unmapped.any():
        logger.warning("  %d rows with unmapped itemid", unmapped.sum())
        df.loc[unmapped, "variable_name"] = (
            "unknown_item_" + df.loc[unmapped, "itemid"].astype(str)
        )

    # Build output schema columns
    df["timestamp"] = df["starttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    # End timestamp (clamped to icu_outtime)
    df["end_timestamp"] = df["endtime"]
    outtimes = df["stay_id"].map(outtime_map)
    exceed_mask = df["end_timestamp"] > outtimes
    if exceed_mask.any():
        logger.info("  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum())
        df.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
    df["end_timestamp_hours"] = compute_hours_since_admission(
        df["end_timestamp"], df["stay_id"], intime_map,
    )

    # Value columns
    df["numeric_value"] = pd.to_numeric(df["amount"], errors="coerce")

    # Build text_value with rate and category info
    rate_str = df["rate"].astype(str).replace("nan", "")
    rateuom_str = df["rateuom"].fillna("")
    cat_str = df["ordercategorydescription"].fillna("")
    df["text_value"] = np.where(
        rate_str != "",
        "rate:" + rate_str + " " + rateuom_str + " | " + cat_str,
        cat_str,
    )
    df["text_value"] = df["text_value"].str.strip(" |")

    df["unit"] = df["amountuom"].fillna("")
    df["domain"] = "action"
    df["source_table"] = "inputevents"
    df["category"] = "fluid_med_input"
    df["is_continuous"] = True

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


def process_procedureevents(
    paths_config: dict,
    chunk_stay_ids_set: set,
    d_items_map: dict[int, str],
    windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process procedureevents (ICU procedures).

    Args:
        paths_config: Loaded mimic_paths.yaml config.
        chunk_stay_ids_set: Set of stay_ids in this chunk.
        d_items_map: Dict mapping itemid -> label.
        windows_df: ICU windows DataFrame.
        intime_map: Dict mapping stay_id -> icu_intime.
        outtime_map: Dict mapping stay_id -> icu_outtime.

    Returns:
        DataFrame in unified output schema.
    """
    logger.info("Processing procedureevents (raw, stay_id, continuous)")

    usecols = [
        "stay_id", "starttime", "endtime", "itemid",
        "value", "valueuom", "ordercategorydescription", "statusdescription",
    ]

    df = load_and_filter_raw_table(
        paths_config, "icu.procedureevents", "stay_id",
        chunk_stay_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No procedureevents data found")
        return empty_events_df()

    # Filter to valid status
    n_before = len(df)
    df = df[df["statusdescription"].isin(VALID_INPUT_STATUS)].reset_index(drop=True)
    logger.info(
        "  Status filter: %d -> %d rows (dropped %d)",
        n_before, len(df), n_before - len(df),
    )

    if len(df) == 0:
        return empty_events_df()

    # Parse timestamps and clip to ICU window
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["endtime"] = pd.to_datetime(df["endtime"])
    df = clip_to_icu_window(df, windows_df, "starttime")

    if len(df) == 0:
        return empty_events_df()

    # Resolve itemid to label
    df["variable_name"] = df["itemid"].map(d_items_map)
    unmapped = df["variable_name"].isna()
    if unmapped.any():
        logger.warning("  %d rows with unmapped itemid", unmapped.sum())
        df.loc[unmapped, "variable_name"] = (
            "unknown_item_" + df.loc[unmapped, "itemid"].astype(str)
        )

    # Build schema columns
    df["timestamp"] = df["starttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    # End timestamp (clamped)
    df["end_timestamp"] = df["endtime"]
    outtimes = df["stay_id"].map(outtime_map)
    exceed_mask = df["end_timestamp"] > outtimes
    if exceed_mask.any():
        logger.info("  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum())
        df.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
    df["end_timestamp_hours"] = compute_hours_since_admission(
        df["end_timestamp"], df["stay_id"], intime_map,
    )

    df["numeric_value"] = pd.to_numeric(df["value"], errors="coerce")
    df["text_value"] = df["ordercategorydescription"].fillna("")
    df["unit"] = df["valueuom"].fillna("")
    df["domain"] = "action"
    df["source_table"] = "procedureevents"
    df["category"] = "procedure"
    df["is_continuous"] = True

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


def process_prescriptions(
    paths_config: dict,
    chunk_hadm_ids_set: set,
    hadm_windows_df: pd.DataFrame,
    intime_map: dict,
    outtime_map: dict,
) -> pd.DataFrame:
    """Process prescriptions (medication orders, hospital level).

    Args:
        paths_config: Loaded mimic_paths.yaml config.
        chunk_hadm_ids_set: Set of hadm_ids in this chunk.
        hadm_windows_df: DataFrame for hadm_id -> stay_id + ICU window matching.
        intime_map: Dict mapping stay_id -> icu_intime.
        outtime_map: Dict mapping stay_id -> icu_outtime.

    Returns:
        DataFrame in unified output schema.
    """
    logger.info("Processing prescriptions (raw, hadm_id, continuous)")

    usecols = [
        "hadm_id", "starttime", "stoptime",
        "drug", "prod_strength", "dose_val_rx", "dose_unit_rx", "route",
    ]

    df = load_and_filter_raw_table(
        paths_config, "hosp.prescriptions", "hadm_id",
        chunk_hadm_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No prescriptions data found")
        return empty_events_df()

    # Map hadm_id -> stay_id and filter to ICU window
    df = assign_stay_ids_from_hadm(df, hadm_windows_df, "starttime")

    if len(df) == 0:
        logger.warning("  No prescriptions rows matched ICU windows")
        return empty_events_df()

    # Parse stoptime and drop rows where starttime > stoptime
    df["stoptime"] = pd.to_datetime(df["stoptime"])
    bad_order = df["starttime"] > df["stoptime"]
    if bad_order.any():
        logger.info(
            "  Dropped %d rows with starttime > stoptime", bad_order.sum(),
        )
        df = df[~bad_order].reset_index(drop=True)

    if len(df) == 0:
        return empty_events_df()

    # Build schema columns
    df["variable_name"] = df["drug"].fillna("unknown_drug")
    df["timestamp"] = df["starttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    # End timestamp (clamped to icu_outtime)
    df["end_timestamp"] = df["stoptime"]
    outtimes = df["stay_id"].map(outtime_map)
    exceed_mask = df["end_timestamp"].notna() & (df["end_timestamp"] > outtimes)
    if exceed_mask.any():
        logger.info("  Clamped %d end_timestamps to icu_outtime", exceed_mask.sum())
        df.loc[exceed_mask, "end_timestamp"] = outtimes[exceed_mask]
    df["end_timestamp_hours"] = compute_hours_since_admission(
        df["end_timestamp"], df["stay_id"], intime_map,
    )

    # Parse dose to numeric (some values are text like "1-2")
    df["numeric_value"] = pd.to_numeric(df["dose_val_rx"], errors="coerce")

    # Build text_value with route and strength info
    route_str = df["route"].fillna("")
    strength_str = df["prod_strength"].fillna("")
    df["text_value"] = np.where(
        route_str != "",
        "route:" + route_str + " | " + strength_str,
        strength_str,
    )
    df["text_value"] = df["text_value"].str.strip(" |")

    df["unit"] = df["dose_unit_rx"].fillna("")
    df["domain"] = "action"
    df["source_table"] = "prescriptions"
    df["category"] = "medication_order"
    df["is_continuous"] = True

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


def process_emar(
    paths_config: dict,
    chunk_hadm_ids_set: set,
    hadm_windows_df: pd.DataFrame,
    intime_map: dict,
) -> pd.DataFrame:
    """Process emar (medication administration records).

    Args:
        paths_config: Loaded mimic_paths.yaml config.
        chunk_hadm_ids_set: Set of hadm_ids in this chunk.
        hadm_windows_df: DataFrame for hadm_id -> stay_id + ICU window matching.
        intime_map: Dict mapping stay_id -> icu_intime.

    Returns:
        DataFrame in unified output schema.
    """
    logger.info("Processing emar (raw, hadm_id, point events)")

    usecols = ["hadm_id", "charttime", "medication", "event_txt"]

    df = load_and_filter_raw_table(
        paths_config, "hosp.emar", "hadm_id",
        chunk_hadm_ids_set, usecols=usecols,
    )

    if len(df) == 0:
        logger.warning("  No emar data found")
        return empty_events_df()

    # Filter to administered events only
    n_before = len(df)
    df = df[df["event_txt"].isin(VALID_EMAR_EVENTS)].reset_index(drop=True)
    logger.info(
        "  Event filter (Administered/Applied): %d -> %d rows (dropped %d)",
        n_before, len(df), n_before - len(df),
    )

    if len(df) == 0:
        return empty_events_df()

    # Map hadm_id -> stay_id and filter to ICU window
    df = assign_stay_ids_from_hadm(df, hadm_windows_df, "charttime")

    if len(df) == 0:
        logger.warning("  No emar rows matched ICU windows")
        return empty_events_df()

    # Build schema columns
    df["variable_name"] = df["medication"].fillna("unknown_medication")
    df["timestamp"] = df["charttime"]
    df["timestamp_hours"] = compute_hours_since_admission(
        df["timestamp"], df["stay_id"], intime_map,
    )

    df["numeric_value"] = np.nan
    df["text_value"] = df["event_txt"].fillna("")
    df["unit"] = ""
    df["end_timestamp"] = pd.NaT
    df["end_timestamp_hours"] = np.nan
    df["domain"] = "action"
    df["source_table"] = "emar"
    df["category"] = "medication_admin"
    df["is_continuous"] = False

    result = df[OUTPUT_COLUMNS].copy()
    logger.info("  Produced %d event rows", len(result))

    del df
    gc.collect()

    return result


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for action extraction."""
    parser = argparse.ArgumentParser(description="Step 03: Extract Actions")
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
        "Step 03: Extract Actions (chunk %d/%d)",
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
    my_hadm_ids_set = set(chunk_cohort["hadm_id"].tolist())

    # Build lookup structures
    intime_map = build_intime_map(chunk_cohort)
    outtime_map = build_outtime_map(chunk_cohort)
    windows_df = build_icu_windows(chunk_cohort)
    hadm_windows_df = build_hadm_windows(chunk_cohort)

    # Load d_items for itemid resolution (inputevents, procedureevents)
    d_items_map = load_d_items(paths_config)

    del cohort_df
    gc.collect()

    # Prepare output directory
    output_dir = ensure_dir(resolve_table_path(paths_config, "output.actions"))

    # Process all action sources
    actions_config = features_config["actions"]
    all_events: dict[int, list[pd.DataFrame]] = {}
    source_counts: dict[str, int] = {}

    for source_name, source_config in actions_config.items():
        is_raw = source_config.get("is_raw_table", False)

        if is_raw:
            # Dedicated handlers for raw tables
            if source_name == "inputevents":
                events = process_inputevents(
                    paths_config, my_stay_ids_set, d_items_map,
                    windows_df, intime_map, outtime_map,
                )
            elif source_name == "procedureevents":
                events = process_procedureevents(
                    paths_config, my_stay_ids_set, d_items_map,
                    windows_df, intime_map, outtime_map,
                )
            elif source_name == "prescriptions":
                events = process_prescriptions(
                    paths_config, my_hadm_ids_set,
                    hadm_windows_df, intime_map, outtime_map,
                )
            elif source_name == "emar":
                events = process_emar(
                    paths_config, my_hadm_ids_set,
                    hadm_windows_df, intime_map,
                )
            else:
                logger.warning("Unknown raw action source: %s", source_name)
                events = empty_events_df()
        else:
            # Concepts table — generic processor
            events = process_concepts_action_source(
                source_name, source_config, paths_config,
                my_stay_ids_set, windows_df, intime_map, outtime_map,
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
        logger.info("    %-40s %d", source_name, count)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
