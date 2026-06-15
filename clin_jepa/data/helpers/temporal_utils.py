"""Temporal utilities for timestamp normalization and ICU window operations.

Handles conversion of raw timestamps to hours-since-ICU-admission,
builds lookup structures for ICU window matching, and provides functions
for clipping events to ICU windows and assigning stay_ids to tables
that join on subject_id or hadm_id.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_icu_windows(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame for subject_id → ICU window matching.

    Used for tables that join on subject_id instead of stay_id.
    Returns a DataFrame with columns: subject_id, stay_id, icu_intime, icu_outtime.

    Args:
        cohort_df: Cohort DataFrame with icu_intime/outtime as datetime.

    Returns:
        DataFrame suitable for merging on subject_id.
    """
    windows = cohort_df[["subject_id", "stay_id", "icu_intime", "icu_outtime"]].copy()
    windows["icu_intime"] = pd.to_datetime(windows["icu_intime"])
    windows["icu_outtime"] = pd.to_datetime(windows["icu_outtime"])
    logger.info(
        "Built ICU windows: %d stays across %d subjects",
        len(windows), windows["subject_id"].nunique(),
    )
    return windows


def build_intime_map(cohort_df: pd.DataFrame) -> dict:
    """Build stay_id → icu_intime mapping.

    Args:
        cohort_df: Cohort DataFrame with icu_intime as datetime.

    Returns:
        Dict mapping stay_id (int) to icu_intime (pd.Timestamp).
    """
    df = cohort_df[["stay_id", "icu_intime"]].copy()
    df["icu_intime"] = pd.to_datetime(df["icu_intime"])
    return dict(zip(df["stay_id"], df["icu_intime"]))


def build_outtime_map(cohort_df: pd.DataFrame) -> dict:
    """Build stay_id → icu_outtime mapping.

    Args:
        cohort_df: Cohort DataFrame with icu_outtime as datetime.

    Returns:
        Dict mapping stay_id (int) to icu_outtime (pd.Timestamp).
    """
    df = cohort_df[["stay_id", "icu_outtime"]].copy()
    df["icu_outtime"] = pd.to_datetime(df["icu_outtime"])
    return dict(zip(df["stay_id"], df["icu_outtime"]))


def build_hadm_windows(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame for hadm_id → ICU window matching.

    Used for raw tables that join on hadm_id (prescriptions, emar).
    Returns a DataFrame with columns: hadm_id, stay_id, icu_intime, icu_outtime.

    Args:
        cohort_df: Cohort DataFrame with hadm_id, stay_id, and ICU times.

    Returns:
        DataFrame suitable for merging on hadm_id.
    """
    windows = cohort_df[["hadm_id", "stay_id", "icu_intime", "icu_outtime"]].copy()
    windows["icu_intime"] = pd.to_datetime(windows["icu_intime"])
    windows["icu_outtime"] = pd.to_datetime(windows["icu_outtime"])
    logger.info(
        "Built hadm windows: %d stays across %d admissions",
        len(windows), windows["hadm_id"].nunique(),
    )
    return windows


def compute_hours_since_admission(
    timestamps: pd.Series,
    stay_ids: pd.Series,
    intime_map: dict,
) -> pd.Series:
    """Convert timestamps to hours since ICU admission (vectorized).

    Args:
        timestamps: Series of datetime64 timestamps.
        stay_ids: Series of stay_id values (same index as timestamps).
        intime_map: Dict mapping stay_id → icu_intime.

    Returns:
        Series of float64 hours since admission.
    """
    intimes = stay_ids.map(intime_map)
    return (timestamps - intimes).dt.total_seconds() / 3600.0


def clip_to_icu_window(
    df: pd.DataFrame,
    windows_df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Filter stay_id-joined table rows to ICU window timestamps.

    stay_id concepts tables may have data timestamped outside the actual
    ICU stay (e.g., SOFA computed post-discharge, vitals charted in ward).
    This clips to [icu_intime, icu_outtime).

    Args:
        df: DataFrame with stay_id and timestamp columns (already parsed).
        windows_df: DataFrame with stay_id, icu_intime, icu_outtime.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame filtered to rows within the ICU window.
    """
    if len(df) == 0:
        return df

    n_before = len(df)
    merged = df.merge(
        windows_df[["stay_id", "icu_intime", "icu_outtime"]],
        on="stay_id",
        how="inner",
    )
    mask = (
        (merged[timestamp_col] >= merged["icu_intime"])
        & (merged[timestamp_col] < merged["icu_outtime"])
    )
    result = merged.loc[mask].drop(columns=["icu_intime", "icu_outtime"])
    n_after = len(result)

    if n_after < n_before:
        logger.info(
            "  ICU window clip: %d → %d rows (dropped %d outside window)",
            n_before, n_after, n_before - n_after,
        )

    return result.reset_index(drop=True)


def assign_stay_ids_vectorized(
    df: pd.DataFrame,
    windows_df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Assign stay_id to rows from subject_id-joined tables.

    Uses vectorized merge + filter: merge on subject_id (one-to-many),
    then filter rows where timestamp falls within [icu_intime, icu_outtime).

    Args:
        df: DataFrame with subject_id and timestamp columns.
        windows_df: DataFrame with subject_id, stay_id, icu_intime, icu_outtime.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with stay_id assigned, rows outside any ICU window dropped.
    """
    if len(df) == 0:
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Merge on subject_id (one measurement row x multiple ICU windows)
    merged = df.merge(windows_df, on="subject_id", how="inner")

    # Filter: timestamp must be within [icu_intime, icu_outtime)
    mask = (
        (merged[timestamp_col] >= merged["icu_intime"])
        & (merged[timestamp_col] < merged["icu_outtime"])
    )
    result = merged.loc[mask].drop(columns=["icu_intime", "icu_outtime"])

    return result.reset_index(drop=True)


def assign_stay_ids_from_hadm(
    df: pd.DataFrame,
    hadm_windows_df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Assign stay_id to rows from hadm_id-joined tables.

    Merges on hadm_id to get stay_id and ICU window, then filters to
    rows where timestamp falls within [icu_intime, icu_outtime).

    Args:
        df: DataFrame with hadm_id and timestamp columns.
        hadm_windows_df: DataFrame with hadm_id, stay_id, icu_intime, icu_outtime
                         (from build_hadm_windows).
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with stay_id assigned, rows outside ICU window dropped.
    """
    if len(df) == 0:
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Merge on hadm_id (one-to-many if multiple ICU stays per admission)
    merged = df.merge(hadm_windows_df, on="hadm_id", how="inner")

    # Filter: timestamp must be within [icu_intime, icu_outtime)
    mask = (
        (merged[timestamp_col] >= merged["icu_intime"])
        & (merged[timestamp_col] < merged["icu_outtime"])
    )
    result = merged.loc[mask].drop(columns=["icu_intime", "icu_outtime"])
    n_dropped = len(df) - len(result)

    if n_dropped > 0:
        logger.info(
            "  hadm_id window matching: %d → %d rows (dropped %d outside ICU)",
            len(df), len(result), n_dropped,
        )

    return result.reset_index(drop=True)
