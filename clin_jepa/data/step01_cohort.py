"""Step 01: Cohort Definition.

Selects the study cohort from MIMIC-IV icustay_detail, applying
inclusion/exclusion criteria. Produces cohort.csv consumed by all
downstream pipeline steps.

Filters applied (in order):
  1. Age >= 18
  2. Valid ICU in/out times (non-null)
  3. First ICU stay per admission (first_icu_stay == "t")
  4. ICU LOS >= 6 hours
  5. Has at least one vitals record (chunked check of vitalsign.csv)

Output includes sepsis3 flag and charlson comorbidity index.

Usage:
    python -m clin_jepa.data.step01_cohort
"""

import argparse
import gc
import json
import logging

import pandas as pd

from clin_jepa.data.helpers.io_utils import load_concepts_table, resolve_table_path
from clin_jepa.utils import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


def _log_filter(name: str, before: int, after: int) -> None:
    """Log a filter step with counts."""
    excluded = before - after
    pct = (excluded / before * 100) if before > 0 else 0.0
    logger.info(
        "  Filter [%s]: %d -> %d (excluded %d, %.1f%%)",
        name, before, after, excluded, pct,
    )


def _get_vitalsign_stay_ids(paths_config: dict) -> set:
    """Get set of stay_ids that have at least one vitals record.

    Reads vitalsign.csv in chunks (stay_id only).
    """
    filepath = resolve_table_path(paths_config, "concepts.measurement.vitalsign")
    logger.info("Scanning vitalsign.csv for stay_ids (chunked)...")

    stay_ids: set = set()
    for chunk in pd.read_csv(filepath, usecols=["stay_id"], chunksize=1_000_000):
        stay_ids.update(chunk["stay_id"].dropna().astype(int).unique())
        del chunk

    logger.info("  Found %d unique stay_ids with vitals records", len(stay_ids))
    return stay_ids


def build_cohort(paths_config: dict, cohort_config: dict) -> pd.DataFrame:
    """Build the study cohort by applying inclusion/exclusion criteria.

    Args:
        paths_config: Loaded mimic_paths.yaml config.
        cohort_config: Loaded cohort.yaml config.

    Returns:
        DataFrame with the filtered cohort and descriptors.
    """
    # --- A. Load icustay_detail ---
    df = load_concepts_table(
        paths_config, "concepts.demographics.icustay_detail"
    )
    logger.info("Starting cohort definition with %d ICU stays", len(df))
    initial_count = len(df)

    # --- B. Data integrity checks ---
    key_cols = ["stay_id", "subject_id", "hadm_id"]
    for col in key_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning("  %d null values in %s — dropping", null_count, col)
            df = df.dropna(subset=[col])

    n_unique = df["stay_id"].nunique()
    if n_unique != len(df):
        logger.warning(
            "  Non-unique stay_ids: %d unique out of %d rows — keeping first",
            n_unique, len(df),
        )
        df = df.drop_duplicates(subset=["stay_id"], keep="first")

    # --- C. Apply filters ---
    count_before = len(df)

    # Filter 1: Age >= 18
    min_age = cohort_config["inclusion"]["min_age"]
    df = df[df["admission_age"] >= min_age]
    _log_filter(f"age >= {min_age}", count_before, len(df))
    count_before = len(df)

    # Filter 2: Valid ICU in/out times
    df = df.dropna(subset=["icu_intime", "icu_outtime"])
    _log_filter("valid ICU times", count_before, len(df))
    count_before = len(df)

    # Filter 3: First ICU stay per admission
    if cohort_config["exclusion"]["keep_first_icu_stay_only"]:
        df = df[df["first_icu_stay"] == "t"]
        _log_filter("first ICU stay only", count_before, len(df))
        count_before = len(df)

    # Filter 4: ICU LOS >= min hours
    min_los_hours = cohort_config["exclusion"]["min_icu_los_hours"]
    df = df[(df["los_icu"] * 24) >= min_los_hours]
    _log_filter(f"ICU LOS >= {min_los_hours}h", count_before, len(df))
    count_before = len(df)

    # Filter 5: Has labs or vitals
    if cohort_config["exclusion"]["require_labs_or_vitals"]:
        vital_stay_ids = _get_vitalsign_stay_ids(paths_config)
        df = df[df["stay_id"].isin(vital_stay_ids)]
        _log_filter("has vitals/labs", count_before, len(df))
        del vital_stay_ids
        gc.collect()

    # --- D. Compute derived columns ---
    max_los_hours = cohort_config["max_icu_los_hours"]
    df["icu_los_hours"] = (df["los_icu"] * 24).clip(upper=max_los_hours)
    logger.info(
        "  ICU LOS hours: median=%.1f, mean=%.1f, max=%.1f (capped at %d)",
        df["icu_los_hours"].median(),
        df["icu_los_hours"].mean(),
        df["icu_los_hours"].max(),
        max_los_hours,
    )

    # --- E. Join sepsis3 ---
    logger.info("Joining sepsis3 data...")
    sepsis_df = load_concepts_table(
        paths_config, "concepts.sepsis.sepsis3",
        usecols=["stay_id", "sepsis3", "sofa_time", "suspected_infection_time"],
    )
    # Convert PostgreSQL boolean strings to Python booleans
    sepsis_df["sepsis3"] = sepsis_df["sepsis3"] == "t"
    sepsis_df = sepsis_df.rename(columns={"sofa_time": "sepsis_sofa_time"})

    df = df.merge(sepsis_df, on="stay_id", how="left")
    df["sepsis3"] = df["sepsis3"].fillna(False).astype(bool)
    logger.info(
        "  Sepsis3 prevalence: %d / %d (%.1f%%)",
        df["sepsis3"].sum(), len(df), df["sepsis3"].mean() * 100,
    )
    del sepsis_df
    gc.collect()

    # --- F. Join charlson comorbidity index ---
    logger.info("Joining charlson comorbidity data...")
    charlson_df = load_concepts_table(
        paths_config, "concepts.comorbidity.charlson",
        usecols=["hadm_id", "charlson_comorbidity_index"],
    )

    df = df.merge(charlson_df, on="hadm_id", how="left")
    charlson_coverage = df["charlson_comorbidity_index"].notna().mean() * 100
    logger.info(
        "  Charlson coverage: %.1f%% non-null, median=%.0f",
        charlson_coverage,
        df["charlson_comorbidity_index"].median(),
    )
    del charlson_df
    gc.collect()

    # --- G. Convert first_icu_stay to boolean ---
    df["first_icu_stay"] = df["first_icu_stay"] == "t"

    # --- H. Select and order output columns ---
    output_cols = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "gender",
        "admission_age",
        "race",
        "icu_intime",
        "icu_outtime",
        "los_icu",
        "icu_los_hours",
        "hospital_expire_flag",
        "first_icu_stay",
        "icustay_seq",
        "sepsis3",
        "sepsis_sofa_time",
        "suspected_infection_time",
        "charlson_comorbidity_index",
    ]
    df = df[output_cols].sort_values("stay_id").reset_index(drop=True)

    logger.info(
        "Final cohort: %d stays (%.1f%% of initial %d)",
        len(df), len(df) / initial_count * 100, initial_count,
    )

    return df


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the cohort."""
    stats = {
        "n_stays": int(len(df)),
        "n_patients": int(df["subject_id"].nunique()),
        "n_admissions": int(df["hadm_id"].nunique()),
        "age": {
            "min": float(df["admission_age"].min()),
            "max": float(df["admission_age"].max()),
            "mean": float(df["admission_age"].mean()),
            "median": float(df["admission_age"].median()),
        },
        "icu_los_hours": {
            "min": float(df["icu_los_hours"].min()),
            "max": float(df["icu_los_hours"].max()),
            "mean": float(df["icu_los_hours"].mean()),
            "median": float(df["icu_los_hours"].median()),
        },
        "gender": df["gender"].value_counts().to_dict(),
        "mortality_rate": float(df["hospital_expire_flag"].mean()),
        "sepsis3_prevalence": float(df["sepsis3"].mean()),
        "charlson": {
            "coverage_pct": float(df["charlson_comorbidity_index"].notna().mean() * 100),
            "mean": float(df["charlson_comorbidity_index"].mean()),
            "median": float(df["charlson_comorbidity_index"].median()),
        },
    }
    return stats


def main() -> None:
    """Main entry point for cohort definition."""
    parser = argparse.ArgumentParser(description="Step 01: Cohort Definition")
    parser.add_argument(
        "--paths_config",
        type=str,
        default="configs/data/mimic_paths.yaml",
        help="Path to mimic_paths.yaml config",
    )
    parser.add_argument(
        "--cohort_config",
        type=str,
        default="configs/data/cohort.yaml",
        help="Path to cohort.yaml config",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 01: Cohort Definition")
    logger.info("=" * 60)

    # Load configs
    paths_config = load_config(args.paths_config)
    cohort_config = load_config(args.cohort_config)

    # Build cohort
    cohort_df = build_cohort(paths_config, cohort_config)

    # Save cohort.csv
    output_dir = ensure_dir(resolve_table_path(paths_config, "output.cohort"))
    output_path = output_dir / "cohort.csv"
    cohort_df.to_csv(output_path, index=False)
    logger.info("Saved cohort to %s", output_path)

    # Compute and save stats
    stats = compute_stats(cohort_df)
    stats_path = output_dir / "cohort_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved cohort stats to %s", stats_path)

    # Log summary
    logger.info("=" * 60)
    logger.info("COHORT SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total stays:        %d", stats["n_stays"])
    logger.info("  Total patients:     %d", stats["n_patients"])
    logger.info("  Total admissions:   %d", stats["n_admissions"])
    logger.info("  Age range:          %.1f - %.1f (median %.1f)",
                stats["age"]["min"], stats["age"]["max"], stats["age"]["median"])
    logger.info("  ICU LOS hours:      %.1f - %.1f (median %.1f)",
                stats["icu_los_hours"]["min"], stats["icu_los_hours"]["max"],
                stats["icu_los_hours"]["median"])
    logger.info("  Gender:             %s", stats["gender"])
    logger.info("  Mortality rate:     %.1f%%", stats["mortality_rate"] * 100)
    logger.info("  Sepsis3 prevalence: %.1f%%", stats["sepsis3_prevalence"] * 100)
    logger.info("  Charlson coverage:  %.1f%%", stats["charlson"]["coverage_pct"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
