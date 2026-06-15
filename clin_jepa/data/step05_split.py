"""Step 05: Train/Val/Test Data Split.

Splits patients (subject_id) into 70/15/15 train/val/test groups,
stratified by mortality and sepsis3 status. Maps subject-level splits
back to stay_ids and writes splits.csv.

Split is on subject_id (not stay_id) to prevent patient leakage across splits.

Input:
  - $CLIN_JEPA_DATA/cohort/cohort.csv  (Step 01)

Output:
  - $CLIN_JEPA_DATA/splits/splits.csv

Usage:
    python -m clin_jepa.data.step05_split
    python -m clin_jepa.data.step05_split --seed 42
"""

import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from clin_jepa.data.helpers.io_utils import resolve_table_path
from clin_jepa.utils import ensure_dir, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for data splitting."""
    parser = argparse.ArgumentParser(
        description="Step 05: Train/Val/Test Data Split",
    )
    parser.add_argument(
        "--paths_config", default="configs/data/mimic_paths.yaml",
        help="Path to mimic_paths.yaml",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    logger.info("=" * 60)
    logger.info("Step 05: Train/Val/Test Data Split")
    logger.info("=" * 60)

    # Load config and cohort
    paths_config = load_config(args.paths_config)
    cohort_dir = resolve_table_path(paths_config, "output.cohort")
    cohort_df = pd.read_csv(cohort_dir / "cohort.csv")
    logger.info("Loaded cohort: %d stays, %d subjects",
                len(cohort_df), cohort_df["subject_id"].nunique())

    subject_labels = (
        cohort_df.groupby("subject_id")
        .agg(
            mortality=("hospital_expire_flag", "max"),
            sepsis3=("sepsis3", "max"),
        )
        .reset_index()
    )

    # Combined stratification label: 4 groups (mort x sepsis)
    subject_labels["strat_label"] = (
        subject_labels["mortality"].astype(int).astype(str)
        + "_"
        + subject_labels["sepsis3"].astype(int).astype(str)
    )
    logger.info("Subject-level strat groups:\n%s",
                subject_labels["strat_label"].value_counts().to_string())

    # --- Two-stage stratified split: 70/30, then 30 -> 50/50 ---
    subjects = subject_labels["subject_id"].values
    strat = subject_labels["strat_label"].values

    train_subj, temp_subj, _, temp_strat = train_test_split(
        subjects, strat,
        test_size=0.30, random_state=args.seed, stratify=strat,
    )
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=0.50, random_state=args.seed, stratify=temp_strat,
    )

    logger.info("Subject splits: train=%d, val=%d, test=%d",
                len(train_subj), len(val_subj), len(test_subj))

    # --- Map subject splits back to stay_ids ---
    train_set = set(train_subj)
    val_set = set(val_subj)
    test_set = set(test_subj)

    def assign_split(subject_id: int) -> str:
        if subject_id in train_set:
            return "train"
        if subject_id in val_set:
            return "val"
        return "test"

    splits_df = cohort_df[["stay_id", "subject_id"]].copy()
    splits_df["split"] = splits_df["subject_id"].map(assign_split)

    # --- Verify no subject leakage ---
    subject_splits = splits_df.groupby("subject_id")["split"].nunique()
    n_leaked = (subject_splits > 1).sum()
    assert n_leaked == 0, f"Subject leakage: {n_leaked} subjects in multiple splits"

    # --- Write output ---
    output_dir = ensure_dir(resolve_table_path(paths_config, "output.splits"))
    output_path = output_dir / "splits.csv"
    splits_df.to_csv(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(splits_df))

    # --- Log split statistics ---
    logger.info("=" * 60)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 60)

    # Merge back cohort metadata for stats
    merged = splits_df.merge(
        cohort_df[["stay_id", "hospital_expire_flag", "sepsis3"]],
        on="stay_id",
    )

    for split_name in ["train", "val", "test"]:
        subset = merged[merged["split"] == split_name]
        n_stays = len(subset)
        n_subjects = subset["subject_id"].nunique()
        mort_rate = subset["hospital_expire_flag"].mean() * 100
        sepsis_rate = subset["sepsis3"].mean() * 100
        logger.info(
            "  %-6s: %6d stays (%5d subjects) | mortality %.1f%% | sepsis %.1f%%",
            split_name, n_stays, n_subjects, mort_rate, sepsis_rate,
        )

    total_stays = len(splits_df)
    for split_name in ["train", "val", "test"]:
        n = (splits_df["split"] == split_name).sum()
        logger.info("  %-6s fraction: %.1f%%", split_name, 100.0 * n / total_stays)

    logger.info("=" * 60)
    logger.info("Step 05 complete.")


if __name__ == "__main__":
    main()
