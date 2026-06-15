# Data preparation

End-to-end checklist to go from a fresh machine to the trajectory shards that
the encoder consumes.

## 1. Get credentialed for MIMIC-IV

MIMIC-IV is hosted on [PhysioNet](https://physionet.org/content/mimiciv/) under
credentialed access. You will need to:

1. Create a PhysioNet account.
2. Complete the *CITI Data or Specimens Only Research* training (free, ~2-3
   hours, valid for several years).
3. Sign the MIMIC-IV data use agreement.

Once your credentialed-access flag is approved, download MIMIC-IV v3.1.

## 2. Download MIMIC-IV v3.1

```bash
# Convenience: install wget (PhysioNet exposes everything over HTTPS)
mkdir -p $MIMIC_RAW
cd $MIMIC_RAW

wget -r -N -c -np --user <PHYSIONET_USER> --ask-password \
    https://physionet.org/files/mimiciv/3.1/
```

This downloads three subdirectories — `icu/`, `hosp/`, `note/` — totalling
~75 GB compressed. See `configs/data/mimic_paths.yaml.example` for the exact files
the pipeline reads.

## 3. Build the mimic-code concepts tables

A subset of the pipeline (`step01_cohort.py`, `step02_observations.py`,
`step03_actions.py`) reads tables that don't ship with the raw MIMIC-IV
release: per-source-table normalised vital signs, lab values, SOFA scores,
medication categorisations, and so on. These are produced by the
[`mimic-code`](https://github.com/MIT-LCP/mimic-code) project.

Two common ways to materialise them:

* **BigQuery** — run the SQL in `mimic-code/mimic-iv/concepts/` against the
  PhysioNet-hosted MIMIC-IV BigQuery dataset, then export each materialised
  view to `.csv`.
* **Local DuckDB** — load the raw MIMIC-IV `.csv.gz` files into DuckDB and
  run the same SQL locally; DuckDB executes the materialised views in
  minutes.

Either way, you should end up with `$MIMIC_RAW/concepts/<category>/<view>.csv`
for every entry listed under `concepts:` in
`configs/data/mimic_paths.yaml.example`.

## 4. Configure paths

```bash
cp .env.example .env                       # then edit the four variables
cp configs/data/mimic_paths.yaml.example configs/data/mimic_paths.yaml
# (paths.yaml works out of the box if $MIMIC_RAW and $CLIN_JEPA_DATA are exported)
```

## 5. Run the pipeline

The six steps are sequential — each reads only what the previous wrote.

```bash
# Step 01: define the ICU cohort.
python -m clin_jepa.data.step01_cohort

# Steps 02-04: chunked parallelism. Each invocation processes one chunk
# of stays out of N. Run with --chunk_idx 0..N-1 in parallel
# (for example, a SLURM array job, or a simple for-loop in shell).
for i in 0 1 2 3 4 5 6 7; do
    python -m clin_jepa.data.step02_observations  --chunk_idx $i --total_chunks 8 &
done; wait
for i in 0 1 2 3 4 5 6 7; do
    python -m clin_jepa.data.step03_actions       --chunk_idx $i --total_chunks 8 &
done; wait
for i in 0 1 2 3 4 5 6 7; do
    python -m clin_jepa.data.step04_discretization --chunk_idx $i --total_chunks 8 &
done; wait

# Step 05: patient-level 70/15/15 train/val/test split.
python -m clin_jepa.data.step05_split

# Step 06: sliding-window trajectories with per-hour state + action text
#   (uses an in-process multiprocessing pool — set --n_workers to taste).
python -m clin_jepa.data.step06_trajectories \
    --config configs/data/trajectories.yaml --n_workers 10
```

Steps 02-04 distribute work across chunks at the command-line boundary
(one OS process per chunk); Step 06 uses an in-process worker pool. On a
single machine 8 chunks is a good default; on a SLURM cluster you can
submit each chunk as a separate array task.

## 6. Verify

After Step 06 finishes you should see roughly:

```
$CLIN_JEPA_DATA/
├── cohort/             # cohort.csv + cohort_stats.json
├── observations/       # 84,497 per-stay parquet files
├── actions/            # 84,497 per-stay parquet files
├── discretized/        # 84,497 per-stay pickle files
├── splits/             # splits.csv (70/15/15)
└── trajectories/       # train/, val/, test/ subdirs of .pt shards
                        # ~280,882 windows total
```

The trajectory shards are the input to the encoder SFT and to all
downstream training (see `docs/training.md`).
