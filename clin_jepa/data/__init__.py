"""Data pipeline — turn raw MIMIC-IV into trajectory shards for training.

The pipeline runs as six sequential steps. Each step reads only the output
of earlier steps (plus, in step 01-03, the raw MIMIC-IV tables) and writes
to a fresh subdirectory under ``$CLIN_JEPA_DATA``.

============ ============================================================
Step         What it does
============ ============================================================
step 01      Define the ICU cohort (age >= 18, LOS >= 6h, first stay only).
step 02      Extract per-stay observation events (vitals, labs, scores).
step 03      Extract per-stay action events (medications, ventilator
             settings, procedures).
step 04      Bin observations and actions into 1-hour discretized buckets
             aligned to ICU admission.
step 05      Patient-level train/val/test split (70/15/15).
step 06      Extract sliding-window trajectories with per-hour state and
             action text + per-hour labels, ready for the encoder.
============ ============================================================

See ``docs/data-preparation.md`` for download / credentialing / wiring
instructions, and the paper (§5 setup + Appendix A feature inventory) for
the design rationale.
"""
