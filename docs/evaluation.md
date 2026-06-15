# Evaluation

The paper reports three independent evaluation axes (§5):

1. **Co-training stability and convergence** (§5.2) — autoregressive
   rollout drift over a 48-hour horizon.
2. **Latent geometry diagnosis** (§5.3) — UMAP cohort discrimination.
3. **Downstream multi-task evaluation** (§5.4) — MLP probes on 7 ICareFM
   EEP tasks (Track 1) and 8 stay-level binary outcomes (Track 2).

This document covers axes 1 and 3 (probes); the UMAP cohort visualisation
in §5.3 is a one-off plotting exercise we leave to the reader and is not
shipped as a packaged pipeline.

The packaged pipeline mixes CLI scripts and Python-library modules. The
four steps marked **CLI** below are runnable directly via ``python -m``;
the rest are library modules
(:mod:`~clin_jepa.evaluation.rollout`,
:mod:`~clin_jepa.evaluation.probes`,
:mod:`~clin_jepa.evaluation.baselines.classical`,
:mod:`~clin_jepa.evaluation.baselines.train_dl`)
called from a small driver script you write to orchestrate one or all
paradigms. Example drivers are sketched in
[`docs/architecture.md`](architecture.md).

## Prerequisites

A trained encoder + predictor for whichever paradigm you want to evaluate
(Clin-JEPA / V-JEPA 2-AC / SFT baseline). Then precompute embeddings for
the test split using the same encoder used at training time:

```bash
# CLI:
python -m clin_jepa.evaluation.precompute_embeddings \
    --config configs/eval/precompute_embeddings.yaml \
    --plan clin_jepa
```

Repeat for each paradigm.

## 1. Drift evaluation (paper §5.2)

The drift metric runs autoregressive latent rollout at four context
lengths (`C ∈ {6, 12, 24, 48}`) on the test split and writes per-step L1
distances against the encoder's own forward outputs.

```python
# Library call (write this in a small driver script):
import torch
from clin_jepa.utils import load_config
from clin_jepa.evaluation.rollout import run_rollout_evaluation

cfg = load_config("configs/eval/rollout.yaml")
plan_cfg = {**cfg, **cfg["plans"]["clin_jepa"]}  # merge plan-specific keys
run_rollout_evaluation(plan_cfg, "absolute", torch.device("cuda"))
```

Outputs land under `$CLIN_JEPA_OUTPUT/evaluation/predictions/<plan>/`.
The paper's §5.2 figure compares the *normalised* drift trajectory (each
horizon's L1 divided by the `h=1` value) across the three paradigms plus
the two ablations.

## 2. Downstream multi-task evaluation (paper §5.4)

### Track 1 — ICareFM 7 EEP tasks

The four non-Clin-JEPA baselines (Ridge, LightGBM, LSTM, TCN) are
library calls; the MLP probe pipeline mixes one library call (training
the probes themselves) with three CLI scripts.

```python
# Train classical baselines (Ridge + LightGBM) on raw clinical features:
from clin_jepa.evaluation.baselines.classical import (
    train_ridge_continuous, train_lightgbm_continuous,
)
ridge_results    = train_ridge_continuous(train_ds, val_ds, test_ds, ...)
lightgbm_results = train_lightgbm_continuous(train_ds, val_ds, test_ds, ...)

# Train DL baselines (LSTM + TCN) on raw clinical features:
from clin_jepa.evaluation.baselines.train_dl import train_dl_continuous
lstm_results = train_dl_continuous(train_ds, val_ds, test_ds, cfg_lstm)
tcn_results  = train_dl_continuous(train_ds, val_ds, test_ds, cfg_tcn)
```

For the MLP probe (Clin-JEPA's downstream classifier):

```bash
# CLI: gather (embedding, label) pairs for each clinical-target variable
python -m clin_jepa.evaluation.collect_probe_data \
    --config configs/eval/probes.yaml --plan clin_jepa
```

```python
# Library call to train one MLP probe per variable:
from clin_jepa.evaluation.probes import train_mlp_probe_one_var
for var in target_vars:
    probe, normaliser, meta = train_mlp_probe_one_var(z, y, var, ...)
```

```bash
# CLI: build the alignment sidecar (run once per paradigm × context):
python -m clin_jepa.evaluation.build_alignment --plan clin_jepa

# CLI: apply trained MLP probes to the rollout outputs → decoded MAE:
python -m clin_jepa.evaluation.apply_probes \
    --config configs/eval/probes.yaml --plan clin_jepa
```

### Track 2 — 8 stay-level binary outcomes

Same flow, but use the ``_binary`` variants of the baseline functions
(``train_ridge_binary``, ``train_lightgbm_binary``, ``train_dl_binary``)
and switch ``cohort_mode`` from ``rolling`` to ``admission`` when
constructing the baseline datasets.

### Bootstrapped confidence intervals

All probe metrics are reported with 95% bootstrap CIs (`n_boot=500`,
clustered by `stay_id`). Settings live under `metrics:` in
`configs/eval/probes.yaml`. The bootstrap is performed inside
:mod:`~clin_jepa.evaluation.apply_probes` and the baseline metric utilities
under :mod:`~clin_jepa.evaluation.baselines.metrics`.

## Reproducing the paper tables

The per-task per-paradigm AUROC and AUPRC tables in Appendix C are produced
by the full pipeline above, repeated for each of the three paradigms
(``clin_jepa``, ``vjepa2ac``, ``sft_baseline``); together they cover every
number the paper reports. Aggregation into the LaTeX tables themselves is
left as a post-processing exercise.
