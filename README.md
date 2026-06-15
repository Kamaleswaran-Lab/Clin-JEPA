# Clin-JEPA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reference implementation of **Clin-JEPA: A Multi-Phase Co-Training Framework
for Joint-Embedding Predictive Pretraining on EHR Patient Trajectories**.

> **Note:** This repository accompanies a paper currently under double-blind
> review. Author, citation, and funding information have been omitted from
> this version and will be added once review concludes.

Clin-JEPA trains a Qwen3-8B-based encoder together with a 92M
action-conditioned Transformer predictor on intensive-care patient
trajectories. At inference, the encoder maps free-text observations and
interventions to a latent state space, and the retained predictor rolls
out future patient states autoregressively in that space. A five-phase
pretraining curriculum addresses the two characteristic failure modes of
naïve encoder + predictor co-training (representation collapse and
online/target space drift).

## Installation

```bash
git clone <repository-url>
cd Clin-JEPA

# Create a Python 3.10+ environment, then:
pip install -r requirements.txt
pip install -e .

# Flash-Attention 2 must be installed separately on a GPU node with the
# CUDA toolkit available — the pre-built PyPI wheels are tied to specific
# torch + CUDA combinations and frequently fail to match local setups, so
# we don't list it in requirements.txt:
pip install flash-attn==2.8.3 --no-build-isolation
```

`requirements.txt` pins the exact dependency versions used for the paper
experiments. `pyproject.toml` is the modern Python package-metadata file
(PEP 621): it declares the project name (`clin-jepa`), the import name
(`clin_jepa`), looser dependency bounds for forward-compatibility, and
the build backend that enables `pip install -e .`.

## Configure paths

The pipeline reads three input/output roots and one HuggingFace cache
path from environment variables. `.env.example` is a template listing
all four; copy it and either source it from your shell or export the
variables manually:

```bash
cp .env.example .env
# Edit .env to point MIMIC_RAW, CLIN_JEPA_DATA, CLIN_JEPA_OUTPUT, HF_HOME
# at your local directories.
set -a; source .env; set +a   # export everything from .env into the shell
```

| Variable           | Used for                                                |
| ------------------ | ------------------------------------------------------- |
| `MIMIC_RAW`        | MIMIC-IV v3.1 from PhysioNet (`icu/`, `hosp/`, `note/`) plus the mimic-code-derived `concepts/`. |
| `CLIN_JEPA_DATA`   | Processed pipeline outputs (cohort, trajectories, embeddings, labels). |
| `CLIN_JEPA_OUTPUT` | Training outputs (checkpoints, logs, evaluation results). |
| `HF_HOME`          | HuggingFace cache (where the Qwen3-8B weights are stored). |

## Quick start

```bash
# 1. Build the trajectory shards from raw MIMIC-IV.
#    Full instructions in docs/data-preparation.md.
python -m clin_jepa.data.step01_cohort
python -m clin_jepa.data.step02_observations  --chunk_idx 0 --total_chunks 8  # repeat for chunk_idx 1..7
python -m clin_jepa.data.step03_actions       --chunk_idx 0 --total_chunks 8  # repeat for chunk_idx 1..7
python -m clin_jepa.data.step04_discretization --chunk_idx 0 --total_chunks 8  # repeat for chunk_idx 1..7
python -m clin_jepa.data.step05_split
python -m clin_jepa.data.step06_trajectories \
    --config configs/data/trajectories.yaml --n_workers 10

# 2. Initialise the encoder LoRA via supervised fine-tuning.
torchrun --nproc_per_node=4 -m clin_jepa.training.encoder_sft \
    --config configs/train/encoder_sft.yaml

# 3. Run the five-phase Clin-JEPA pretraining.
torchrun --nproc_per_node=4 -m clin_jepa.training.pretrain_clin_jepa \
    --config configs/train/pretrain_clin_jepa.yaml

# 4. Evaluate — paper §5.2 (rollout drift) and §5.4 (downstream multi-task
#    probes). §5.3 (UMAP latent-geometry) is a one-off plotting exercise,
#    not shipped here; see docs/evaluation.md.
python -m clin_jepa.evaluation.precompute_embeddings \
    --config configs/eval/precompute_embeddings.yaml --plan clin_jepa
python -m clin_jepa.evaluation.rollout \
    --config configs/eval/rollout.yaml --plan clin_jepa

# Downstream multi-task MLP probes (collect data, train, align, apply):
python -m clin_jepa.evaluation.collect_probe_data --plan clin_jepa
python -m clin_jepa.evaluation.probes            --plan clin_jepa
python -m clin_jepa.evaluation.build_alignment   --plan clin_jepa
python -m clin_jepa.evaluation.apply_probes      --plan clin_jepa
```

Full per-step instructions live in `docs/`:

* [`docs/data-preparation.md`](docs/data-preparation.md) — PhysioNet
  credentialing, MIMIC-IV download, mimic-code concepts setup, and the
  six-step preprocessing pipeline.
* [`docs/training.md`](docs/training.md) — Clin-JEPA pretraining, the two
  baselines (V-JEPA 2-AC and SFT), and the two ablations.
* [`docs/evaluation.md`](docs/evaluation.md) — drift evaluation (§5.2)
  and downstream multi-task probes (§5.4); the §5.3 UMAP latent-geometry
  diagnosis is a not-shipped plotting exercise, noted there for completeness.
* [`docs/architecture.md`](docs/architecture.md) — quick reference for
  the model components.

For SLURM-managed clusters, see `slurm-examples/train.slurm.template`
for a generic job template.

## Project structure

```
clin_jepa/
├── model/             ACTransformerPredictor + disposable JEPAPredictor
├── training/          Encoder SFT + Clin-JEPA pretraining + V-JEPA 2-AC and
│                      SFT baselines + 2 ablations + encoder LoRA / EMA target
│                      utilities + dataset classes
├── evaluation/        Embedding precompute + autoregressive rollout +
│                      MLP probes (continuous + binary outcomes) + alignment
│                      sidecar builder + paper baselines under baselines/
│                      (Ridge, LightGBM, LSTM, TCN)
└── data/              Six-step MIMIC-IV preprocessing pipeline + helpers

configs/
├── data/              Cohort, features, paths, trajectory configs
├── train/             Per-training-script configs (one YAML per script)
└── eval/              Rollout, probes, embedding-precompute configs

docs/                  Setup, training, evaluation, architecture
slurm-examples/        Generic SLURM template (cluster-agnostic placeholders)
```

## Data

Experiments use MIMIC-IV v3.1
([Johnson et al., 2023](https://physionet.org/content/mimiciv/3.1/)),
accessed via PhysioNet under credentialed access. The raw data is not
redistributed in this repository; see
[`docs/data-preparation.md`](docs/data-preparation.md) for download
instructions.

## Citation

Citation information has been omitted while this work is under double-blind
review, and will be added once review concludes.

## License

This code is released under the [MIT License](LICENSE). MIMIC-IV data is
governed separately by the PhysioNet Credentialed Health Data License.

## Acknowledgments

We thank the MIMIC-IV team at the MIT Laboratory for Computational Physiology
and Beth Israel Deaconess Medical Center for making the dataset publicly
available. Funding acknowledgments have been omitted for anonymity and will be
added once review concludes.
