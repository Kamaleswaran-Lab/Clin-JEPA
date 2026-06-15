# Training

The Clin-JEPA framework involves several training scripts. The headline
result (paper §4-§5) comes from this sequence:

1. **Encoder SFT** — next-token-prediction fine-tuning to initialise the
   encoder LoRA.
2. **Clin-JEPA pretraining** — five-phase joint co-training of the encoder
   LoRA and the AC Transformer predictor.

Two baselines and two ablations are also reproducible:

* **V-JEPA 2-AC baseline** = encoder SFT → JEPA refinement → train predictor
  on frozen encoder.
* **SFT baseline** = encoder SFT → train predictor on frozen encoder
  (no JEPA refinement).
* **Clin-JEPA w/o warmup** = Clin-JEPA without Phase 1 (warmup).
* **Clin-JEPA w/o alignment** = Clin-JEPA without Phase 3 (alignment) +
  Phase 4 (hard sync).

All training scripts accept a single `--config` argument. The canonical
config for each script lives under `configs/train/`.

## Hardware + dependencies

* 4 or 8x NVIDIA H100 / H200 (the Qwen3-8B encoder needs ~80 GB VRAM per
  GPU during joint co-training; the predictor-only scripts are much
  cheaper).
* CUDA 12.x.
* The Liger-Kernel fused linear+CE kernel is required for any script that
  trains the encoder — without it the Qwen3-8B 152K-vocab logits will OOM
  even on an H200. `pip install -r requirements.txt` covers this.
* Flash-Attention 2 (`pip install flash-attn==2.8.3 --no-build-isolation`,
  on a GPU node with the CUDA toolkit available).

## 1. Encoder SFT

```bash
torchrun --nproc_per_node=4 -m clin_jepa.training.encoder_sft \
    --config configs/train/encoder_sft.yaml
```

Outputs the SFT-initialised encoder LoRA to
`$CLIN_JEPA_OUTPUT/encoder_sft/{best,final,latest}/`. Wall-clock: ~10 hours
on 4x H200.

## 2. Clin-JEPA pretraining (headline)

```bash
torchrun --nproc_per_node=4 -m clin_jepa.training.pretrain_clin_jepa \
    --config configs/train/pretrain_clin_jepa.yaml
```

Runs the five-phase curriculum end-to-end in a single job. Outputs both the
co-trained encoder LoRA and the 92M AC Transformer predictor to
`$CLIN_JEPA_OUTPUT/clin_jepa_pretrain/checkpoints/{best,final}/`. Wall-clock:
~54 GPU-hours total (~14 h on 4x H200, ~7 h on 8x H200).

Phase mapping (paper terminology vs YAML keys):

| Paper          | YAML key              | Description                                   |
|----------------|-----------------------|-----------------------------------------------|
| Phase 1 warmup | `phase_1_steps`       | Encoder frozen, predictor warmup              |
| Phase 2 co-train | `phase_2a_steps`    | Encoder + predictor jointly trained           |
| Phase 3 align  | `phase_2b_steps`      | Encoder re-frozen, EMA target catches up      |
| Phase 4 sync   | (instantaneous)       | `target.load_state_dict(online.state_dict())` |
| Phase 5 final  | `phase_4_steps`       | Predictor trained under native AR rollout     |

The YAML keys use a finer sub-phase split (`phase_2a`/`phase_2b`); the paper
groups these into the cleaner Phase 1-5 enumeration.

## 3. Baselines

### V-JEPA 2-AC

Two sequential scripts (run on the same SFT-initialised encoder as above):

```bash
# Step 1: random-mask JEPA encoder refinement (~3 h on 4x H200).
torchrun --nproc_per_node=4 -m clin_jepa.training.refine_encoder_vjepa \
    --config configs/train/refine_encoder_vjepa.yaml

# Step 2: precompute embeddings with the refined encoder.
python -m clin_jepa.evaluation.precompute_embeddings \
    --config configs/eval/precompute_embeddings.yaml \
    --plan vjepa2ac

# Step 3: train AC predictor on cached embeddings (~1 h on 1 H200).
python -m clin_jepa.training.train_predictor_on_frozen \
    --config configs/train/predictor_vjepa2ac.yaml
```

### SFT baseline

Same as the V-JEPA 2-AC predictor stage, but precompute embeddings directly
from the SFT-initialised encoder (no JEPA refinement):

```bash
python -m clin_jepa.evaluation.precompute_embeddings \
    --config configs/eval/precompute_embeddings.yaml \
    --plan sft_baseline

python -m clin_jepa.training.train_predictor_on_frozen \
    --config configs/train/predictor_sft_baseline.yaml
```

## 4. Ablations

```bash
torchrun --nproc_per_node=4 -m clin_jepa.training.ablation_no_warmup \
    --config configs/train/ablation_no_warmup.yaml

torchrun --nproc_per_node=4 -m clin_jepa.training.ablation_no_alignment \
    --config configs/train/ablation_no_alignment.yaml
```

## SLURM

A generic SLURM template lives at `slurm-examples/train.slurm.template`.
Edit the placeholders for your cluster's account, partition, and
walltime defaults.
