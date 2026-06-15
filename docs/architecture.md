# Architecture

Clin-JEPA has two deployed model components plus one training-only target
network. See paper §3 (model) and §4 (training) for the full design
rationale; this document is a fast reference for readers of the code.

## Components

### Encoder

A Qwen3-8B language model with lightweight LoRA adapters (r=16, alpha=32,
on `q_proj`, `k_proj`, `v_proj`, `o_proj` — ~30.7M trainable parameters
out of 8.2B). The base weights are frozen throughout. Each input text
(per-hour state, per-hour action, or per-stay demographics) is mapped to
a 4096-dim embedding by taking the last-token hidden state.

The encoder lives in :mod:`clin_jepa.training.encoder_ops` (the LoRA setup
and forward-pass utilities are bundled with the training infrastructure
because the encoder is constructed and used exclusively inside training
and embedding-precompute scripts).

### AC Transformer predictor

A 6-layer pre-norm Transformer encoder (~92M params; 1024 hidden,
8 heads, FFN 4096, dropout 0.15/0.10) that operates entirely in the
encoder's 4096-dim latent space. It consumes the interleaved sequence
`[d, s_1, u_1, s_2, u_2, ..., s_T, u_T]` of length `2T+1`, applies
block-causal attention, and outputs the next-state prediction
$\hat z_{t+1}$ from each state position.

At inference, the predictor is retained and rolls out autoregressively:
each predicted $\hat z_{t+h}$ is fed back as the state input for step
`t+h+1`. See :class:`clin_jepa.model.predictor.ACTransformerPredictor`.

### EMA target encoder (training-only)

A parameter copy of the encoder LoRA held in fp32 in a second PEFT
adapter. Updated as an exponential moving average of the online encoder
with `tau=0.996`. Serves only as a stop-gradient anchor for the JEPA L1
loss; carries no projection head; is discarded at inference.

See :func:`clin_jepa.training.encoder_ops.create_target_encoder_fp32`.

## Training-only auxiliary networks

### Disposable JEPA predictor (V-JEPA 2-AC baseline only)

A ~16.9M-parameter bidirectional cross-attention decoder used only during
the encoder-refinement stage of the V-JEPA 2-AC baseline. Predicts
encoder embeddings at random-masked positions, then is discarded.

See :class:`clin_jepa.model.jepa_predictor.JEPAPredictor`.

## Data flow at a glance

```
                       Trajectory shards
                                |
                                v
                         Encoder SFT  -- (Qwen3-8B + LoRA)
                                |
                                v
                  Clin-JEPA pretraining (five phases)
                                |
                                v
              Encoder LoRA  +  AC Transformer predictor
                          \   /
                           v v
                       Autoregressive
                          rollout
                                |
                                v
                  Downstream MLP probes
```

For the V-JEPA 2-AC and SFT baselines the encoder is frozen after the
JEPA refinement / SFT stage respectively, and the AC predictor is trained
on cached embeddings. The probe stage is identical across paradigms.
