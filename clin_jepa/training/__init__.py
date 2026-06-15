"""Training scripts and shared infrastructure for Clin-JEPA.

Top-level scripts (intended to be invoked via ``python -m``):

* :mod:`~clin_jepa.training.encoder_sft` — supervised next-token-prediction
  fine-tuning that initialises the encoder LoRA on per-hour clinical texts.
* :mod:`~clin_jepa.training.pretrain_clin_jepa` — the five-phase Clin-JEPA
  joint co-training of encoder LoRA and AC Transformer predictor.
* :mod:`~clin_jepa.training.refine_encoder_vjepa` — V-JEPA 2-AC baseline,
  random-mask JEPA encoder-refinement stage.
* :mod:`~clin_jepa.training.train_predictor_on_frozen` — V-JEPA 2-AC
  baseline (predictor stage) and SFT baseline; trains the AC Transformer
  predictor on cached embeddings from a frozen encoder.
* :mod:`~clin_jepa.training.ablation_no_warmup` — Clin-JEPA ablation that
  removes Phase 1 warmup.
* :mod:`~clin_jepa.training.ablation_no_alignment` — Clin-JEPA ablation
  that removes Phase 3 alignment + Phase 4 hard sync.

Shared infrastructure used by the above:

* :mod:`~clin_jepa.training.encoder_ops` — encoder LoRA loading, PEFT
  multi-adapter EMA target, ``encode_texts_batched`` forward.
* :mod:`~clin_jepa.training.trajectory_dataset` — yields trajectory windows
  with per-hour state/action/demographics texts (used by all
  encoder-touching scripts).
* :mod:`~clin_jepa.training.predictor_dataset` — yields trajectory windows
  with cached encoder embeddings (used by
  :mod:`~clin_jepa.training.train_predictor_on_frozen`).
* :mod:`~clin_jepa.training.sampler` — ``TokenBudgetBatchSampler`` for
  efficient variable-length training.
"""
