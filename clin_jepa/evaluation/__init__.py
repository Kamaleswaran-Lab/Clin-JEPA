"""Evaluation pipelines for Clin-JEPA.

Stages: precompute_embeddings (cache encoder embeddings) → rollout
(autoregressive latent-rollout drift evaluation, paper §5.2) →
build_alignment / collect_probe_data / probes (shallow MLP probes for the
multi-task downstream evaluation, paper §5.4) → apply_probes (per-step
decoded MAE in original clinical units). ``baselines`` packages the four
non-Clin-JEPA baselines (Ridge, LightGBM, LSTM, TCN).
"""
