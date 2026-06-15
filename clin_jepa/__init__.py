"""Clin-JEPA: Joint-Embedding Predictive Pretraining on EHR Patient Trajectories.

A multi-phase co-training framework that pretrains a Qwen3-8B-based encoder
together with a 92M action-conditioned Transformer predictor on intensive-care
patient trajectories, then uses the retained predictor at inference for
autoregressive latent-space rollout.
"""

__version__ = "0.1.0"
