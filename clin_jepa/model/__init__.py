"""Model classes for Clin-JEPA.

The Clin-JEPA framework comprises two deployed model components and one
training-only model:

* :class:`~clin_jepa.model.predictor.ACTransformerPredictor` — the 92M
  action-conditioned Transformer that operates in the encoder's 4096-dim
  latent space. Used at inference for autoregressive trajectory rollout.
* :class:`~clin_jepa.model.jepa_predictor.JEPAPredictor` — the disposable
  16.9M bidirectional cross-attention predictor used only during the V-JEPA
  2-AC baseline's random-mask encoder-refinement stage.

The encoder itself is a Qwen3-8B language model with LoRA adapters; it lives
in :mod:`clin_jepa.training.encoder_ops` because it is constructed and used
exclusively inside the training and embedding-precompute scripts.
"""

from clin_jepa.model.predictor import ACTransformerPredictor, build_block_causal_mask
from clin_jepa.model.jepa_predictor import JEPAPredictor

__all__ = [
    "ACTransformerPredictor",
    "JEPAPredictor",
    "build_block_causal_mask",
]
