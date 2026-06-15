"""Disposable bidirectional JEPA predictor for the V-JEPA 2-AC baseline.

A small (~16.9M-param) cross-attention Transformer decoder used during the
random-mask JEPA encoder-refinement stage. It is discarded after refinement;
only the refined encoder LoRA is kept for downstream training of the
:class:`~clin_jepa.model.predictor.ACTransformerPredictor`.

Architecture:

* Input: encoder embeddings of *visible* positions form the memory; position
  ids of *masked* positions form the queries.
* ``4096 -> 512`` input projection (bottleneck).
* Learnable position embeddings shared between memory and queries so that a
  position id always means the same thing to the predictor.
* 3-layer ``nn.TransformerDecoder`` with cross-attention (no causal mask).
* ``512 -> 4096`` output projection.

This implements the V-JEPA 2-AC Stage 1 predictor (Assran et al., 2025),
adapted to text inputs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """Disposable bidirectional cross-attention JEPA predictor.

    Args:
        embed_dim: Encoder embedding dimension (4096 for Qwen3-8B).
        hidden_dim: Internal transformer hidden dimension (512 — bottleneck).
        num_layers: Number of cross-attention decoder layers (3).
        num_heads: Number of attention heads per layer (8).
        ffn_dim: Feed-forward intermediate dimension (2048 = 4 x hidden).
        dropout: Dropout rate (0.1).
        max_position_id: Largest position id we expect. The interleaved
            ``[d, s_1, a_1, ...]`` sequence at the default ``max_timesteps=72``
            reaches position ``2 * 72 + 1 = 145``; the default of ``256`` is a
            safety buffer.
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_position_id: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_position_id = max_position_id

        # Input projection: encoder dim -> internal hidden (bottleneck)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Learnable position embeddings, shared between memory and queries.
        self.pos_embed = nn.Embedding(max_position_id, hidden_dim)

        # Cross-attention decoder layers (no causal mask)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection: hidden -> encoder dim
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        memory_embeddings: torch.Tensor,
        memory_positions: torch.Tensor,
        query_positions: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attention forward pass.

        Args:
            memory_embeddings: ``(B, V, embed_dim)`` encoder embeddings of
                visible positions. ``V`` may vary by sample; pad and pass
                ``memory_key_padding_mask`` when batching variable ``V``.
            memory_positions: ``(B, V)`` int64 position ids for memory tokens
                (used to add positional encoding to memory).
            query_positions: ``(B, Q)`` int64 position ids for masked
                positions to predict.
            memory_key_padding_mask: Optional ``(B, V)`` boolean mask, where
                ``True`` marks padded memory positions to ignore.

        Returns:
            ``(B, Q, embed_dim)`` predicted encoder embeddings at the query
            positions.
        """
        memory = self.input_proj(memory_embeddings.float())
        memory = memory + self.pos_embed(memory_positions)

        queries = self.pos_embed(query_positions)

        # Cross-attention decoder, no causal mask.
        out = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.output_proj(out)

    def num_params(self) -> int:
        """Total parameter count (~16.9M at default hyperparameters)."""
        return sum(p.numel() for p in self.parameters())
