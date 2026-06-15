"""Token-budget batch sampler for variable-length training.

Groups samples by sequence length so short samples get larger batches and
long samples get smaller batches, all bounded by a target token budget.
"""

import logging
import random
from typing import Iterator

logger = logging.getLogger(__name__)


class TokenBudgetBatchSampler:
    """Creates variable-size batches bounded by a total token budget.

    Two modes:
    - padding mode (packing=False): budget = max_len_in_batch × batch_size
    - packing mode (packing=True): budget = sum of all sample lengths

    Args:
        lengths: Per-sample token lengths.
        max_tokens: Maximum total tokens per micro-batch.
        max_batch_size: Hard cap on batch size (for gradient memory limits).
        shuffle: Whether to shuffle batch order each epoch.
        seed: Random seed for reproducibility.
        drop_last: Drop the last incomplete batch.
        packing: If True, use sum-based budget (no padding waste).
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens: int = 32768,
        max_batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        packing: bool = False,
    ) -> None:
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.packing = packing
        self.rng = random.Random(seed)

        self._batches = self._create_batches()
        self._log_stats()

    def _create_batches(self) -> list[list[int]]:
        """Create batches by sorting by length and bin-packing."""
        # Sort by length, longest first
        sorted_indices = sorted(
            range(len(self.lengths)),
            key=lambda i: self.lengths[i],
            reverse=True,
        )

        batches = []
        current_batch: list[int] = []
        current_max_len = 0
        current_sum_len = 0

        for idx in sorted_indices:
            sample_len = self.lengths[idx]

            if self.packing:
                # Packing: budget = sum of lengths
                new_total_tokens = current_sum_len + sample_len
            else:
                # Padding mode: budget = max_len × batch_size
                new_max_len = max(current_max_len, sample_len)
                new_total_tokens = (len(current_batch) + 1) * new_max_len

            if (new_total_tokens > self.max_tokens
                    or len(current_batch) >= self.max_batch_size):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len
                current_sum_len = sample_len
            else:
                current_batch.append(idx)
                current_max_len = max(current_max_len, sample_len)
                current_sum_len += sample_len

        # Handle last batch
        if current_batch:
            if not self.drop_last or len(current_batch) >= 1:
                batches.append(current_batch)

        return batches

    def _log_stats(self) -> None:
        """Log batch statistics."""
        batch_sizes = [len(b) for b in self._batches]
        if not batch_sizes:
            return

        avg_bs = sum(batch_sizes) / len(batch_sizes)
        min_bs = min(batch_sizes)
        max_bs = max(batch_sizes)

        # Compute effective tokens per batch (padded)
        token_counts = []
        for batch in self._batches:
            max_len = max(self.lengths[i] for i in batch)
            token_counts.append(len(batch) * max_len)
        avg_tokens = sum(token_counts) / len(token_counts)

        logger.info(
            "TokenBudgetBatchSampler: %d batches, batch_size min=%d avg=%.1f max=%d, "
            "avg_tokens_per_batch=%.0f (budget=%d)",
            len(self._batches), min_bs, avg_bs, max_bs,
            avg_tokens, self.max_tokens,
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batch index lists, optionally shuffled."""
        batches = self._batches.copy()
        if self.shuffle:
            self.rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        """Reset RNG seed for deterministic shuffling across epochs."""
        self.rng = random.Random(42 + epoch)
