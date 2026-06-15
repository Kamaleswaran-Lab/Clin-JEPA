"""Trajectory-window dataset.

Yields full trajectory windows with per-hour state and action texts, a
per-stay demographics text, per-hour clinical labels, and per-stay binary
outcomes. Used by the JEPA encoder-refinement step (V-JEPA 2-AC baseline)
and by the Clin-JEPA joint co-training pretraining.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _build_demo_text(age: int, gender: str, race: str) -> str:
    """Build a demographics text string from cohort fields.

    Produces the same demographics text as ``build_demographics_text`` in
    :mod:`clin_jepa.evaluation.precompute_embeddings`. The race string is kept
    as-is (uppercase cohort form).

    Examples:
        >>> _build_demo_text(53, "M", "WHITE")
        'Age: 53 years. Gender: Male. Race: WHITE.'
        >>> _build_demo_text(71, "F", "BLACK/AFRICAN AMERICAN")
        'Age: 71 years. Gender: Female. Race: BLACK/AFRICAN AMERICAN.'

    Args:
        age: Patient age in years (already cast to int).
        gender: Cohort raw gender code ("M" or "F").
        race: Cohort raw race string (uppercase).

    Returns:
        Single-line demographics text used as the ``d`` token in the predictor.
    """
    gender_text = "Male" if gender == "M" else "Female"
    return f"Age: {age} years. Gender: {gender_text}. Race: {race}."


class TrajectoryWindowDataset(Dataset):
    """Yields trajectory windows with raw text + demographics + labels.

    Each item is one trajectory window referring to a ``.pt`` shard produced
    by the data pipeline (:mod:`clin_jepa.data.step06_trajectories`).

    Args:
        shard_dir: Path to the parent dir holding ``train/``, ``val/``, and
            ``test/`` subdirs of ``.pt`` shards. The split subdir is appended
            internally.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        max_windows: If set, stop loading after this many windows
            (useful for smoke tests).
        min_length: Skip windows shorter than this. The trajectory-extraction
            step already enforces a minimum length, so this is a safety net.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        split: str = "train",
        max_windows: Optional[int] = None,
        min_length: int = 6,
    ) -> None:
        shard_dir = Path(shard_dir) / split
        shard_files = sorted(shard_dir.glob("*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No .pt shards found in {shard_dir}")

        self.shards: list[dict] = []
        self.window_refs: list[dict] = []

        for path in shard_files:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            shard_idx = len(self.shards)
            self.shards.append(shard)

            for w in shard["windows"]:
                length = int(w["length"])
                if length < min_length:
                    continue
                self.window_refs.append({
                    "_shard_idx":  shard_idx,
                    "stay_index":  int(w["stay_index"]),
                    "start_step":  int(w["start_step"]),
                    "length":      length,
                    "hour_indices": w["hour_indices"],  # np.int32 array, kept as-is
                })

                if max_windows and len(self.window_refs) >= max_windows:
                    break

            if max_windows and len(self.window_refs) >= max_windows:
                break

        logger.info(
            "TrajectoryWindowDataset: %d windows from %d shards (%s)",
            len(self.window_refs), len(self.shards), shard_dir,
        )

    def __len__(self) -> int:
        return len(self.window_refs)

    def __getitem__(self, idx: int) -> dict:
        """Lazy gather: pull texts + demographics + labels on demand.

        Returns:
            Dict with keys:
                stay_id (int)
                start_step (int)
                length (int) — number of hours in this window
                state_texts (list[str], len=length) — one state text per hour
                action_texts (list[str], len=length) — one action text per hour
                demographics_text (str) — single-line demographics descriptor
                labels (dict[str, Tensor(length,) float32]) — clinical-target labels
                binary_labels (dict[str, bool]) — stay-level binary outcomes
        """
        w     = self.window_refs[idx]
        shard = self.shards[w["_shard_idx"]]
        ph    = shard["per_hour"]
        phl   = shard["per_hour_labels"]
        ps    = shard["per_stay"]

        hi = w["hour_indices"].tolist()  # length L
        si = w["stay_index"]

        # Gather per-hour text lists
        state_texts  = [ph["state_texts"][h]  for h in hi]
        action_texts = [ph["action_texts"][h] for h in hi]

        labels = {
            k: torch.from_numpy(phl[k][hi].copy())
            for k in phl
        }

        # Demographics text (same format as precompute_embeddings.build_demographics_text)
        demographics_text = _build_demo_text(
            age=int(round(float(ps["ages"][si]))),
            gender=ps["genders"][si],
            race=ps["races"][si],
        )

        # Stay-level binary outcomes
        binary_labels = {
            "icu_mortality":      bool(ps["icu_mortalities"][si]),
            "hospital_mortality": bool(ps["hospital_mortalities"][si]),
            "prolonged_stay":     bool(ps["prolonged_stays"][si]),
            "sepsis3":            bool(ps["sepsis3s"][si]),
        }

        return {
            "stay_id":           int(ps["stay_ids"][si]),
            "start_step":        w["start_step"],
            "length":            w["length"],
            "state_texts":       state_texts,
            "action_texts":      action_texts,
            "demographics_text": demographics_text,
            "labels":            labels,
            "binary_labels":     binary_labels,
        }


def trajectory_collate_fn(batch: list[dict]) -> dict:
    """Collate trajectory windows: keep texts as lists, stack lengths.

    Returns:
        Dict with:
            batch (list[dict]): the original list of window dicts
            lengths (LongTensor, B): trajectory length per window
            max_length (int): max length in batch
    """
    lengths = torch.tensor([w["length"] for w in batch], dtype=torch.long)
    return {
        "batch":      batch,
        "lengths":    lengths,
        "max_length": int(lengths.max().item()),
        "batch_size": len(batch),
    }
