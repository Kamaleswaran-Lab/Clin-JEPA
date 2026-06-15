"""Autoregressive rollout engine — paper §5.2 drift evaluation.

Performs full autoregressive rollout at every future step for each context
length ``C ∈ {6, 12, 24, 48}``, saving predicted embeddings, L1 distances
against the encoder's own forward outputs, and per-step labels. Consumes the
two-table embedding schema from
:mod:`clin_jepa.evaluation.precompute_embeddings`; outputs feed the
downstream probes (paper §5.4).
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "label_sofa_total", "label_sofa_resp", "label_sofa_cardio",
    "label_sofa_renal", "label_sofa_hepatic", "label_sofa_coag",
    "label_sofa_neuro", "label_hr", "label_map", "label_sbp", "label_rr",
]

BINARY_LABEL_COLUMNS = [
    "icu_mortality", "hospital_mortality", "prolonged_stay", "sepsis3",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_predictor(
    checkpoint_path: str | Path,
    device: torch.device,
    model_config: Optional[dict] = None,
) -> ACTransformerPredictor:
    """Load a trained AC Transformer predictor from a checkpoint.

    Supports two formats: a wrapped dict (``{"model_state_dict", "config",
    ...}``, architecture read from the checkpoint) and a bare ``OrderedDict``
    (flat state_dict; architecture must be supplied via ``model_config``).
    Returns the model in eval mode with all parameters frozen.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info("Loading predictor from %s", checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_wrapped = isinstance(ckpt, dict) and "model_state_dict" in ckpt

    if is_wrapped:
        state_dict = ckpt["model_state_dict"]
        ckpt_config = ckpt.get("config", {}).get("model")
        if ckpt_config is not None and model_config is not None:
            for k in (
                "state_dim", "action_dim", "hidden_dim", "num_layers",
                "num_heads", "ffn_dim", "max_timesteps", "prediction_mode",
            ):
                if k in ckpt_config and k in model_config and ckpt_config[k] != model_config[k]:
                    logger.warning(
                        "Config %s mismatch: checkpoint=%s vs user_config=%s. "
                        "Using checkpoint value.",
                        k, ckpt_config[k], model_config[k],
                    )
        arch = ckpt_config or model_config
        val_loss = ckpt.get("val_loss", "?")
        epoch = ckpt.get("epoch", "?")
        logger.info(
            "Detected wrapped checkpoint, val_loss=%s, epoch=%s",
            val_loss, epoch,
        )
    else:
        if model_config is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is a bare state_dict but no "
                "model_config was provided. Set ``model_config`` for this "
                "plan in the rollout YAML."
            )
        state_dict = ckpt
        arch = model_config
        logger.info("Detected bare OrderedDict checkpoint")

    if arch is None:
        raise ValueError(
            f"Could not determine architecture from checkpoint or config "
            f"for {checkpoint_path}"
        )

    model = ACTransformerPredictor(
        state_dim=arch["state_dim"],
        action_dim=arch["action_dim"],
        static_dim=arch.get("static_dim", arch["state_dim"]),
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        ffn_dim=arch["ffn_dim"],
        max_timesteps=arch["max_timesteps"],
        ffn_dropout=0.0,
        attn_dropout=0.0,
        prediction_mode=arch.get("prediction_mode", "absolute"),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded predictor: %s mode, %.1fM params (device=%s)",
        arch.get("prediction_mode", "absolute"), n_params / 1e6, device,
    )
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(
    embedding_dir: str | Path,
    split: str = "test",
    max_windows: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Load all embedding shards for one paradigm and one split.

    Returns ``(shards, windows)`` where ``windows`` is a flat list of window
    dicts each annotated with ``_shard_idx``.
    """
    split_dir = Path(embedding_dir) / split
    shard_files = sorted(split_dir.glob("embeddings_c*.pt"))
    if not shard_files:
        # Fall back to any .pt file
        shard_files = sorted(split_dir.glob("*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No embedding shards in {split_dir}")

    logger.info("Loading %d shards from %s", len(shard_files), split_dir)
    shards: list[dict] = []
    windows: list[dict] = []

    required_keys = {
        "z_state_embeddings", "z_action_embeddings", "z_statics", "windows",
    }

    for f in shard_files:
        shard = torch.load(f, map_location="cpu", weights_only=False)

        missing = required_keys - set(shard.keys())
        if missing:
            raise ValueError(f"Shard {f} missing keys: {missing}")

        shard_idx = len(shards)
        shards.append(shard)

        for w in shard["windows"]:
            w_copy = dict(w)
            w_copy["_shard_idx"] = shard_idx
            windows.append(w_copy)
            if max_windows and len(windows) >= max_windows:
                break
        if max_windows and len(windows) >= max_windows:
            break

    logger.info(
        "Loaded %d windows from %d shards (%s split)",
        len(windows), len(shards), split,
    )
    return shards, windows


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def create_batches(
    windows: list[dict],
    context_length: int,
    batch_size: int,
) -> tuple[list[list[int]], int]:
    """Bucket windows by length for efficient rollout.

    Filters windows with ``length <= context_length``; sorts survivors by
    descending length to minimize padding.
    """
    valid_indices = [
        i for i, w in enumerate(windows) if w["length"] > context_length
    ]
    n_skipped = len(windows) - len(valid_indices)
    if n_skipped > 0:
        logger.info("Skipped %d windows with length <= %d", n_skipped, context_length)

    valid_indices.sort(key=lambda i: -windows[i]["length"])

    batches = []
    for start in range(0, len(valid_indices), batch_size):
        batches.append(valid_indices[start : start + batch_size])

    logger.info(
        "Created %d batches from %d valid windows (batch_size=%d)",
        len(batches), len(valid_indices), batch_size,
    )
    return batches, n_skipped


def gather_batch_data(
    windows: list[dict],
    shards: list[dict],
    batch_indices: list[int],
    device: torch.device,
) -> dict:
    """Gather a batch of windows and pad for rollout (two-table schema).

    States and actions are both 4096-D encoder outputs gathered from the
    deduplicated ``z_state_embeddings`` and ``z_action_embeddings`` tables.

    Returns:
        Dict with ``z_states (B, T_max, 4096) fp16``,
        ``z_static (B, 4096) fp16``, ``actions (B, T_max, 4096) fp16``,
        ``lengths (B,)``, ``stay_ids (list)``, ``labels`` (NaN-padded
        per-step), ``binary_labels`` (per-window).
    """
    B = len(batch_indices)
    T_max = max(windows[i]["length"] for i in batch_indices)
    D = 4096

    z_states = torch.zeros(B, T_max, D, dtype=torch.float16)
    z_static = torch.zeros(B, D, dtype=torch.float16)
    actions = torch.zeros(B, T_max, D, dtype=torch.float16)
    lengths = torch.zeros(B, dtype=torch.long)
    stay_ids: list[int] = []

    labels = {col: torch.full((B, T_max), float("nan"), dtype=torch.float32)
              for col in LABEL_COLUMNS}
    binary_labels = {col: torch.zeros(B, dtype=torch.bool)
                     for col in BINARY_LABEL_COLUMNS}

    for b, idx in enumerate(batch_indices):
        w = windows[idx]
        shard = shards[w["_shard_idx"]]
        L = w["length"]

        z_states[b, :L] = shard["z_state_embeddings"][w["emb_indices_state"]]
        z_static[b] = shard["z_statics"][w["static_idx"]]
        actions[b, :L] = shard["z_action_embeddings"][w["emb_indices_action"]]

        lengths[b] = L
        stay_ids.append(w["stay_id"])

        for col in LABEL_COLUMNS:
            lbl = w["labels"][col]
            lbl_t = (torch.as_tensor(lbl, dtype=torch.float32)
                     if not isinstance(lbl, torch.Tensor) else lbl.float())
            labels[col][b, :L] = lbl_t

        for col in BINARY_LABEL_COLUMNS:
            binary_labels[col][b] = bool(w["binary_labels"][col])

    return {
        "z_states": z_states.to(device),
        "z_static": z_static.to(device),
        "actions": actions.to(device),
        "lengths": lengths.to(device),
        "stay_ids": stay_ids,
        "labels": {k: v for k, v in labels.items()},
        "binary_labels": binary_labels,
    }


# ---------------------------------------------------------------------------
# Autoregressive rollout (core algorithm)
# ---------------------------------------------------------------------------

@torch.no_grad()
def autoregressive_rollout(
    model: ACTransformerPredictor,
    batch: dict,
    context_length: int,
    use_bf16: bool = True,
) -> dict:
    """Roll out the predictor autoregressively across the batch.

    ``k = context_length - 1`` is the last context index; at step ``s`` feed
    action ``k+s``, predict state ``k+s+1`` with horizon ``s+1``.

    Returns:
        Dict with ``per_step_l1 (B, max_rollout) fp32`` (NaN for inactive
        steps), ``predicted_z (B, max_rollout, 4096) fp16``,
        ``valid_mask (B, max_rollout) bool``,
        ``rollout_lengths (B,)`` int64.
    """
    z_states = batch["z_states"]
    z_static = batch["z_static"]
    actions = batch["actions"]
    lengths = batch["lengths"]
    B = z_states.shape[0]
    D = z_states.shape[2]

    k = context_length - 1

    rollout_lengths = (lengths - context_length).clamp(min=0)
    max_rollout = int(rollout_lengths.max().item())

    if max_rollout <= 0:
        return _empty_rollout_result(B)

    z_hist = z_states[:, :k, :].float()
    a_hist = actions[:, :k, :].clone()
    z_current = z_states[:, k, :].float()

    per_step_l1 = torch.full((B, max_rollout), float("nan"), device="cpu")
    predicted_z = torch.zeros(B, max_rollout, D, dtype=torch.float16, device="cpu")
    valid_mask = torch.zeros(B, max_rollout, dtype=torch.bool, device="cpu")

    for step in range(max_rollout):
        active = (step < rollout_lengths)
        if not active.any():
            break

        a_next = actions[:, k + step, :]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            z_pred = model.forward_rollout_step(
                z_hist, z_static.float(), a_hist, a_next, z_current,
            )
        z_pred = z_pred.float()

        target_idx = k + step + 1
        z_real = z_states[:, target_idx, :].float()

        l1 = (z_pred - z_real).abs().sum(dim=-1)
        active_cpu = active.cpu()
        per_step_l1[active_cpu, step] = l1[active].cpu()

        predicted_z[active_cpu, step] = z_pred[active].cpu().half()
        valid_mask[active_cpu, step] = True

        z_hist = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        a_hist = torch.cat([a_hist, a_next.unsqueeze(1)], dim=1)
        z_current = z_pred

    return {
        "per_step_l1": per_step_l1,
        "predicted_z": predicted_z,
        "valid_mask": valid_mask,
        "rollout_lengths": rollout_lengths.cpu(),
    }


def _empty_rollout_result(B: int) -> dict:
    return {
        "per_step_l1": torch.full((B, 0), float("nan")),
        "predicted_z": torch.zeros(B, 0, 4096, dtype=torch.float16),
        "valid_mask": torch.zeros(B, 0, dtype=torch.bool),
        "rollout_lengths": torch.zeros(B, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def compute_baselines(
    batch: dict,
    context_length: int,
    max_rollout: int,
) -> dict:
    """Compute the two latent-space baselines used in §5.2.

    * Copy-forward: ``ẑ_{C+h} = z_C``.
    * Linear extrapolation: ``ẑ_{C+h} = z_C + h * (z_C - z_{C-1})``.
    """
    z_states = batch["z_states"].float()
    lengths = batch["lengths"]
    k = context_length - 1
    B = z_states.shape[0]

    if context_length < 2:
        raise ValueError(
            f"context_length must be >= 2 for the linear-extrapolation "
            f"baseline, got {context_length}"
        )

    z_C = z_states[:, k, :]
    z_C_prev = z_states[:, k - 1, :]
    delta = z_C - z_C_prev

    copy_forward_l1 = torch.full((B, max_rollout), float("nan"), device="cpu")
    linear_extrap_l1 = torch.full((B, max_rollout), float("nan"), device="cpu")

    for step in range(max_rollout):
        h = step + 1
        target_idx = k + h
        valid = (target_idx < lengths)
        if not valid.any():
            break

        safe_idx = min(target_idx, z_states.shape[1] - 1)
        z_real = z_states[:, safe_idx, :]

        cf_l1 = (z_C - z_real).abs().sum(dim=-1)
        copy_forward_l1[valid.cpu(), step] = cf_l1[valid].cpu()

        z_extrap = z_C + h * delta
        le_l1 = (z_extrap - z_real).abs().sum(dim=-1)
        linear_extrap_l1[valid.cpu(), step] = le_l1[valid].cpu()

    return {
        "copy_forward_l1": copy_forward_l1,
        "linear_extrap_l1": linear_extrap_l1,
    }


# ---------------------------------------------------------------------------
# Result aggregation and saving
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    predictor_name: str,
    all_results: list[dict],
    all_baselines: list[dict],
    all_stay_ids: list[list[int]],
    all_lengths: list[torch.Tensor],
    all_labels: list[dict],
    all_binary_labels: list[dict],
    context_length: int,
    elapsed: float,
) -> Path:
    """Aggregate per-batch rollout results and write them to disk.

    Writes one ``predictions_<predictor_name>.pt`` (the full payload) and
    one ``metadata_<predictor_name>.json`` (summary metrics at every step)
    into ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_rollout_global = max(r["per_step_l1"].shape[1] for r in all_results)

    def _pad_2d(t: torch.Tensor, fill: float = float("nan")) -> torch.Tensor:
        B, M = t.shape
        if M >= max_rollout_global:
            return t
        pad = torch.full((B, max_rollout_global - M), fill, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    def _pad_3d(t: torch.Tensor) -> torch.Tensor:
        B, M, D = t.shape
        if M >= max_rollout_global:
            return t
        pad = torch.zeros(B, max_rollout_global - M, D, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    per_step_l1 = torch.cat(
        [_pad_2d(r["per_step_l1"]) for r in all_results], dim=0)

    predicted_z = torch.cat(
        [_pad_3d(r["predicted_z"]) for r in all_results], dim=0)
    for r in all_results:
        r.pop("predicted_z", None)
    gc.collect()

    valid_mask = torch.cat(
        [_pad_2d(r["valid_mask"].float(), fill=0.0).bool() for r in all_results], dim=0)
    for r in all_results:
        r.pop("valid_mask", None)
    gc.collect()

    copy_forward_l1 = torch.cat(
        [_pad_2d(b["copy_forward_l1"]) for b in all_baselines], dim=0)
    linear_extrap_l1 = torch.cat(
        [_pad_2d(b["linear_extrap_l1"]) for b in all_baselines], dim=0)
    for b in all_baselines:
        b.pop("copy_forward_l1", None)
        b.pop("linear_extrap_l1", None)
    gc.collect()

    stay_ids = np.array(
        [sid for batch_sids in all_stay_ids for sid in batch_sids], dtype=np.int64)
    lengths_arr = torch.cat(all_lengths, dim=0).numpy().astype(np.int32)

    N = per_step_l1.shape[0]

    k = context_length - 1
    labels_at_step: dict[str, torch.Tensor] = {}
    for col in LABEL_COLUMNS:
        col_all = torch.full((N, max_rollout_global), float("nan"), dtype=torch.float32)
        row_offset = 0
        for lbl_batch in all_labels:
            batch_vals = lbl_batch[col]
            B_batch = batch_vals.shape[0]
            T_batch = batch_vals.shape[1]
            for step in range(max_rollout_global):
                target_idx = k + step + 1
                if target_idx < T_batch:
                    col_all[row_offset:row_offset + B_batch, step] = batch_vals[:, target_idx]
            row_offset += B_batch
        labels_at_step[col] = col_all

    binary_labels_agg: dict[str, np.ndarray] = {}
    for col in BINARY_LABEL_COLUMNS:
        binary_labels_agg[col] = torch.cat(
            [b[col] for b in all_binary_labels], dim=0).numpy()

    save_dict = {
        "stay_ids": stay_ids,
        "lengths": lengths_arr,
        "context_length": context_length,
        "max_rollout": max_rollout_global,
        "per_step_l1": per_step_l1.numpy(),
        "predicted_z": predicted_z.numpy(),
        "valid_mask": valid_mask.numpy(),
        "baselines": {
            "copy_forward_l1": copy_forward_l1.numpy(),
            "linear_extrap_l1": linear_extrap_l1.numpy(),
        },
        "labels_at_step": {col: v.numpy() for col, v in labels_at_step.items()},
        "binary_labels": binary_labels_agg,
    }

    pred_path = output_dir / f"predictions_{predictor_name}.pt"
    torch.save(save_dict, pred_path, pickle_protocol=4)
    file_size_gb = pred_path.stat().st_size / 1e9
    logger.info("Saved predictions to %s (%.2f GB)", pred_path, file_size_gb)

    per_step_l1_np = per_step_l1.numpy()
    cf_l1_np = copy_forward_l1.numpy()
    valid_np = valid_mask.numpy()

    per_step_mean_l1 = []
    per_step_mean_cf_l1 = []
    per_step_mase = []
    per_step_n_valid = []

    for step in range(max_rollout_global):
        mask = valid_np[:, step] & ~np.isnan(per_step_l1_np[:, step])
        n = int(mask.sum())
        if n > 0:
            ml1 = float(np.mean(per_step_l1_np[mask, step]))
            cfl1 = float(np.mean(cf_l1_np[mask, step]))
            mase = ml1 / cfl1 if cfl1 > 0 else None
        else:
            ml1, cfl1, mase = None, None, None
        per_step_mean_l1.append(ml1)
        per_step_mean_cf_l1.append(cfl1)
        per_step_mase.append(mase)
        per_step_n_valid.append(n)

    metadata = {
        "predictor_name": predictor_name,
        "context_length": context_length,
        "max_rollout": max_rollout_global,
        "n_windows_total": N,
        "elapsed_seconds": round(elapsed, 1),
        "per_step_mean_l1": per_step_mean_l1,
        "per_step_mean_cf_l1": per_step_mean_cf_l1,
        "per_step_mase": per_step_mase,
        "per_step_n_valid": per_step_n_valid,
    }

    meta_path = output_dir / f"metadata_{predictor_name}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    return pred_path


# ---------------------------------------------------------------------------
# Single-context orchestrator
# ---------------------------------------------------------------------------

def run_single_context(
    config: dict,
    plan: str,
    context_length: int,
    device: torch.device,
    shards: list[dict],
    windows: list[dict],
    max_windows: Optional[int] = None,
) -> dict:
    """Run rollout for one paradigm at one context length."""
    t_start = time.time()

    rollout_cfg = config["rollout"]
    batch_size = rollout_cfg["batch_size"]
    use_bf16 = rollout_cfg.get("precision", "bf16") == "bf16"
    zero_actions = rollout_cfg.get("zero_actions", False)

    plan_cfg = config["plans"][plan]
    output_dir = Path(plan_cfg["output_dir"]) / f"context_{context_length:02d}"
    ckpt_path = plan_cfg["predictor_checkpoint"]
    model_arch_config = plan_cfg.get("model_config")

    model = load_predictor(ckpt_path, device, model_config=model_arch_config)

    batches, n_skipped = create_batches(windows, context_length, batch_size)
    if not batches:
        logger.warning(
            "Plan %s C=%d: no valid windows (all %d skipped). Skipping.",
            plan, context_length, n_skipped,
        )
        return {"n_windows_total": 0, "context_length": context_length, "plan": plan}

    all_results: list[dict] = []
    all_baselines: list[dict] = []
    all_stay_ids: list[list[int]] = []
    all_lengths: list[torch.Tensor] = []
    all_labels: list[dict] = []
    all_binary_labels: list[dict] = []

    logger.info(
        "Starting rollout: plan=%s C=%d, %d batches, %d valid windows, "
        "bf16=%s, zero_actions=%s",
        plan, context_length, len(batches),
        sum(len(b) for b in batches), use_bf16, zero_actions,
    )
    if zero_actions:
        logger.warning(
            "ZERO ACTIONS MODE: replacing each action embedding with a "
            "4096-D zero vector (unconditional rollout ablation)."
        )

    for batch_idx, batch_indices in enumerate(batches):
        batch = gather_batch_data(windows, shards, batch_indices, device)

        if zero_actions:
            batch["actions"] = torch.zeros_like(batch["actions"])

        result = autoregressive_rollout(model, batch, context_length, use_bf16)
        max_rollout = result["per_step_l1"].shape[1]
        baseline = compute_baselines(batch, context_length, max_rollout)

        all_results.append(result)
        all_baselines.append(baseline)
        all_stay_ids.append(batch["stay_ids"])
        all_lengths.append(batch["lengths"].cpu())
        all_labels.append({col: v.cpu() for col, v in batch["labels"].items()})
        all_binary_labels.append(batch["binary_labels"])

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            elapsed = time.time() - t_start
            pct = (batch_idx + 1) / len(batches) * 100
            logger.info(
                "  plan=%s C=%d batch %d/%d (%.0f%%) — %.1fs elapsed",
                plan, context_length, batch_idx + 1, len(batches), pct, elapsed,
            )

    elapsed = time.time() - t_start
    logger.info(
        "plan=%s C=%d rollout complete in %.1fs, saving...",
        plan, context_length, elapsed,
    )

    predictor_name = "absolute"

    save_results(
        output_dir, predictor_name,
        all_results, all_baselines,
        all_stay_ids, all_lengths,
        all_labels, all_binary_labels,
        context_length, elapsed,
    )

    meta_path = output_dir / f"metadata_{predictor_name}.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["plan"] = plan

    logger.info("=== Plan %s C=%d Summary ===", plan, context_length)
    for h in [1, 6, 12, 24, 48]:
        step_idx = h - 1
        if step_idx < len(metadata["per_step_mean_l1"]):
            ml1 = metadata["per_step_mean_l1"][step_idx]
            cfl1 = metadata["per_step_mean_cf_l1"][step_idx]
            mase = metadata["per_step_mase"][step_idx]
            n = metadata["per_step_n_valid"][step_idx]
            logger.info(
                "  h=%2d: model_L1=%.2f, copy_fwd_L1=%.2f, MASE=%.3f (n=%d)",
                h, ml1 or 0, cfl1 or 0, mase or 0, n,
            )

    return metadata


# ---------------------------------------------------------------------------
# Multi-context orchestrator
# ---------------------------------------------------------------------------

def run_rollout_evaluation(
    config: dict,
    plan: str,
    device: torch.device,
    max_windows: Optional[int] = None,
) -> dict:
    """Run rollout across all configured context lengths for one paradigm."""
    context_lengths = config["rollout"]["context_lengths"]
    plan_cfg = config["plans"][plan]

    shards, windows = load_test_data(
        plan_cfg["embedding_dir"],
        config["data"].get("split", "test"),
        max_windows=max_windows,
    )

    all_metadata: dict[int, dict] = {}
    for C in context_lengths:
        logger.info("=" * 60)
        logger.info("Plan %s — context length C=%d", plan, C)
        logger.info("=" * 60)

        metadata = run_single_context(
            config, plan, C, device,
            shards, windows,
            max_windows=max_windows,
        )
        all_metadata[C] = metadata

    return all_metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-context autoregressive rollout evaluation (paper §5.2)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a rollout YAML (see configs/eval/rollout.yaml).",
    )
    parser.add_argument(
        "--plan",
        choices=["clin_jepa", "vjepa2ac", "sft_baseline", "all"],
        required=True,
        help="Which paradigm to evaluate. 'all' loops over all three.",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU device ID (ignored if CUDA is unavailable).",
    )
    parser.add_argument(
        "--context", type=int, nargs="+", default=None,
        help="Override the context lengths from the config.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Override data.split from the config.",
    )
    parser.add_argument(
        "--max_windows", type=int, default=None,
        help="Limit the number of windows (smoke testing only).",
    )
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)

    if args.context:
        config["rollout"]["context_lengths"] = args.context
    if args.split is not None:
        config["data"]["split"] = args.split

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(
            "Using GPU %d: %s",
            args.gpu_id, torch.cuda.get_device_name(args.gpu_id),
        )
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available — running on CPU (slow).")

    plans_to_run = (
        ["clin_jepa", "vjepa2ac", "sft_baseline"] if args.plan == "all"
        else [args.plan]
    )

    for plan in plans_to_run:
        if plan not in config["plans"]:
            logger.error(
                "Plan '%s' not found in config['plans']. Available: %s",
                plan, list(config["plans"].keys()),
            )
            sys.exit(1)

        logger.info("=" * 70)
        logger.info(
            "Evaluating plan=%s | split=%s | contexts=%s",
            plan, config["data"].get("split", "test"),
            config["rollout"]["context_lengths"],
        )
        logger.info("=" * 70)

        all_metadata = run_rollout_evaluation(
            config, plan, device, max_windows=args.max_windows,
        )
        total_time = sum(m.get("elapsed_seconds", 0) for m in all_metadata.values())
        logger.info(
            "Done: plan=%s (total %.1fs across %d contexts)",
            plan, total_time, len(all_metadata),
        )

    logger.info("All rollout evaluations complete.")


if __name__ == "__main__":
    main()
