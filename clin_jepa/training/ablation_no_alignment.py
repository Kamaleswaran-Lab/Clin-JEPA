"""Clin-JEPA ablation: remove Phase 3 (alignment) and the Phase 4 hard sync.

Paper §5.2 ablation.

This script reuses the Clin-JEPA pretraining loop from
:mod:`clin_jepa.training.pretrain_clin_jepa` but removes Phase 3
(alignment) and the Phase 4 hard sync. The alignment-specific differences:

1. Each phase-transition gate is guarded by
   ``and phase_steps["phase_X"] > 0`` so configuring ``phase_2b_steps=0``
   cleanly transitions ``phase_2a -> phase_4`` in one step.
2. The Phase 4 ``hard_sync_target_from_online`` call is removed. The
   phase_4 transition saves a ``phase_2a_to_4_no_sync`` checkpoint and
   preserves the accumulated online/target gap into phase_4. Phase 4
   then runs native-chaining rollout in this mixed space, surfacing the
   drift the hard sync would have prevented.
3. The resume-checkpoint ``schema_version`` is ``"ablation_no_alignment_v1"``
   so that Clin-JEPA pretraining resume state cannot be cross-loaded into
   this script (and vice versa).

Phase 1 warmup is kept (only the warmup ablation removes it).

Usage::

    # Single-GPU smoke:
    python -m clin_jepa.training.ablation_no_alignment \\
        --config configs/train/ablation_no_alignment.yaml \\
        --max_windows 100 --smoke

    # Multi-GPU production (4x H200 DDP):
    torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \\
        -m clin_jepa.training.ablation_no_alignment \\
        --config configs/train/ablation_no_alignment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.encoder_ops import (
    PROJECT_ROOT,
    _build_online_target_param_pairs,
    atomic_torch_save,
    collect_lora_adapter_state,
    create_target_encoder_fp32,
    ema_update,
    encode_texts_batched,
    load_sft_initialized_encoder,
    manual_allreduce_grads,
    restore_lora_adapter_state,
    restore_rng_state,
    snapshot_rng_state,
    verify_ema_working,
)
from clin_jepa.training.trajectory_dataset import (
    TrajectoryWindowDataset,
    trajectory_collate_fn,
)
from clin_jepa.utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema version for the no-alignment ablation resume state.
# ---------------------------------------------------------------------------
RESUME_SCHEMA_VERSION = "ablation_no_alignment_v1"


# ---------------------------------------------------------------------------
# Preemption signal handling
# ---------------------------------------------------------------------------
_PREEMPTED: bool = False


def _preempt_signal_handler(signum: int, frame) -> None:  # noqa: ARG001
    global _PREEMPTED
    _PREEMPTED = True


# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, bool]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Encoder forward helpers (shared with pretrain_clin_jepa)
# ---------------------------------------------------------------------------

def encode_window_three_streams(model, tokenizer, window, device, max_seq_len,
                                train_mode, output_dtype):
    L = window["length"]
    all_texts = (
        [window["demographics_text"]]
        + window["state_texts"]
        + window["action_texts"]
    )
    z = encode_texts_batched(
        model, tokenizer, all_texts, device=device,
        max_seq_len=max_seq_len, train_mode=train_mode, output_dtype=output_dtype,
    )
    z_static = z[0]
    z_states = z[1 : 1 + L]
    z_actions = z[1 + L : 1 + 2 * L]
    return z_states, z_actions, z_static


def encode_batch_via_online(model, tokenizer, batch_windows, device, max_seq_len,
                            train_mode, output_dtype):
    B = len(batch_windows)
    lengths = torch.tensor([w["length"] for w in batch_windows], dtype=torch.long)
    T_max = int(lengths.max().item())

    z_states_padded = torch.zeros(B, T_max, 4096, dtype=output_dtype, device=device)
    z_actions_padded = torch.zeros(B, T_max, 4096, dtype=output_dtype, device=device)
    z_static_padded = torch.zeros(B, 4096, dtype=output_dtype, device=device)
    padding_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)

    for i, window in enumerate(batch_windows):
        L = window["length"]
        zs, za, zd = encode_window_three_streams(
            model, tokenizer, window, device, max_seq_len, train_mode, output_dtype,
        )
        z_states_padded[i, :L] = zs
        z_actions_padded[i, :L] = za
        z_static_padded[i] = zd
        padding_mask[i, :L] = True

    return {
        "z_states": z_states_padded,
        "z_actions": z_actions_padded,
        "z_static": z_static_padded,
        "lengths": lengths.to(device),
        "padding_mask": padding_mask,
    }


# ---------------------------------------------------------------------------
# Loss functions (shared with pretrain_clin_jepa)
# ---------------------------------------------------------------------------

def teacher_forcing_loss(preds, targets, mask):
    l1 = (preds - targets).abs().sum(dim=-1)
    loss = (l1 * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return loss


def compute_rollout_loss_native(predictor, z_states, z_static, z_actions, lengths,
                                rollout_horizon=2):
    """Native autoregressive chaining. Valid only when online == target."""
    B, T_max, _ = z_states.shape
    device = z_states.device

    min_len = int(lengths.min().item())
    if min_len < rollout_horizon + 2:
        return torch.tensor(0.0, device=device)

    k = random.randint(1, min_len - rollout_horizon - 1)

    z_hist = z_states[:, :k, :]
    za_hist = z_actions[:, :k, :]
    z_current = z_states[:, k, :]

    total_loss = torch.tensor(0.0, device=device)

    for step in range(rollout_horizon):
        za_next = z_actions[:, k + step, :]
        z_seq = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        za_seq = torch.cat([za_hist, za_next.unsqueeze(1)], dim=1)
        seq_lengths = torch.full(
            (z_seq.shape[0],), z_seq.shape[1],
            device=device, dtype=torch.long,
        )
        z_pred = predictor(
            z_seq, z_static, za_seq, seq_lengths,
            return_last_only=True,
        )

        z_target = z_states[:, k + step + 1, :]
        step_loss = (z_pred - z_target).abs().sum(dim=-1).mean()
        total_loss = total_loss + step_loss

        z_hist = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        za_hist = torch.cat([za_hist, za_next.unsqueeze(1)], dim=1)
        z_current = z_pred

    return total_loss


def compute_rollout_loss_teacher_forced(predictor, z_states_online, z_actions_online,
                                        z_states_target, z_static_online, lengths,
                                        rollout_horizon=2):
    """Teacher-forced rollout for mixed-space (phase 2a/2b)."""
    B, T_max, _ = z_states_online.shape
    device = z_states_online.device

    min_len = int(lengths.min().item())
    if min_len < rollout_horizon + 2:
        return torch.tensor(0.0, device=device)

    k = random.randint(1, min_len - rollout_horizon - 1)

    total_loss = torch.tensor(0.0, device=device)

    for step in range(rollout_horizon):
        z_hist_through_curr = z_states_online[:, : k + step + 1, :]
        za_hist_through_curr = z_actions_online[:, : k + step + 1, :]
        seq_lengths = torch.full(
            (z_hist_through_curr.shape[0],), z_hist_through_curr.shape[1],
            device=device, dtype=torch.long,
        )

        z_pred = predictor(
            z_hist_through_curr, z_static_online, za_hist_through_curr, seq_lengths,
            return_last_only=True,
        )

        z_target = z_states_target[:, k + step + 1, :]
        step_loss = (z_pred - z_target).abs().sum(dim=-1).mean()
        total_loss = total_loss + step_loss

    return total_loss


# ---------------------------------------------------------------------------
# Phase-specific step functions (shared with pretrain_clin_jepa)
# ---------------------------------------------------------------------------

def phase_1_step(raw_online, predictor, tokenizer, batch_windows, device,
                 max_seq_len, rollout_horizon):
    raw_online.set_adapter("online")
    with torch.no_grad():
        enc = encode_batch_via_online(
            raw_online, tokenizer, batch_windows, device, max_seq_len,
            train_mode=False, output_dtype=torch.bfloat16,
        )
    z_states = enc["z_states"]
    z_actions = enc["z_actions"]
    z_static = enc["z_static"]
    lengths = enc["lengths"]
    pad_mask = enc["padding_mask"]
    preds = predictor(z_states, z_static, z_actions, lengths)
    targets = z_states[:, 1:, :]
    tf_loss = teacher_forcing_loss(preds, targets, pad_mask[:, 1:])
    roll_loss = compute_rollout_loss_native(
        predictor, z_states, z_static, z_actions, lengths, rollout_horizon,
    )
    return tf_loss + roll_loss


def phase_2a_step(raw_online, predictor, tokenizer, batch_windows, device,
                  max_seq_len, rollout_horizon):
    raw_online.set_adapter("online")
    enc_online = encode_batch_via_online(
        raw_online, tokenizer, batch_windows, device, max_seq_len,
        train_mode=True, output_dtype=torch.bfloat16,
    )
    raw_online.set_adapter("target")
    with torch.no_grad():
        enc_target = encode_batch_via_online(
            raw_online, tokenizer, batch_windows, device, max_seq_len,
            train_mode=False, output_dtype=torch.float32,
        )
    raw_online.set_adapter("online")

    z_states_o = enc_online["z_states"]
    z_actions_o = enc_online["z_actions"]
    z_static_o = enc_online["z_static"]
    lengths = enc_online["lengths"]
    pad_mask = enc_online["padding_mask"]
    z_states_t = enc_target["z_states"].detach()

    preds = predictor(z_states_o, z_static_o, z_actions_o, lengths)
    targets = z_states_t[:, 1:, :]
    tf_loss = teacher_forcing_loss(preds, targets, pad_mask[:, 1:])

    roll_loss = compute_rollout_loss_teacher_forced(
        predictor=predictor,
        z_states_online=z_states_o,
        z_actions_online=z_actions_o,
        z_states_target=z_states_t,
        z_static_online=z_static_o,
        lengths=lengths,
        rollout_horizon=rollout_horizon,
    )
    return tf_loss + roll_loss


def phase_2b_step(raw_online, predictor, tokenizer, batch_windows, device,
                  max_seq_len, rollout_horizon):
    """Phase 2b (unused: this ablation sets phase_2b_steps=0)."""
    raw_online.set_adapter("online")
    with torch.no_grad():
        enc_online = encode_batch_via_online(
            raw_online, tokenizer, batch_windows, device, max_seq_len,
            train_mode=False, output_dtype=torch.bfloat16,
        )
    raw_online.set_adapter("target")
    with torch.no_grad():
        enc_target = encode_batch_via_online(
            raw_online, tokenizer, batch_windows, device, max_seq_len,
            train_mode=False, output_dtype=torch.float32,
        )
    raw_online.set_adapter("online")

    z_states_o = enc_online["z_states"]
    z_actions_o = enc_online["z_actions"]
    z_static_o = enc_online["z_static"]
    lengths = enc_online["lengths"]
    pad_mask = enc_online["padding_mask"]
    z_states_t = enc_target["z_states"].detach()

    preds = predictor(z_states_o, z_static_o, z_actions_o, lengths)
    targets = z_states_t[:, 1:, :]
    tf_loss = teacher_forcing_loss(preds, targets, pad_mask[:, 1:])

    roll_loss = compute_rollout_loss_teacher_forced(
        predictor=predictor,
        z_states_online=z_states_o,
        z_actions_online=z_actions_o,
        z_states_target=z_states_t,
        z_static_online=z_static_o,
        lengths=lengths,
        rollout_horizon=rollout_horizon,
    )
    return tf_loss + roll_loss


def phase_4_step(raw_online, predictor, tokenizer, batch_windows, device,
                 max_seq_len, rollout_horizon):
    """Phase 4: encoder frozen, predictor finalize. Native chaining rollout
    (mixed space — no hard sync in this ablation)."""
    raw_online.set_adapter("online")
    with torch.no_grad():
        enc = encode_batch_via_online(
            raw_online, tokenizer, batch_windows, device, max_seq_len,
            train_mode=False, output_dtype=torch.bfloat16,
        )

    z_states = enc["z_states"]
    z_actions = enc["z_actions"]
    z_static = enc["z_static"]
    lengths = enc["lengths"]
    pad_mask = enc["padding_mask"]

    preds = predictor(z_states, z_static, z_actions, lengths)
    targets = z_states[:, 1:, :]
    tf_loss = teacher_forcing_loss(preds, targets, pad_mask[:, 1:])

    roll_loss = compute_rollout_loss_native(
        predictor, z_states, z_static, z_actions, lengths, rollout_horizon,
    )
    return tf_loss + roll_loss


# ---------------------------------------------------------------------------
# Encoder freeze toggle (shared with pretrain_clin_jepa)
# ---------------------------------------------------------------------------

def set_encoder_trainable(online: nn.Module, trainable: bool) -> None:
    for name, p in online.named_parameters():
        if "lora_" in name and ".online." in name:
            p.requires_grad = trainable


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ablation(
    raw_online,
    predictor: nn.Module,
    tokenizer,
    val_loader: DataLoader,
    device: torch.device,
    max_seq_len: int,
    max_windows: int,
    use_target: bool,
) -> dict:
    """Compute val tf loss + z_std collapse monitor."""
    predictor.eval()
    n_windows = 0
    total_loss = 0.0
    z_std_samples = []

    for collated in val_loader:
        if n_windows >= max_windows:
            break

        batch_windows = collated["batch"]
        if not batch_windows:
            continue

        raw_online.set_adapter("online")
        enc_online = encode_batch_via_online(
            raw_online, tokenizer, batch_windows, device, max_seq_len,
            train_mode=False, output_dtype=torch.bfloat16,
        )

        if use_target:
            raw_online.set_adapter("target")
            enc_target = encode_batch_via_online(
                raw_online, tokenizer, batch_windows, device, max_seq_len,
                train_mode=False, output_dtype=torch.float32,
            )
            raw_online.set_adapter("online")
            target_states = enc_target["z_states"]
        else:
            target_states = enc_online["z_states"]

        preds = predictor(
            enc_online["z_states"], enc_online["z_static"],
            enc_online["z_actions"], enc_online["lengths"],
        )
        targets = target_states[:, 1:, :].float()
        loss = teacher_forcing_loss(preds, targets, enc_online["padding_mask"][:, 1:])
        total_loss += loss.item()
        n_windows += len(batch_windows)

        z_state = enc_online["z_states"].float()
        z_std_samples.append(z_state.std(dim=1).mean().detach())

    predictor.train()

    return {
        "val_loss": total_loss / max(1, len(z_std_samples)),
        "z_std": torch.stack(z_std_samples).mean().item() if z_std_samples else 0.0,
        "n_windows": n_windows,
    }


# ---------------------------------------------------------------------------
# Save checkpoint (same layout as pretrain_clin_jepa)
# ---------------------------------------------------------------------------

def save_ablation_checkpoint(
    raw_online, predictor: nn.Module, tokenizer,
    save_dir: Path, tag: str, step: int, phase: str,
    val_loss: float, z_std: float, elapsed_seconds: float,
) -> None:
    out = save_dir / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)

    raw_online.set_adapter("online")
    raw_online.save_pretrained(out, selected_adapters=["online"])

    online_dir = out / "online"
    tokenizer.save_pretrained(online_dir)

    torch.save(predictor.state_dict(), out / "predictor.pt")

    meta = {
        "tag": tag,
        "step": step,
        "phase": phase,
        "val_loss": val_loss,
        "z_std": z_std,
        "elapsed_seconds": elapsed_seconds,
        "ablation": "no_alignment",
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Resume state save / load (uses ABLATION schema_version)
# ---------------------------------------------------------------------------

def save_resume_state_ablation(
    save_dir: Path, raw_online, raw_pred, optimizer, scheduler,
    global_step, micro_step, epoch, micro_step_in_epoch,
    current_phase, last_phase_2b_val_loss, last_phase_4_val_loss,
    best_val_loss, config,
) -> None:
    state = {
        "schema_version":         RESUME_SCHEMA_VERSION,
        "lora_adapter_state":     collect_lora_adapter_state(raw_online),
        "predictor_state":        raw_pred.state_dict(),
        "optimizer_state":        optimizer.state_dict(),
        "scheduler_state":        scheduler.state_dict(),
        "global_step":            global_step,
        "micro_step":             micro_step,
        "epoch":                  epoch,
        "micro_step_in_epoch":    micro_step_in_epoch,
        "current_phase":          current_phase,
        "last_phase_2b_val_loss": last_phase_2b_val_loss,
        "last_phase_4_val_loss":  last_phase_4_val_loss,
        "best_val_loss":          best_val_loss,
        "rng_state":              snapshot_rng_state(),
        "config_snapshot":        config,
    }
    atomic_torch_save(state, save_dir / "resume.pt")


def load_resume_state_ablation(
    resume_path: Path, raw_online, raw_pred, optimizer, scheduler, device,
) -> dict:
    state = torch.load(resume_path, map_location="cpu", weights_only=False)
    sv = state.get("schema_version")
    if sv != RESUME_SCHEMA_VERSION:
        raise RuntimeError(
            f"resume.pt schema_version {sv!r} does not match expected "
            f"{RESUME_SCHEMA_VERSION!r}. Refuse to load — wrong ablation script?"
        )

    restore_lora_adapter_state(raw_online, state["lora_adapter_state"])

    raw_pred.load_state_dict(state["predictor_state"])
    raw_pred.to(device)

    optimizer.load_state_dict(state["optimizer_state"])
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)

    scheduler.load_state_dict(state["scheduler_state"])
    restore_rng_state(state["rng_state"])

    return {
        "global_step":            int(state["global_step"]),
        "micro_step":             int(state["micro_step"]),
        "epoch":                  int(state["epoch"]),
        "micro_step_in_epoch":    int(state.get("micro_step_in_epoch", 0)),
        "current_phase":          str(state["current_phase"]),
        "last_phase_2b_val_loss": state.get("last_phase_2b_val_loss"),
        "last_phase_4_val_loss":  state.get("last_phase_4_val_loss"),
        "best_val_loss":          float(state["best_val_loss"]),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, args: argparse.Namespace) -> None:
    rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    if is_main:
        setup_logging()
        logger.info(
            "Clin-JEPA w/o alignment — phase_2b + phase_3 hard sync removed "
            "(%d GPUs)", world_size,
        )

    seed = config["training"]["seed"]
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    online, tokenizer = load_sft_initialized_encoder(config, device)
    online = create_target_encoder_fp32(online)
    if is_main:
        verify_ema_working(online, n_updates=10)
    if is_distributed:
        dist.barrier()
    ema_pairs = _build_online_target_param_pairs(online)

    pred_cfg = config["predictor"]
    predictor = ACTransformerPredictor(
        state_dim=pred_cfg.get("state_dim", 4096),
        action_dim=pred_cfg.get("action_dim", 4096),
        static_dim=pred_cfg.get("static_dim", 4096),
        hidden_dim=pred_cfg.get("hidden_dim", 1024),
        num_layers=pred_cfg.get("num_layers", 6),
        num_heads=pred_cfg.get("num_heads", 8),
        ffn_dim=pred_cfg.get("ffn_dim", 4096),
        max_timesteps=pred_cfg.get("max_timesteps", 72),
        ffn_dropout=pred_cfg.get("ffn_dropout", 0.15),
        attn_dropout=pred_cfg.get("attn_dropout", 0.1),
        prediction_mode=pred_cfg.get("prediction_mode", "absolute"),
    ).to(device)

    if is_main:
        logger.info(
            "AC Transformer Predictor: %.1fM params (target ~92M)",
            predictor.num_params() / 1e6,
        )

    if is_distributed:
        with torch.no_grad():
            for p in predictor.parameters():
                dist.broadcast(p.data, src=0)
        if is_main:
            logger.info("Broadcast predictor weights from rank 0 to %d ranks", world_size)

    raw_online = online
    raw_pred = predictor

    data_cfg = config["data"]
    train_ds = TrajectoryWindowDataset(
        shard_dir=PROJECT_ROOT / data_cfg["train_dir"],
        split="train",
        max_windows=args.max_windows,
    )
    val_ds = TrajectoryWindowDataset(
        shard_dir=PROJECT_ROOT / data_cfg["val_dir"],
        split="val",
        max_windows=config["logging"].get("val_windows", 1000),
    )

    train_cfg = config["training"]
    per_gpu_batch = train_cfg["per_gpu_batch_size"]
    grad_accum = train_cfg.get("grad_accum", 1)
    if args.per_gpu_batch_size is not None:
        per_gpu_batch = args.per_gpu_batch_size
    if args.grad_accum is not None:
        grad_accum = args.grad_accum

    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=per_gpu_batch, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=trajectory_collate_fn,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=per_gpu_batch, shuffle=False,
        collate_fn=trajectory_collate_fn, num_workers=2,
    )

    phases_cfg = train_cfg["phases"]
    if args.smoke:
        phase_steps = {
            "phase_1": args.smoke_phase_steps if args.smoke_phase_steps else 5,
            "phase_2a": args.smoke_phase_steps if args.smoke_phase_steps else 5,
            "phase_2b": 0,  # no-alignment ablation: phase_2b has zero steps
            "phase_4": args.smoke_phase_steps if args.smoke_phase_steps else 5,
        }
    else:
        phase_steps = {
            "phase_1":  phases_cfg["phase_1_steps"],
            "phase_2a": phases_cfg["phase_2a_steps"],
            "phase_2b": phases_cfg["phase_2b_steps"],
            "phase_4":  phases_cfg["phase_4_steps"],
        }
    total_steps = sum(phase_steps.values())

    boundary_phase_2a = phase_steps["phase_1"]
    boundary_phase_2b = boundary_phase_2a + phase_steps["phase_2a"]
    boundary_phase_4 = boundary_phase_2b + phase_steps["phase_2b"]

    if is_main:
        logger.info(
            "phase schedule: 1=%d, 2a=%d, 2b=%d, 3=skipped, 4=%d, total=%d",
            phase_steps["phase_1"], phase_steps["phase_2a"],
            phase_steps["phase_2b"], phase_steps["phase_4"], total_steps,
        )
        if phase_steps["phase_2b"] != 0:
            logger.warning(
                "This script expects phase_2b_steps=0 (and skips the phase_3 "
                "hard sync) to test the no-alignment hypothesis. Config has "
                "phase_2b_steps=%d. Hard sync will still be skipped regardless. "
                "For a full Clin-JEPA run, use pretrain_clin_jepa.py.",
                phase_steps["phase_2b"],
            )

    encoder_params = [online_p for online_p, _ in ema_pairs]
    predictor_params = list(raw_pred.parameters())
    wd = train_cfg["weight_decay"]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params,   "lr": train_cfg["lr_encoder"],   "weight_decay": wd},
        {"params": predictor_params, "lr": train_cfg["lr_predictor"], "weight_decay": wd},
    ])

    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_cfg = config.get("logging", {})
    log_every = args.log_every if args.log_every else log_cfg.get("log_every", 25)
    eval_every = args.eval_every if args.eval_every else log_cfg.get("eval_every", 200)
    eval_max_windows = log_cfg.get("eval_max_windows", 200)

    z_std_abort_threshold = log_cfg.get("z_std_abort_threshold", -1.0)  # disabled

    save_cfg = config["save"]
    save_dir = PROJECT_ROOT / save_cfg["dir"]
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    grad_clip_enc = train_cfg["gradient_clip_encoder"]
    grad_clip_pred = train_cfg["gradient_clip_predictor"]
    rollout_horizon = train_cfg.get("rollout_horizon", 2)
    max_seq_len = data_cfg["max_seq_len"]

    tau = train_cfg["ema"]["tau_phase_2a"]

    # Standard entry point — phase_1 + frozen encoder (this ablation keeps phase_1)
    set_encoder_trainable(online, False)
    current_phase = "phase_1"
    if is_main:
        logger.info("=== ENTERING PHASE 1 (predictor warmup, encoder frozen) ===")

    best_val_loss = float("inf")
    global_step = 0
    micro_step = 0
    epoch = 0
    start_micro_in_epoch = 0
    last_phase_2b_val_loss = None
    last_phase_4_val_loss = None

    resume_path: Optional[Path] = None
    if args.resume_from is not None:
        if args.resume_from.lower() != "none":
            resume_path = Path(args.resume_from)
    else:
        candidate = save_dir / "resume.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None and resume_path.exists():
        if is_main:
            logger.info("=== RESUMING from %s ===", resume_path)
        loaded = load_resume_state_ablation(
            resume_path, raw_online, raw_pred, optimizer, scheduler, device,
        )
        global_step             = loaded["global_step"]
        micro_step              = loaded["micro_step"]
        epoch                   = loaded["epoch"]
        start_micro_in_epoch    = loaded["micro_step_in_epoch"]
        current_phase           = loaded["current_phase"]
        last_phase_2b_val_loss  = loaded["last_phase_2b_val_loss"]
        last_phase_4_val_loss   = loaded["last_phase_4_val_loss"]
        best_val_loss           = loaded["best_val_loss"]
        if current_phase == "phase_2a":
            set_encoder_trainable(online, True)
        else:
            set_encoder_trainable(online, False)
        if is_main:
            logger.info(
                "  resumed: phase=%s, epoch=%d, micro_in_epoch=%d, global_step=%d, "
                "best_val=%.4f",
                current_phase, epoch, start_micro_in_epoch, global_step, best_val_loss,
            )
        if is_distributed:
            dist.barrier()

    signal.signal(signal.SIGUSR1, _preempt_signal_handler)
    signal.signal(signal.SIGTERM, _preempt_signal_handler)
    if is_main:
        logger.info("Registered SIGUSR1/SIGTERM handlers for preemption recovery")

    raw_online.train()
    raw_pred.train()

    t0 = time.time()

    if is_distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    if start_micro_in_epoch > 0:
        if is_main:
            logger.info(
                "  resume: skipping first %d micro-batches in epoch %d",
                start_micro_in_epoch, epoch,
            )
        skipped = 0
        try:
            for _ in range(start_micro_in_epoch):
                next(train_iter)
                skipped += 1
        except StopIteration:
            if is_main:
                logger.info(
                    "  resume skip hit end of epoch %d after %d batches; advancing",
                    epoch, skipped,
                )
            epoch += 1
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)

    micro_step_in_epoch_counter = start_micro_in_epoch

    while global_step < total_steps:
        try:
            collated = next(train_iter)
            micro_step_in_epoch_counter += 1
        except StopIteration:
            epoch += 1
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            collated = next(train_iter)
            micro_step_in_epoch_counter = 1

        batch_windows = collated["batch"]
        if not batch_windows:
            continue

        if current_phase == "phase_1":
            loss = phase_1_step(raw_online, raw_pred, tokenizer,
                                batch_windows, device, max_seq_len, rollout_horizon)
        elif current_phase == "phase_2a":
            loss = phase_2a_step(raw_online, raw_pred, tokenizer,
                                 batch_windows, device, max_seq_len, rollout_horizon)
        elif current_phase == "phase_2b":
            loss = phase_2b_step(raw_online, raw_pred, tokenizer,
                                 batch_windows, device, max_seq_len, rollout_horizon)
        elif current_phase == "phase_4":
            loss = phase_4_step(raw_online, raw_pred, tokenizer,
                                batch_windows, device, max_seq_len, rollout_horizon)
        else:
            raise RuntimeError(f"Unknown phase: {current_phase}")

        loss = loss / grad_accum
        loss.backward()

        micro_step += 1

        if micro_step % grad_accum == 0:
            if is_distributed:
                if current_phase == "phase_2a":
                    manual_allreduce_grads(encoder_params)
                manual_allreduce_grads(predictor_params)

            if current_phase == "phase_2a":
                torch.nn.utils.clip_grad_norm_(encoder_params, grad_clip_enc)
            torch.nn.utils.clip_grad_norm_(predictor_params, grad_clip_pred)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if current_phase in ("phase_2a", "phase_2b"):
                ema_update(online, tau=tau, pairs=ema_pairs)

            # ====================================================================
            # Phase transitions (gated on phase_steps[X] > 0)
            # ====================================================================
            new_phase = current_phase
            if global_step == boundary_phase_2a and phase_steps["phase_2a"] > 0 \
                    and phase_steps["phase_1"] > 0:
                new_phase = "phase_2a"
            elif global_step == boundary_phase_2b and phase_steps["phase_2b"] > 0:
                new_phase = "phase_2b"
            elif global_step == boundary_phase_4 and phase_steps["phase_4"] > 0:
                new_phase = "phase_4"

            if new_phase != current_phase:
                if is_main:
                    save_ablation_checkpoint(
                        raw_online, raw_pred, tokenizer, save_dir,
                        tag=f"{current_phase}_end",
                        step=global_step, phase=current_phase,
                        val_loss=best_val_loss, z_std=0.0,
                        elapsed_seconds=time.time() - t0,
                    )

                if new_phase == "phase_2a":
                    set_encoder_trainable(online, True)
                    if is_main:
                        logger.info("=== ENTERING PHASE 2a (joint, encoder trainable, tau=0.996) ===")
                elif new_phase == "phase_2b":
                    set_encoder_trainable(online, False)
                    if is_main:
                        logger.info("=== ENTERING PHASE 2b (online frozen, target EMA chases) ===")
                elif new_phase == "phase_4":
                    # ====================================================================
                    # Phase 3 hard sync is removed here.
                    # ====================================================================
                    if is_main:
                        # log the preserved gap
                        max_gap = 0.0
                        for online_p, target_p in ema_pairs:
                            gap = (online_p.float() - target_p).abs().max().item()
                            max_gap = max(max_gap, gap)
                        logger.warning(
                            "Skipping phase 3 hard sync. "
                            "Preserving online != target gap: max |online - target| = %.2e",
                            max_gap,
                        )
                        save_ablation_checkpoint(
                            raw_online, raw_pred, tokenizer, save_dir,
                            tag="phase_2a_to_4_no_sync", step=global_step,
                            phase="phase_2a_to_4_no_sync",
                            val_loss=best_val_loss, z_std=0.0,
                            elapsed_seconds=time.time() - t0,
                        )
                    set_encoder_trainable(online, False)
                    if is_distributed:
                        dist.barrier()
                        with torch.no_grad():
                            for online_p, target_p in ema_pairs:
                                dist.broadcast(online_p.data, src=0)
                                dist.broadcast(target_p.data, src=0)
                        dist.barrier()
                    if is_main:
                        logger.info(
                            "=== ENTERING PHASE 4 (NO ALIGNMENT, native chaining in mixed space) ===",
                        )

                # DDP re-sync after requires_grad toggle
                if is_distributed and new_phase != "phase_4":
                    dist.barrier()
                    with torch.no_grad():
                        for online_p, target_p in ema_pairs:
                            dist.broadcast(online_p.data, src=0)
                            dist.broadcast(target_p.data, src=0)
                    dist.barrier()

                current_phase = new_phase

            if is_main and global_step % log_every == 0:
                lr_enc = scheduler.get_last_lr()[0]
                lr_pred = scheduler.get_last_lr()[1]
                elapsed = time.time() - t0
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                logger.info(
                    "Step %d/%d [%s] | loss=%.4f | lr_enc=%.1e lr_pred=%.1e | "
                    "VRAM=%.1fG | %.1f min",
                    global_step, total_steps, current_phase, loss.item() * grad_accum,
                    lr_enc, lr_pred, mem_gb, elapsed / 60,
                )

            if global_step % 50 == 0:
                if is_distributed:
                    dist.barrier()
                if is_main:
                    try:
                        save_resume_state_ablation(
                            save_dir=save_dir, raw_online=raw_online, raw_pred=raw_pred,
                            optimizer=optimizer, scheduler=scheduler,
                            global_step=global_step, micro_step=micro_step,
                            epoch=epoch, micro_step_in_epoch=micro_step_in_epoch_counter,
                            current_phase=current_phase,
                            last_phase_2b_val_loss=last_phase_2b_val_loss,
                            last_phase_4_val_loss=last_phase_4_val_loss,
                            best_val_loss=best_val_loss, config=config,
                        )
                    except Exception as save_err:
                        logger.error("periodic save_resume_state_ablation failed: %s", save_err)
                if is_distributed:
                    dist.barrier()

            if global_step == 25:
                if is_distributed:
                    dist.barrier()
                if is_main:
                    logger.info("=== EARLY FORCE-SAVE at step 25 (first resume checkpoint) ===")
                    try:
                        save_resume_state_ablation(
                            save_dir=save_dir, raw_online=raw_online, raw_pred=raw_pred,
                            optimizer=optimizer, scheduler=scheduler,
                            global_step=global_step, micro_step=micro_step,
                            epoch=epoch, micro_step_in_epoch=micro_step_in_epoch_counter,
                            current_phase=current_phase,
                            last_phase_2b_val_loss=last_phase_2b_val_loss,
                            last_phase_4_val_loss=last_phase_4_val_loss,
                            best_val_loss=best_val_loss, config=config,
                        )
                        logger.info("Early force-save complete")
                    except Exception as _early_save_err:
                        logger.error("Early force-save FAILED: %s", _early_save_err)
                if is_distributed:
                    dist.barrier()

            if global_step % eval_every == 0:
                if is_distributed:
                    dist.barrier()

                if is_main:
                    val_metrics = None
                    eval_failed = False
                    try:
                        use_target = current_phase in ("phase_2a", "phase_2b")
                        val_metrics = evaluate_ablation(
                            raw_online, raw_pred, tokenizer, val_loader, device,
                            max_seq_len, eval_max_windows, use_target=use_target,
                        )
                    except Exception as eval_err:
                        logger.error("=" * 60)
                        logger.error(
                            "EVAL FAILED at step %d [%s]: %s: %s",
                            global_step, current_phase,
                            type(eval_err).__name__, eval_err,
                        )
                        logger.error("=" * 60)
                        global _PREEMPTED
                        _PREEMPTED = True
                        eval_failed = True

                    if not eval_failed and val_metrics is not None:
                        val_loss = val_metrics["val_loss"]
                        z_std = val_metrics["z_std"]

                        logger.info(
                            "  EVAL [%s] | val_loss=%.4f | z_std=%.4f | best=%.4f",
                            current_phase, val_loss, z_std, best_val_loss,
                        )

                        # Optional z_std abort (disabled by default for this ablation)
                        if z_std_abort_threshold > 0 and z_std < z_std_abort_threshold \
                                and global_step > 100:
                            logger.warning(
                                "z_std=%.4f below threshold %.4f — aborting",
                                z_std, z_std_abort_threshold,
                            )
                            try:
                                save_ablation_checkpoint(
                                    raw_online, raw_pred, tokenizer, save_dir,
                                    tag="collapsed", step=global_step,
                                    phase=f"{current_phase}_collapsed",
                                    val_loss=val_loss, z_std=z_std,
                                    elapsed_seconds=time.time() - t0,
                                )
                            except Exception as save_err:
                                logger.error("save collapsed checkpoint failed: %s", save_err)
                            _PREEMPTED = True

                        if current_phase == "phase_2b":
                            last_phase_2b_val_loss = val_loss
                        if current_phase == "phase_4":
                            last_phase_4_val_loss = val_loss

                        try:
                            save_ablation_checkpoint(
                                raw_online, raw_pred, tokenizer, save_dir,
                                tag="latest", step=global_step, phase=current_phase,
                                val_loss=val_loss, z_std=z_std,
                                elapsed_seconds=time.time() - t0,
                            )
                        except Exception as save_err:
                            logger.error("save latest failed: %s", save_err)

                        if val_loss < best_val_loss and save_cfg.get("save_best", True):
                            best_val_loss = val_loss
                            try:
                                save_ablation_checkpoint(
                                    raw_online, raw_pred, tokenizer, save_dir,
                                    tag="best", step=global_step, phase=current_phase,
                                    val_loss=val_loss, z_std=z_std,
                                    elapsed_seconds=time.time() - t0,
                                )
                                logger.info("  -> saved best (val=%.4f)", val_loss)
                            except Exception as save_err:
                                logger.error("save best failed: %s", save_err)

                    try:
                        save_resume_state_ablation(
                            save_dir=save_dir, raw_online=raw_online, raw_pred=raw_pred,
                            optimizer=optimizer, scheduler=scheduler,
                            global_step=global_step, micro_step=micro_step,
                            epoch=epoch, micro_step_in_epoch=micro_step_in_epoch_counter,
                            current_phase=current_phase,
                            last_phase_2b_val_loss=last_phase_2b_val_loss,
                            last_phase_4_val_loss=last_phase_4_val_loss,
                            best_val_loss=best_val_loss, config=config,
                        )
                    except Exception as save_err:
                        logger.error("save_resume_state_ablation failed: %s", save_err)

                    raw_online.train()
                    raw_pred.train()

                if is_distributed:
                    dist.barrier()

            if args.max_global_steps is not None and global_step >= args.max_global_steps:
                if is_distributed:
                    dist.barrier()
                if is_main:
                    logger.warning(
                        "=== max_global_steps=%d reached [%s] — saving resume.pt and exiting ===",
                        args.max_global_steps, current_phase,
                    )
                    save_resume_state_ablation(
                        save_dir=save_dir, raw_online=raw_online, raw_pred=raw_pred,
                        optimizer=optimizer, scheduler=scheduler,
                        global_step=global_step, micro_step=micro_step,
                        epoch=epoch, micro_step_in_epoch=micro_step_in_epoch_counter,
                        current_phase=current_phase,
                        last_phase_2b_val_loss=last_phase_2b_val_loss,
                        last_phase_4_val_loss=last_phase_4_val_loss,
                        best_val_loss=best_val_loss, config=config,
                    )
                if is_distributed:
                    dist.barrier()
                cleanup_distributed(is_distributed)
                sys.exit(0)

            if _PREEMPTED:
                if is_distributed:
                    dist.barrier()
                if is_main:
                    logger.warning(
                        "=== PREEMPTION caught at step %d [%s] — saving resume.pt ===",
                        global_step, current_phase,
                    )
                    save_resume_state_ablation(
                        save_dir=save_dir, raw_online=raw_online, raw_pred=raw_pred,
                        optimizer=optimizer, scheduler=scheduler,
                        global_step=global_step, micro_step=micro_step,
                        epoch=epoch, micro_step_in_epoch=micro_step_in_epoch_counter,
                        current_phase=current_phase,
                        last_phase_2b_val_loss=last_phase_2b_val_loss,
                        last_phase_4_val_loss=last_phase_4_val_loss,
                        best_val_loss=best_val_loss, config=config,
                    )
                    logger.warning(
                        "Preemption save complete. Restart with same job to auto-resume.",
                    )
                if is_distributed:
                    dist.barrier()
                cleanup_distributed(is_distributed)
                sys.exit(0)

    if is_main:
        save_ablation_checkpoint(
            raw_online, raw_pred, tokenizer, save_dir,
            tag="final", step=global_step, phase=current_phase,
            val_loss=best_val_loss, z_std=0.0,
            elapsed_seconds=time.time() - t0,
        )
        # log final online-target gap
        final_max_gap = 0.0
        for online_p, target_p in ema_pairs:
            gap = (online_p.float() - target_p).abs().max().item()
            final_max_gap = max(final_max_gap, gap)
        meta = {
            "ablation": "no_alignment",
            "total_steps": global_step,
            "best_val_loss": best_val_loss,
            "final_online_target_max_gap": final_max_gap,
            "elapsed_seconds": time.time() - t0,
            "phase_steps": phase_steps,
            "config": config,
        }
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(
            "Clin-JEPA w/o alignment complete. Best val: %.4f, %d steps, %.1f min, "
            "final online-target gap: %.2e",
            best_val_loss, global_step, (time.time() - t0) / 60, final_max_gap,
        )

    cleanup_distributed(is_distributed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clin-JEPA w/o alignment: phase_2b + phase_3 hard sync removed",
    )
    parser.add_argument("--config", required=True,
                        help="Path to configs/train/ablation_no_alignment.yaml")
    parser.add_argument("--max_windows", type=int, default=None, help="Limit windows (smoke)")
    parser.add_argument("--smoke", action="store_true", help="Use scaled-down phase steps")
    parser.add_argument("--smoke_phase_steps", type=int, default=None,
                        help="Override smoke phase step count")
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to resume.pt (default: auto-detect save_dir/resume.pt). "
                             "Pass 'none' to force training from scratch.")
    parser.add_argument("--max_global_steps", type=int, default=None,
                        help="If set, exit cleanly after this many opt steps")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override config save.dir")
    parser.add_argument("--per_gpu_batch_size", type=int, default=None,
                        help="Override per_gpu_batch_size from config")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Override grad_accum from config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    hf_cfg = config.get("hf", {})
    if hf_cfg.get("cache_dir"):
        os.environ["HF_HOME"] = hf_cfg["cache_dir"]
    if hf_cfg.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    train(config, args)


if __name__ == "__main__":
    main()
