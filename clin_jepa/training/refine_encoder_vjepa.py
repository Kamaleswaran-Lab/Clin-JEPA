"""V-JEPA 2-AC baseline: random-mask JEPA encoder refinement.

Refines the SFT-initialised encoder via masked temporal JEPA prediction —
the encoder-side stage of the V-JEPA 2-AC baseline (paper §5.2).

* The **online encoder** is Qwen3-8B + LoRA, warm-started from the SFT
  initialisation produced by :mod:`clin_jepa.training.encoder_sft`.
* The **target encoder** is an fp32 EMA copy held in a second PEFT
  adapter alongside the online adapter (see
  :mod:`clin_jepa.training.encoder_ops`).
* The **JEPA predictor** is a disposable 16.9M bidirectional
  cross-attention decoder
  (:class:`~clin_jepa.model.jepa_predictor.JEPAPredictor`) that predicts
  encoder embeddings at masked positions. Loss is L1 against the target
  encoder's embeddings.

After training, only the refined online encoder LoRA is kept; the JEPA
predictor is discarded. The downstream predictor for the V-JEPA 2-AC
baseline is then trained on cached embeddings produced from this
refined encoder (see
:mod:`clin_jepa.training.train_predictor_on_frozen`).

Usage::

    # Single GPU smoke (~50 windows, no DDP):
    python -m clin_jepa.training.refine_encoder_vjepa \\
        --config configs/train/refine_encoder_vjepa.yaml \\
        --max_windows 50 --eval_every 10 --log_every 5

    # Multi-GPU production (4x H200 DDP):
    torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \\
        -m clin_jepa.training.refine_encoder_vjepa \\
        --config configs/train/refine_encoder_vjepa.yaml
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from clin_jepa.model.jepa_predictor import JEPAPredictor
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
# Preemption signal handling (SIGUSR1 from SLURM, SIGTERM on cancel)
# ---------------------------------------------------------------------------
# preemption flag set by signal handler
_PREEMPTED: bool = False


def _preempt_signal_handler(signum: int, frame) -> None:  # noqa: ARG001
    """Set preemption flag."""
    global _PREEMPTED
    _PREEMPTED = True


# ---------------------------------------------------------------------------
# DDP setup
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
# Symmetric masking
# ---------------------------------------------------------------------------

def apply_symmetric_mask(
    length: int,
    mask_ratio: float = 0.40,
    min_visible: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], list[int]]:
    """Randomly mask `mask_ratio` fraction of timesteps, keeping at least
    min_visible timesteps visible. Both state and action at a masked timestep
    are hidden together (symmetric).

    Returns:
        (visible_timesteps, masked_timesteps) — both sorted ascending.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_mask = int(round(mask_ratio * length))
    n_mask = min(n_mask, length - min_visible)  # at least min_visible visible
    n_mask = max(n_mask, 0)

    if n_mask == 0:
        return list(range(length)), []

    masked = rng.choice(length, size=n_mask, replace=False)
    masked_set = set(int(x) for x in masked)
    visible = [i for i in range(length) if i not in masked_set]
    return visible, sorted(masked_set)


# ---------------------------------------------------------------------------
# Position id layout for predictor
# ---------------------------------------------------------------------------
# Position 0     : demographics
# Position 1+2t  : state at hour t (odd positions)
# Position 2+2t  : action at hour t (even positions)
# So total positions for length L = 1 + 2L

def _state_pos(t: int) -> int:
    return 1 + 2 * t


def _action_pos(t: int) -> int:
    return 2 + 2 * t


# ---------------------------------------------------------------------------
# Per-window forward + loss
# ---------------------------------------------------------------------------

def encode_window_texts(
    raw_online: torch.nn.Module,
    tokenizer,
    window: dict,
    device: torch.device,
    max_seq_len: int,
    train_mode: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Encode all (demo + L state + L action) = 2L+1 texts of one window.

    Returns:
        (2L+1, 4096) tensor. Layout:
            [0]        demo
            [1..L]     state_0, state_1, ..., state_{L-1}
            [L+1..2L]  action_0, action_1, ..., action_{L-1}
    """
    L = window["length"]
    all_texts = (
        [window["demographics_text"]]
        + window["state_texts"]
        + window["action_texts"]
    )
    z = encode_texts_batched(
        raw_online, tokenizer, all_texts, device=device,
        max_seq_len=max_seq_len, train_mode=train_mode, output_dtype=output_dtype,
    )
    return z  # (2L+1, 4096)


def compute_window_loss(
    raw_online: torch.nn.Module,    # raw PEFT model, for set_adapter + ALL forwards
    raw_jepa: nn.Module,             # raw JEPA predictor, for forward
    tokenizer,
    window: dict,
    device: torch.device,
    max_seq_len: int,
    rng: np.random.Generator,
    mask_ratio: float = 0.40,
    min_visible: int = 2,
) -> Optional[torch.Tensor]:
    """Compute the JEPA L1 loss for a single trajectory window.

    Both encoder and JEPA predictor are raw (not DDP-wrapped); grads synced
    via manual_allreduce_grads.

    Steps:
      1. Choose visible/masked timesteps via symmetric_mask.
      2. Encode all 2L+1 texts via online encoder (gradients flow, raw).
      3. Encode all 2L+1 texts via target encoder (no_grad, fp32).
      4. Build memory (visible state+action+demo from online).
      5. Build queries (masked state+action position ids).
      6. JEPA predictor forward → predicted (Q, 4096).
      7. Loss = L1 sum-over-feature, mean over Q queries.
         Targets are from target encoder, .detach()'d.

    Returns None if the window has fewer than 1 masked timestep (skip).
    """
    L = window["length"]
    visible, masked = apply_symmetric_mask(L, mask_ratio, min_visible, rng)

    if not masked:
        return None  # skip windows with nothing to predict

    # --- ENCODE: online (trainable, RAW model — manual allreduce later) ---
    raw_online.set_adapter("online")
    z_online = encode_window_texts(
        raw_online, tokenizer, window, device,
        max_seq_len=max_seq_len, train_mode=True,
        output_dtype=torch.bfloat16,
    )  # (2L+1, 4096) bf16

    # --- ENCODE: target (no_grad, fp32, raw) ---
    raw_online.set_adapter("target")
    with torch.no_grad():
        z_target = encode_window_texts(
            raw_online, tokenizer, window, device,
            max_seq_len=max_seq_len, train_mode=False,
            output_dtype=torch.float32,
        )  # (2L+1, 4096) fp32
    raw_online.set_adapter("online")  # back to default for the next window

    DEMO_IDX = 0
    state_slice  = lambda t: 1 + t           # state_t at index 1+t
    action_slice = lambda t: 1 + L + t       # action_t at index 1+L+t

    # --- Build memory: demo + visible state + visible action ---
    memory_emb_list: list[torch.Tensor] = [z_online[DEMO_IDX]]
    memory_pos_list: list[int] = [0]  # demo at predictor pos 0

    for t in visible:
        memory_emb_list.append(z_online[state_slice(t)])
        memory_pos_list.append(_state_pos(t))
        memory_emb_list.append(z_online[action_slice(t)])
        memory_pos_list.append(_action_pos(t))

    memory_emb = torch.stack(memory_emb_list).unsqueeze(0)         # (1, V, 4096)
    memory_pos = torch.tensor(memory_pos_list, device=device,
                              dtype=torch.long).unsqueeze(0)       # (1, V)

    query_pos_list: list[int] = []
    target_emb_list: list[torch.Tensor] = []

    for t in masked:
        query_pos_list.append(_state_pos(t))
        target_emb_list.append(z_target[state_slice(t)])

        query_pos_list.append(_action_pos(t))
        target_emb_list.append(z_target[action_slice(t)])

    query_pos = torch.tensor(query_pos_list, device=device,
                             dtype=torch.long).unsqueeze(0)        # (1, Q)
    targets = torch.stack(target_emb_list).unsqueeze(0)            # (1, Q, 4096) fp32

    # --- JEPA predictor forward (RAW model — manual allreduce later) ---
    predicted = raw_jepa(memory_emb, memory_pos, query_pos)  # (1, Q, 4096)

    # --- L1 loss: sum over feature dim, mean over queries ---
    loss = (predicted.float() - targets.detach().float()).abs().sum(dim=-1).mean()

    return loss


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    raw_online: torch.nn.Module,
    raw_jepa: nn.Module,
    tokenizer,
    val_loader: DataLoader,
    device: torch.device,
    max_seq_len: int,
    max_windows: int,
    mask_ratio: float,
    min_visible: int,
    seed: int = 0,
) -> dict:
    """Compute validation L1 loss + z_std collapse monitor.

    Uses a deterministic RNG so the masking pattern is the same across
    eval calls (so val loss is comparable step-to-step).
    """
    raw_jepa.eval()
    rng = np.random.default_rng(seed)
    total_loss = 0.0
    n_windows = 0

    z_std_samples: list[torch.Tensor] = []

    for collated in val_loader:
        for window in collated["batch"]:
            if n_windows >= max_windows:
                break

            # Force per-window to use no-grad inside compute_window_loss too
            L = window["length"]
            visible, masked = apply_symmetric_mask(L, mask_ratio, min_visible, rng)
            if not masked:
                continue

            # Online encoder forward (no_grad here too — eval)
            raw_online.set_adapter("online")
            z_online = encode_window_texts(
                raw_online, tokenizer, window, device,
                max_seq_len=max_seq_len, train_mode=False, output_dtype=torch.bfloat16,
            )

            raw_online.set_adapter("target")
            z_target = encode_window_texts(
                raw_online, tokenizer, window, device,
                max_seq_len=max_seq_len, train_mode=False, output_dtype=torch.float32,
            )
            raw_online.set_adapter("online")

            # z_std for collapse monitoring (use online state embeddings)
            z_state_online = z_online[1:1+L].float()  # (L, 4096)
            z_std_samples.append(z_state_online.std(dim=0).mean().detach())

            # Build memory + queries (same as training)
            memory_emb_list = [z_online[0]]
            memory_pos_list = [0]
            for t in visible:
                memory_emb_list.append(z_online[1 + t])
                memory_pos_list.append(_state_pos(t))
                memory_emb_list.append(z_online[1 + L + t])
                memory_pos_list.append(_action_pos(t))
            memory_emb = torch.stack(memory_emb_list).unsqueeze(0)
            memory_pos = torch.tensor(memory_pos_list, device=device,
                                      dtype=torch.long).unsqueeze(0)

            query_pos_list = []
            target_emb_list = []
            for t in masked:
                query_pos_list.append(_state_pos(t))
                target_emb_list.append(z_target[1 + t])
                query_pos_list.append(_action_pos(t))
                target_emb_list.append(z_target[1 + L + t])
            query_pos = torch.tensor(query_pos_list, device=device,
                                     dtype=torch.long).unsqueeze(0)
            targets = torch.stack(target_emb_list).unsqueeze(0)

            predicted = raw_jepa(memory_emb, memory_pos, query_pos)
            loss = (predicted.float() - targets.float()).abs().sum(dim=-1).mean()

            total_loss += loss.item()
            n_windows += 1

        if n_windows >= max_windows:
            break

    raw_jepa.train()

    avg_loss = total_loss / max(n_windows, 1)
    avg_z_std = torch.stack(z_std_samples).mean().item() if z_std_samples else 0.0

    return {
        "val_loss": avg_loss,
        "z_std": avg_z_std,
        "n_windows": n_windows,
    }


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_checkpoint(
    raw_online: torch.nn.Module,
    raw_jepa: nn.Module,
    tokenizer,
    save_dir: Path,
    tag: str,
    step: int,
    val_loss: float,
    z_std: float,
    elapsed_seconds: float,
) -> None:
    """Save online LoRA + JEPA predictor + meta to save_dir/checkpoints/{tag}/.

    PEFT save layout (PEFT 0.18+):
      out/                          ← `out` passed to save_pretrained
      ├── online/                   ← PEFT auto-creates this subdir because
      │                               adapter_name != "default"
      │   ├── adapter_config.json
      │   ├── adapter_model.safetensors
      │   └── (tokenizer files saved here too)
      ├── jepa_predictor.pt
      └── meta.json

    the embedding-precompute stage then reads `out/online/` directly via PeftModel.from_pretrained.
    """
    out = save_dir / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)

    raw_online.set_adapter("online")
    raw_online.save_pretrained(out, selected_adapters=["online"])

    online_dir = out / "online"
    tokenizer.save_pretrained(online_dir)

    # Save JEPA predictor
    torch.save(raw_jepa.state_dict(), out / "jepa_predictor.pt")

    # Save meta
    meta = {
        "tag": tag,
        "step": step,
        "val_loss": val_loss,
        "z_std": z_std,
        "elapsed_seconds": elapsed_seconds,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Resume state save / load
# ---------------------------------------------------------------------------

def save_resume_state(
    save_dir: Path,
    raw_online: torch.nn.Module,
    raw_jepa: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    micro_step: int,
    epoch: int,
    micro_step_in_epoch: int,
    best_val_loss: float,
    patience_counter: int,
    config: dict,
) -> None:
    """Save full training state to save_dir/resume.pt for crash/preempt recovery.

    Captures:
      - online + target LoRA adapter weights (fp32 target preserved exactly)
      - JEPA predictor state_dict
      - optimizer + scheduler state
      - global_step (counts opt steps), micro_step (counts micro-batches),
        epoch, micro_step_in_epoch (= next batch_idx to process), best_val_loss,
        patience_counter
      - RNG state (torch CPU + CUDA + numpy + python)
      - config snapshot for sanity check on resume
    """
    state = {
        "schema_version":      "clin_jepa_vjepa_refine_v1",
        "lora_adapter_state":  collect_lora_adapter_state(raw_online),
        "jepa_predictor_state": raw_jepa.state_dict(),
        "optimizer_state":     optimizer.state_dict(),
        "scheduler_state":     scheduler.state_dict(),
        "global_step":         global_step,
        "micro_step":          micro_step,
        "epoch":               epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "best_val_loss":       best_val_loss,
        "patience_counter":    patience_counter,
        "rng_state":           snapshot_rng_state(),
        "config_snapshot":     config,
    }
    atomic_torch_save(state, save_dir / "resume.pt")


def load_resume_state(
    resume_path: Path,
    raw_online: torch.nn.Module,
    raw_jepa: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> dict:
    """Load resume.pt and restore mutable training state in place.

    Restores LoRA adapters, predictor, optimizer, scheduler, RNG.
    Returns a small dict the caller uses to overwrite local variables:
      {global_step, micro_step, epoch, best_val_loss, patience_counter}

    Raises if the schema_version doesn't match — refuses to silently load
    an incompatible old snapshot.
    """
    state = torch.load(resume_path, map_location="cpu", weights_only=False)
    sv = state.get("schema_version")
    if sv != "clin_jepa_vjepa_refine_v1":
        raise RuntimeError(
            f"resume.pt schema_version {sv!r} does not match expected "
            f"'clin_jepa_vjepa_refine_v1'. Refuse to load."
        )

    restore_lora_adapter_state(raw_online, state["lora_adapter_state"])

    raw_jepa.load_state_dict(state["jepa_predictor_state"])
    raw_jepa.to(device)

    optimizer.load_state_dict(state["optimizer_state"])
    # Move optimizer state tensors to the current device
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)

    scheduler.load_state_dict(state["scheduler_state"])
    restore_rng_state(state["rng_state"])

    return {
        "global_step":         int(state["global_step"]),
        "micro_step":          int(state["micro_step"]),
        "epoch":               int(state["epoch"]),
        "micro_step_in_epoch": int(state.get("micro_step_in_epoch", 0)),
        "best_val_loss":       float(state["best_val_loss"]),
        "patience_counter":    int(state.get("patience_counter", 0)),
    }


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(config: dict, args: argparse.Namespace) -> None:
    rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    if is_main:
        setup_logging()
        logger.info("V-JEPA 2-AC encoder refinement (%d GPUs)", world_size)

    # Seed
    seed = config["training"]["seed"]
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    # --- Encoder warm-start ---
    online, tokenizer = load_sft_initialized_encoder(config, device)
    online = create_target_encoder_fp32(online)

    if is_main:
        verify_ema_working(online, n_updates=10)
    if is_distributed:
        dist.barrier()

    # Cache online/target param pairs once for fast EMA access
    ema_pairs = _build_online_target_param_pairs(online)

    # --- JEPA predictor ---
    jp_cfg = config["jepa_predictor"]
    jepa_pred = JEPAPredictor(
        embed_dim=jp_cfg.get("embed_dim", 4096),
        hidden_dim=jp_cfg.get("hidden_dim", 512),
        num_layers=jp_cfg.get("num_layers", 3),
        num_heads=jp_cfg.get("num_heads", 8),
        ffn_dim=jp_cfg.get("ffn_dim", 2048),
        dropout=jp_cfg.get("dropout", 0.1),
        max_position_id=jp_cfg.get("max_position_id", 256),
    ).to(device)

    if is_main:
        logger.info(
            "JEPA predictor: %.1fM params (target 16.9M)",
            jepa_pred.num_params() / 1e6,
        )

    if is_distributed:
        with torch.no_grad():
            for p in jepa_pred.parameters():
                dist.broadcast(p.data, src=0)
        if is_main:
            logger.info("Broadcast JEPA predictor weights from rank 0 to %d ranks", world_size)

    raw_online = online
    raw_jepa = jepa_pred

    # --- Datasets + loaders ---
    data_cfg = config["data"]

    train_ds = TrajectoryWindowDataset(
        shard_dir=PROJECT_ROOT / data_cfg["train_dir"],
        split="train",
        max_windows=args.max_windows,
    )
    val_ds = TrajectoryWindowDataset(
        shard_dir=PROJECT_ROOT / data_cfg["val_dir"],
        split="val",
        max_windows=config["logging"].get("val_windows", 2000),
    )

    train_cfg = config["training"]
    per_gpu_batch = train_cfg["per_gpu_batch_size"]
    grad_accum = train_cfg.get("grad_accum", 1)
    # CLI overrides for batch tuning experiments
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
        train_ds,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=trajectory_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_gpu_batch,
        shuffle=False,
        collate_fn=trajectory_collate_fn,
        num_workers=2,
    )

    # --- Optimizer (separate lr groups for encoder LoRA and JEPA predictor) ---
    encoder_params = [online_p for online_p, _ in ema_pairs]
    predictor_params = list(raw_jepa.parameters())

    wd = train_cfg["weight_decay"]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params,   "lr": train_cfg["lr_encoder"],   "weight_decay": wd},
        {"params": predictor_params, "lr": train_cfg["lr_predictor"], "weight_decay": wd},
    ])

    total_micro_steps = len(train_loader) * train_cfg["epochs"]
    total_steps = total_micro_steps // grad_accum
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training settings ---
    log_cfg = config.get("logging", {})
    log_every = log_cfg.get("log_every", 25)
    eval_every = log_cfg.get("eval_every", 250)
    eval_max_windows = log_cfg.get("eval_max_windows", 500)

    # CLI overrides for smoke
    if args.eval_every is not None:
        eval_every = args.eval_every
    if args.log_every is not None:
        log_every = args.log_every
    if args.eval_max_windows is not None:
        eval_max_windows = args.eval_max_windows

    save_cfg = config["save"]
    save_dir = PROJECT_ROOT / save_cfg["dir"]
    if args.save_dir is not None:
        # CLI override (used by resume test to avoid polluting production save_dir)
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Encoder collapse abort threshold (JEPA failure mode) ---
    z_std_abort_threshold = float(log_cfg.get("z_std_abort_threshold", 0.05))

    # --- Early stopping config (config holds defaults, CLI overrides) ---
    es_cfg = log_cfg.get("early_stop", {}) or {}
    early_stop_enabled  = es_cfg.get("enabled", True)
    early_stop_patience = int(es_cfg.get("patience", 5))
    early_stop_min_steps = int(es_cfg.get("min_steps", 500))
    early_stop_min_delta = float(es_cfg.get("min_delta", 0.001))
    if args.no_early_stop:
        early_stop_enabled = False
    if args.early_stop_patience is not None:
        early_stop_patience = args.early_stop_patience
    if args.early_stop_min_steps is not None:
        early_stop_min_steps = args.early_stop_min_steps
    if args.early_stop_min_delta is not None:
        early_stop_min_delta = args.early_stop_min_delta

    grad_clip_enc = train_cfg["gradient_clip_encoder"]
    grad_clip_pred = train_cfg["gradient_clip_predictor"]
    mask_ratio = train_cfg["mask_ratio"]
    min_visible = train_cfg.get("min_visible", 2)
    tau_init = train_cfg["tau_init"]
    tau_final = train_cfg["tau_final"]
    max_seq_len = data_cfg["max_seq_len"]

    if is_main:
        logger.info(
            "Training: %d epochs, %d steps/epoch, %d total steps",
            train_cfg["epochs"], len(train_loader), total_steps,
        )
        logger.info(
            "Warmup: %d steps, lr_enc=%.0e lr_pred=%.0e, wd=%.2f",
            warmup_steps, train_cfg["lr_encoder"], train_cfg["lr_predictor"], wd,
        )
        logger.info(
            "Mask ratio: %.2f, tau init->final: %.4f -> %.4f",
            mask_ratio, tau_init, tau_final,
        )
        if early_stop_enabled:
            logger.info(
                "Early stopping: enabled, patience=%d, min_steps=%d, min_delta=%.4f",
                early_stop_patience, early_stop_min_steps, early_stop_min_delta,
            )
        else:
            logger.info("Early stopping: disabled")

    best_val_loss = float("inf")
    global_step = 0
    micro_step = 0
    start_epoch = 0
    start_micro_in_epoch = 0
    patience_counter = 0

    # Auto-detect resume.pt unless --resume_from is explicitly given (or "none").
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
        loaded = load_resume_state(
            resume_path, raw_online, raw_jepa, optimizer, scheduler, device,
        )
        global_step          = loaded["global_step"]
        micro_step           = loaded["micro_step"]
        start_epoch          = loaded["epoch"]
        start_micro_in_epoch = loaded["micro_step_in_epoch"]
        best_val_loss        = loaded["best_val_loss"]
        patience_counter     = loaded["patience_counter"]
        if is_main:
            logger.info(
                "  resumed: epoch=%d, micro_in_epoch=%d, global_step=%d, "
                "best_val=%.4f, patience=%d",
                start_epoch, start_micro_in_epoch, global_step,
                best_val_loss, patience_counter,
            )
        if is_distributed:
            dist.barrier()

    # --- Register preemption signal handlers (after model+state setup) ---
    signal.signal(signal.SIGUSR1, _preempt_signal_handler)
    signal.signal(signal.SIGTERM, _preempt_signal_handler)
    if is_main:
        logger.info("Registered SIGUSR1/SIGTERM handlers for preemption recovery")

    # --- Training loop ---
    raw_online.train()
    raw_jepa.train()

    t0 = time.time()
    rng = np.random.default_rng(seed + rank)
    epoch_loss_sum = 0.0
    epoch_n_windows = 0
    should_stop = False  # set by early stopping or preemption

    for epoch in range(start_epoch, train_cfg["epochs"]):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Skip-ahead on resume (only for the FIRST epoch after resume)
        skip_n = start_micro_in_epoch if epoch == start_epoch else 0
        if skip_n > 0 and is_main:
            logger.info("  resume: skipping first %d micro-batches in epoch %d", skip_n, epoch)

        for batch_idx, collated in enumerate(train_loader):
            if batch_idx < skip_n:
                continue
            # Per-window loss accumulation
            batch_losses: list[torch.Tensor] = []

            for window in collated["batch"]:
                loss_w = compute_window_loss(
                    raw_online=raw_online,    # raw PEFT model (no DDP)
                    raw_jepa=raw_jepa,         # raw JEPA predictor (no DDP)
                    tokenizer=tokenizer,
                    window=window,
                    device=device,
                    max_seq_len=max_seq_len,
                    rng=rng,
                    mask_ratio=mask_ratio,
                    min_visible=min_visible,
                )
                if loss_w is not None:
                    batch_losses.append(loss_w)

            if not batch_losses:
                continue

            # Average loss across windows in this micro-batch, then divide by grad_accum
            micro_loss = torch.stack(batch_losses).mean() / grad_accum
            micro_loss.backward()

            epoch_loss_sum += micro_loss.item() * grad_accum * len(batch_losses)
            epoch_n_windows += len(batch_losses)

            micro_step += 1

            # Optimizer step every `grad_accum` micro-batches
            if micro_step % grad_accum == 0:
                if is_distributed:
                    manual_allreduce_grads(encoder_params)
                    manual_allreduce_grads(predictor_params)

                torch.nn.utils.clip_grad_norm_(encoder_params, grad_clip_enc)
                torch.nn.utils.clip_grad_norm_(predictor_params, grad_clip_pred)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # τ annealing 0.996 → 1.0 linear over total optimizer steps
                tau = tau_init + (tau_final - tau_init) * (global_step / max(total_steps, 1))
                ema_update(online, tau=tau, pairs=ema_pairs)

                if global_step % 50 == 0:
                    if is_distributed:
                        dist.barrier()
                    if is_main:
                        try:
                            save_resume_state(
                                save_dir=save_dir,
                                raw_online=raw_online,
                                raw_jepa=raw_jepa,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                micro_step=micro_step,
                                epoch=epoch,
                                micro_step_in_epoch=batch_idx + 1,
                                best_val_loss=best_val_loss,
                                patience_counter=patience_counter,
                                config=config,
                            )
                        except Exception as save_err:
                            logger.error("periodic save_resume_state failed: %s", save_err)
                    if is_distributed:
                        dist.barrier()

                # --- Logging ---
                if is_main and global_step % log_every == 0:
                    avg_loss = epoch_loss_sum / max(epoch_n_windows, 1)
                    lr_enc = scheduler.get_last_lr()[0]
                    lr_pred = scheduler.get_last_lr()[1]
                    elapsed = time.time() - t0
                    mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                    logger.info(
                        "Step %d/%d | loss=%.4f | tau=%.4f | lr_enc=%.2e lr_pred=%.2e | "
                        "VRAM=%.1fG | %.1f min",
                        global_step, total_steps, avg_loss, tau, lr_enc, lr_pred,
                        mem_gb, elapsed / 60,
                    )

                if global_step == 25:
                    if is_distributed:
                        dist.barrier()
                    if is_main:
                        logger.info("=== EARLY FORCE-SAVE at step 25 (first resume checkpoint) ===")
                        try:
                            save_resume_state(
                                save_dir=save_dir,
                                raw_online=raw_online,
                                raw_jepa=raw_jepa,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                micro_step=micro_step,
                                epoch=epoch,
                                micro_step_in_epoch=batch_idx + 1,
                                best_val_loss=best_val_loss,
                                patience_counter=patience_counter,
                                config=config,
                            )
                            logger.info("Early force-save complete")
                        except Exception as _early_save_err:
                            logger.error("Early force-save FAILED: %s", _early_save_err)
                    if is_distributed:
                        dist.barrier()

                if global_step % eval_every == 0:
                    if is_distributed:
                        dist.barrier()  # rendezvous: all ranks pause here

                    if is_main:
                        val_metrics = None
                        eval_failed = False
                        try:
                            val_metrics = evaluate(
                                raw_online=raw_online,
                                raw_jepa=raw_jepa,
                                tokenizer=tokenizer,
                                val_loader=val_loader,
                                device=device,
                                max_seq_len=max_seq_len,
                                max_windows=eval_max_windows,
                                mask_ratio=mask_ratio,
                                min_visible=min_visible,
                                seed=seed,
                            )
                        except Exception as eval_err:
                            logger.error("=" * 60)
                            logger.error(
                                "EVAL FAILED at step %d: %s: %s",
                                global_step, type(eval_err).__name__, eval_err,
                            )
                            logger.error(
                                "Likely cause: DataLoader worker killed by SIGTERM "
                                "( preemption). Forcing _PREEMPTED=True and "
                                "saving resume.pt before exit.",
                            )
                            logger.error("=" * 60)
                            global _PREEMPTED
                            _PREEMPTED = True
                            eval_failed = True

                        if not eval_failed and val_metrics is not None:
                            val_loss = val_metrics["val_loss"]
                            z_std = val_metrics["z_std"]

                            logger.info(
                                "  EVAL | val_loss=%.4f | z_std=%.4f | best=%.4f",
                                val_loss, z_std, best_val_loss,
                            )

                            if z_std < z_std_abort_threshold:
                                logger.error("=" * 60)
                                logger.error("CRITICAL: ENCODER REPRESENTATION COLLAPSE")
                                logger.error(
                                    "z_std = %.6f < threshold %.4f at step %d",
                                    z_std, z_std_abort_threshold, global_step,
                                )
                                logger.error(
                                    "Aborting to save compute. Investigate encoder LoRA, "
                                    "EMA tau, lr, mask ratio.",
                                )
                                logger.error("=" * 60)
                                should_stop = True

                            # Always save latest
                            try:
                                save_checkpoint(
                                    raw_online, raw_jepa, tokenizer, save_dir,
                                    tag="latest", step=global_step, val_loss=val_loss, z_std=z_std,
                                    elapsed_seconds=time.time() - t0,
                                )
                            except Exception as save_err:
                                logger.error("save latest failed: %s", save_err)

                            improvement_threshold = best_val_loss * (1.0 - early_stop_min_delta)
                            if val_loss < improvement_threshold:
                                best_val_loss = val_loss
                                patience_counter = 0
                                if save_cfg.get("save_best", True):
                                    try:
                                        save_checkpoint(
                                            raw_online, raw_jepa, tokenizer, save_dir,
                                            tag="best", step=global_step,
                                            val_loss=val_loss, z_std=z_std,
                                            elapsed_seconds=time.time() - t0,
                                        )
                                        logger.info(
                                            "  -> saved best (val=%.4f, patience reset)", val_loss,
                                        )
                                    except Exception as save_err:
                                        logger.error("save best failed: %s", save_err)
                            else:
                                patience_counter += 1
                                logger.info(
                                    "  no improvement (val=%.4f >= %.4f) | patience %d/%d",
                                    val_loss, improvement_threshold,
                                    patience_counter, early_stop_patience,
                                )
                                if (
                                    early_stop_enabled
                                    and global_step >= early_stop_min_steps
                                    and patience_counter >= early_stop_patience
                                ):
                                    logger.info(
                                        "=== EARLY STOP at step %d (patience %d evals "
                                        "without improvement, best val=%.4f) ===",
                                        global_step, patience_counter, best_val_loss,
                                    )
                                    should_stop = True

                        try:
                            save_resume_state(
                                save_dir=save_dir,
                                raw_online=raw_online,
                                raw_jepa=raw_jepa,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                micro_step=micro_step,
                                epoch=epoch,
                                micro_step_in_epoch=batch_idx + 1,
                                best_val_loss=best_val_loss,
                                patience_counter=patience_counter,
                                config=config,
                            )
                        except Exception as save_err:
                            logger.error("save_resume_state failed: %s", save_err)

                        raw_online.train()
                        raw_jepa.train()

                    if is_distributed:
                        stop_tensor = torch.tensor(
                            [1 if should_stop else 0],
                            device=device, dtype=torch.long,
                        )
                        dist.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())
                        dist.barrier()  # rendezvous before resuming training

                if args.max_global_steps is not None and global_step >= args.max_global_steps:
                    if is_distributed:
                        dist.barrier()
                    if is_main:
                        logger.warning(
                            "=== max_global_steps=%d reached — saving resume.pt and exiting "
                            "(simulated preemption for resume testing) ===",
                            args.max_global_steps,
                        )
                        save_resume_state(
                            save_dir=save_dir,
                            raw_online=raw_online,
                            raw_jepa=raw_jepa,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            micro_step=micro_step,
                            epoch=epoch,
                            micro_step_in_epoch=batch_idx + 1,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter,
                            config=config,
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
                            "=== PREEMPTION caught at step %d — saving resume.pt ===",
                            global_step,
                        )
                        save_resume_state(
                            save_dir=save_dir,
                            raw_online=raw_online,
                            raw_jepa=raw_jepa,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            micro_step=micro_step,
                            epoch=epoch,
                            micro_step_in_epoch=batch_idx + 1,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter,
                            config=config,
                        )
                        logger.warning(
                            "Preemption save complete. Restart with same job to auto-resume.",
                        )
                    if is_distributed:
                        dist.barrier()
                    cleanup_distributed(is_distributed)
                    sys.exit(0)

            if should_stop:
                break  # break inner micro-batch loop

        if should_stop:
            break  # break outer epoch loop

        # End of epoch: log epoch summary
        if is_main:
            avg_epoch_loss = epoch_loss_sum / max(epoch_n_windows, 1)
            logger.info(
                "Epoch %d done | %d windows | avg_loss=%.4f",
                epoch, epoch_n_windows, avg_epoch_loss,
            )

    # --- Final save ---
    if is_main:
        save_checkpoint(
            raw_online, raw_jepa, tokenizer, save_dir,
            tag="final", step=global_step, val_loss=best_val_loss, z_std=0.0,
            elapsed_seconds=time.time() - t0,
        )

        # Training metadata
        meta = {
            "total_steps": global_step,
            "best_val_loss": best_val_loss,
            "elapsed_seconds": time.time() - t0,
            "config": config,
        }
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(
            "Training complete. Best val: %.4f, Total: %d steps, %.1f min",
            best_val_loss, global_step, (time.time() - t0) / 60,
        )

    cleanup_distributed(is_distributed)


def main() -> None:
    parser = argparse.ArgumentParser(description="V-JEPA 2-AC encoder refinement")
    parser.add_argument("--config", required=True, help="Path to refine_encoder_vjepa.yaml")
    parser.add_argument("--max_windows", type=int, default=None, help="Limit train+val windows (smoke)")
    parser.add_argument("--eval_every", type=int, default=None, help="Override eval cadence")
    parser.add_argument("--log_every", type=int, default=None, help="Override log cadence")
    parser.add_argument("--eval_max_windows", type=int, default=None,
                        help="Override eval_max_windows (smoke: use small value to keep eval fast)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to resume.pt (default: auto-detect save_dir/resume.pt). "
                             "Pass 'none' to force training from scratch even if resume.pt exists.")
    parser.add_argument("--max_global_steps", type=int, default=None,
                        help="If set, exit cleanly after this many opt steps "
                             "(saves resume.pt first). Used for resume testing.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override config save.dir (used by resume test to "
                             "avoid polluting production save_dir).")
    parser.add_argument("--per_gpu_batch_size", type=int, default=None,
                        help="Override per_gpu_batch_size from config (used to "
                             "test VRAM headroom without editing yaml).")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Override grad_accum from config.")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable patience-based early stopping (overrides config)")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Override early stopping patience (consecutive evals without improvement)")
    parser.add_argument("--early_stop_min_steps", type=int, default=None,
                        help="Override min global_step before early stopping can trigger")
    parser.add_argument("--early_stop_min_delta", type=float, default=None,
                        help="Override min relative val_loss improvement to count (e.g. 0.001 = 0.1%%)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set HF env vars
    hf_cfg = config.get("hf", {})
    if hf_cfg.get("cache_dir"):
        os.environ["HF_HOME"] = hf_cfg["cache_dir"]
    if hf_cfg.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    train(config, args)


if __name__ == "__main__":
    main()
