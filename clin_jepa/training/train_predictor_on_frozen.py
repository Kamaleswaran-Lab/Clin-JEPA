"""Train the AC Transformer predictor on cached, frozen-encoder embeddings.

Trains :class:`~clin_jepa.model.predictor.ACTransformerPredictor` (~92M
parameters) on the embedding shards produced by
:mod:`clin_jepa.evaluation.precompute_embeddings`. The encoder is not touched: this
script implements the predictor-training stage for two baselines:

* the **V-JEPA 2-AC baseline** — uses embeddings produced by the encoder
  after JEPA refinement (see
  :mod:`clin_jepa.training.refine_encoder_vjepa`).
* the **SFT baseline** — uses embeddings produced directly by the
  SFT-initialised encoder, without any JEPA refinement.

The two configurations differ only in the input ``embedding_dir``; the
training loop, optimizer settings, and architecture are identical.

Loss: teacher-forced L1 + 2-step rollout L1 (paper §4.1).
Optimizer: AdamW with cosine schedule, ``lr=5e-4``,
``weight_decay=0.04``, 10% warmup (paper Appendix B).
Embedding noise is applied to both state and action embeddings during
training.

Usage::

    python -m clin_jepa.training.train_predictor_on_frozen \\
        --config configs/train/predictor_vjepa2ac.yaml --gpu_id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.predictor_dataset import (
    PredictorDataset,
    predictor_collate_fn,
)
from clin_jepa.utils import setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def teacher_forcing_loss(
    preds: torch.Tensor,    # (B, T-1, 4096)
    targets: torch.Tensor,  # (B, T-1, 4096)
    mask: torch.Tensor,     # (B, T-1) bool, True=valid
) -> torch.Tensor:
    """L1 sum-over-feature, mean-over-valid."""
    l1 = (preds - targets).abs().sum(dim=-1)  # (B, T-1)
    loss = (l1 * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return loss


def compute_rollout_loss(
    model: ACTransformerPredictor,
    z_states: torch.Tensor,
    z_static: torch.Tensor,
    z_actions: torch.Tensor,
    lengths: torch.Tensor,
    rollout_horizon: int = 2,
) -> torch.Tensor:
    """Batched k-step autoregressive rollout loss with shared random k."""
    B, T_max, _ = z_states.shape
    device = z_states.device

    min_len = int(lengths.min().item())
    if min_len < rollout_horizon + 2:
        return torch.tensor(0.0, device=device)

    k = random.randint(1, min_len - rollout_horizon - 1)

    z_hist = z_states[:, :k, :].float()
    za_hist = z_actions[:, :k, :].float()
    z_current = z_states[:, k, :].float()

    total_loss = torch.tensor(0.0, device=device)

    for step in range(rollout_horizon):
        z_pred = model.forward_rollout_step(
            z_history=z_hist,
            z_static=z_static,
            za_history=za_hist,
            za_next=z_actions[:, k + step, :].float(),
            z_last=z_current,
        )

        z_target = z_states[:, k + step + 1, :].float()
        step_loss = (z_pred - z_target).abs().sum(dim=-1).mean()
        total_loss = total_loss + step_loss

        # extend history
        z_hist = torch.cat([z_hist, z_current.unsqueeze(1)], dim=1)
        za_hist = torch.cat([za_hist, z_actions[:, k + step:k + step + 1, :].float()], dim=1)
        z_current = z_pred  # autoregressive chain

    return total_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ACTransformerPredictor,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 100,
    zero_actions: bool = False,
) -> float:
    """Compute validation teacher forcing loss.

    Args:
        zero_actions: If True, set z_actions to torch.zeros (unconditional ablation).
            Default False preserves normal behavior.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break

        z_states = batch["z_states"].to(device)
        z_actions = batch["z_actions"].to(device)
        if zero_actions:
            z_actions = torch.zeros_like(z_actions)
        z_static = batch["z_static"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["padding_mask"].to(device)

        preds = model(z_states, z_static, z_actions, lengths)
        targets = z_states[:, 1:, :].float()
        loss_mask = mask[:, 1:]

        loss = teacher_forcing_loss(preds, targets, loss_mask)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict, args: argparse.Namespace) -> None:
    """Main training function (single GPU)."""
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    setup_logging()

    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Model ---
    model_cfg = config["model"]
    model = ACTransformerPredictor(
        state_dim=model_cfg.get("state_dim", 4096),
        action_dim=model_cfg.get("action_dim", 4096),
        static_dim=model_cfg.get("static_dim", 4096),
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        max_timesteps=model_cfg["max_timesteps"],
        ffn_dropout=model_cfg.get("ffn_dropout", 0.15),
        attn_dropout=model_cfg.get("attn_dropout", 0.1),
        prediction_mode=model_cfg.get("prediction_mode", "absolute"),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "AC Transformer Predictor: %s mode, %.1fM params, device=%s",
        model_cfg.get("prediction_mode", "absolute"), n_params / 1e6, device,
    )

    # --- Data ---
    data_cfg = config["data"]

    train_dataset = PredictorDataset(
        embedding_dir=PROJECT_ROOT / data_cfg["embedding_dir"],
        split="train",
        max_windows=args.max_windows,
    )
    val_dataset = PredictorDataset(
        embedding_dir=PROJECT_ROOT / data_cfg["embedding_dir"],
        split="val",
        max_windows=min(args.max_windows or 10000, 10000),
    )

    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=predictor_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=predictor_collate_fn,
        num_workers=2,
    )

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    rollout_horizon = train_cfg.get("rollout_horizon", 2)
    embedding_noise_std = train_cfg.get("embedding_noise_std", 0.0)

    zero_actions = train_cfg.get("zero_actions", False)
    if zero_actions:
        logger.warning("=" * 70)
        logger.warning("ZERO ACTIONS MODE — z_actions replaced with torch.zeros (4096D)")
        logger.warning("=" * 70)

    # Compute embedding std for noise scaling
    if embedding_noise_std > 0:
        sample_states = train_dataset[0]["z_states"].float()
        emb_std = sample_states.std().item()
        noise_std = embedding_noise_std * emb_std
        logger.info(
            "Embedding noise std: %.4f (%.2f × emb_std %.2f)",
            noise_std, embedding_noise_std, emb_std,
        )
    else:
        noise_std = 0.0

    # --- Logging + save ---
    log_cfg = config.get("logging", {})
    log_every = log_cfg.get("log_every", 50)

    save_cfg = config["save"]
    save_dir = PROJECT_ROOT / save_cfg["dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    logger.info(
        "Training: %d epochs, %d steps/epoch, %d total steps",
        train_cfg["epochs"], len(train_loader), total_steps,
    )
    logger.info(
        "Warmup: %d, lr=%s, wd=%s, batch=%d",
        warmup_steps, train_cfg["lr"], train_cfg["weight_decay"], batch_size,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 5)
    global_step = 0
    t0 = time.time()

    use_bf16 = train_cfg.get("precision", "bf16") == "bf16"
    model.train()

    for epoch in range(train_cfg["epochs"]):
        epoch_tf_loss = 0.0
        epoch_roll_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            z_states = batch["z_states"].to(device)
            z_actions = batch["z_actions"].to(device)
            if zero_actions:
                z_actions = torch.zeros_like(z_actions)
            z_static = batch["z_static"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["padding_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                preds = model(z_states, z_static, z_actions, lengths, noise_std=noise_std)
                targets = z_states[:, 1:, :].float()
                loss_mask = mask[:, 1:]
                tf_loss = teacher_forcing_loss(preds, targets, loss_mask)

                roll_loss = compute_rollout_loss(
                    model, z_states, z_static, z_actions, lengths, rollout_horizon,
                )

                loss = tf_loss + roll_loss

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_tf_loss += tf_loss.item()
            epoch_roll_loss += roll_loss.item() if isinstance(roll_loss, torch.Tensor) else roll_loss
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                avg_tf = epoch_tf_loss / epoch_steps
                avg_roll = epoch_roll_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                logger.info(
                    "Step %d/%d | tf=%.4f roll=%.4f total=%.4f | lr=%.2e | %.1f min",
                    global_step, total_steps, avg_tf, avg_roll, avg_tf + avg_roll,
                    lr, elapsed / 60,
                )

        # End of epoch
        avg_tf = epoch_tf_loss / max(epoch_steps, 1)
        avg_roll = epoch_roll_loss / max(epoch_steps, 1)
        logger.info(
            "Epoch %d/%d | tf=%.4f roll=%.4f | %d steps",
            epoch + 1, train_cfg["epochs"], avg_tf, avg_roll, epoch_steps,
        )

        # Validation
        val_loss = evaluate(model, val_loader, device, zero_actions=zero_actions)
        logger.info("  Val loss: %.4f (best: %.4f)", val_loss, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "config": config,
            }, save_dir / "best.pt")
            logger.info("  -> saved best (val=%.4f)", val_loss)
        else:
            patience_counter += 1
            logger.info("  No improvement (%d/%d)", patience_counter, patience)

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch + 1,
        "val_loss": val_loss,
        "config": config,
    }, save_dir / "final.pt")

    meta = {
        "total_steps": global_step,
        "total_epochs": epoch + 1,
        "final_train_tf_loss": avg_tf,
        "final_train_roll_loss": avg_roll,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": time.time() - t0,
        "n_params": n_params,
        "prediction_mode": model_cfg.get("prediction_mode", "absolute"),
        "config": config,
    }
    with open(save_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(
        "Training complete. Best val: %.4f, %.1f min",
        best_val_loss, (time.time() - t0) / 60,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="predictor training (V-JEPA 2-AC style / SFT baseline): AC predictor training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_windows", type=int, default=None, help="Smoke test limit")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args)


if __name__ == "__main__":
    main()
