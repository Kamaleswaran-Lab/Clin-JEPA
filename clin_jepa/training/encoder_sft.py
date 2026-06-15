"""Encoder SFT initialisation: next-token prediction on per-hour clinical texts.

Trains a fresh LoRA adapter on top of Qwen3-8B by next-token prediction over
two independent text streams from the trajectory shards: per-hour *state*
texts (vitals, labs, severity scores) and per-hour *action* texts
(medications, ventilator settings, procedures). Each ``(stay, hour)`` pair
contributes two training samples — one state, one action. See paper §3.1.

Usage::

    # Single-GPU smoke test (~100 samples):
    python -m clin_jepa.training.encoder_sft \\
        --config configs/train/encoder_sft.yaml --max_samples 100

    # Multi-GPU production (4x H200 DDP):
    torchrun --nproc_per_node=4 -m clin_jepa.training.encoder_sft \\
        --config configs/train/encoder_sft.yaml
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import yaml
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening

from liger_kernel.transformers import apply_liger_kernel_to_qwen3
apply_liger_kernel_to_qwen3()

from clin_jepa.training.sampler import TokenBudgetBatchSampler
from clin_jepa.utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — reads .pt shards, pools state + action texts
# ---------------------------------------------------------------------------

class PerHourSFTDataset(Dataset):
    """Per-hour SFT dataset, reading trajectory .pt shards directly.

    The ``text_source`` argument controls which texts to include:
      - ``"state"``: only state_texts
      - ``"action"``: only action_texts
      - ``"both"``: state_texts + action_texts pooled
    """

    def __init__(
        self,
        shard_dir: str | Path,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 4096,
        text_source: Literal["state", "action", "both"] = "both",
        max_samples: int | None = None,
        compute_lengths: bool = True,
    ) -> None:
        if text_source not in ("state", "action", "both"):
            raise ValueError(f"text_source must be state/action/both, got {text_source!r}")

        shard_dir = Path(shard_dir)
        files = sorted(shard_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt shards in {shard_dir}")

        logger.info(
            "Loading %d trajectory shards from %s (text_source=%s)",
            len(files), shard_dir, text_source,
        )

        self.texts: list[str] = []
        n_state_total = 0
        n_action_total = 0

        for shard_path in files:
            shard = torch.load(shard_path, weights_only=False, map_location="cpu")
            state_texts = shard["per_hour"]["state_texts"]
            action_texts = shard["per_hour"]["action_texts"]
            if len(state_texts) != len(action_texts):
                raise ValueError(
                    f"Shard {shard_path.name}: state_texts ({len(state_texts)}) "
                    f"and action_texts ({len(action_texts)}) length mismatch"
                )

            if text_source == "both":
                # Interleave state/action: [s0, a0, s1, a1, ...]
                for s, a in zip(state_texts, action_texts):
                    self.texts.append(s)
                    self.texts.append(a)
                n_state_total += len(state_texts)
                n_action_total += len(action_texts)
            elif text_source == "state":
                self.texts.extend(state_texts)
                n_state_total += len(state_texts)
            elif text_source == "action":
                self.texts.extend(action_texts)
                n_action_total += len(action_texts)

        if max_samples:
            # Truncate to max_samples
            self.texts = self.texts[:max_samples]

        logger.info(
            "Loaded %d samples (text_source=%s; state=%d, action=%d)",
            len(self.texts), text_source, n_state_total, n_action_total,
        )

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if compute_lengths:
            logger.info("Pre-tokenizing %d samples for length estimation...", len(self.texts))
            self.token_lengths: list[int] = []
            for text in self.texts:
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                n_tokens = min(n_tokens, max_seq_len)
                self.token_lengths.append(n_tokens)

            avg_len = sum(self.token_lengths) / len(self.token_lengths)
            total_tokens = sum(self.token_lengths)
            logger.info(
                "Dataset: %d samples, avg %.0f tokens, total %.1fM tokens",
                len(self.texts), avg_len, total_tokens / 1e6,
            )
        else:
            self.token_lengths = None
            logger.info("Dataset: %d samples (length estimation skipped — DDP mode)", len(self.texts))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        # tokenize, leaving room for EOS
        ids = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_seq_len - 1,
            add_special_tokens=True,
        )["input_ids"]
        ids.append(self.tokenizer.eos_token_id)
        return {"input_ids": ids, "labels": ids.copy()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, bool]:
    """Initialize distributed training if available."""
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


def create_model(config: dict, device: torch.device) -> tuple:
    """Load Qwen3-8B + fresh LoRA."""
    model_cfg = config["model"]
    hf_cfg = config.get("hf", {})

    logger.info("Loading base model: %s", model_cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg["attn_implementation"],
        local_files_only=hf_cfg.get("offline", False),
    )

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    # Add fresh LoRA
    lora_cfg = model_cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        local_files_only=hf_cfg.get("offline", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device,
             max_batches: int = 50) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(config: dict, args: argparse.Namespace) -> None:
    """Main training function."""
    rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if is_main:
        setup_logging()
        logger.info("Encoder SFT — %d GPUs", world_size)

    # --- Model ---
    model, tokenizer = create_model(config, device)

    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    raw_model = model.module if is_distributed else model

    # --- Data ---
    data_cfg = config["data"]
    project_root = Path(__file__).resolve().parents[2]

    need_lengths = not is_distributed

    train_dataset = PerHourSFTDataset(
        shard_dir=project_root / data_cfg["train_dir"],
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        text_source="both",
        max_samples=args.max_samples,
        compute_lengths=need_lengths,
    )

    val_max_samples = min(args.max_samples or 5000, 5000)
    val_state_dataset = PerHourSFTDataset(
        shard_dir=project_root / data_cfg["val_dir"],
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        text_source="state",
        max_samples=val_max_samples,
        compute_lengths=False,
    )
    val_action_dataset = PerHourSFTDataset(
        shard_dir=project_root / data_cfg["val_dir"],
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        text_source="action",
        max_samples=val_max_samples,
        compute_lengths=False,
    )

    collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

    train_cfg = config["training"]

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("max_batch_size", 256),
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    else:
        # Single GPU fallback: use TokenBudgetBatchSampler with packing
        budget_sampler = TokenBudgetBatchSampler(
            lengths=train_dataset.token_lengths,
            max_tokens=train_cfg.get("token_budget", 32768),
            max_batch_size=train_cfg.get("max_batch_size", 256),
            shuffle=True,
            seed=train_cfg.get("seed", 42),
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=budget_sampler,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True,
        )

    # Val loaders: same packing, smaller batch
    val_state_loader = DataLoader(
        val_state_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )
    val_action_loader = DataLoader(
        val_action_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training Loop ---
    log_cfg = config.get("logging", {})
    save_cfg = config.get("save", {})
    save_dir = project_root / save_cfg.get("dir", "checkpoints/encoder_sft")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_every = log_cfg.get("log_every", 50)
    eval_every = log_cfg.get("eval_every", 500)
    eval_max_batches = log_cfg.get("eval_max_batches", 50)
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    # CLI overrides (for smoke tests)
    if getattr(args, "eval_every", None) is not None:
        eval_every = args.eval_every
        if is_main:
            logger.info("CLI override: eval_every = %d", eval_every)
    if getattr(args, "log_every", None) is not None:
        log_every = args.log_every
        if is_main:
            logger.info("CLI override: log_every = %d", log_every)

    best_val_loss = float("inf")
    global_step = 0
    t0 = time.time()

    if is_main:
        logger.info("Training: %d epochs, %d steps/epoch, %d total steps",
                     train_cfg["epochs"], len(train_loader), total_steps)
        logger.info("Warmup: %d steps, LR: %s, WD: %s",
                     warmup_steps, train_cfg["lr"], train_cfg.get("weight_decay", 0.01))

    model.train()
    for epoch in range(train_cfg["epochs"]):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_steps = 0

        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in raw_model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            n_tokens = (batch["labels"] != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            epoch_steps += 1
            global_step += 1

            # Log
            if is_main and global_step % log_every == 0:
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                tokens_per_sec = epoch_tokens / elapsed
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
                logger.info(
                    "Step %d/%d | loss=%.4f | lr=%.2e | %.0f tok/s | peak VRAM %.1f GB | %.1f min elapsed",
                    global_step, total_steps, avg_loss, lr, tokens_per_sec, mem_gb, elapsed / 60,
                )

            # Eval (state + action separately, then combined for best-checkpoint selection)
            if is_main and global_step % eval_every == 0:
                eval_model = raw_model if not is_distributed else model
                val_loss_state = evaluate(eval_model, val_state_loader, device,
                                          max_batches=eval_max_batches)
                val_loss_action = evaluate(eval_model, val_action_loader, device,
                                           max_batches=eval_max_batches)
                val_loss = (val_loss_state + val_loss_action) / 2
                logger.info(
                    "  Val loss: %.4f (state=%.4f, action=%.4f) (best: %.4f)",
                    val_loss, val_loss_state, val_loss_action, best_val_loss,
                )

                latest_path = save_dir / "latest"
                raw_model.save_pretrained(latest_path)
                tokenizer.save_pretrained(latest_path)
                # Also store a small marker file so we can tell which step this is
                with open(save_dir / "latest_meta.json", "w") as f:
                    json.dump({
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_loss_state": val_loss_state,
                        "val_loss_action": val_loss_action,
                        "elapsed_seconds": time.time() - t0,
                    }, f, indent=2)

                # Save "best" only when val loss improves
                if val_loss < best_val_loss and save_cfg.get("save_best", True):
                    best_val_loss = val_loss
                    best_path = save_dir / "best"
                    raw_model.save_pretrained(best_path)
                    tokenizer.save_pretrained(best_path)
                    logger.info("  Saved best checkpoint to %s", best_path)

                model.train()

        # End of epoch
        if is_main:
            avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
            logger.info(
                "Epoch %d complete | avg_loss=%.4f | %d steps | %.1fM tokens",
                epoch, avg_epoch_loss, epoch_steps, epoch_tokens / 1e6,
            )

    # --- Save final checkpoint ---
    if is_main and save_cfg.get("save_final", True):
        final_path = save_dir / "final"
        raw_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info("Saved final checkpoint to %s", final_path)

        # Save training metadata
        meta = {
            "total_steps": global_step,
            "total_tokens": epoch_tokens,
            "final_train_loss": epoch_loss / max(epoch_tokens, 1),
            "best_val_loss": best_val_loss,
            "config": config,
            "elapsed_seconds": time.time() - t0,
        }
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    cleanup_distributed(is_distributed)
    if is_main:
        logger.info("Encoder SFT complete. Total time: %.1f min", (time.time() - t0) / 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder SFT (NTP on state + action texts)")
    parser.add_argument("--config", required=True, help="Path to configs/train/encoder_sft.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (testing)")
    parser.add_argument("--eval_every", type=int, default=None, help="Override eval cadence (for smoke)")
    parser.add_argument("--log_every", type=int, default=None, help="Override log cadence (for smoke)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set HF environment
    hf_cfg = config.get("hf", {})
    if hf_cfg.get("cache_dir"):
        os.environ["HF_HOME"] = hf_cfg["cache_dir"]
    if hf_cfg.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    train(config, args)


if __name__ == "__main__":
    main()
