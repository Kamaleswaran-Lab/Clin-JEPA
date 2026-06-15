"""Encoder forward pass + PEFT multi-adapter EMA target.

Shared infrastructure used by all encoder-touching training scripts:

* the V-JEPA 2-AC baseline's encoder-refinement step
  (:mod:`clin_jepa.training.refine_encoder_vjepa`),
* the embedding-precompute step (:mod:`clin_jepa.evaluation.precompute_embeddings`),
* the Clin-JEPA five-phase joint co-training
  (:mod:`clin_jepa.training.pretrain_clin_jepa`).

Key components:

* :func:`encode_texts_batched` — chunked bin-packed encoder forward; returns
  the last-token hidden state per input text, in original order.
* :func:`load_sft_initialized_encoder` — warm-start LoRA loader; renames the
  default PEFT adapter to ``"online"``.
* :func:`create_target_encoder_fp32` — adds an fp32 ``"target"`` PEFT adapter.
* :func:`ema_update` / :func:`hard_sync_target_from_online` — fp32 EMA update
  and the explicit hard sync of target to online.
* :func:`verify_ema_working` — smoke check that the EMA moves the target.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from liger_kernel.transformers import apply_liger_kernel_to_qwen3
apply_liger_kernel_to_qwen3()

from peft import PeftModel  # noqa: E402  (after liger)
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithFlattening,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# PEFT multi-adapter parameter name resolution
# ---------------------------------------------------------------------------

_LORA_NAME_PAT = re.compile(r"(.*\.lora_[AB])\.(\w+)\.weight$")


def _split_lora_name(full_name: str) -> Optional[tuple[str, str]]:
    """Parse a PEFT LoRA param name into (prefix, adapter).

    Examples:
        'base_model...q_proj.lora_A.online.weight' -> ('base_model...q_proj.lora_A', 'online')
        'base_model...q_proj.lora_B.target.weight' -> ('base_model...q_proj.lora_B', 'target')
        'base_model.model.embed_tokens.weight'     -> None  (not a LoRA param)

    Returns:
        Tuple (prefix, adapter_name) if the name matches the LoRA pattern,
        else None.
    """
    m = _LORA_NAME_PAT.match(full_name)
    if m is None:
        return None
    return (m.group(1), m.group(2))


def _build_online_target_param_pairs(
    model: nn.Module,
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    """Return list of (online_param, target_param) pairs for all LoRA layers.

    Walks all named parameters, parses each via _split_lora_name(), groups
    by prefix, then pairs up "online" and "target" within each prefix. Raises
    if any prefix is missing one of the two adapters.
    """
    prefix_to_adapters: dict[str, dict[str, nn.Parameter]] = {}
    for name, p in model.named_parameters():
        parsed = _split_lora_name(name)
        if parsed is None:
            continue
        prefix, adapter = parsed
        prefix_to_adapters.setdefault(prefix, {})[adapter] = p

    pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    for prefix, d in prefix_to_adapters.items():
        if "online" not in d or "target" not in d:
            raise RuntimeError(
                f"PEFT multi-adapter setup is broken at prefix {prefix!r}: "
                f"found adapters {list(d.keys())}, expected both 'online' and 'target'. "
                f"Did create_target_encoder_fp32() run correctly?"
            )
        pairs.append((d["online"], d["target"]))

    if not pairs:
        raise RuntimeError(
            "No LoRA online/target param pairs found. "
            "Either the model has no LoRA adapters, or the PEFT adapter "
            "rename to 'online' did not run."
        )
    return pairs


def _rename_peft_adapter(model: nn.Module, old: str, new: str) -> None:
    """Rename a PEFT adapter from `old` to `new` in-place.

    PEFT stores adapter-indexed dicts on each LoraLayer (e.g.,
    `self.lora_A: nn.ModuleDict({"default": ...})`). We walk all submodules
    and swap the key in every such dict, plus rename in `model.peft_config`
    and update active_adapter / _active_adapter.

    Use before any forward pass and before adding a second adapter.
    """
    adapter_dict_attrs = (
        # nn.ModuleDict / nn.ParameterDict entries
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",  # DoRA
        "lora_dropout",
        # Plain Python dict entries
        "scaling",
        "r",
        "lora_alpha",
        "lora_bias",              # bias != "none"
        "use_rslora",             # rslora
        "use_dora",               # DoRA flag
    )

    for module in model.modules():
        for attr in adapter_dict_attrs:
            if not hasattr(module, attr):
                continue
            d = getattr(module, attr)
            if isinstance(d, (nn.ModuleDict, nn.ParameterDict)) and old in d:
                d[new] = d.pop(old)
            elif isinstance(d, dict) and old in d:
                d[new] = d.pop(old)

    # Rename in peft_config
    if hasattr(model, "peft_config") and old in model.peft_config:
        model.peft_config[new] = model.peft_config.pop(old)

    # Rename in active_adapter / _active_adapter (PEFT internal state)
    for active_attr in ("active_adapter", "_active_adapter"):
        if hasattr(model, active_attr):
            cur = getattr(model, active_attr)
            if isinstance(cur, str) and cur == old:
                setattr(model, active_attr, new)
            elif isinstance(cur, list):
                setattr(model, active_attr, [new if a == old else a for a in cur])

    # Some PEFT versions also have an "active_adapters" list on the base model
    if hasattr(model, "active_adapters"):
        try:
            cur = model.active_adapters
            if isinstance(cur, list):
                model.active_adapters = [new if a == old else a for a in cur]
        except Exception:
            pass  # property without setter — fall through

    # Recursively rename on submodules' active_adapter too
    for module in model.modules():
        if module is model:
            continue
        for active_attr in ("active_adapter", "_active_adapter"):
            if hasattr(module, active_attr):
                cur = getattr(module, active_attr)
                if isinstance(cur, str) and cur == old:
                    try:
                        setattr(module, active_attr, new)
                    except Exception:
                        pass
                elif isinstance(cur, list):
                    try:
                        setattr(module, active_attr, [new if a == old else a for a in cur])
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_sft_initialized_encoder(
    config: dict,
    device: torch.device,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load the encoder-SFT LoRA as the trainable "online" encoder for JEPA pretraining.

    Steps:
      1. Load Qwen3-8B base in bf16 with FA2.
      2. Load the SFT LoRA via PEFT.from_pretrained() — auto-named "default".
      3. Rename adapter "default" -> "online" so create_target_encoder_fp32()
         can add a second "target" adapter alongside.
      4. Set active adapter to "online".
      5. Enable gradient checkpointing.
      6. Load tokenizer from same checkpoint dir (preserves any added tokens).

    The base model is frozen; only LoRA params train. Caller must wrap in DDP
    if distributed.

    Args:
        config: Full config dict. Reads:
            config["model"]["base_model"] (default "Qwen/Qwen3-8B")
            config["model"]["checkpoint_dir"] (relative to PROJECT_ROOT)
            config["model"]["attn_implementation"] (default "flash_attention_2")
            config["model"]["gradient_checkpointing"] (default True)
            config["hf"]["offline"] (default False)

    Returns:
        (online_model, tokenizer). online_model.peft_config has "online" key.
    """
    model_cfg = config["model"]
    hf_cfg = config.get("hf", {})

    base_name = model_cfg.get("base_model", "Qwen/Qwen3-8B")
    ckpt_rel = model_cfg["checkpoint_dir"]
    ckpt_abs = PROJECT_ROOT / ckpt_rel
    if not ckpt_abs.exists():
        raise FileNotFoundError(f"encoder-SFT checkpoint not found at {ckpt_abs}")

    logger.info("Loading base model: %s", base_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        local_files_only=hf_cfg.get("offline", False),
    )

    # Freeze base — only LoRA trains
    for p in base.parameters():
        p.requires_grad = False

    logger.info("Loading encoder-SFT LoRA from: %s", ckpt_abs)
    online = PeftModel.from_pretrained(
        base,
        str(ckpt_abs),
        is_trainable=True,
    )
    online = online.to(device)

    # Rename "default" -> "online" so we can add "target" alongside
    if "default" in online.peft_config and "online" not in online.peft_config:
        _rename_peft_adapter(online, old="default", new="online")
    elif "online" in online.peft_config:
        logger.info("Adapter already named 'online' — skipping rename")
    else:
        raise RuntimeError(
            f"Unexpected PEFT adapter state after from_pretrained: "
            f"peft_config keys = {list(online.peft_config.keys())}"
        )

    online.set_adapter("online")

    # Verify rename succeeded
    assert "online" in online.peft_config, "rename to 'online' failed (config)"
    assert "default" not in online.peft_config, "rename did not pop 'default'"
    found_lora = False
    for name, _ in online.named_parameters():
        if ".lora_A.online.weight" in name or ".lora_B.online.weight" in name:
            found_lora = True
            break
    if not found_lora:
        # Print first 10 lora params for debugging
        sample = [n for n, _ in online.named_parameters() if "lora_" in n][:10]
        raise RuntimeError(
            "Rename to 'online' did not produce expected param names. "
            f"First 10 lora params: {sample}"
        )

    # Enable gradient checkpointing for memory
    if model_cfg.get("gradient_checkpointing", True):
        online.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # Tokenizer from same checkpoint dir
    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt_abs),
        local_files_only=hf_cfg.get("offline", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_lora_params = sum(
        p.numel() for n, p in online.named_parameters()
        if "lora_" in n and p.requires_grad
    )
    logger.info(
        "Loaded online encoder: %s + LoRA (%.1fM trainable)",
        base_name, n_lora_params / 1e6,
    )

    return online, tokenizer


def create_target_encoder_fp32(online: PeftModel) -> PeftModel:
    """Add a 'target' adapter as fp32 EMA copy of 'online'.

    Steps:
      1. Add a new adapter "target" with the same LoRA config as "online".
      2. Set active adapter back to "online" (add_adapter may switch it).
      3. Explicit copy: target params <- online params, cast bf16 -> fp32.
      4. Freeze target params (no gradients ever).
      5. Verify: target_p.dtype == fp32, max |online - target| < 1e-6,
         target.requires_grad == False.

    Args:
        online: PeftModel with adapter "online" already loaded.

    Returns:
        Same PeftModel object (mutated in place), now with two adapters:
        "online" (bf16, trainable) and "target" (fp32, frozen).
    """
    if "online" not in online.peft_config:
        raise RuntimeError(
            f"create_target_encoder_fp32 requires 'online' adapter; "
            f"found {list(online.peft_config.keys())}"
        )

    if "target" in online.peft_config:
        raise RuntimeError(
            "Adapter 'target' already exists. create_target_encoder_fp32 should "
            "only be called once per model."
        )

    # Step 1: add "target" with same config as "online"
    online_config = online.peft_config["online"]
    online.add_adapter("target", online_config)

    # Step 2: PEFT may switch active adapter when adding; force back to online
    online.set_adapter("online")

    # Step 3 + 4: explicit copy online -> target, cast to fp32, freeze
    pairs = _build_online_target_param_pairs(online)
    with torch.no_grad():
        for online_p, target_p in pairs:
            target_p.data = online_p.detach().float().clone()  # bf16 -> fp32
            target_p.requires_grad = False

    # Step 5: verification
    for online_p, target_p in pairs:
        if target_p.dtype != torch.float32:
            raise RuntimeError(
                f"target param dtype is {target_p.dtype}, expected float32"
            )
        max_gap = (online_p.detach().float() - target_p).abs().max().item()
        if max_gap > 1e-6:
            raise RuntimeError(
                f"target init copy verification failed: max |online - target| "
                f"= {max_gap:.2e}, expected < 1e-6"
            )
        if target_p.requires_grad:
            raise RuntimeError("target param has requires_grad=True; should be False")

    online.set_adapter("online")  # ensure online is active for downstream forward

    n_target_params = sum(p.numel() for _, p in pairs)
    logger.info(
        "Created target adapter as fp32 EMA copy of online (%.1fM params, frozen)",
        n_target_params / 1e6,
    )

    return online


# ---------------------------------------------------------------------------
# EMA update + hard sync
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(
    online: PeftModel,
    tau: float,
    pairs: Optional[list[tuple[nn.Parameter, nn.Parameter]]] = None,
) -> None:
    """Update target adapter: target <- tau * target + (1-tau) * online.

    Online params are bf16, target params fp32; the online side is cast to
    fp32 before the interpolation.

    Args:
        online: PeftModel with both 'online' and 'target' adapters.
        tau: EMA momentum (1.0 freezes target).
        pairs: Optional cached param pairs from _build_online_target_param_pairs(model).
            If None, walks named_parameters() (slower).
    """
    if pairs is None:
        pairs = _build_online_target_param_pairs(online)
    for online_p, target_p in pairs:
        online_fp32 = online_p.detach().float()  # bf16 -> fp32
        target_p.data.mul_(tau).add_(online_fp32, alpha=1.0 - tau)


@torch.no_grad()
def manual_allreduce_grads(params: list[nn.Parameter]) -> None:
    """Manually all-reduce gradients across DDP ranks via dist.all_reduce.

    Skips params with requires_grad=False or grad=None; uses ReduceOp.AVG.
    Call once per optimizer step, before optimizer.step().

    Args:
        params: Parameters whose grads should be all-reduced (encoder LoRA +
            predictor params).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return  # single-GPU mode, no allreduce needed

    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)


@torch.no_grad()
def hard_sync_target_from_online(
    online: PeftModel,
    pairs: Optional[list[tuple[nn.Parameter, nn.Parameter]]] = None,
) -> None:
    """Direct copy: target <- online (hard sync). target == online in fp32."""
    if pairs is None:
        pairs = _build_online_target_param_pairs(online)
    for online_p, target_p in pairs:
        target_p.data.copy_(online_p.detach().float())


def verify_ema_working(
    online: PeftModel,
    n_updates: int = 10,
    perturbation_scale: float = 0.001,
) -> bool:
    """Smoke-test the EMA mechanism: verify the target params actually move.

    Procedure:
      1. Snapshot target params.
      2. Perturb online params slightly (simulates one step of training).
      3. Run n_updates EMA updates with tau=0.996.
      4. Assert target params have moved by > 1e-6 in total.
      5. Restore online to original values (so this is a no-op for training).

    Args:
        online: PeftModel with both adapters created.
        n_updates: How many EMA updates to perform during the test.
        perturbation_scale: Magnitude of random online perturbation.

    Returns:
        True on success.

    Raises:
        RuntimeError if target params don't move enough.
    """
    pairs = _build_online_target_param_pairs(online)

    # Snapshot
    target_before = [t.detach().clone() for _, t in pairs]
    online_before = [o.detach().clone() for o, _ in pairs]

    try:
        # Perturb online slightly
        with torch.no_grad():
            for online_p, _ in pairs:
                online_p.data += torch.randn_like(online_p) * perturbation_scale

        # Run N EMA updates
        for _ in range(n_updates):
            ema_update(online, tau=0.996, pairs=pairs)

        # Assert target moved
        total_delta = 0.0
        for (_, target_p), tb in zip(pairs, target_before):
            total_delta += (target_p - tb).abs().sum().item()

        if total_delta < 1e-6:
            raise RuntimeError(
                f"verify_ema_working FAILED: target delta = {total_delta:.2e} "
                f"after {n_updates} EMA updates. Either bf16 precision bug or "
                f"PEFT name resolution bug — DO NOT proceed with training."
            )

        logger.info(
            "verify_ema_working PASSED: target moved by %.2e total over %d updates",
            total_delta, n_updates,
        )

    finally:
        # Restore online and target to pre-test state
        with torch.no_grad():
            for (online_p, target_p), ob, tb in zip(pairs, online_before, target_before):
                online_p.data.copy_(ob)
                target_p.data.copy_(tb)

    return True


# ---------------------------------------------------------------------------
# Encoder forward (chunked packed)
# ---------------------------------------------------------------------------

# Module-level collator
_COLLATOR: Optional[DataCollatorWithFlattening] = None


def _get_collator() -> DataCollatorWithFlattening:
    global _COLLATOR
    if _COLLATOR is None:
        _COLLATOR = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
    return _COLLATOR


def encode_texts_batched(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_seq_len: int = 4096,
    train_mode: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Encode a list of texts into (N, 4096) last-token hidden state embeddings.

    Uses DataCollatorWithFlattening + FA2 to pack multiple texts into one
    sequence per forward. Bin-packs greedily into chunks of <= max_seq_len
    total tokens, so very long single texts get their own chunk.

    Args:
        model: PeftModel (online or target adapter, set externally via set_adapter).
        tokenizer: HF tokenizer (left-padding not required).
        texts: List of text strings to encode.
        device: CUDA device.
        max_seq_len: Max packed token count per chunk AND per single text truncation.
        train_mode: If True, gradients flow through the encoder forward.
            If False, runs under torch.no_grad().
        output_dtype: Dtype of returned embeddings (bf16 for training, fp16 for cache).

    Returns:
        Tensor (len(texts), 4096) on `device` in `output_dtype`.
    """
    if not texts:
        # empty input -> zero-row tensor
        hidden_size = model.config.hidden_size if hasattr(model, "config") else 4096
        return torch.zeros((0, hidden_size), device=device, dtype=output_dtype)

    collator = _get_collator()

    # Tokenize all texts first (so we know each one's length for bin-packing)
    token_ids_list: list[list[int]] = []
    for text in texts:
        ids = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )["input_ids"]
        token_ids_list.append(ids)

    # Greedy bin-pack into chunks of <= max_seq_len total tokens
    chunks: list[list[int]] = []  # each chunk is a list of indices into texts
    current: list[int] = []
    current_tokens = 0
    for idx, ids in enumerate(token_ids_list):
        n_tok = len(ids)
        if current and current_tokens + n_tok > max_seq_len:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += n_tok
    if current:
        chunks.append(current)

    # Encode each chunk, collect embeddings in original order
    all_embeddings: list[Optional[torch.Tensor]] = [None] * len(texts)

    for chunk_indices in chunks:
        features = [
            {"input_ids": token_ids_list[i], "labels": token_ids_list[i]}
            for i in chunk_indices
        ]
        batch = collator(features)
        # Move to device, but skip non-tensor entries (max_length_q/k are Python ints)
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        forward_kwargs = dict(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            cu_seq_lens_q=batch["cu_seq_lens_q"],
            cu_seq_lens_k=batch["cu_seq_lens_k"],
            max_length_q=batch["max_length_q"],
            max_length_k=batch["max_length_k"],
            output_hidden_states=True,
            use_cache=False,
        )

        if train_mode:
            outputs = model(**forward_kwargs)
        else:
            with torch.no_grad():
                outputs = model(**forward_kwargs)

        # Extract last-token of each sub-sequence via cu_seqlens
        cu = batch["cu_seq_lens_q"]                    # (n_sub + 1,) int32
        last_positions = (cu[1:] - 1).long()           # (n_sub,)
        last_hidden = outputs.hidden_states[-1]        # (1, total_tokens, 4096)
        chunk_embs = last_hidden[0, last_positions, :] # (n_sub, 4096)

        # Cast to output dtype and assign in original order
        chunk_embs = chunk_embs.to(output_dtype)
        for out_idx, text_idx in enumerate(chunk_indices):
            all_embeddings[text_idx] = chunk_embs[out_idx]

        # Free intermediate tensors
        del outputs, batch, last_hidden, chunk_embs

    # All slots should be filled
    assert all(e is not None for e in all_embeddings), "encode_texts_batched dropped some texts"

    return torch.stack(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Resume support: collect / restore PEFT adapter state + RNG snapshot
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_lora_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot all LoRA adapter weights (online + target) for resume.

    Walks ``named_parameters()`` and collects every tensor whose name matches
    the LoRA pattern (``.lora_A.{adapter}.weight`` or ``.lora_B.{adapter}.weight``).
    Returns CPU tensors so the snapshot can be torch.save()'d safely.

    The target adapter has ``requires_grad=False`` but we still capture it via
    ``named_parameters()`` because PEFT registers all adapter params there
    regardless of grad state.

    Returns:
        ``{full_param_name: cpu_tensor}`` covering both online and target adapters.
    """
    state: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if _split_lora_name(name) is None:
            continue
        state[name] = p.detach().cpu().clone()
    return state


@torch.no_grad()
def restore_lora_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore LoRA adapter weights from a snapshot produced by collect_lora_adapter_state.

    Iterates the model's named_parameters, looks each up in ``state`` by name,
    and copies the saved tensor into the live param. Preserves dtype/device of
    the live param via ``copy_`` (which dtype-casts as needed).

    Asserts every LoRA param in the model has a matching saved entry.
    """
    name_to_param = {n: p for n, p in model.named_parameters() if _split_lora_name(n)}
    missing = [n for n in name_to_param if n not in state]
    extra = [n for n in state if n not in name_to_param]
    if missing:
        raise RuntimeError(
            f"restore_lora_adapter_state: {len(missing)} live params missing from snapshot. "
            f"First few: {missing[:5]}"
        )
    if extra:
        logger.warning(
            "restore_lora_adapter_state: %d snapshot entries have no live param "
            "(ignored). First few: %s",
            len(extra), extra[:5],
        )
    for name, p in name_to_param.items():
        p.data.copy_(state[name].to(p.device))


def snapshot_rng_state() -> dict:
    """Capture torch CPU + CUDA + numpy + python RNG state for resume."""
    import random as _py_random
    return {
        "torch_cpu":  torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy":      __import__("numpy").random.get_state(),
        "python":     _py_random.getstate(),
    }


def restore_rng_state(state: dict) -> None:
    """Restore RNG state from a snapshot produced by snapshot_rng_state.

    Handles cross-world-size resume (e.g. save on 8 GPU, resume on 4 GPU): the
    saved torch_cuda list has length = save-time device_count, while at restore
    time torch.cuda.device_count() may differ. If saved list is longer than
    currently available, we clip; if shorter, we pass through (PyTorch will
    restore only the available prefix and leave others at their default init).
    """
    import random as _py_random
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        saved_cuda_states = state["torch_cuda"]
        n_available = torch.cuda.device_count()
        if len(saved_cuda_states) > n_available:
            logger.warning(
                "restore_rng_state: saved %d CUDA RNG states but only %d devices "
                "available (cross-world-size resume); clipping to %d.",
                len(saved_cuda_states), n_available, n_available,
            )
            saved_cuda_states = saved_cuda_states[:n_available]
        torch.cuda.set_rng_state_all(saved_cuda_states)
    if "numpy" in state:
        __import__("numpy").random.set_state(state["numpy"])
    if "python" in state:
        _py_random.setstate(state["python"])


def atomic_torch_save(state: dict, path: "Path") -> None:
    """torch.save with atomic rename. Avoids half-written resume files on crash."""
    from pathlib import Path as _P
    path = _P(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
