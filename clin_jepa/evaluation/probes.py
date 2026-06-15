"""MLP probe utilities.

MLP architecture: 4096 → 512 (ReLU + dropout 0.1) → 1.

Saved-probe schema:
    {
        "state_dict": MLPProbe.state_dict(),
        "normalizer": {"mean": float, "std": float, "log_transform": bool},
        "metadata": {
            "var": str,
            "n_train": int, "n_val": int,
            "epochs_trained": int,
            "best_val_loss_norm": float,
            "final_val_mae_orig": float,
            "log_transform": bool,
            "ceiling": {"mae": float},  # = final_val_mae_orig (probe ceiling)
            "training_time_seconds": float,
            "arch": "mlp_4096_512_1_relu_dropout_0.1",
        }
    }

The normalizer uses the keys ``mean`` / ``std``.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLPProbe(nn.Module):
    """4096 -> 512 (ReLU + dropout) -> 1, output squeezed."""
    def __init__(self, in_dim: int = 4096, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_mlp_probe(
    probe: MLPProbe,
    normalizer: dict,
    metadata: dict,
    path: str | Path,
) -> None:
    """Save MLP probe + normalizer + metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": probe.state_dict(),
        "normalizer": dict(normalizer),
        "metadata": dict(metadata),
    }, path)


def load_mlp_probe(
    path: str | Path,
    device: str = "cpu",
    in_dim: int = 4096,
    hidden: int = 512,
    dropout: float = 0.1,
) -> tuple[MLPProbe, dict, dict]:
    """Load MLP probe + normalizer + metadata."""
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    probe = MLPProbe(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    return probe, ckpt["normalizer"], ckpt["metadata"]


def apply_mlp_probe(
    probe: MLPProbe,
    z: torch.Tensor,
    normalizer: dict,
    device: str = "cuda",
    batch_size: int = 65536,
) -> np.ndarray:
    """Apply a trained MLP probe → predictions in original clinical units."""
    probe = probe.to(device).eval()
    preds_norm: list[np.ndarray] = []
    n = z.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = z[i : i + batch_size].to(device).float()
            preds_norm.append(probe(batch).cpu().numpy())
    preds_norm_arr = np.concatenate(preds_norm, axis=0)
    preds_unit = preds_norm_arr * normalizer["std"] + normalizer["mean"]
    if normalizer.get("log_transform", False):
        preds_unit = np.expm1(preds_unit)
    return preds_unit


def train_mlp_probe_one_var(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_vl: np.ndarray,
    y_vl: np.ndarray,
    var_name: str,
    log_transform: bool,
    device: str = "cuda",
    max_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    patience: int = 10,
    hidden: int = 512,
    dropout: float = 0.1,
    seed: int = 42,
) -> tuple[MLPProbe, dict, dict]:
    """Train one MLP probe with z-score-normalized target + early stopping.

    Returns: (trained_probe, normalizer_dict, metadata_dict)
    """
    import time
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    t0 = time.time()
    if log_transform:
        y_tr_t = np.log1p(np.maximum(y_tr.astype(np.float32), 0))
        y_vl_t = np.log1p(np.maximum(y_vl.astype(np.float32), 0))
    else:
        y_tr_t = y_tr.astype(np.float32)
        y_vl_t = y_vl.astype(np.float32)
    y_mean = float(np.mean(y_tr_t))
    y_std = float(np.std(y_tr_t) + 1e-8)
    y_tr_n = (y_tr_t - y_mean) / y_std
    y_vl_n = (y_vl_t - y_mean) / y_std

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Ztr = torch.from_numpy(z_tr).to(dev).float()
    Ytr = torch.from_numpy(y_tr_n).to(dev).float()
    Zvl = torch.from_numpy(z_vl).to(dev).float()
    Yvl = torch.from_numpy(y_vl_n).to(dev).float()

    in_dim = Ztr.shape[1]
    probe = MLPProbe(in_dim=in_dim, hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state: dict | None = None
    pat = 0
    n = Ztr.shape[0]
    epochs_trained = 0
    history: list[float] = []

    for epoch in range(1, max_epochs + 1):
        probe.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = F.mse_loss(probe(Ztr[idx]), Ytr[idx])
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            vl = F.mse_loss(probe(Zvl), Yvl).item()
        history.append(vl)
        epochs_trained = epoch

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.clone().cpu() for k, v in probe.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    assert best_state is not None
    probe.load_state_dict(best_state)
    probe.eval()

    # Final val MAE in original units
    normalizer = {"mean": y_mean, "std": y_std, "log_transform": log_transform}
    with torch.no_grad():
        vp_norm = probe(Zvl).cpu().numpy()
    vp_unit = vp_norm * y_std + y_mean
    if log_transform:
        vp_unit = np.expm1(vp_unit)
    final_val_mae_orig = float(np.mean(np.abs(vp_unit - y_vl.astype(np.float32))))
    final_val_rmse_orig = float(np.sqrt(np.mean((vp_unit - y_vl.astype(np.float32)) ** 2)))

    metadata = {
        "var": var_name,
        "n_train": int(z_tr.shape[0]),
        "n_val": int(z_vl.shape[0]),
        "epochs_trained": epochs_trained,
        "best_val_loss_norm": float(best_val),
        "final_val_mae_orig": final_val_mae_orig,
        "final_val_rmse_orig": final_val_rmse_orig,
        "log_transform": log_transform,
        "ceiling": {"mae": final_val_mae_orig},  # MLP probe ceiling
        "training_time_seconds": float(time.time() - t0),
        "arch": f"mlp_{in_dim}_{hidden}_1_relu_dropout_{dropout}",
        "history": history,
    }
    return probe.to(dev), normalizer, metadata


# ---------------------------------------------------------------------------
# CLI entry point — train one MLP probe per continuous clinical variable
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import time
    import yaml

    parser = argparse.ArgumentParser(
        description=(
            "Train MLP probes (one per continuous clinical variable) from the "
            "(embedding, label) pairs written by "
            "`clin_jepa.evaluation.collect_probe_data`."
        ),
    )
    parser.add_argument(
        "--plan",
        choices=["clin_jepa", "vjepa2ac", "sft_baseline"],
        required=True,
        help="Which paradigm's probe-training data to consume and where to "
             "write the trained probes.",
    )
    parser.add_argument(
        "--config",
        default="configs/eval/probes.yaml",
        help="Path to probes.yaml (per-plan paths are read from "
             "cfg['plans'][plan]).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Subset of variables to train (for smoke tests). Defaults to "
             "every variable listed in cfg['variables'].",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the per-variable probe-training RNG (torch/numpy).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    plan_cfg = cfg["plans"][args.plan]
    probe_data_dir = Path(plan_cfg["probe_data_dir"])
    probes_dir = Path(plan_cfg["probes_dir"])
    probes_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg.get("training", {})
    var_cfg = cfg.get("variables", {})

    all_vars = list(var_cfg.get("existing_11", [])) + list(var_cfg.get("new_4", []))
    log_transform_vars = set(var_cfg.get("log_transform", []))
    if args.variables:
        all_vars = [v for v in args.variables if v in all_vars]
        if not all_vars:
            raise ValueError(
                f"No valid variables from --variables={args.variables}. "
                f"Known: {all_vars}"
            )

    logger.info("MLP probe training")
    logger.info("Plan:           %s", args.plan)
    logger.info("Probe-data dir: %s", probe_data_dir)
    logger.info("Probes dir:     %s", probes_dir)
    logger.info("Variables:      %d", len(all_vars))

    t_total = time.time()
    for var in all_vars:
        t0 = time.time()
        log_transform = var in log_transform_vars

        train_path = probe_data_dir / f"train_{var}.pt"
        val_path = probe_data_dir / f"val_{var}.pt"
        if not train_path.exists() or not val_path.exists():
            logger.warning(
                "Skipping %s — missing %s or %s. Run "
                "`python -m clin_jepa.evaluation.collect_probe_data` first.",
                var, train_path, val_path,
            )
            continue

        train = torch.load(train_path, map_location="cpu", weights_only=False)
        val = torch.load(val_path, map_location="cpu", weights_only=False)
        z_tr = train["z"].numpy()
        y_tr = train["y"].numpy()
        z_vl = val["z"].numpy()
        y_vl = val["y"].numpy()

        probe, normalizer, metadata = train_mlp_probe_one_var(
            z_tr, y_tr, z_vl, y_vl,
            var_name=var,
            log_transform=log_transform,
            device=train_cfg.get("device", "cuda"),
            max_epochs=int(train_cfg.get("max_epochs", 100)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            batch_size=int(train_cfg.get("batch_size", 4096)),
            patience=int(train_cfg.get("patience", 10)),
            seed=args.seed,
        )

        out_path = probes_dir / f"probe_{var}.pt"
        save_mlp_probe(probe, normalizer, metadata, out_path)
        logger.info(
            "[%s] val_MAE=%.4f epochs=%d time=%.1fs → %s",
            var, metadata["final_val_mae_orig"], metadata["epochs_trained"],
            time.time() - t0, out_path,
        )

    logger.info("All probes trained in %.1f min", (time.time() - t_total) / 60)


if __name__ == "__main__":
    main()
