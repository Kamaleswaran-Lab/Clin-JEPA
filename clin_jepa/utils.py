"""Shared utilities for Clin-JEPA: config loading, path expansion, logging, seeding.

Config paths use shell-style ``${VAR}`` references resolved against the
process environment (see ``.env.example``).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def expand_path(path_str: str | Path) -> Path:
    """Expand ``~`` and ``${VAR}`` references in a path string.

    Args:
        path_str: A path that may contain a leading ``~`` or any number of
            ``${VAR}`` / ``$VAR`` references.

    Returns:
        An absolute :class:`pathlib.Path` with all references expanded
        against the process environment.

    Raises:
        ValueError: If the expanded string still contains an unresolved
            ``${...}`` reference.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(path_str)))
    if "${" in expanded:
        raise ValueError(
            f"Unresolved environment variable in path: {expanded!r}. "
            "Make sure the referenced variable is exported (see .env.example)."
        )
    return Path(expanded).resolve()


def _expand_in_obj(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` references inside a nested config object."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_in_obj(v) for v in obj]
    return obj


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand ``${VAR}`` references in string values.

    Args:
        config_path: Path to a YAML file.

    Returns:
        Parsed configuration with environment variables resolved.
    """
    path = Path(config_path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return _expand_in_obj(cfg) if cfg is not None else {}


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and any missing parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and (if available) PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure root logger to write to stderr and optionally to a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def count_parameters(model: Any, *, trainable_only: bool = True) -> int:
    """Count parameters in a PyTorch model.

    Args:
        model: Any module with a ``parameters()`` iterator.
        trainable_only: If ``True`` (default), count only ``requires_grad``
            parameters; otherwise count all parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
