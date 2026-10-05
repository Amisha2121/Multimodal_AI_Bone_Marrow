"""
configs/config_loader.py
─────────────────────────────────────────────────────────────────────────────
Utility to load and merge YAML configs using OmegaConf.

Usage:
    from configs.config_loader import load_config
    cfg = load_config()                    # loads default.yaml
    cfg = load_config("configs/colab.yaml")  # merges colab overrides on top
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("omegaconf is required. Run: pip install omegaconf")

_DEFAULT_CFG = Path(__file__).parent / "default.yaml"


def load_config(override_path: str | Path | None = None) -> DictConfig:
    """
    Load default.yaml and optionally merge an override config on top.

    Args:
        override_path: Path to an override YAML (e.g. 'configs/colab.yaml').
                       Only keys present in the override are changed.

    Returns:
        Merged OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(_DEFAULT_CFG)
    if override_path is not None:
        override = OmegaConf.load(override_path)
        cfg = OmegaConf.merge(cfg, override)
    return cfg


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and (if available) PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_cfg_and_seed(override_path: str | Path | None = None) -> DictConfig:
    """Convenience: load config and immediately set the global seed."""
    cfg = load_config(override_path)
    set_global_seed(cfg.seed)
    return cfg
