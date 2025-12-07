"""Utility helpers shared across training scripts."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a python dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


@dataclass
class RunningLoss:
    """Keeps a running average without storing all observations."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def get_accelerator(config: Dict[str, Any]) -> Accelerator:
    """Instantiate Accelerator using common keys from config."""
    hw = config.get("hardware", {})
    return Accelerator(
        gradient_accumulation_steps=hw.get("gradient_accumulation_steps", 1),
        mixed_precision=hw.get("mixed_precision"),
        log_with="wandb" if config.get("logging", {}).get("use_wandb") else None,
        project_dir=config.get("logging", {}).get("log_dir"),
    )


def prepare_output_dirs(*paths: str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def flatten_dict(config: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def broadcast_config(accelerator: Accelerator, config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every process sees identical configuration."""
    return accelerator.broadcast(config)  # type: ignore[arg-type]


def save_accelerator_state(accelerator: Accelerator, output_dir: str, tag: str) -> None:
    accelerator.save_state(
        output_dir=Path(output_dir) / tag,
    )


def log_rank_zero(accelerator: Accelerator, message: str) -> None:
    if accelerator.is_main_process:
        print(message)


def yield_batch(iterable: Iterable[Dict[str, Any]], accelerator: Accelerator) -> Iterable[Dict[str, Any]]:
    """Shard iterable across processes when not using native data loaders."""
    for batch in iterable:
        yield accelerator.prepare(batch)
