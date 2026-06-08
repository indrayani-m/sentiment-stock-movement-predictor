"""Shared utilities for configuration, paths, logging, and reproducibility."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_directories(config: dict[str, Any]) -> None:
    """Create every configured output directory if it does not exist."""
    for path in config["paths"].values():
        resolve_path(path).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "notebooks").mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging with a compact format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logging.getLogger(__name__).warning("PyTorch is not installed; skipping torch seed.")


def clean_text(text: str) -> str:
    """Normalize Reddit text using beginner-friendly rules."""
    import re

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s$]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_read_csv(path: Path) -> Any:
    """Read a CSV with a clearer error if it is missing."""
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)
