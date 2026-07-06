"""
Common utility functions used across the project.

This module centralizes functionality such as:
- loading the project configuration
- setting random seeds
- creating output directories
- saving/loading checkpoints

Keeping these functions here avoids duplication across
train.py, evaluate.py and app.py.
"""

from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(config_path: Path) -> dict:
    """
    Load the project configuration from config.yaml.

    Parameters
    ----------
    config_path : Path
        Path to config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory to create.
    """

    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_score: float,
    save_path: Path,
):
    """
    Save a model checkpoint.

    Parameters
    ----------
    model : torch.nn.Module

    optimizer : torch.optim.Optimizer

    epoch : int

    best_score : float

    save_path : Path
    """

    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )