"""Utility functions for loading models and configs."""

import torch
import yaml
from transformer_lens import HookedTransformer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(config: dict, device: str | None = None) -> HookedTransformer:
    """Load the HookedTransformer model."""
    if device is None:
        device = get_device()
    print(f"Using device: {device}")
    model = HookedTransformer.from_pretrained(config["model"]["name"], device=device)
    return model
