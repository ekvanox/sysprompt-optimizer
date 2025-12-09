import torch
import yaml
from sae_lens import SAE
from transformer_lens import HookedTransformer


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_hooked_model(config: dict, device: str | torch.device = "cuda"):
    release = config["model"]["sae"]["name"]
    sae_id = config["model"]["sae"]["sae_id"]
    sae = SAE.from_pretrained(release, sae_id, device=device)
    model = HookedTransformer.from_pretrained(config["model"]["name"])
    return model, sae
