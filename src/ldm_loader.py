"""
ldm_loader.py

Handles loading of Latent Diffusion Models (LDM) using CompVis code.
"""

import os
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def load_ldm(config_path: str, checkpoint_path: str):
    """
    Load a latent diffusion model from a given config (.yaml) and checkpoint (.ckpt) file.

    :param config_path: Path to the .yaml config file (e.g., 'configs/latent-diffusion/ffhq-ldm-vq-4.yaml')
    :param checkpoint_path: Path to the .ckpt file (e.g., 'models/ldm/ffhq256/model.ckpt')
    :return: Loaded model (usually an instance of LatentDiffusion)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config
    config = OmegaConf.load(config_path)
    
    # Create model
    model = instantiate_from_config(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys when loading model:", missing)
    if unexpected:
        print("Unexpected keys when loading model:", unexpected)

    model.eval()
    return model

