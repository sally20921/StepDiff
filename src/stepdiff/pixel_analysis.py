"""
pixel_analysis.py

Illustrates how to do pixel/latent difference analysis. This code is simplified
and can be extended to do actual per-pixel diffs across partial steps.
"""

import os
import torch
import numpy as np
from PIL import Image
from ldm.models.diffusion.ddim import DDIMSampler

def run_pixel_analysis(
    model,
    total_steps: int,
    step_size: int,
    outdir: str,
    seed: int = 42,
    eta: float = 1.0
):
    """
    Runs partial-step sampling and computes pixel-level differences between step intervals.

    :param model: LDM model
    :param total_steps: e.g. 200
    :param step_size: Save/compare every step_size steps
    :param outdir: Where to save difference images
    :param seed: Random seed
    :param eta: DDIM's eta
    """
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    sampler = DDIMSampler(model)
    shape = [4, 64, 64]
    with torch.no_grad():
        with model.ema_scope():
            x = torch.randn((1, *shape), device=model.device)
            sampler.make_schedule(ddim_num_steps=total_steps, ddim_eta=eta, verbose=False)
            samples, intermediates = sampler.sample(
                S=total_steps,
                conditioning=None,
                batch_size=1,
                shape=shape,
                eta=eta,
                x_T=x,
                return_intermediates=True
            )

    # latents at each step
    latents_list = intermediates['xs']  # list of length total_steps+1
    # decode them all first
    decoded_images = []
    for i, lat in enumerate(latents_list):
        decoded_images.append(decode_latent_to_np(model, lat))

    # We'll do pixel diffs between consecutive intervals (e.g. step, step-step_size)
    # or you can do any pair of steps you like.
    for i, lat_img in enumerate(decoded_images):
        t = total_steps - i
        if t % step_size == 0 and t != 0:
            prev_t = t - step_size
            if prev_t >= 0:
                prev_idx = total_steps - prev_t
                diff_img = compute_pixel_diff(decoded_images[i], decoded_images[prev_idx])
                outpath = os.path.join(outdir, f"pixel_diff_{t}_minus_{prev_t}.png")
                Image.fromarray(diff_img).save(outpath)
                print(f"Saved {outpath}")

def decode_latent_to_np(model, latent):
    """
    Decodes a latent to a numpy array of shape (H, W, 3) in [0,255].
    """
    x_recon = model.decode_first_stage(latent)
    x_recon = (x_recon.clamp(-1., 1.) + 1.0) / 2.0
    x_recon = x_recon.cpu().numpy().transpose(0,2,3,1)[0] * 255
    x_recon = x_recon.astype(np.uint8)
    return x_recon

def compute_pixel_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute simple absolute difference between two decoded images (img1, img2).
    Returns a numpy array in the same shape.
    """
    # shape is (H, W, 3)
    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32)).astype(np.uint8)
    return diff

