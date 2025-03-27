"""
grid_visualization.py

Creates a multi-column grid of partial-step outputs, e.g. for steps 40, 80, 120, 160, 200.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import trange
from ldm.models.diffusion.ddim import DDIMSampler

def create_step_grid(
    model,
    total_steps: int,
    step_intervals: list,
    outdir: str,
    seed: int = 42,
    eta: float = 1.0,
    grid_name: str = "grid.png"
):
    """
    Creates a single image grid containing decodes at the specified step intervals.

    :param model: Loaded LDM model
    :param total_steps: The maximum number of steps (e.g. 200)
    :param step_intervals: A list of steps to visualize (e.g. [40, 80, 120, 160, 200])
    :param outdir: Where to save the final grid image
    :param seed: Random seed for reproducibility
    :param eta: DDIM's eta parameter
    :param grid_name: Filename for the grid image
    """
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    sampler = DDIMSampler(model)
    shape = [4, 64, 64]  # adapt to your model
    with torch.no_grad():
        with model.ema_scope():
            x = torch.randn((1, *shape), device=model.device)
            sampler.make_schedule(ddim_num_steps=total_steps, ddim_eta=eta, verbose=False)
            samples, intermediates = sampler.sample(
                S=total_steps,
                conditioning=None,  # or your prompt conditioning
                batch_size=1,
                shape=shape,
                verbose=False,
                eta=eta,
                x_T=x,
                return_intermediates=True
            )

    latents_per_step = intermediates['xs']  # list of latents
    # latents_per_step[i] => step = total_steps - i

    # decode each step in step_intervals
    images = []
    for step in step_intervals:
        idx = total_steps - step
        lat = latents_per_step[idx]
        img = decode_latent_to_pil(model, lat)
        images.append(img)

    # Now create a horizontal grid (one row, len(step_intervals) columns)
    w, h = images[0].size
    total_width = w * len(images)
    grid = Image.new("RGB", (total_width, h))
    x_offset = 0
    for i, im in enumerate(images):
        grid.paste(im, (x_offset, 0))
        x_offset += w

    outpath = os.path.join(outdir, grid_name)
    grid.save(outpath)
    print(f"Saved grid to {outpath}")


def decode_latent_to_pil(model, latent):
    """
    Decodes a latent tensor into a PIL image.
    (Same logic as in partial_steps.py - you could import from partial_steps to avoid duplication.)
    """
    x_recon = model.decode_first_stage(latent)
    x_recon = (x_recon.clamp(-1., 1.) + 1.0) / 2.0
    x_recon = x_recon.cpu().numpy().transpose(0,2,3,1)[0] * 255
    x_recon = x_recon.astype(np.uint8)
    return Image.fromarray(x_recon)

