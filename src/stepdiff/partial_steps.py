"""
partial_steps.py

Provides functionality for partial-step diffusion sampling, decoding, and saving images
at specified intervals (e.g., every --step_size steps).
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import trange

# The DDIMSampler or PLMSSampler is found in the compvis code, e.g. ldm.models.diffusion.ddim
# but we assume you've installed 'latent-diffusion' so you can import from it:
from ldm.models.diffusion.ddim import DDIMSampler

def sample_partial_steps(
    model,
    total_steps: int,
    step_size: int,
    outdir: str,
    seed: int = 42,
    eta: float = 1.0
):
    """
    Runs partial-step sampling using a DDIMSampler and saves intermediate images.

    :param model: Loaded LDM model
    :param total_steps: e.g. 200 (total DDIM steps)
    :param step_size: e.g. 40 (save images every 40 steps)
    :param outdir: Where to save the output .png images
    :param seed: Random seed for reproducibility
    :param eta: DDIM's 'eta' parameter (0.0 = deterministic, 1.0 = more stochastic)
    """
    os.makedirs(outdir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create DDIMSampler or you can choose PLMSSampler, etc.
    sampler = DDIMSampler(model)

    # You may want to define an initial noise z, or let the sampler handle it.
    # Here we assume you're sampling a single image. If you want a batch, expand shape accordingly.
    shape = [4, 64, 64]  # For a 256x256 latent (4x64x64 for example), depends on your config
    # If your LDM uses other shapes (like 3, 256, 256 for pixel-based diffusion), adapt accordingly.

    with torch.no_grad():
        with model.ema_scope():
            # Start from random noise
            x = torch.randn((1, *shape), device=model.device)
            # We'll use DDIM steps = total_steps
            sampler.make_schedule(ddim_num_steps=total_steps, ddim_eta=eta, verbose=False)

            # We can manually step from t = total_steps down to t = 1 in increments
            # or use sampler.ddim_sampling to do it. We'll illustrate manual stepping.
            # For simplicity, let's do full generation at once, but store intermediate steps.

            # The sampler normally returns x at each step if we pass 'True' for return_intermediates
            samples, intermediates = sampler.sample(
                S=total_steps,
                conditioning=None,    # No prompt for unconditional, or supply your own cond
                batch_size=1,
                shape=shape,
                verbose=False,
                eta=eta,
                x_T=x,
                return_intermediates=True
            )

            # 'intermediates' is typically a dict with 'xs' storing latents at each step.
            # We'll decode latents at each step_size interval and save them.
            step_latents = intermediates['xs']  # list of latents at each step

            for i, lat in enumerate(step_latents):
                # i = 0..total_steps, lat is a latent tensor
                t = total_steps - i  # T goes from total_steps..1

                if t % step_size == 0 or t == 0:
                    # decode and save
                    img = decode_latent_to_pil(model, lat)
                    outpath = os.path.join(outdir, f"image_step_{t}.png")
                    img.save(outpath)
                    print(f"Saved {outpath}")

            # Also save the final image
            final_img = decode_latent_to_pil(model, samples)
            final_out = os.path.join(outdir, f"image_step_{0}.png")
            final_img.save(final_out)
            print(f"Saved final {final_out}")


def decode_latent_to_pil(model, latent):
    """
    Decodes a latent tensor into a PIL image.

    :param model: The LDM model with a 'first_stage_model' or 'decode' method
    :param latent: A single latent tensor of shape [1, 4, 64, 64], or similar
    :return: A PIL.Image
    """
    # Some LDM models have a function decode_first_stage(), others do something else.
    # We'll assume decode_first_stage is available:
    x_recon = model.decode_first_stage(latent)
    # x_recon is typically [1, 3, 256, 256] in range [-1,1]
    x_recon = (x_recon.clamp(-1., 1.) + 1.0) / 2.0
    x_recon = x_recon.cpu().numpy().transpose(0,2,3,1)[0] * 255
    x_recon = x_recon.astype(np.uint8)
    return Image.fromarray(x_recon)

