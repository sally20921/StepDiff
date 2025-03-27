"""
attn_and_sal.py

Code for extracting cross-attention maps and gradient-based saliency at partial steps.
This is a simplified example. You can expand it to match your real hooking logic.
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import trange
from ldm.models.diffusion.ddim import DDIMSampler
import torch.nn.functional as F

def save_attention_and_saliency(
    model,
    total_steps: int,
    step_size: int,
    outdir: str,
    seed: int = 42,
    eta: float = 1.0
):
    """
    Runs partial-step sampling, computing cross-attention maps and gradient-based saliency
    at selected intervals, then saves them.

    :param model: LDM model
    :param total_steps: e.g. 200
    :param step_size: e.g. 40
    :param outdir: output directory
    :param seed: random seed
    :param eta: DDIM's eta
    """
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # We'll demonstrate hooking a single cross-attention layer
    # The real model has multiple. You can replicate or iterate over them.

    attention_maps = []
    saliency_maps = []

    def forward_hook_attn(module, input, output):
        # output shape might be [batch, heads, tokens, tokens]
        # We'll just store it in a list
        attention_maps.append(output.detach().cpu())

    # Register a hook on a cross-attention layer (this is an example name!)
    # You need to find the actual name of a cross-attn block in your model
    cross_attn_layer = find_cross_attention_layer(model)
    hook_handle = cross_attn_layer.attn.register_forward_hook(forward_hook_attn)

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

    # We'll do saliency by hooking gradients for the final reconstruction
    # For a generative model, one approach: sum of outputs, call backward, get dL/d(latent)
    # For simplicity, we won't do a full "Grad-CAM" here, just an example

    # For each step, decode, compute grad, etc. We'll do a simplified pass:
    latents_list = intermediates['xs']
    for i, lat in enumerate(latents_list):
        t = total_steps - i
        if t % step_size == 0 or t == 0:
            sal_map = compute_fake_saliency(model, lat)
            # Save saliency as a grayscale
            sal_img = (sal_map*255).astype(np.uint8)
            outpath = os.path.join(outdir, f"saliency_step_{t}.png")
            Image.fromarray(sal_img).save(outpath)
            print(f"Saved {outpath}")

    # Now, attention_maps has collected all cross-attention outputs for all steps
    # We can attempt to save them at intervals.
    # The hooking here might store all steps in one big list. 
    # For demonstration, let's assume we save the last one for final step:
    if attention_maps:
        last_attn = attention_maps[-1][0, 0]  # pick batch=0, head=0
        # Normalize and save
        attn_norm = (last_attn - last_attn.min())/(last_attn.max()-last_attn.min()+1e-8)
        attn_img = (attn_norm*255).cpu().numpy().astype(np.uint8)
        # It's token x token, so 64x64 or something. We'll just store it as an image:
        attn_pil = Image.fromarray(attn_img)
        out_attn = os.path.join(outdir, "attention_final.png")
        attn_pil.save(out_attn)
        print(f"Saved {out_attn}")

    # Remove the hook
    hook_handle.remove()


def find_cross_attention_layer(model):
    """
    This function is a placeholder that finds a cross-attention block in your LDM.
    You need to adapt it to your actual model architecture.
    """
    for name, module in model.named_modules():
        if "attn2" in name or "crossattention" in name:
            return module
    raise RuntimeError("No cross-attention layer found. Adapt the search logic if needed.")


def compute_fake_saliency(model, latent):
    """
    Placeholder for a gradient-based saliency method. Real saliency would require
    a forward + backward pass. Here we just do a simple absolute value of latent
    as 'saliency' for demonstration.
    """
    lat_np = latent.detach().cpu().numpy()
    sal = np.abs(lat_np).mean(axis=1)[0]  # shape [H, W], average across channels
    # normalize 0..1
    mi, ma = sal.min(), sal.max()
    sal_norm = (sal - mi)/(ma - mi + 1e-8)
    return sal_norm

