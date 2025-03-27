import torch
import argparse
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import save_image
import torchvision

# LDM imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

##############################################################################
# 1) Saliency & Attention Extractors
##############################################################################
class SaliencyExtractor:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.feature_maps = None
        self.hooks = []

    def clear(self):
        self.gradients = None
        self.feature_maps = None
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register_hooks(self):
        def forward_hook(module, inp, out):
            self.feature_maps = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer = self.model.model.diffusion_model  # main UNet
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook))


class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.hooks = []

    def clear(self):
        self.attention_maps = []
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register_hooks(self):
        def attention_hook(module, inp, out):
            self.attention_maps.append(out.detach())

        # Hook any sub‐module containing 'attn' in its name
        for name, submod in self.model.model.diffusion_model.named_modules():
            if 'attn' in name:
                h = submod.register_forward_hook(attention_hook)
                self.hooks.append(h)

##############################################################################
# 2) Overlay Helpers (to mimic a Grad‐CAM-like look)
##############################################################################
def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """
    Utility: take a single image in [0,1], shape (3,H,W) or (1,H,W),
    and make a uint8 RGB or gray array of shape (H,W,3)/(H,W).
    """
    c, h, w = img.shape
    arr = img.cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    if c == 1:
        arr = (arr[0] * 255).astype(np.uint8)
    else:
        arr = (arr.transpose(1,2,0) * 255).astype(np.uint8)
    return arr

def simple_color_map(gray: torch.Tensor):
    """
    Turn a [1,H,W] saliency or attention map in [0,1] into a pseudo‐color [3,H,W].
    Here we do a simple grayscale->(R,G,B) approach.
    """
    # grayscale -> color by stacking channels in a naive way
    # You might prefer something more advanced, e.g. applying a matplotlib colormap.
    c, h, w = 3, gray.shape[-2], gray.shape[-1]
    # Expand the single‐channel map to 3 channels: e.g. red for “high,” etc.
    # For simplicity, let's do: color = [gray, zeros, 1-gray] => a red->blue look
    # You can do anything you like here.
    color = torch.zeros((3, h, w), device=gray.device)
    color[0] = gray[0]            # R channel
    color[2] = 1.0 - gray[0]      # B channel
    return color.clamp(0,1)

def overlay_heatmap(rgb_img: torch.Tensor, heatmap: torch.Tensor, alpha=0.5):
    """
    alpha‐blend the heatmap (already in 3‐channel form [3,H,W], 0..1)
    onto the rgb_img ([3,H,W], 0..1). Return shape [3,H,W].
    """
    return (1.0 - alpha)*rgb_img + alpha*heatmap

##############################################################################
# 3) Saliency / Attention Computation
##############################################################################
def compute_saliency_map(latent, model, timestep, extractor):
    extractor.clear()
    extractor.register_hooks()

    latent = latent.clone().requires_grad_(True)
    ts = torch.tensor([timestep], device=model.device, dtype=torch.long)

    # Some LDM models have a logvar buffer
    if hasattr(model, "logvar"):
        model.logvar = model.logvar.to(model.device)

    out = model(latent, ts)
    if isinstance(out, tuple):
        prediction = out[0]
    else:
        prediction = out

    loss = prediction.sum()
    loss.backward()

    grads = extractor.gradients   # shape [B,C,H,W]
    feats = extractor.feature_maps
    # simplistic measure: mean abs(grad)*abs(feat)
    sal_map = grads.abs().mean(dim=1, keepdim=True) * feats.abs().mean(dim=1, keepdim=True)
    # min‐max
    s = sal_map[0]
    mn, mx = s.min(), s.max()
    if (mx - mn) < 1e-8:
        s_norm = torch.zeros_like(s)
    else:
        s_norm = (s - mn)/(mx - mn)
    return s_norm  # shape [1,H,W], in [0,1]

def compute_attention_map(latent, model, timestep, extractor):
    extractor.clear()
    extractor.register_hooks()
    ts = torch.tensor([timestep], device=model.device, dtype=torch.long)
    model(latent, ts)
    if len(extractor.attention_maps) == 0:
        return None
    # just take the last attn map
    attn = extractor.attention_maps[-1]  # shape [B,heads,?,?]
    # average over heads if shape is 4D
    if attn.dim() == 4:
        attn = attn.mean(dim=1, keepdim=True)  # shape [B,1,?,?]
    a = attn[0]
    mn, mx = a.min(), a.max()
    if (mx - mn) < 1e-8:
        a_norm = torch.zeros_like(a)
    else:
        a_norm = (a - mn)/(mx - mn)
    return a_norm  # shape [1,H,W] or [1,tokens,tokens]

##############################################################################
# 4) Decoding & Grid
##############################################################################
def decode_image(model, latent):
    """
    Decodes a latent to an RGB image in [0,1].
    """
    with torch.no_grad():
        x_rec = model.decode_first_stage(latent)
        x_rec = (x_rec + 1.)/2.  # shift [-1,1] -> [0,1]
    return x_rec  # shape [B,3,H,W]

def make_grid(tensor_list, nrow=5):
    """
    Convert a list of 3‐channel images ([3,H,W]) to a single grid.
    """
    batch = torch.stack(tensor_list, dim=0)  # shape [B,3,H,W]
    grid = torchvision.utils.make_grid(batch, nrow=nrow)
    return grid

##############################################################################
# 5) Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--total_steps", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config & model
    config = OmegaConf.load(args.config)
    print(f"[INFO] Loading model from {args.checkpoint}")
    pl_sd = torch.load(args.checkpoint, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Sampler
    sampler = DDIMSampler(model)

    # For ffhq256, shape=(3,64,64) in latent space, batch=1
    shape = [model.channels, model.image_size, model.image_size]

    # Actually run the sampling
    with torch.no_grad():
        samples, intermediates = sampler.sample(
            S=args.total_steps,
            conditioning=None,
            batch_size=1,
            shape=shape,
            verbose=True,
            eta=args.eta,
            return_intermediates=True,
            log_every_t=1
        )

    # x_inter is the list of latents at each step
    x_inter = intermediates["x_inter"]
    total_saved = len(x_inter)

    # We'll grab steps 40,80,120,160,200, plus final if not included
    step_list = [40,80,120,160,200]
    if total_saved not in step_list:
        step_list.append(total_saved)

    # Saliency + attention hooks
    sal_extractor = SaliencyExtractor(model)
    attn_extractor = AttentionExtractor(model)

    # We'll store 5 images each for the final big grids
    image_list = []
    saliency_list = []
    attention_list = []

    for s in step_list:
        if s <= 0 or s > total_saved:
            continue
        latent = x_inter[s-1]  # zero-based indexing

        # We'll guess the UNet's integer time index as (total_steps - s)
        # if the code doesn't store actual steps. That is approximate.
        approx_t = max(args.total_steps - s, 0)

        # 1) decode
        img = decode_image(model, latent)  # shape [1,3,H,W]
        img_0 = img[0]                     # shape [3,H,W], [0,1]
        image_list.append(img_0.cpu())

        # 2) saliency
        s_map = compute_saliency_map(latent, model, approx_t, sal_extractor)  # [1,H,W]
        # convert single‐channel to color, then overlay
        sal_color = simple_color_map(s_map)  # shape [3,H,W]
        sal_overlay = overlay_heatmap(img_0, sal_color, alpha=0.5).clamp(0,1)
        saliency_list.append(sal_overlay.cpu())

        # 3) attention
        a_map = compute_attention_map(latent, model, approx_t, attn_extractor)
        if a_map is not None and a_map.dim() == 3:
            # e.g. shape [1,H,W]. We can reshape if it's [1,tokens,tokens].
            # If it's [1,64,64] we can overlay directly.
            # If it's [1, X, X] with X != 64, it's trickier to overlay in image space.
            if a_map.shape[-1] == img_0.shape[-1]:
                # direct overlay
                attn_color = simple_color_map(a_map)
                attn_overlay = overlay_heatmap(img_0, attn_color, alpha=0.5).clamp(0,1)
            else:
                # mismatch shape => just produce a black overlay or skip
                attn_overlay = torch.zeros_like(img_0)
        else:
            # no attention
            attn_overlay = torch.zeros_like(img_0)

        attention_list.append(attn_overlay.cpu())

        print(f"[Saved step={s}] approx_t={approx_t}")

    # Now create 3 big grids for the 5 (or 6) steps
    step_str = "(" + ",".join(str(x) for x in step_list) + ")"

    # image_grid
    img_grid = make_grid(image_list, nrow=len(image_list))
    save_image(img_grid, os.path.join(args.outdir, f"image_step{step_str}.png"))

    # saliency_grid
    sal_grid = make_grid(saliency_list, nrow=len(saliency_list))
    save_image(sal_grid, os.path.join(args.outdir, f"saliency_step{step_str}.png"))

    # attention_grid
    attn_grid = make_grid(attention_list, nrow=len(attention_list))
    save_image(attn_grid, os.path.join(args.outdir, f"attention_step{step_str}.png"))

    print("[DONE] Wrote multi‐column PNGs to", args.outdir)


if __name__ == "__main__":
    main()

