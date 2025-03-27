import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torchvision
from torchvision.utils import save_image

# LDM imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

##############################################################################
# Saliency & Attention Extractors
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

        target_layer = self.model.model.diffusion_model  # the UNet
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

        # Hook any sub-module with 'attn' in its name
        for name, submod in self.model.model.diffusion_model.named_modules():
            if 'attn' in name:
                self.hooks.append(submod.register_forward_hook(attention_hook))

##############################################################################
# Color Mapping & Overlay
##############################################################################
def simple_color_map(gray_map: torch.Tensor):
    """
    Turn a 1×H×W map [0..1] into a 3×H×W color map.
    For instance, R=gray, G=0, B=1-gray -> a naive red→blue scheme.
    """
    # gray_map shape: [1,H,W]
    c, h, w = 3, gray_map.shape[-2], gray_map.shape[-1]
    color = torch.zeros((3, h, w), device=gray_map.device)
    color[0] = gray_map[0]          # R channel
    color[2] = 1.0 - gray_map[0]    # B channel
    return color.clamp(0,1)

def overlay_heatmap(rgb_img: torch.Tensor, heatmap: torch.Tensor, alpha=0.5):
    """
    alpha‐blend the heatmap (3×H×W) with the original rgb_img (3×H×W).
    Return shape (3×H×W).
    """
    return (1.0 - alpha)*rgb_img + alpha*heatmap

##############################################################################
# Saliency / Attention Computation
##############################################################################
def compute_saliency_map(latent, model, timestep, extractor):
    extractor.clear()
    extractor.register_hooks()

    lat = latent.clone().requires_grad_(True)
    t_tensor = torch.tensor([timestep], device=model.device, dtype=torch.long)

    if hasattr(model, "logvar"):
        model.logvar = model.logvar.to(model.device)

    out = model(lat, t_tensor)
    if isinstance(out, tuple):
        prediction = out[0]
    else:
        prediction = out

    loss = prediction.sum()
    loss.backward()

    grads = extractor.gradients   # [B, C, H, W]
    feats = extractor.feature_maps
    # Simple measure: mean abs(grad)*abs(feat)
    sal_map = grads.abs().mean(dim=1, keepdim=True) * feats.abs().mean(dim=1, keepdim=True)
    # Min‐max
    s = sal_map[0]  # shape [1, H, W]
    mn, mx = s.min(), s.max()
    if (mx - mn) < 1e-8:
        s_norm = torch.zeros_like(s)
    else:
        s_norm = (s - mn)/(mx - mn)
    return s_norm  # [1,H,W] in [0..1], but only 64×64 for the latent

def compute_attention_map(latent, model, timestep, extractor):
    extractor.clear()
    extractor.register_hooks()

    t_tensor = torch.tensor([timestep], device=model.device, dtype=torch.long)
    model(latent, t_tensor)

    if len(extractor.attention_maps) == 0:
        return None

    attn = extractor.attention_maps[-1]  # shape e.g. [B, heads, tokens, tokens]
    # average over heads if 4D
    if attn.dim() == 4:
        attn = attn.mean(dim=1, keepdim=True)
    # attn shape: [B,1,?,?]
    a = attn[0]  # [1,?,?]
    mn, mx = a.min(), a.max()
    if (mx - mn) < 1e-8:
        return torch.zeros_like(a)
    else:
        return (a - mn)/(mx - mn)

##############################################################################
# Decoding & Sampler
##############################################################################
def decode_image(model, latent):
    with torch.no_grad():
        x_rec = model.decode_first_stage(latent)
        x_rec = (x_rec + 1.0)/2.0  # -> [0,1]
    return x_rec  # shape [B,3,256,256] typically

def make_grid(tensor_list, nrow=5):
    batch = torch.stack(tensor_list, dim=0)
    grid = torchvision.utils.make_grid(batch, nrow=nrow)
    return grid

##############################################################################
# Main
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

    config = OmegaConf.load(args.config)
    print(f"[INFO] Loading model from {args.checkpoint}")
    pl_sd = torch.load(args.checkpoint, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    sampler = DDIMSampler(model)
    shape = [model.channels, model.image_size, model.image_size]  # e.g. [3,64,64]

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

    x_inter = intermediates["x_inter"]
    total_saved = len(x_inter)

    step_list = [40,80,120,160,200]
    if total_saved not in step_list:
        step_list.append(total_saved)

    # Setup extractors
    sal_extractor = SaliencyExtractor(model)
    attn_extractor = AttentionExtractor(model)

    # For final grids:
    image_list = []
    saliency_list = []
    attention_list = []

    for s in step_list:
        if s <= 0 or s > total_saved:
            continue

        latent = x_inter[s-1]
        # Approx “t” = total_steps - s
        approx_t = max(args.total_steps - s, 0)

        # 1) decode to 256×256
        img = decode_image(model, latent)  # shape [1,3,256,256]
        img_0 = img[0].clamp(0,1)          # shape [3,256,256]
        image_list.append(img_0.cpu())

        # 2) Compute saliency in latent (64×64), then upsample to 256
        s_map_64 = compute_saliency_map(latent, model, approx_t, sal_extractor)  # [1,64,64]
        # Upsample to [1,256,256]
        s_map_256 = F.interpolate(s_map_64.unsqueeze(0),  # shape [B=1, C=1, H=64, W=64]
                                  size=(img_0.shape[-2], img_0.shape[-1]),
                                  mode='bilinear',
                                  align_corners=False).squeeze(0)
        # Now shape [1,256,256]
        sal_color = simple_color_map(s_map_256)          # [3,256,256]
        sal_overlay = overlay_heatmap(img_0, sal_color, alpha=0.5).clamp(0,1)
        saliency_list.append(sal_overlay.cpu())

        # 3) Compute attention similarly
        a_map_64 = compute_attention_map(latent, model, approx_t, attn_extractor)
        if a_map_64 is not None and a_map_64.dim() == 3:
            # a_map_64 shape [1,64,64], typically
            a_map_256 = F.interpolate(a_map_64.unsqueeze(0), 
                                      size=(img_0.shape[-2], img_0.shape[-1]),
                                      mode='bilinear',
                                      align_corners=False).squeeze(0)
            attn_color = simple_color_map(a_map_256)   # [3,256,256]
            attn_overlay = overlay_heatmap(img_0, attn_color, alpha=0.5).clamp(0,1)
        else:
            attn_overlay = torch.zeros_like(img_0)

        attention_list.append(attn_overlay.cpu())

        print(f"[Step {s}] t={approx_t} → Sal/Attn upsampled to 256×256")

    # Create 3 multi‐column images
    step_str = "(" + ",".join(str(x) for x in step_list) + ")"

    # image grid
    img_grid = make_grid(image_list, nrow=len(image_list))
    save_image(img_grid, os.path.join(args.outdir, f"image_step{step_str}.png"))

    # saliency grid
    sal_grid = make_grid(saliency_list, nrow=len(saliency_list))
    save_image(sal_grid, os.path.join(args.outdir, f"saliency_step{step_str}.png"))

    # attention grid
    attn_grid = make_grid(attention_list, nrow=len(attention_list))
    save_image(attn_grid, os.path.join(args.outdir, f"attention_step{step_str}.png"))

    print("[DONE] See", args.outdir)


if __name__ == "__main__":
    main()

