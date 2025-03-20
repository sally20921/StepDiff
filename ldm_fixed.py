import torch
import argparse
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
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
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer = self.model.model.diffusion_model
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook))


class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.hooks = []

    def clear(self):
        self.attention_maps = []
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def register_hooks(self):
        def attention_hook(module, input, output):
            self.attention_maps.append(output.detach())

        target_layers = [
            m for (n, m) in self.model.model.diffusion_model.named_modules() if 'attn' in n
        ]
        for layer in target_layers:
            self.hooks.append(layer.register_forward_hook(attention_hook))


##############################################################################
# Utility Functions
##############################################################################
def decode_image(model, latent):
    """
    Decodes a latent to an RGB image in [0,1].
    """
    with torch.no_grad():
        x_rec = model.decode_first_stage(latent)
        x_rec = (x_rec + 1.0) / 2.0  # shift [-1,1] -> [0,1]
    return x_rec

def compute_saliency_map(latent, model, timestep, extractor):
    extractor.clear()
    extractor.register_hooks()

    latent = latent.clone().requires_grad_(True)
    t_tensor = torch.tensor([timestep], device=model.device, dtype=torch.long)

    # Some LDM models have a logvar buffer
    if hasattr(model, "logvar"):
        model.logvar = model.logvar.to(model.device)

    out = model(latent, t_tensor)
    # Usually out is (prediction, ...)
    predicted = out[0] if isinstance(out, tuple) else out

    loss = predicted.sum()
    loss.backward()

    grads = extractor.gradients  # shape [B, C, H, W]
    feats = extractor.feature_maps

    # For a simple saliency measure: mean of abs(grad)*abs(activation)
    sal_map = grads.abs().mean(dim=1, keepdim=True) * feats.abs().mean(dim=1, keepdim=True)
    sal_img = sal_map[0]  # just the first in batch
    # Normalize to [0,1]
    s_min, s_max = sal_img.min(), sal_img.max()
    if (s_max - s_min) < 1e-8:
        sal_img_norm = torch.zeros_like(sal_img)
    else:
        sal_img_norm = (sal_img - s_min) / (s_max - s_min)
    return sal_img_norm  # shape [1,H,W]


def compute_attention_map(latent, model, timestep, extractor):
    extractor.clear()
    extractor.register_hooks()

    t_tensor = torch.tensor([timestep], device=model.device, dtype=torch.long)
    model(latent, t_tensor)

    if len(extractor.attention_maps) == 0:
        return None

    # Take the last layer's attn
    attn = extractor.attention_maps[-1]  # shape [B, heads, tokens, tokens] usually
    if attn.dim() == 4:
        # average over heads
        attn = attn.mean(dim=1, keepdim=True)  # shape [B,1,tokens,tokens]
    attn_0 = attn[0]  # shape [1,tokens,tokens] or [tokens,tokens]

    a_min, a_max = attn_0.min(), attn_0.max()
    if (a_max - a_min) < 1e-8:
        attn_norm = torch.zeros_like(attn_0)
    else:
        attn_norm = (attn_0 - a_min) / (a_max - a_min)
    return attn_norm


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

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    config = OmegaConf.load(args.config)
    print(f"Loading model from {args.checkpoint}")
    pl_sd = torch.load(args.checkpoint, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval().to(device)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # sampler
    sampler = DDIMSampler(model)

    # shape (batch=1, model.channels=3, 64,64 for ffhq256)
    shape = [model.channels, model.image_size, model.image_size]

    # sample
    with torch.no_grad():
        samples, intermediates = sampler.sample(
            S=args.total_steps,
            conditioning=None,     # unconditional
            batch_size=1,
            shape=shape,
            verbose=True,
            eta=args.eta,
            return_intermediates=True,
            log_every_t=1
        )

    x_inter = intermediates["x_inter"]  # list of latents at each step
    total_saved = len(x_inter)
    # We want steps step_size, 2*step_size, ... and final
    chosen_steps = []
    for i in range(1, total_saved + 1):
        if i % args.step_size == 0:
            chosen_steps.append(i)
    if total_saved not in chosen_steps:
        chosen_steps.append(total_saved)

    print(f"[INFO] # of intermediates: {total_saved}. We will output at: {chosen_steps}")

    # init saliency & attn
    sal_extractor = SaliencyExtractor(model)
    attn_extractor = AttentionExtractor(model)

    # for each chosen step
    for step_idx in chosen_steps:
        latent = x_inter[step_idx - 1]  # zero-based
        # "actual_t" is not stored in older repos. We'll do a simple guess:
        actual_t = max(args.total_steps - step_idx, 0)

        # decode image
        image = decode_image(model, latent)
        outname_img = os.path.join(args.outdir, f"image_step_{step_idx}.png")
        save_image(image, outname_img)

        # saliency
        sal_map = compute_saliency_map(latent, model, actual_t, sal_extractor)
        sal_map_img = (sal_map.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(sal_map_img).save(
            os.path.join(args.outdir, f"saliency_step_{step_idx}.png")
        )

        # attention
        attn_map = compute_attention_map(latent, model, actual_t, attn_extractor)
        if attn_map is None:
            attn_img_array = np.zeros_like(sal_map_img)
        else:
            attn_img_array = (attn_map.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(attn_img_array).save(
            os.path.join(args.outdir, f"attention_step_{step_idx}.png")
        )

        print(f"[Saved] step={step_idx}, used t={actual_t}")

    print("Done.")


if __name__ == "__main__":
    main()

