import argparse, os
import torch
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

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

        # Register a forward hook on any sub-module that has "attn" in its name
        target_layers = [
            module for name, module in self.model.model.diffusion_model.named_modules() if "attn" in name
        ]
        for layer in target_layers:
            self.hooks.append(layer.register_forward_hook(attention_hook))


def compute_saliency_map(latent, model, timestep, saliency_extractor):
    saliency_extractor.clear()
    saliency_extractor.register_hooks()

    latent.requires_grad_()
    timestep_tensor = torch.tensor([timestep], device=model.device).long()

    # Some LDM models keep a logvar buffer. Make sure it's on the correct device
    if hasattr(model, "logvar"):
        model.logvar = model.logvar.to(model.device)

    outputs = model(latent, timestep_tensor)
    predicted_latent = outputs[0]  # Assuming the first element is the prediction

    # A dummy loss to backprop through
    loss = predicted_latent.sum()
    loss.backward()

    gradients = saliency_extractor.gradients
    feature_maps = saliency_extractor.feature_maps
    # Simple pointwise multiplication of mean abs gradients and mean abs activations
    saliency_map = gradients.abs().mean(dim=1, keepdim=True) * feature_maps.abs().mean(dim=1, keepdim=True)

    return saliency_map


def compute_attention_map(latent, model, timestep, attention_extractor):
    attention_extractor.clear()
    timestep_tensor = torch.tensor([timestep], device=model.device).long()
    model(latent, timestep_tensor)
    return attention_extractor.attention_maps


def decode_latent(model, latent):
    """
    Decodes the latent into an image. Adjust based on your actual LDM model structure.
    """
    with torch.no_grad():
        # For many LDM models, the first_stage_model handles decoding:
        x_rec = model.first_stage_model.decode(latent)
        # x_rec is typically in [-1, 1]. Shift to [0,1] for saving as an image:
        x_rec = (x_rec.clamp(-1., 1.) + 1.0) / 2.0
    return x_rec


def normalize_map(tensor_map):
    """
    Normalizes a single or batch of maps to [0,1] for easier visualization.
    """
    # If shape is (B, 1, H, W) or (B, H, W), do a simple min-max per image
    b = tensor_map.shape[0]
    out_maps = []
    for i in range(b):
        this_map = tensor_map[i]
        # Ensure shape is (1, H, W) by forcing channels
        if this_map.dim() == 3:
            pass  # shape is [C, H, W]
        elif this_map.dim() == 2:
            this_map = this_map.unsqueeze(0)

        # Min-max normalize each single map across its entire area
        m_min = this_map.min()
        m_max = this_map.max()
        if (m_max - m_min) < 1e-8:
            norm_map = this_map * 0.0
        else:
            norm_map = (this_map - m_min) / (m_max - m_min)
        out_maps.append(norm_map)
    out = torch.stack(out_maps, dim=0)
    return out


def create_grid(tensor_list, nrow=5):
    """
    Given a list of torch tensors (each shape like (C,H,W)), stack them into
    a single batch and make a grid.
    """
    # Concatenate on batch dimension
    batch = torch.stack(tensor_list, dim=0)
    grid = torchvision.utils.make_grid(batch, nrow=nrow)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/latent-diffusion/ffhq-ldm-vq-4.yaml")
    parser.add_argument("--checkpoint", type=str, default="./models/ldm/ffhq256/model.ckpt")
    parser.add_argument("--total_steps", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="output_maps")
    parser.add_argument("--seed", type=int, default=1125)
    parser.add_argument("--eta", type=float, default=1.0)  # Not used here, but included for completeness
    opt = parser.parse_args()

    # Hard-code the steps you want: 40, 80, 120, 160, 200
    # (Or compute them from total_steps & step_size in a range)
    step_list = [40, 80, 120, 160, 200]

    # Create output directory
    os.makedirs(opt.outdir, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    # Load config and model
    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    sd = torch.load(opt.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Instantiate extractors
    saliency_extractor = SaliencyExtractor(model)
    attention_extractor = AttentionExtractor(model)

    # Create some random latent. Adjust size as needed for your model
    latent = torch.randn(1, model.first_stage_model.embed_dim, 64, 64).to(device)

    # Lists to hold the 5-step images for final concatenation
    image_list = []
    saliency_list = []
    attention_list = []

    for step_num in step_list:
        # 1) Decode (or partially denoise) latent as "the image" at this step
        #    For demonstration, we're just decoding the *same* latent at each step.
        #    You might do actual diffusion steps if you want the partially denoised images.
        x_rec = decode_latent(model, latent)
        # We'll just take the first image in the batch
        image_list.append(x_rec[0].cpu())

        # 2) Compute and store saliency
        saliency_map = compute_saliency_map(latent.clone(), model, step_num, saliency_extractor)
        # saliency_map shape: (B, 1, H, W) or (B, C, H, W)
        # We'll just visualize the first in the batch
        sal_norm = normalize_map(saliency_map)[0]
        saliency_list.append(sal_norm.cpu())

        # 3) Compute and store attention
        attn_maps = compute_attention_map(latent.clone(), model, step_num, attention_extractor)
        # attn_maps is a list of attention features from each attention layer
        # Here, for simplicity, we just take the last one, average over heads if needed
        if len(attn_maps) > 0:
            last_map = attn_maps[-1]
            # last_map shape might be (B, nHeads, tokens, tokens) or something similar
            # We'll do a very rough approach: mean over heads, then choose the first batch
            if last_map.dim() == 4:
                attn_mean = last_map.mean(dim=1, keepdim=True)  # average over heads
                # Now we have shape (B, 1, tokens, tokens)
                # Make it square or interpret tokens as e.g. (H*W).
                # For demonstration, we just do min-max and store it directly:
                attn_norm = normalize_map(attn_mean)[0]
                attention_list.append(attn_norm.cpu())
            else:
                # If you have no heads dimension, do a simpler approach
                attn_norm = normalize_map(last_map)[0]
                attention_list.append(attn_norm.cpu())
        else:
            # In case there's no attention, just store zero
            zeros = torch.zeros_like(saliency_map[0])
            attention_list.append(zeros.cpu())

        print(f"Done step {step_num}.")

    # -------------------------------------------------------
    # Now create one final grid image for each category
    # -------------------------------------------------------
    #   image_step(40,80,120,160,200).png
    #   saliency_step(40,80,120,160,200).png
    #   attention_step(40,80,120,160,200).png
    # -------------------------------------------------------

    step_str = "(" + ",".join(str(s) for s in step_list) + ")"  # e.g. (40,80,120,160,200)

    # 1) Images
    image_grid = create_grid(image_list, nrow=len(step_list))
    save_image(image_grid, os.path.join(opt.outdir, f"image_step{step_str}.png"))
    print(f"Saved {os.path.join(opt.outdir, f'image_step{step_str}.png')}")

    # 2) Saliency
    saliency_grid = create_grid(saliency_list, nrow=len(step_list))
    save_image(saliency_grid, os.path.join(opt.outdir, f"saliency_step{step_str}.png"))
    print(f"Saved {os.path.join(opt.outdir, f'saliency_step{step_str}.png')}")

    # 3) Attention
    attention_grid = create_grid(attention_list, nrow=len(step_list))
    save_image(attention_grid, os.path.join(opt.outdir, f"attention_step{step_str}.png"))
    print(f"Saved {os.path.join(opt.outdir, f'attention_step{step_str}.png')}")

if __name__ == "__main__":
    main()

