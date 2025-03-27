import torch
import argparse
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

# Import necessary modules from the LDM codebase
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, device):
    """
    Loads the model from a given checkpoint and configuration file.

    Args:
        config: OmegaConf object with model configuration.
        ckpt: Path to the model checkpoint.
        device: Torch device to load the model on.

    Returns:
        Loaded model in evaluation mode.
    """
    print(f"Loading model from {ckpt}")
    # Load the checkpoint
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    # Extract the state dictionary
    sd = pl_sd["state_dict"]
    # Instantiate the model from configuration
    model = instantiate_from_config(config.model)
    # Load the state dictionary into the model
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    model.to(device)
    model.eval()
    return model

def main():
    # Argument parser for command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--step1", type=int, required=True, help="First inference step to capture")
    parser.add_argument("--step2", type=int, required=True, help="Second inference step to capture")
    parser.add_argument("--total_steps", type=int, default=50, help="Total number of sampling steps")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eta", type=float, default=1.0, help="Eta parameter for DDIM sampling (controls stochasticity)")
    args = parser.parse_args()

    # Ensure that step1 and step2 are within the total number of steps
    if args.step1 >= args.step2 or args.step2 > args.total_steps:
        print("Error: Ensure that step1 < step2 <= total_steps")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Set device to GPU if available, else CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load configuration and model
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.checkpoint, device)

    # Set random seed
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Initialize the sampler
    sampler = DDIMSampler(model)

    # Prepare conditioning (for unconditional model, use None)
    cond = None

    # Sampling parameters
    shape = [1, model.channels, model.image_size, model.image_size]

    with torch.no_grad():
        # Perform sampling
        samples, intermediates = sampler.sample(
            S=args.total_steps,
            conditioning=cond,
            batch_size=1,
            shape=shape[1:],
            verbose=True,
            eta=args.eta,
            return_intermediates=True,  # Ensure intermediates are returned
            log_every_t=1,              # Log intermediates at every timestep
        )

        # Decode the final sample
        final_image = model.decode_first_stage(samples)

        # Access intermediate latents
        x_inter = intermediates['x_inter']
        steps = len(x_inter)

        print(f"Number of intermediate steps available: {steps}")

        if args.step1 <= steps and args.step2 <= steps:
            latent_step1 = x_inter[args.step1 - 1].clone().detach()
            latent_step2 = x_inter[args.step2 - 1].clone().detach()
            image_step1 = model.decode_first_stage(latent_step1)
            image_step2 = model.decode_first_stage(latent_step2)
        else:
            print(f"Error: Specified steps ({args.step1}, {args.step2}) exceed the number of available intermediate steps ({steps}).")
            return

    # Compute differences between latents and images
    latent_diff = latent_step2 - latent_step1
    image_diff = image_step2 - image_step1

    # Process and save latent difference as grayscale image
    latent_diff_image = latent_diff.squeeze(0).mean(0)  # Average over channels
    latent_diff_image = latent_diff_image.cpu().numpy()
    # Normalize to [0, 255]
    latent_diff_image = (latent_diff_image - latent_diff_image.min()) / (latent_diff_image.max() - latent_diff_image.min() + 1e-8) * 255
    latent_diff_image = latent_diff_image.astype(np.uint8)
    latent_diff_image = Image.fromarray(latent_diff_image)
    latent_diff_image.save(os.path.join(args.outdir, 'latent_difference.png'))

    # Process and save image difference as grayscale image
    image_diff = image_diff.squeeze(0).cpu()
    image_diff_image = image_diff.mean(0)  # Average over channels
    image_diff_image = (image_diff_image - image_diff_image.min()) / (image_diff_image.max() - image_diff_image.min() + 1e-8) * 255
    image_diff_image = image_diff_image.numpy().astype(np.uint8)
    image_diff_image = Image.fromarray(image_diff_image)
    image_diff_image.save(os.path.join(args.outdir, 'image_difference.png'))

    # Optionally, save the intermediate images
    # Scale images to [0,1] range
    image_step1 = (image_step1 + 1.0) / 2.0
    image_step2 = (image_step2 + 1.0) / 2.0
    final_image = (final_image + 1.0) / 2.0

    # Save images
    save_image(image_step1, os.path.join(args.outdir, 'image_step1.png'))
    save_image(image_step2, os.path.join(args.outdir, 'image_step2.png'))
    save_image(final_image, os.path.join(args.outdir, 'final_image.png'))

if __name__ == "__main__":
    main()

