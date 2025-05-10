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
    parser.add_argument("--total_steps", type=int, default=50, help="Total number of sampling steps")
    parser.add_argument("--step_size", type=int, default=10, help="Interval of steps at which to save images")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eta", type=float, default=1.0, help="Eta parameter for DDIM sampling (controls stochasticity)")
    args = parser.parse_args()

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
        final_latent = samples.clone()
        final_image = model.decode_first_stage(samples)

        # Access intermediate latents
        x_inter = intermediates['x_inter']
        steps = len(x_inter)

        print(f"Number of intermediate steps available: {steps}")

        # Generate list of step numbers where images will be saved
        save_steps = [s for s in range(1, steps + 1) if s % args.step_size == 0]

        latents_at_steps = {}
        images_at_steps = {}

        for step_num in save_steps:
            idx = step_num - 1  # Adjust for zero-based indexing
            latent = x_inter[idx].clone().detach()
            image = model.decode_first_stage(latent)

            # Save latent and image
            latents_at_steps[step_num] = latent
            images_at_steps[step_num] = image

            # Save latent image (as grayscale for visualization)
            latent_image = latent.squeeze(0).mean(0)  # Average over channels
            latent_image = latent_image.cpu().numpy()
            latent_image = (latent_image - latent_image.min()) / (latent_image.max() - latent_image.min() + 1e-8) * 255
            latent_image = latent_image.astype(np.uint8)
            latent_image = Image.fromarray(latent_image)
            latent_image.save(os.path.join(args.outdir, f'latent_step_{step_num}.png'))

            # Save decoded image
            image_to_save = (image + 1.0) / 2.0  # Scale to [0,1]
            save_image(image_to_save, os.path.join(args.outdir, f'image_step_{step_num}.png'))

        # Save final latent and image
        latents_at_steps[args.total_steps] = final_latent
        images_at_steps[args.total_steps] = final_image

        # Save final latent image
        latent_image = final_latent.squeeze(0).mean(0)  # Average over channels
        latent_image = latent_image.cpu().numpy()
        latent_image = (latent_image - latent_image.min()) / (latent_image.max() - latent_image.min() + 1e-8) * 255
        latent_image = latent_image.astype(np.uint8)
        latent_image = Image.fromarray(latent_image)
        latent_image.save(os.path.join(args.outdir, f'latent_step_{args.total_steps}.png'))

        # Save final decoded image
        image_to_save = (final_image + 1.0) / 2.0  # Scale to [0,1]
        save_image(image_to_save, os.path.join(args.outdir, f'image_step_{args.total_steps}.png'))

        # Compute and save differences
        # Differences between the final step and each saved intermediate step
        for step_num in save_steps:
            # Compute latent difference
            latent_diff = latents_at_steps[args.total_steps] - latents_at_steps[step_num]
            # Save latent difference image
            latent_diff_image = latent_diff.squeeze(0).mean(0)  # Average over channels
            latent_diff_image = latent_diff_image.cpu().numpy()
            latent_diff_image = (latent_diff_image - latent_diff_image.min()) / (latent_diff_image.max() - latent_diff_image.min() + 1e-8) * 255
            latent_diff_image = latent_diff_image.astype(np.uint8)
            latent_diff_image = Image.fromarray(latent_diff_image)
            latent_diff_image.save(os.path.join(args.outdir, f'latent_diff_{args.total_steps}_minus_{step_num}.png'))

            # Compute image difference
            image_diff = images_at_steps[args.total_steps] - images_at_steps[step_num]
            # Save image difference
            image_diff_image = image_diff.squeeze(0).cpu()
            image_diff_image = image_diff_image.mean(0)  # Average over channels
            image_diff_image = (image_diff_image - image_diff_image.min()) / (image_diff_image.max() - image_diff_image.min() + 1e-8) * 255
            image_diff_image = image_diff_image.numpy().astype(np.uint8)
            image_diff_image = Image.fromarray(image_diff_image)
            image_diff_image.save(os.path.join(args.outdir, f'image_diff_{args.total_steps}_minus_{step_num}.png'))

        # Optionally, compute all possible differences between saved steps
        # Uncomment the following code if you want all combinations
        '''
        from itertools import combinations

        step_numbers = save_steps.copy()
        step_numbers.append(args.total_steps)  # Include the final step
        for step1, step2 in combinations(step_numbers, 2):
            # Compute latent difference
            latent_diff = latents_at_steps[step2] - latents_at_steps[step1]
            # Save latent difference image
            latent_diff_image = latent_diff.squeeze(0).mean(0)  # Average over channels
            latent_diff_image = latent_diff_image.cpu().numpy()
            latent_diff_image = (latent_diff_image - latent_diff_image.min()) / (latent_diff_image.max() - latent_diff_image.min() + 1e-8) * 255
            latent_diff_image = latent_diff_image.astype(np.uint8)
            latent_diff_image = Image.fromarray(latent_diff_image)
            latent_diff_image.save(os.path.join(args.outdir, f'latent_diff_{step2}_minus_{step1}.png'))

            # Compute image difference
            image_diff = images_at_steps[step2] - images_at_steps[step1]
            # Save image difference
            image_diff_image = image_diff.squeeze(0).cpu()
            image_diff_image = image_diff_image.mean(0)  # Average over channels
            image_diff_image = (image_diff_image - image_diff_image.min()) / (image_diff_image.max() - image_diff_image.min() + 1e-8) * 255
            image_diff_image = image_diff_image.numpy().astype(np.uint8)
            image_diff_image = Image.fromarray(image_diff_image)
            image_diff_image.save(os.path.join(args.outdir, f'image_diff_{step2}_minus_{step1}.png'))
        '''

if __name__ == "__main__":
    main()

