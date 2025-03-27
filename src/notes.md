# Notes on each code


# `ldm_loader.py`

- uses `omegaconf` and `instantiate_from_config` from the CompVis codebase.
- `strict=False` because some pretrained models might have minor key mismatches.

# `partial_steps.py`
- `sample_partial_steps` uses `DDIMSampler` from the LDM code to do partial steps.
- We decode latents at each step interval and save them.
- Adjust the latent `shape` to match your model's config.

# `grid_visualization.py`

- We store each step in `step_intervals` like `[40,80,120,160,200]` and create a single row grid.
- You can modify to create multi-row grid if you want.

# `pixel_analysis.py`

- this code saves partial latents, decodes them, and optionally computes differences.
- `latent_to_grayscale` is a simplistic function just to show how you might visualize latent channels.

# `attn_and_sal.py`

- in a real script, you'd do a forward + backward pass to compute true Grad-CAM.
- `find_cross_attention_layer` must be adapted to your LDM's naming structure.



