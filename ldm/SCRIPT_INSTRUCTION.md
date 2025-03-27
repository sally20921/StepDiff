# Consolidated Overview of Each Script

Below is a consolidated overview of various scripts I used for initial experiment.
For each, I explain

1. What the source code does 
2. How you typically run it from the command line 
3. What output it produces 
4. Additional notes on internal workings 

## `ldm_fixed.py`

### What It Does

- this is a variation of the latent diffusion sampling script. 
 1. loads a latent diffusion model (LDM) from config + checkpoint
 2. performs partial-step sampling with a certain number of total steps (`--total_steps`) and save images at intervals (`--step_size`). 
 3. optinally collects latent difference, image difference, or partial decodes at each step. 

### Outputs

- saves `image_step_{n}.png`, `latent_step_{n}.png` for each logged step.
- possibly saves `image_diff_{m}_minus_{n}.png`, `latent_diff_{m}_minus_{n}.png`. 
- final output image is image at `--total_steps`, so it is stored as `image_step_{TOTAL_STEPS}.png`.

### How to Run

```bash
python ldm_fixed.py --config ./configs/latent-diffusion/ffhq-ldm-vq-4.yaml --checkpoint ./models/ldm/ffhq256/model.ckpt --total_steps 200 --step_size 40 --outdir 2025-03-25 --seed 1125 --eta 1.0
```

- `--config`: points to LDM config (e.g. ffhq)
- `--checkpoint`: path to the `.ckpt` model file
- `--total_steps`: e.g. 200 DDIM steps
- `--step_size`: save intermediate results every 40 steps
- `--outdir`: output directory for images, differences, etc.
- `--eta`: the DDIM `eta` parameter (stochasticity).
- `--seed`: random seed for reproduction of the same image. 


### How It Works
- internally, it uses a DDIMSampler or similar to generate partial latents at each step. 
- decodes them to 256x256 images. 
- (optionally) computes differences between pixels. 

## `ldm_grid.py`

### What It Does

- another variation focusing on creating multi-column grid outputs
 1. runs partial-step sampling like `ldm_fixed.py`
 2. collects intermediate latents, and decodes them.
 3. assembles them into a single `.png` grid with columns for steps `[40,80,120,160,200]`. 


### Outputs

- `image_step(40,80,120,160,200).png`: a simple PNG with 5 columns.
- `latent_step(40,80,120,160,200).png`.

### How to Run

```bash
python ldm_grid.py --config ./configs/latent-diffusion/ffhq-ldm-vq-4.yaml --checkpoint ./models/ldm/ffhq256/model.ckpt --total_steps 200 --step_size 40 --outdir 2025-01-25-grid-results --seed 1125 --eta 1.0
```

### How It Works

- partial-step sampling approach is similar to `ldm_fixed.py`
- summarizes them in one grid image instead of separate PNGs per step.
- this is handy for side-by-side visualization of progressive denoising. 

 partial-step sampling approach is similar to `ldm_fixed.py`
 - summarizes them in one grid image instead of separate PNGs per step.
 - this is handy for side-by-side visualization of progressive denoising. 

  partial-step sampling approach is similar to `ldm_fixed.py`
  - summarizes them in one grid image instead of separate PNGs per step.
  - this is handy for side-by-side visualization of progressive denoising. 

   partial-step sampling approach is similar to `ldm_fixed.py`
   - summarizes them in one grid image instead of separate PNGs per step.
   - this is handy for side-by-side visualization of progressive denoising. 

## `ldm_upsample.py`

### What It Does

This is a well-rounded script that
 1. Does partial-step sampling
 2. Computes saliency via a gradient-based approach at each step.
 3. Computes attention at each step. 
 4. Upsamples the saliency/attention from the latent resolution (64x64) to the final 256x256 for overlaying.
 5. Create final multi-column PNGs: `image_step(40,80,120,160,200).png`, `saliency_step(40,80,120,160,200).png`, `attention_step(40,80,120,160,200).png`.

### Outputs

Three PNGs if you request image/saliency/attention.
- `image_step(40,80,120,160,200).png`
- `saliency_step(40,80,120,160,200).png`
- `attention_step(40,80,120,160,200).png`
Each shows a grid (5 columns) for partial steps. Saliency map is overlaid in a pseudo-code color or alpha-blend manner. 

### How to Run

```bash
```






























