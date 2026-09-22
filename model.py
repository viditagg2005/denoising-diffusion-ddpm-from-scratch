"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    betas = torch.linspace(beta_start,beta_end,T)
    return betas
    pass

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    alphas = 1.0-betas
    return alphas
    pass

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    alpha_bar = torch.cumprod(alphas,dim =0)
    return alpha_bar
    pass

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    batch_schedule = a.gather(0,t.long())
    batch_schedule = batch_schedule.reshape(-1,1,1,1)
    return batch_schedule
    pass

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    batch_t  = extract_into_batch(alphas_cumprod, t, x0)
    x_t  = torch.sqrt(batch_t)*x0 + torch.sqrt(1-batch_t)*noise
    return x_t
    pass

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = torch.linspace(beta_start,beta_end,T)
    alphas = 1- betas
    alphas_cumprod = torch.cumprod(alphas,dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1-alphas_cumprod)
    d = {'alphas': alphas, 'betas': betas, 'alphas_cumprod': alphas_cumprod, 'sqrt_alphas_cumprod': sqrt_alphas_cumprod, 'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod, 'T': T}
    return d
    pass

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    loss = F.mse_loss(noise_pred, noise, reduction = 'mean')
    return loss
    pass

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    x_t = q_sample(x0,t,noise,alphas_cumprod)
    noise_pred = model(x_t, t)
    loss = noise_prediction_loss(noise_pred, noise)
    return loss

    pass

# Step 9 - timestep_embedding
import torch
import math
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    half = dim//2
    if half == 1:
        freqs = torch.ones(half)
    else:
        indices = torch.arange(half)
        freqs = torch.exp(-math.log(10000.0) * indices/ (half -1))
    args = t.to(torch.float32)[:, None] * freqs[None, :]
    embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim = -1)
    return embeddings
    pass

# Step 10 - init_tiny_unet (not yet solved)
# TODO: implement

# Step 11 - tiny_unet_forward (not yet solved)
# TODO: implement

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

