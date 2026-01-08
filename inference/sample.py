import torch
import numpy as np

from model import UNet
from sampler import ddim_sample


# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 128
TIMESTEPS = 1000
DDIM_STEPS = 50
CFG_SCALE = 4.0

EMA_PATH = "../outputs/attribute_face_ddpm_ema_130k.pt"
OUT_PATH = "sample.png"
# --------------------------------------


def cosine_schedule(T, device):
    import math
    t = torch.linspace(0, T, T + 1, device=device)
    f = torch.cos((t / T + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = torch.clamp(1 - f[1:] / f[:-1], 1e-4, 0.9999)
    alphas = 1 - betas
    return torch.cumprod(alphas, dim=0)


model = UNet().to(DEVICE)
ema_state = torch.load(EMA_PATH, map_location=DEVICE)

alphas_bar = cosine_schedule(TIMESTEPS, DEVICE)

# example attributes (replace with real ones)
attributes = torch.randn(16, 40, device=DEVICE)

ddim_sample(
    model=model,
    ema_state=ema_state,
    alphas_bar=alphas_bar,
    attributes=attributes,
    img_size=IMG_SIZE,
    steps=DDIM_STEPS,
    cfg_scale=CFG_SCALE,
    device=DEVICE,
    out_path=OUT_PATH
)
