import os
import math
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
T = 1000  # diffusion steps used in training
CFG_SCALE = 4.0
MODEL_PATH = #EMA Weights
ATTR_PATH = #Attributes NPZ file of celeba dataset
N_SAMPLES = 4  # 2x2 grid
DDIM_STEPS = 100  # increase for better quality but slower

OUT_PATH = "C:/Users/Udayan/Downloads/sample_2x2.png"

# --- Load conditioning attributes ---
data = np.load(ATTR_PATH, allow_pickle=True)
attrs = data["attributes"].astype(np.float32)

# --- Helpers ---
def timestep_embedding(t, dim=512):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def gn(c):
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c)

# --- Blocks ---
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init__()
        self.emb = nn.Linear(emb_dim, out_c)
        self.block = nn.Sequential(
            gn(in_c), nn.SiLU(),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            gn(out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, emb):
        h = self.block(x)
        h = h + self.emb(emb)[:, :, None, None]
        return h + self.skip(x)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = gn(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.matmul(q.transpose(-1, -2), k) * (c ** -0.5), dim=-1)
        out = torch.matmul(v, attn.transpose(-1, -2))
        out = out.reshape(b, c, h, w)
        return self.proj(out) + x

# --- UNet model ---
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = 192
        chs = [192, 384, 768, 768]
        time_dim = 512
        attn_levels = {2, 3}

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Linear(40, time_dim)
        self.conv_in = nn.Conv2d(3, base, 3, padding=1)

        # down path
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleDict()
        for i in range(4):
            in_c  = base if i == 0 else chs[i-1]
            out_c = chs[i]
            level = nn.ModuleList([ResBlock(in_c, out_c, time_dim),
                                   ResBlock(out_c, out_c, time_dim)])
            self.down_blocks.append(level)
            if i in attn_levels:
                self.down_attn[str(i)] = Attention(out_c)

        # mid
        self.mid = ResBlock(chs[-1], chs[-1], time_dim)

        # up path
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleDict()
        for i in reversed(range(4)):
            skip_ch = chs[i]
            prev_ch = chs[i+1] if i < 3 else chs[-1]
            in_c = skip_ch + prev_ch
            out_c = chs[i]
            level = nn.ModuleList([
                ResBlock(in_c,  out_c, time_dim),
                ResBlock(out_c, out_c, time_dim)
            ])
            self.up_blocks.append(level)
            if i in attn_levels:
                self.up_attn[str(i)] = Attention(out_c)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.pool = nn.AvgPool2d(2)
        self.conv_out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, t, cond):
        emb = self.time_mlp(timestep_embedding(t, 512)) + self.cond_mlp(cond)
        hs = []
        h = self.conv_in(x)

        # Down
        for i, level in enumerate(self.down_blocks):
            for block in level:
                h = block(h, emb)
            hs.append(h)
            if str(i) in self.down_attn:
                h = self.down_attn[str(i)](h)
            if i != len(self.down_blocks) - 1:
                h = self.pool(h)

        # Mid
        h = self.mid(h, emb)

        # Up
        for i, level in enumerate(self.up_blocks):
            skip = hs.pop()
            h = self.up(h)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = level[0](h, emb)
            h = level[1](h, emb)
            attn_idx = str(len(self.up_blocks) - 1 - i)
            if attn_idx in self.up_attn:
                h = self.up_attn[attn_idx](h)

        return self.conv_out(h)

# --- Diffusion schedules ---
def cosine_schedule(T):
    s = 0.008
    t = torch.linspace(0, T, T+1)
    f = torch.cos((t/T + s)/(1+s) * math.pi/2) ** 2
    return torch.clamp(1 - f[1:] / f[:-1], 1e-4, 0.9999)

betas = cosine_schedule(T).to(DEVICE)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# --- EMA class ---
class EMA:
    def __init__(self, model):
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(0.9999).add_(v, alpha=1-0.9999)

    def load_state_dict(self, state, device):
        self.shadow = {k: v.to(device) for k,v in state.items()}

# --- Load model & EMA weights ---
model = UNet().to(DEVICE)
ema = EMA(model)
ema_state = torch.load(MODEL_PATH, map_location=DEVICE)
ema.load_state_dict(ema_state, DEVICE)

# --- Sampling function with tqdm and ETA ---
@torch.no_grad()
def sample_ddim(model, ema, cond, steps=DDIM_STEPS, cfg_scale=CFG_SCALE):
    model.eval()
    model.load_state_dict(ema.shadow)

    n = cond.shape[0]
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    ts = torch.linspace(T-1, 0, steps, device=DEVICE).long()

    # Unconditional zero conditioning
    uncond = torch.zeros_like(cond)

    start_time = time.time()

    for i in tqdm(range(len(ts)-1), desc="Generating", unit="step"):
        t = ts[i].repeat(n)
        t_next = ts[i+1].repeat(n)
        ab = alphas_bar[t].view(-1,1,1,1)
        ab_next = alphas_bar[t_next].view(-1,1,1,1)

        step_start = time.time()

        # Model prediction with CFG
        v_uncond = model(x, t, uncond)
        v_cond = model(x, t, cond)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        x0 = torch.sqrt(ab) * x - torch.sqrt(1 - ab) * v
        eps = torch.sqrt(1 - ab) * x + torch.sqrt(ab) * v

        x = torch.sqrt(ab_next) * x0 + torch.sqrt(1 - ab_next) * eps

        step_time = time.time() - step_start
        elapsed = time.time() - start_time
        remaining = step_time * (len(ts)-1 - i - 1)

        tqdm.write(f"Step {i+1}/{len(ts)-1} | Step time: {step_time:.2f}s | ETA: {remaining:.1f}s")

    model.train()
    return x.clamp(-1, 1).add(1).div(2).mul(255).byte()

# --- Prepare conditioning for sampling ---
np.random.seed(42)
idx = np.random.randint(0, len(attrs), N_SAMPLES)
cond = torch.from_numpy(attrs[idx]).to(DEVICE)

# --- Generate images ---
samples = sample_ddim(model, ema, cond, steps=DDIM_STEPS)

# --- Save ---
grid = samples.reshape(2, 2, 3, IMG_SIZE, IMG_SIZE).permute(0, 3, 1, 4, 2).reshape(2*IMG_SIZE, 2*IMG_SIZE, 3)
Image.fromarray(grid.cpu().numpy()).save(OUT_PATH)

print(f"Saved generated 2x2 image grid at {OUT_PATH}")
