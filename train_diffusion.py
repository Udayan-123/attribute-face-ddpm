# =========================================================
#IMPORTS
# =========================================================

import os, math, random, tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# =========================================================
# CONFIG
# =========================================================
IMG_DIR = "/content/celeba/128x128" #Images Resized to 128x128
ATTR_PATH = "/content/celeba_attributes_01.npz" #CelebA attributes converted to [0,1]
OUT_DIR = "/content/drive/MyDrive/celeba_ddpm"

IMG_SIZE = 128
BATCH_SIZE = 4
LR = 1e-4
T = 1000
CFG_DROP_PROB = 0.15
CFG_SCALE = 4.0
EMA_DECAY = 0.9999

SAVE_EVERY = 10000
SAMPLE_EVERY = 3000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# DATASET
# =========================================================
class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_path):
        data = np.load(attr_path, allow_pickle=True)
        self.files = data["filenames"]
        self.attrs = data["attributes"].astype(np.float32)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.files[idx])).convert("RGB")
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 127.5 - 1
        cond = torch.from_numpy(self.attrs[idx])
        return img, cond

# =========================================================
# HELPERS
# =========================================================
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def gn(c):
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c)

# =========================================================
# BLOCKS
# =========================================================
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

# =========================================================
# UNET 
# =========================================================
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
            # channels coming from previous layer (or mid) + skip channels
            skip_ch = chs[i]
            prev_ch = chs[i+1] if i < 3 else chs[-1]  # mid layer for first up block
            in_c = skip_ch + prev_ch  # because of concatenation with skip
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

# =========================================================
# DIFFUSION
# =========================================================
def cosine_schedule(T):
    s = 0.008
    t = torch.linspace(0, T, T+1)
    f = torch.cos((t/T + s)/(1+s) * math.pi/2) ** 2
    return torch.clamp(1 - f[1:] / f[:-1], 1e-4, 0.9999)

betas = cosine_schedule(T).to(DEVICE)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# =========================================================
# EMA
# =========================================================
class EMA:
    def __init__(self, model):
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(EMA_DECAY).add_(v, alpha=1-EMA_DECAY)

    def cpu_state(self):
        return {k: v.cpu() for k,v in self.shadow.items()}
    
    def load_state_dict(self, state, device):
        self.shadow = {k: v.to(device) for k, v in state.items()}

# =========================================================
# CHECKPOINTING
# =========================================================
def save_checkpoint(state, name):
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=OUT_DIR)
    torch.save(state, tmp.name)
    os.replace(tmp.name, os.path.join(OUT_DIR, name))

# =========================================================
# DDIM SAMPLING
# =========================================================
@torch.no_grad()
def sample_ddim(model, ema, dataset, step, n=16, steps=50):
    model.eval()
    original_weights = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.shadow)

    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    idx = np.random.randint(0, len(dataset), n)
    cond = torch.from_numpy(dataset.attrs[idx]).to(DEVICE)
    uncond = torch.zeros_like(cond)
    ts = torch.linspace(T-1, 0, steps, device=DEVICE).long()

    for i in range(len(ts)-1):
        t = ts[i].repeat(n).to(DEVICE)
        t_next = ts[i+1].repeat(n).to(DEVICE)
        ab = alphas_bar[t].view(-1,1,1,1)
        ab_next = alphas_bar[t_next].view(-1,1,1,1)

        with autocast(device_type="cuda"):
            v_u = model(x, t, uncond)
            v_c = model(x, t, cond)
            v = v_u + CFG_SCALE*(v_c - v_u)

        x0  = torch.sqrt(ab) * x - torch.sqrt(1 - ab) * v
        eps = torch.sqrt(1 - ab) * x + torch.sqrt(ab) * v

        x = (
            torch.sqrt(ab_next) * x0 +
            torch.sqrt(1 - ab_next) * eps
        ) 
    img = (x.clamp(-1,1)+1)/2
    img = (img*255).byte()
    grid = img.reshape(4,4,3,IMG_SIZE,IMG_SIZE).permute(0,3,1,4,2).reshape(4*IMG_SIZE,4*IMG_SIZE,3)
    Image.fromarray(grid.cpu().numpy()).save(f"{OUT_DIR}/sample_{step}.png")
    model.load_state_dict(original_weights)
    model.train()

# =========================================================
# TRAINING LOOP
# =========================================================
dataset = CelebADataset(IMG_DIR, ATTR_PATH)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

model = UNet().to(DEVICE)
ema = EMA(model)
opt = torch.optim.AdamW(model.parameters(), LR, betas=(0.9,0.99), weight_decay=0.01)
scaler = GradScaler()

# Resume checkpoint if exists
latest_ckpt = os.path.join(OUT_DIR, "latest.pt")
if os.path.exists(latest_ckpt):
    ckpt = torch.load(
      latest_ckpt, 
      map_location=DEVICE, 
      weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.shadow = {k:v.to(DEVICE) for k,v in ckpt["ema"].items()}
    opt.load_state_dict(ckpt["opt"])
    scaler.load_state_dict(ckpt["scaler"])
    step = ckpt["step"]
    # --- Restore RNG states safely (PyTorch 2.6 compatible) ---

    torch_rng = ckpt["rng"]["torch"]

    if isinstance(torch_rng, torch.Tensor):
        torch_rng = torch_rng.detach().cpu()
    else:
        torch_rng = torch.ByteTensor(torch_rng)

    torch.set_rng_state(torch_rng)

    if torch.cuda.is_available():
        cuda_rng = ckpt["rng"]["cuda"]
        torch.cuda.set_rng_state_all([
            s.detach().cpu() if isinstance(s, torch.Tensor)
            else torch.ByteTensor(s)
            for s in cuda_rng
          ])

    np.random.set_state(ckpt["rng"]["numpy"])
    random.setstate(ckpt["rng"]["random"])


    print(f"Resumed from step {step}")
else:
    step = 0
    print("No checkpoint found, starting from scratch.")

model.train()

while True:
    for x, cond in loader:
        step += 1
        x = x.to(DEVICE)
        cond = cond.to(DEVICE)

        if random.random() < CFG_DROP_PROB:
            cond.zero_()

        t = torch.randint(0, T, (x.size(0),), device=DEVICE)
        noise = torch.randn_like(x)
        ab = alphas_bar[t].view(-1,1,1,1)
        xt = torch.sqrt(ab)*x + torch.sqrt(1-ab)*noise
        v = torch.sqrt(ab)*noise - torch.sqrt(1-ab)*x

        with autocast(device_type="cuda"):
            pred = model(xt, t, cond)
            loss = F.mse_loss(pred, v)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

        if step % SAVE_EVERY == 0:
            state = {
                "model": model.state_dict(),
                "ema": ema.cpu_state(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "random": random.getstate(),
                }
            }
            save_checkpoint(state, "latest.pt")
            save_checkpoint(state, f"step_{step}.pt")
  
        # -------------------------------------------------
        # CLEAN INFERENCE CHECKPOINTS
        # -------------------------------------------------
        if step in {150_000, 200_000}:
            torch.save(
                model.state_dict(),
                f"{OUT_DIR}/attribute_face_ddpm_model_{step//1000}k.pt"
            )
            torch.save(
                ema.cpu_state(),
                f"{OUT_DIR}/attribute_face_ddpm_ema_{step//1000}k.pt"
            )
            print(f"Saved clean model + EMA checkpoints at step {step}")


        if step % SAMPLE_EVERY == 0:
            sample_ddim(model, ema, dataset, step)

#End of file