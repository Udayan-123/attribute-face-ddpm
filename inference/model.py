import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- helpers ----------------
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def group_norm(c):
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c)


# ---------------- blocks ----------------
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init__()
        self.emb = nn.Linear(emb_dim, out_c)
        self.block = nn.Sequential(
            group_norm(in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            group_norm(out_c),
            nn.SiLU(),
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
        self.norm = group_norm(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(
            torch.matmul(q.transpose(-1, -2), k) * (c ** -0.5),
            dim=-1
        )
        out = torch.matmul(v, attn.transpose(-1, -2))
        return self.proj(out.reshape(b, c, h, w)) + x


# ---------------- UNet ----------------
class UNet(nn.Module):
    def __init__(self, cond_dim=40):
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

        self.cond_mlp = nn.Linear(cond_dim, time_dim)
        self.conv_in = nn.Conv2d(3, base, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleDict()

        for i in range(4):
            in_c = base if i == 0 else chs[i - 1]
            out_c = chs[i]
            self.down_blocks.append(nn.ModuleList([
                ResBlock(in_c, out_c, time_dim),
                ResBlock(out_c, out_c, time_dim)
            ]))
            if i in attn_levels:
                self.down_attn[str(i)] = Attention(out_c)

        self.mid = ResBlock(chs[-1], chs[-1], time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleDict()

        for i in reversed(range(4)):
            prev_c = chs[i + 1] if i < 3 else chs[-1]
            in_c = chs[i] + prev_c
            out_c = chs[i]
            self.up_blocks.append(nn.ModuleList([
                ResBlock(in_c, out_c, time_dim),
                ResBlock(out_c, out_c, time_dim)
            ]))
            if i in attn_levels:
                self.up_attn[str(i)] = Attention(out_c)

        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, t, cond):
        emb = self.time_mlp(timestep_embedding(t, 512)) + self.cond_mlp(cond)

        h = self.conv_in(x)
        skips = []

        for i, level in enumerate(self.down_blocks):
            for block in level:
                h = block(h, emb)
            skips.append(h)
            if str(i) in self.down_attn:
                h = self.down_attn[str(i)](h)
            if i < len(self.down_blocks) - 1:
                h = self.pool(h)

        h = self.mid(h, emb)

        for i, level in enumerate(self.up_blocks):
            skip = skips.pop()
            h = self.up(h)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in level:
                h = block(h, emb)
            idx = str(len(self.up_blocks) - 1 - i)
            if idx in self.up_attn:
                h = self.up_attn[idx](h)

        return self.conv_out(h)
