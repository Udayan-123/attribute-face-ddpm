import torch
from torch.amp import autocast
from PIL import Image


@torch.no_grad()
def ddim_sample(
    model,
    ema_state,
    alphas_bar,
    attributes,
    img_size,
    steps,
    cfg_scale,
    device,
    out_path
):
    model.eval()

    # ---- swap EMA weights ----
    original = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state)

    n = attributes.shape[0]
    x = torch.randn(n, 3, img_size, img_size, device=device)
    uncond = torch.zeros_like(attributes)

    ts = torch.linspace(
        len(alphas_bar) - 1, 0, steps, device=device
    ).long()

    for i in range(len(ts) - 1):
        t = ts[i].repeat(n)
        t_next = ts[i + 1].repeat(n)

        ab = alphas_bar[t].view(-1, 1, 1, 1)
        ab_next = alphas_bar[t_next].view(-1, 1, 1, 1)

        with autocast(device_type="cuda"):
            v_u = model(x, t, uncond)
            v_c = model(x, t, attributes)
            v = v_u + cfg_scale * (v_c - v_u)

        x0 = torch.sqrt(ab) * x - torch.sqrt(1 - ab) * v
        eps = torch.sqrt(1 - ab) * x + torch.sqrt(ab) * v
        x = torch.sqrt(ab_next) * x0 + torch.sqrt(1 - ab_next) * eps

    img = (x.clamp(-1, 1) + 1) / 2
    img = (img * 255).byte()

    grid = img.reshape(4, 4, 3, img_size, img_size)
    grid = grid.permute(0, 3, 1, 4, 2)
    grid = grid.reshape(4 * img_size, 4 * img_size, 3)

    Image.fromarray(grid.cpu().numpy()).save(out_path)

    # ---- restore training weights ----
    model.load_state_dict(original)
    model.train()
