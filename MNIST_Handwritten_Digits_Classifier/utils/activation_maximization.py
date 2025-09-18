
# =============== Activation Maximization Utilities ===============
# Visualizing activations of the MLP layers
import torch
import torch.nn as nn
import torch.nn.functional as F

def _mlp_forward_logits_and_hidden(x_img, w1, b1, w2, b2, activation_fn):
    """
    x_img: (B, 1, 28, 28) in [0,1]
    Returns:
      logits: (B, 10)
      hidden: (B, hidden_dim)
    """
    x = x_img.view(x_img.size(0), -1)  # (B, 784)
    h = activation_fn(x @ w1 + b1)     # (B, hidden)
    logits = h @ w2 + b2               # (B, 10)
    return logits, h

def _total_variation_loss(x_img, tv_strength=1.0):
    """
    Isotropic TV loss for smoothness.
    x_img: (B, 1, H, W)
    """
    dx = x_img[:, :, :, 1:] - x_img[:, :, :, :-1]
    dy = x_img[:, :, 1:, :] - x_img[:, :, :-1, :]
    return tv_strength * (dx.abs().mean() + dy.abs().mean())

@torch.no_grad()
def _to_display(x_img):
    """Detach to CPU numpy for visualization."""
    x = x_img.clamp(0, 1).squeeze(0).squeeze(0).cpu().numpy()
    return x

def activation_maximize(
    w1, b1, w2, b2,
    target_type="class",          # "class" or "neuron"
    target_index=0,               # class index (0..9) or hidden neuron index
    activation_fn=None,           # e.g., nn.LeakyReLU(0.1); default ReLU
    steps=400,
    lr=0.1,
    l2_weight=1e-3,               # L2 penalty on image
    tv_weight=1e-2,               # Total variation penalty on image
    jitter_pixels=2,              # random shift each step; 0 to disable
    seed=42,
    device=None,
    verbose_every=50
):
    """
    Perform activation maximization to synthesize an input image that:
      - maximizes a class logit (target_type='class'), or
      - maximizes a hidden neuron activation (target_type='neuron').

    Returns:
      result: dict with keys:
        'image' (torch.Tensor, 1x1x28x28),
        'history' (list of dicts), and 'target_type', 'target_index'
    """
    if device is None:
        device = w1.device

    if activation_fn is None:
        activation_fn = nn.ReLU()

    # Parameterize image via unconstrained tensor and map through sigmoid to [0,1]
    g = torch.Generator(device=device).manual_seed(seed)
    img_param = torch.randn(1, 1, 28, 28, device=device, generator=g, requires_grad=True)

    opt = torch.optim.Adam([img_param], lr=lr)
    history = []

    for t in range(1, steps + 1):
        opt.zero_grad()

        # Map to [0,1], apply random jitter (roll) for robustness
        img = torch.sigmoid(img_param)
        if jitter_pixels and jitter_pixels > 0:
            shift_x = int(torch.randint(-jitter_pixels, jitter_pixels + 1, (1,), generator=g, device=device).item())
            shift_y = int(torch.randint(-jitter_pixels, jitter_pixels + 1, (1,), generator=g, device=device).item())
            img_j = torch.roll(img, shifts=(shift_y, shift_x), dims=(2, 3))
        else:
            img_j = img

        logits, hidden = _mlp_forward_logits_and_hidden(img_j, w1, b1, w2, b2, activation_fn)

        if target_type == "class":
            target_value = logits[0, target_index]
        elif target_type == "neuron":
            target_value = hidden[0, target_index]
        else:
            raise ValueError("target_type must be 'class' or 'neuron'")

        # Regularization
        l2 = (img_j**2).mean()
        tv = _total_variation_loss(img_j)

        # We maximize target, so loss is negative target + regularizers
        loss = -target_value + l2_weight * l2 + tv_weight * tv
        loss.backward()
        opt.step()

        with torch.no_grad():
            hist_item = {
                "step": t,
                "target": float(target_value.detach().item()),
                "loss": float(loss.detach().item()),
                "l2": float(l2.detach().item()),
                "tv": float(tv.detach().item()),
            }
            history.append(hist_item)
            if verbose_every and t % verbose_every == 0:
                print(f"[AM] step {t:4d} | target={hist_item['target']:.4f} "
                      f"| loss={hist_item['loss']:.4f} | l2={hist_item['l2']:.4f} | tv={hist_item['tv']:.4f}")

    with torch.no_grad():
        img_final = torch.sigmoid(img_param)  # 1x1x28x28 in [0,1]

    return {
        "image": img_final,
        "history": history,
        "target_type": target_type,
        "target_index": target_index,
    }