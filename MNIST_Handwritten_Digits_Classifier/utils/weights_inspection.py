import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
This module covers:
1.Visualizing first Linear layer weights as images (e.g., MNIST 28×28).
2.Listing top-k strongest neurons (by weight norm).
3.Inspecting deeper Linear layers by mapping a chosen output neuron back to the input space with a simple linear back-projection (product of weight matrices)—useful for a rough “what input pattern would this logit like?” view.

Usage overview
1. For image inputs (e.g., MNIST), call plot_first_layer_filters_as_images(...).
2. To see which first-layer neurons are “strongest,” call topk_first_layer_neurons(...).
3. To peek at deeper layers, call backproject_to_input(...) and then show_as_image(...).
"""

# ---------- Helpers ----------

def _to_numpy(t):
    return t.detach().cpu().numpy()

def _minmax01(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)

def show_as_image(vec, input_shape=(1, 28, 28), title=None):
    """
    vec: 1D tensor/np array shaped like flattened input (C*H*W) or same shape as input.
    input_shape: e.g. (1,28,28) for MNIST, or (3,32,32) for RGB.
    """
    v = vec
    if isinstance(v, torch.Tensor):
        v = _to_numpy(v)

    if v.ndim == 1:
        v = v.reshape(input_shape)
    # collapse channels for display if needed
    if v.ndim == 3 and v.shape[0] == 1:
        v = v[0]  # grayscale
        cmap = 'gray'
    else:
        cmap = None  # matplotlib will show RGB if (3,H,W)

    v = _minmax01(v)
    plt.imshow(v if cmap is None else v, cmap=cmap)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# ---------- 1) First-layer visualization as images ----------
def plot_first_layer_filters_as_images(first_linear: nn.Linear,
                                       input_shape=(1, 28, 28),
                                       cols=8,
                                       sort_by='l2',  # 'l2' | 'none' | 'absmean'
                                       max_neurons=None,
                                       weight=None):
    """
    Visualize each neuron’s weight vector in the first Linear as an image.

    Accepts one of:
      - first_linear: nn.Linear
      - weight: torch.Tensor of shape [out_features, in_features] or [in_features, out_features]
      - (for convenience) first_linear can also be a torch.Tensor weight matrix

    first_linear.weight: shape [out_features, in_features]
    input_shape: image shape (C,H,W) that matches in_features == C*H*W
    sort_by:
      - 'none': original order
      - 'l2': sort neurons by L2 norm descending
      - 'absmean': sort by mean absolute weight descending
    max_neurons: limit how many to display (for large layers)

    Notes:
      - This is most meaningful when the input is image-like.
      - If your model has a Flatten before first_linear, pass the actual Linear that consumes flattened pixels.
    """
    # Resolve weight matrix W as [out_features, in_features]
    W_raw = None
    if weight is not None:
        W_raw = weight
    elif isinstance(first_linear, nn.Linear):
        W_raw = first_linear.weight
    elif isinstance(first_linear, torch.Tensor):
        W_raw = first_linear
    else:
        raise TypeError("plot_first_layer_filters_as_images expects nn.Linear, a weight tensor, "
                        "or use the 'weight=' argument.")

    W = W_raw.detach().cpu()
    C, H, Wimg = input_shape
    in_expected = C * H * Wimg

    # Auto-handle [out, in] vs [in, out]
    if W.ndim != 2:
        raise ValueError(f"Weight must be 2D, got shape {tuple(W.shape)}")

    if W.shape[1] == in_expected:
        # already [out, in]
        out_features, in_features = W.shape
    elif W.shape[0] == in_expected:
        # transpose from [in, out] -> [out, in]
        W = W.t().contiguous()
        out_features, in_features = W.shape
    else:
        raise AssertionError(f"Weight shape {tuple(W.shape)} incompatible with input_shape={input_shape}. "
                             f"Expected one dimension to equal C*H*W={in_expected}.")

    # choose order
    if sort_by == 'l2':
        norms = torch.norm(W, dim=1)
        idx = torch.argsort(norms, descending=True)
    elif sort_by == 'absmean':
        absmean = W.abs().mean(dim=1)
        idx = torch.argsort(absmean, descending=True)
    else:
        idx = torch.arange(out_features)

    if max_neurons is not None:
        idx = idx[:max_neurons]

    weights = W[idx]  # [k, in]

    k = weights.shape[0]
    cols = min(cols, k)
    rows = math.ceil(k / cols)

    plt.figure(figsize=(1.8*cols, 1.8*rows))
    for i in range(k):
        w = weights[i].reshape(C, H, Wimg)
        # display as grayscale if single-channel
        if C == 1:
            img = _minmax01(_to_numpy(w[0]))
            cmap = 'gray'
        else:
            img = _minmax01(_to_numpy(w))
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            cmap = None
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    plt.suptitle("First-layer weight 'filters'")
    plt.tight_layout()
    plt.show()

def plot_first_layer_filters_as_images_backup(first_linear: nn.Linear,
                                       input_shape=(1, 28, 28),
                                       cols=8,
                                       sort_by='l2',  # 'l2' | 'none' | 'absmean'
                                       max_neurons=None):
    """
    Visualize each neuron’s weight vector in the first Linear as an image.

    first_linear.weight: shape [out_features, in_features]
    input_shape: image shape (C,H,W) that matches in_features == C*H*W
    sort_by:
      - 'none': original order
      - 'l2': sort neurons by L2 norm descending
      - 'absmean': sort by mean absolute weight descending
    max_neurons: limit how many to display (for large layers)

    Notes:
      - This is most meaningful when the input is image-like.
      - If your model has a Flatten before first_linear, pass the actual Linear that consumes flattened pixels.
    """
    W = first_linear.weight.detach().cpu()  # [out, in]
    out_features, in_features = W.shape
    C, H, Wimg = input_shape
    assert in_features == C * H * Wimg, f"in_features={in_features} != C*H*W={C*H*Wimg}"

    # choose order
    if sort_by == 'l2':
        norms = torch.norm(W, dim=1)
        idx = torch.argsort(norms, descending=True)
    elif sort_by == 'absmean':
        absmean = W.abs().mean(dim=1)
        idx = torch.argsort(absmean, descending=True)
    else:
        idx = torch.arange(out_features)

    if max_neurons is not None:
        idx = idx[:max_neurons]

    weights = W[idx]  # [k, in]

    k = weights.shape[0]
    cols = min(cols, k)
    rows = math.ceil(k / cols)

    plt.figure(figsize=(1.8*cols, 1.8*rows))
    for i in range(k):
        w = weights[i].reshape(C, H, Wimg)
        # display as grayscale if single-channel
        if C == 1:
            img = _minmax01(_to_numpy(w[0]))
            cmap = 'gray'
        else:
            img = _minmax01(_to_numpy(w))
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            cmap = None
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
    plt.suptitle("First-layer weight 'filters'")
    plt.tight_layout()
    plt.show()

def topk_first_layer_neurons(first_linear: nn.Linear, k=10, metric='l2'):
    """
    Return indices of top-k strongest neurons in the first layer.
    metric: 'l2' or 'absmean'
    """
    W = first_linear.weight.detach().cpu()
    if metric == 'l2':
        scores = torch.norm(W, dim=1)
    elif metric == 'absmean':
        scores = W.abs().mean(dim=1)
    else:
        raise ValueError("metric must be 'l2' or 'absmean'")
    vals, idx = torch.topk(scores, k)
    return idx.tolist(), vals.tolist()

# ---------- 2) Back-project deeper layer / class logit to input space ----------

def backproject_to_input(linears, target_vector):
    """
    Compute a simple linear back-projection to input space.

    Args:
      linears: list of nn.Linear layers that connect input->...->target,
               in forward order. Example for MLP:
                 [Linear(in->h1), Linear(h1->h2), Linear(h2->out)]
               If your MLP has ReLUs etc., we ignore them here for a linearized view.
      target_vector: 1D tensor of shape equal to the last Linear's out_features.
                     For example, to visualize the j-th class logit, use a one-hot
                     vector e_j.

    Returns:
      A 1D tensor in input space (same length as in_features of the first Linear).
      Conceptually: W_eff^T * target, where W_eff = W_last @ W_(L-1) @ ... @ W_first
    """
    assert isinstance(linears, (list, tuple)) and len(linears) >= 1
    # Build effective matrix product (last @ ... @ first)
    with torch.no_grad():
        W_eff = None
        for L in linears:
            W = L.weight  # [out, in]
            W_eff = W if W_eff is None else (W @ W_eff)
        # Now W_eff maps input -> target_space
        # We want a vector in input space that, when dotted with input, aligns with target_vector.
        # A reasonable linearized visualization is W_eff^T @ target_vector
        v = W_eff.t() @ target_vector  # [in_features]
    return v

def visualize_class_logit_as_input_pattern(linears, class_idx, input_shape=(1,28,28), title=None):
    """
    Convenience: one-hot on class_idx, back-project, and show as image.
    """
    out_dim = linears[-1].out_features
    e = torch.zeros(out_dim)
    e[class_idx] = 1.0
    v = backproject_to_input(linears, e)
    show_as_image(v, input_shape=input_shape, title=title or f"Linearized pattern for class {class_idx}")

# ---------- 3) Tabular-oriented inspection ----------

def top_features_for_first_neuron(first_linear: nn.Linear, neuron_idx: int, feature_names=None, k=20):
    """
    For tabular inputs, list features with largest |weight| for a chosen neuron.
    """
    w = first_linear.weight.detach().cpu()[neuron_idx]  # [in_features]
    vals = torch.abs(w)
    topv, topi = torch.topk(vals, k)
    topi = topi.tolist()
    topv = topv.tolist()
    rows = []
    for rank, (i, v) in enumerate(zip(topi, topv), 1):
        name = feature_names[i] if feature_names is not None else f"feat_{i}"
        rows.append((rank, name, float(w[i]), float(v)))
    # Pretty print
    print(f"Top-{k} features for neuron {neuron_idx}:")
    print(f"{'rank':>4}  {'feature':<24}  {'weight':>12}  {'|weight|':>12}")
    for r in rows:
        print(f"{r[0]:>4}  {r[1]:<24}  {r[2]:>12.6f}  {r[3]:>12.6f}")
    return rows
