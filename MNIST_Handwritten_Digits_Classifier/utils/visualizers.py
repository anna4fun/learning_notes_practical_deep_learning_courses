import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from fastai.vision.all import *

## # Plotting the training stats, train loss, valid loss and valid accuracy of a Learner class
def plot_epoch_stats(plt, epoch_num, lr, learner_name, learner):
    train_loss = L(learner.recorder.values).itemgot(0)
    valid_loss = L(learner.recorder.values).itemgot(1)
    valid_acc = L(learner.recorder.values).itemgot(2)
    # Find best epoch (based on highest validation accuracy)
    best_epoch = int(np.argmax(valid_acc))
    best_acc = valid_acc[best_epoch]

    # Create 3 horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_loss)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(valid_loss)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch {best_epoch}")
    axes[1].grid(True)

    axes[2].plot(valid_acc)
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].axvline(best_epoch, color="red", linestyle="--")
    axes[2].plot(best_epoch, best_acc, "ro", label=f"Best Acc {best_acc:.2f}")
    axes[2].legend()
    axes[2].grid(True)

    # Add a global figure title
    fig.suptitle(f"{learner_name} Training Results (Epochs={epoch_num}, Learning Rate={lr})", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt


## Plots loss and accuracy on one chart with a twin y-axis for my hand-rolled 3-layer NN MLP
def _coerce_and_sort(d):
    """Return (epochs, values) as sorted numpy arrays. Keys can be str/int/float."""
    xs, ys = [], []
    for k, v in d.items():
        # try numeric key
        try:
            e = float(k)
        except Exception:
            # extract first number from string key
            m = re.search(r"[-+]?\d*\.?\d+", str(k))
            if not m:
                continue
            e = float(m.group())
        xs.append(e)
        ys.append(float(v))
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    order = np.argsort(xs)
    return xs[order], ys[order]

def plot_loss_and_accuracy(valid_loss: dict, valid_accuracy: dict, title="Validation Curves", savepath=None):
    epochs_l, loss = _coerce_and_sort(valid_loss)
    epochs_a, acc  = _coerce_and_sort(valid_accuracy)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs_l, loss, label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs_a, acc, linestyle="--", label="Accuracy")
    ax2.set_ylabel("Accuracy", color="orange")
    ax2.tick_params(axis="y", colors="orange")
    ax2.spines["right"].set_color("orange")


    # single legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
    return(plt)

## Helper functions for showing an image
def print_image_as_pixel(image_tensor, decimal=2):
    # only works in Jupyter env
    tdf = pd.DataFrame(image_tensor)
    styler = tdf.style.format(
        lambda x: f"{x:.{decimal}f}".rstrip('0').rstrip('.') if isinstance(x, float) else x
    )
    return (styler
            .set_properties(**{'font-size':'7pt'})
            .background_gradient(cmap='Greys'))

def show_image(image_tensor):
    # works in Python console
    arr = getattr(image_tensor, 'detach', lambda: image_tensor)().cpu().numpy() \
          if hasattr(image_tensor, 'cpu') else np.asarray(image_tensor)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            r, g, b = arr[0], arr[1], arr[2]
            arr = 0.2989*r + 0.5870*g + 0.1140*b
    plt.imshow(arr, cmap='gray')
    plt.axis('off')
    plt.show()

def file_size(image):
    return PILImage.create(image).size


## other misc
def _to_float(x):
    # Works for plain numbers, torch scalars (x.item), numpy scalars, etc.
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    if isinstance(x, numbers.Number):
        return float(x)
    return float(str(x))

def _to_epoch(k):
    # Prefer int→float→string (so 1, "1", 1.0 all sort correctly)
    try:
        return int(k)
    except Exception:
        try:
            return float(k)
        except Exception:
            return str(k)

def plot_losses(train_losses: dict, valid_losses: dict, title="Training & Validation Loss"):
    # Convert & sort
    tr = sorted(((_to_epoch(k), _to_float(v)) for k, v in train_losses.items()), key=lambda t: (isinstance(t[0], str), t[0]))
    va = sorted(((_to_epoch(k), _to_float(v)) for k, v in valid_losses.items()), key=lambda t: (isinstance(t[0], str), t[0]))

    tr_x, tr_y = zip(*tr) if tr else ([], [])
    va_x, va_y = zip(*va) if va else ([], [])

    # Plot (one chart, default colors)
    plt.figure(figsize=(8, 5))
    if tr_x: plt.plot(tr_x, tr_y, label="Train loss")
    if va_x: plt.plot(va_x, va_y, label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return(plt)

