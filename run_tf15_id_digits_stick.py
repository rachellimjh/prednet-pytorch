"""
Runs the Moving MNIST trained model on ID_STICK.npz with 15 ground-truth
conditioning frames (teacher forcing k=15), then saves predicted vs actual
frame plots for a handful of sequences.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from prednet import PredNet
from mnist_training.mnist_dataset import MovingMNISTDataset

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
WEIGHTS_FILE  = "models/mnist_id_digits/prednet_mmnist_best.pth"
NPZ_FILE      = "data/custom/ID_STICK.npz"
RESULTS_DIR   = "results/id_digits/id_digits_stick/tf15_prediction_plots"
NT            = 20
CONTEXT_K     = 15   # teacher forcing horizon
BATCH_SIZE    = 8
N_PLOT        = 5    # number of sequences to plot
# -----------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
stack_sizes = (1, 48, 96, 192)
model = PredNet(
    stack_sizes, stack_sizes, (3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3),
    output_mode="prediction",
    extrap_start_time=CONTEXT_K,
)
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
model.to(device)
model.eval()
print(f"Loaded weights from {WEIGHTS_FILE}")
print(f"Teacher forcing k={CONTEXT_K}  (frames 0-{CONTEXT_K-1} = ground truth, frames {CONTEXT_K}-{NT-1} = predicted)")

# Load data
dataset = MovingMNISTDataset(npz_file=NPZ_FILE, nt=NT, split="all")
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Run one batch and collect frames for plotting
with torch.no_grad():
    inputs = next(iter(loader)).to(device)
    if inputs.dtype != torch.float32:
        inputs = inputs.float()
    if inputs.max() > 1.0:
        inputs = inputs / 255.0
    preds = model(inputs)

X     = inputs.cpu().numpy()   # (B, T, 1, H, W)
X_hat = preds.cpu().numpy()

# Plot
for i in range(min(N_PLOT, X.shape[0])):
    fig = plt.figure(figsize=(NT * 1.1, 4))
    gs  = gridspec.GridSpec(2, NT, wspace=0.05, hspace=0.1)

    axes_grid = {}
    for t in range(NT):
        # Predicted (top row)
        ax = fig.add_subplot(gs[0, t])
        axes_grid[(0, t)] = ax
        ax.imshow(X_hat[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if t == 0:
            ax.set_title("Predicted", loc="left", fontsize=9)

        # Ground truth (bottom row)
        ax = fig.add_subplot(gs[1, t])
        axes_grid[(1, t)] = ax
        ax.imshow(X[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if t == 0:
            ax.set_title("Ground Truth", loc="left", fontsize=9)

        # Frame number label
        fig.text(
            (t + 0.5) / NT, 0.01, str(t + 1),
            ha="center", va="bottom", fontsize=7, color="gray"
        )

    # Vertical line between frame CONTEXT_K (last context) and CONTEXT_K+1 (first predicted)
    fig.canvas.draw()
    x_right = axes_grid[(0, CONTEXT_K - 1)].get_position().x1
    x_left  = axes_grid[(0, CONTEXT_K)].get_position().x0
    x_mid   = (x_right + x_left) / 2
    y_bottom = axes_grid[(1, CONTEXT_K - 1)].get_position().y0
    y_top    = axes_grid[(0, CONTEXT_K - 1)].get_position().y1
    fig.add_artist(plt.Line2D(
        [x_mid, x_mid], [y_bottom, y_top],
        transform=fig.transFigure, color="red", linewidth=2, linestyle="--", zorder=10,
    ))

    fig.suptitle(
        f"ID Digits Stick — Seq {i} — k={CONTEXT_K} context frames "
        f"(red line = start of autoregressive prediction)",
        fontsize=9, y=1.01
    )

    out_path = os.path.join(RESULTS_DIR, f"seq_{i}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

print("Done.")
