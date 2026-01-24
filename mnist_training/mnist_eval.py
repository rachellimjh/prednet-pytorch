import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as ssim

from prednet import PredNet
from .mnist_dataset import MovingMNISTDataset
from .mnist_settings import *

# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_file = os.path.join(WEIGHTS_DIR, "prednet_mmnist_best.pth")

nt = 20
batch_size = 3
n_plot = 3

# --------------------
# Model params (must match training)
# --------------------
stack_sizes = (1, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# --------------------
# Load model
# --------------------
model = PredNet(
    stack_sizes,
    R_stack_sizes,
    A_filt_sizes,
    Ahat_filt_sizes,
    R_filt_sizes,
    output_mode="prediction",
)

model.load_state_dict(torch.load(weights_file, map_location=device))
model.to(device)
model.eval()

print("Loaded trained Moving MNIST model")

# --------------------
# Dataset
# --------------------
dataset = MovingMNISTDataset(
    data_dir=DATA_DIR,
    nt=nt,
    split="val",
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
)

# --------------------
# Metrics
# --------------------
all_mse_model = []
all_mse_prev = []
all_snr = []
all_ssim = []

with torch.no_grad():
    inputs = next(iter(loader))  # (B, T, 1, H, W)
    inputs = inputs.to(device)

    predictions = model(inputs)

    X = inputs.cpu().numpy()
    X_hat = predictions.cpu().numpy()

    # ---- Loop over batch + time (skip t=0) ----
    for b in range(batch_size):
        for t in range(1, nt):
            gt = X[b, t, 0]
            pred = X_hat[b, t, 0]
            prev = X[b, t - 1, 0]

            # ---- MSE ----
            mse_model = np.mean((gt - pred) ** 2)
            mse_prev = np.mean((gt - prev) ** 2)

            all_mse_model.append(mse_model)
            all_mse_prev.append(mse_prev)

            # ---- SNR ----
            signal_power = np.mean(gt ** 2)
            noise_power = np.mean((gt - pred) ** 2) + 1e-8
            snr = 10 * np.log10(signal_power / noise_power)
            all_snr.append(snr)

            # ---- SSIM ----
            ssim_val = ssim(
                gt,
                pred,
                data_range=1.0,
            )
            all_ssim.append(ssim_val)

# --------------------
# Results
# --------------------
print("===== Moving MNIST Evaluation =====")
print(f"Model MSE: {np.mean(all_mse_model):.6f}")
print(f"Prev-frame MSE: {np.mean(all_mse_prev):.6f}")
print(f"MSE ratio (model / prev): {np.mean(all_mse_model) / np.mean(all_mse_prev):.3f}")
print(f"SNR (dB): {np.mean(all_snr):.2f}")
print(f"SSIM: {np.mean(all_ssim):.4f}")

# --------------------
# Save results to text file
# --------------------
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
results_txt = os.path.join(RESULTS_SAVE_DIR, "mnist_eval_metrics.txt")

with open(results_txt, "w") as f:
    f.write("===== Moving MNIST Evaluation =====\n")
    f.write(f"Model MSE: {np.mean(all_mse_model):.6f}\n")
    f.write(f"Prev-frame MSE: {np.mean(all_mse_prev):.6f}\n")
    f.write(
        f"MSE ratio (model / prev): "
        f"{np.mean(all_mse_model) / np.mean(all_mse_prev):.3f}\n"
    )
    f.write(f"SNR (dB): {np.mean(all_snr):.2f}\n")
    f.write(f"SSIM: {np.mean(all_ssim):.4f}\n")

print(f"Saved evaluation metrics to {results_txt}")

# --------------------
# Plot predictions
# --------------------
save_dir = os.path.join(RESULTS_SAVE_DIR, "mnist_prediction_plots")
os.makedirs(save_dir, exist_ok=True)

for i in range(n_plot):
    fig = plt.figure(figsize=(nt, 4))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0.05, hspace=0.05)

    # ---- Title using results directory name ----
    exp_name = os.path.basename(RESULTS_SAVE_DIR.rstrip("/"))
    fig.suptitle(
        f"{exp_name} | Moving MNIST Prediction (Sequence {i})",
        fontsize=12,
        y=1.05,
    )

    for t in range(nt):
        # Predicted (top)
        ax = plt.subplot(gs[t])
        ax.imshow(X_hat[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if t == 0:
            ax.set_ylabel("Predicted", fontsize=10)

        # Actual (bottom)
        ax = plt.subplot(gs[t + nt])
        ax.imshow(X[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if t == 0:
            ax.set_ylabel("Actual", fontsize=10)

    plt.savefig(
        os.path.join(save_dir, f"sequence_{i}.png"),
        bbox_inches="tight",
    )
    plt.close()

