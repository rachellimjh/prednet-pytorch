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
from mnist_training.mnist_dataset import MovingMNISTDataset
from mnist_training.mnist_settings import *

# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure this points to the correct 1-channel model weights
weights_file = os.path.join(WEIGHTS_DIR, "prednet_mmnist_best.pth")

nt = 20
batch_size = 8
n_plot = 3

# --------------------
# Model params (1 Channel)
# --------------------
stack_sizes = (1, 48, 96, 192)  # 1 Channel for standard MNIST
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

if os.path.exists(weights_file):
    model.load_state_dict(torch.load(weights_file, map_location=device))
    print(f"Loaded weights from {weights_file}")
else:
    print(f"ERROR: Weights file not found at {weights_file}")
    # exit()

model.to(device)
model.eval()

# --------------------
# Dataset
# --------------------
# split="all" uses the entire file provided in the dataset class
dataset = MovingMNISTDataset(data_dir=DATA_DIR, nt=nt, split="all")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# --------------------
# Anomaly evaluation config
# --------------------
EVAL_ANOMALY_ONLY = EVAL_ANOMALY_ONLY  # Set to True to evaluate from anomaly frame onwards
ANOMALY_T = nt // 2

# --------------------
# Metrics Loop
# --------------------
# Model Metrics
all_mse = []
all_snr = []
all_ssim = []

# Baseline Metrics (Previous Frame Copy)
all_mse_prev = []
all_snr_prev = []
all_ssim_prev = []

# Storage for plotting the first batch
plot_inputs = None
plot_preds = None

print("Starting evaluation loop (Model vs Baseline)...")

with torch.no_grad():
    for i, inputs in enumerate(loader):
        inputs = inputs.to(device)

        # [FIX] Ensure inputs are Float and 0-1
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        if inputs.max() > 1.0:
            inputs = inputs / 255.0

        # Run Model
        predictions = model(inputs)

        # Save first batch for plotting
        if i == 0:
            plot_inputs = inputs.cpu().numpy()
            plot_preds = predictions.cpu().numpy()

        # Convert to Numpy
        X = inputs.cpu().numpy()
        X_hat = predictions.cpu().numpy()

        # Iterate through batch
        for b in range(X.shape[0]):
            
            # --- MODIFIED LOGIC: Time Selection ---
            if EVAL_ANOMALY_ONLY:
                # Calculate scores for ALL frames from the Anomaly onwards
                time_indices = range(ANOMALY_T, nt)
            else:
                # Calculate scores for all frames (skipping the very first one)
                time_indices = range(1, nt)

            for t in time_indices:
                gt = X[b, t, 0]
                pred = X_hat[b, t, 0]
                prev = X[b, t - 1, 0] # Previous frame (Baseline)

                # ==========================
                # 1. MSE
                # ==========================
                mse_model = np.mean((gt - pred) ** 2)
                mse_base  = np.mean((gt - prev) ** 2)

                all_mse.append(mse_model)
                all_mse_prev.append(mse_base)

                # ==========================
                # 2. SNR
                # ==========================
                p_signal = np.mean(gt ** 2)
                
                # Model SNR
                p_noise_model = mse_model + 1e-12
                if p_signal == 0: snr_model = 0.0
                else: snr_model = 10 * np.log10(p_signal / p_noise_model)
                
                # Baseline SNR
                p_noise_base = mse_base + 1e-12
                if p_signal == 0: snr_base = 0.0
                else: snr_base = 10 * np.log10(p_signal / p_noise_base)

                all_snr.append(snr_model)
                all_snr_prev.append(snr_base)

                # ==========================
                # 3. SSIM
                # ==========================
                ssim_model = ssim(gt, pred, data_range=1.0)
                ssim_base  = ssim(gt, prev, data_range=1.0)

                all_ssim.append(ssim_model)
                all_ssim_prev.append(ssim_base)

# --------------------
# Results Aggregation
# --------------------
avg_mse = np.mean(all_mse)
avg_snr = np.mean(all_snr)
avg_ssim = np.mean(all_ssim)

avg_mse_prev = np.mean(all_mse_prev)
avg_snr_prev = np.mean(all_snr_prev)
avg_ssim_prev = np.mean(all_ssim_prev)

print("\n" + "="*50)
print(f"FINAL RESULTS ({'Anomaly -> End' if EVAL_ANOMALY_ONLY else 'Full Sequence'})")
print("="*50)
print(f"{'Metric':<10} | {'Model':<15} | {'Prev Frame (Base)':<20}")
print("-" * 50)
print(f"{'MSE':<10} | {avg_mse:.6f}        | {avg_mse_prev:.6f}")
print(f"{'SNR':<10} | {avg_snr:.4f} dB      | {avg_snr_prev:.4f} dB")
print(f"{'SSIM':<10} | {avg_ssim:.4f}        | {avg_ssim_prev:.4f}")
print("="*50)

# --------------------
# Save Text Results
# --------------------
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
results_txt = os.path.join(RESULTS_SAVE_DIR, "mnist_eval_metrics.txt")

with open(results_txt, "w") as f:
    f.write(f"Evaluation Mode: {'Anomaly -> End' if EVAL_ANOMALY_ONLY else 'Full Sequence'}\n")
    f.write("==================================================\n")
    f.write(f"{'Metric':<10} | {'Model':<15} | {'Prev Frame':<20}\n")
    f.write("--------------------------------------------------\n")
    f.write(f"{'MSE':<10} | {avg_mse:.6f}        | {avg_mse_prev:.6f}\n")
    f.write(f"{'SNR':<10} | {avg_snr:.4f}        | {avg_snr_prev:.4f}\n")
    f.write(f"{'SSIM':<10} | {avg_ssim:.4f}        | {avg_ssim_prev:.4f}\n")

print(f"Metrics saved to {results_txt}")

# --------------------
# Plotting
# --------------------
save_dir = os.path.join(RESULTS_SAVE_DIR, "mnist_prediction_plots")
os.makedirs(save_dir, exist_ok=True)

# Use the saved batch
X = plot_inputs
X_hat = plot_preds

print(f"Plotting {n_plot} sequences...")

for i in range(min(n_plot, X.shape[0])):
    fig = plt.figure(figsize=(nt, 4))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0.05, hspace=0.05)

    # Calculate specific metrics for this specific sequence (averaged over time)
    # We grab the indices corresponding to this sequence in the flattened list
    # Note: This is an approximation if batch shuffling was on, but here shuffle=False so it works.
    idx_start = i * len(time_indices)
    idx_end = idx_start + len(time_indices)
    
    seq_mse = np.mean(all_mse[idx_start:idx_end])
    seq_snr = np.mean(all_snr[idx_start:idx_end])
    seq_ssim = np.mean(all_ssim[idx_start:idx_end])

    fig.suptitle(f"MNIST Model Prediction (Sequence {i})", y=0.95)

    for t in range(nt):
        # --- PREDICTED (Top Row) ---
        ax_pred = plt.subplot(gs[0, t]) 
        ax_pred.imshow(X_hat[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax_pred.axis("off")
        
        if t == 0:
            ax_pred.set_title("Predicted", fontsize=10, loc='left')

        # --- ACTUAL (Bottom Row) ---
        ax_gt = plt.subplot(gs[1, t])
        ax_gt.imshow(X[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax_gt.axis("off")
        
        if t == 0:
            ax_gt.set_title("Actual", fontsize=10, loc='left')

    plt.savefig(os.path.join(save_dir, f"sequence_{i}.png"), bbox_inches="tight")
    plt.close()

print(f"Plots saved to {save_dir}")