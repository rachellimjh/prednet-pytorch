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

weights_file = os.path.join(WEIGHTS_DIR, "prednet_kitti_to_mmnist_best.pth")

nt = 20
batch_size = 8
n_plot = 3

# --------------------
# Model params (MUST match training - 3 Channel)
# --------------------
stack_sizes = (3, 48, 96, 192) # 3 Channels because it was finetuned from KITTI
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
    print(f"Loaded finetuned weights from {weights_file}")
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
EVAL_ANOMALY_ONLY = EVAL_ANOMALY_ONLY # Set True to evaluate from anomaly frame onwards
ANOMALY_T = nt // 2

# --------------------
# Metrics Loop
# --------------------
# Model Metrics Only
all_mse = []
all_snr = []
all_ssim = []

# Storage for plotting
plot_inputs = None
plot_preds = None

print("Starting evaluation loop...")

with torch.no_grad():
    for i, inputs in enumerate(loader):
        inputs = inputs.to(device)

        # 1. Normalization (Safety)
        if inputs.max() > 1.0:
            inputs = inputs.float() / 255.0

        # 2. Channel Adaptation (1 -> 3)
        # Finetuned model expects 3 channels, MNIST gives 1
        inputs_3ch = inputs.repeat(1, 1, 3, 1, 1)

        # Run Model
        predictions = model(inputs_3ch)

        # Save first batch for plotting
        if i == 0:
            plot_inputs = inputs_3ch.cpu().numpy()
            plot_preds = predictions.cpu().numpy()

        # Convert to Numpy (We compare on 3 channels, or just channel 0)
        # For fairness, we extract Channel 0 (Grayscale) for metrics
        X_true = inputs_3ch.cpu().numpy()
        X_hat = predictions.cpu().numpy()

        # Iterate through batch
        for b in range(X_true.shape[0]):
            
            # --- MODIFIED LOGIC: Time Selection ---
            if EVAL_ANOMALY_ONLY:
                # Calculate scores for ALL frames from the Anomaly onwards
                time_indices = range(ANOMALY_T, nt)
            else:
                # Calculate scores for all frames (skipping the very first one)
                time_indices = range(1, nt)

            for t in time_indices:
                # Extract Grayscale (Channel 0)
                gt = X_true[b, t, 0]
                pred = X_hat[b, t, 0]

                # --- 1. MSE ---
                mse_model = np.mean((gt - pred) ** 2)
                all_mse.append(mse_model)

                # --- 2. SNR ---
                p_signal = np.mean(gt ** 2)
                p_noise_model = mse_model + 1e-12
                
                if p_signal == 0: 
                    snr_model = 0.0
                else: 
                    snr_model = 10 * np.log10(p_signal / p_noise_model)
                
                all_snr.append(snr_model)

                # --- 3. SSIM ---
                ssim_model = ssim(gt, pred, data_range=1.0)
                all_ssim.append(ssim_model)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} batches...")

# --------------------
# Results Aggregation
# --------------------
avg_mse = np.mean(all_mse)
avg_snr = np.mean(all_snr)
avg_ssim = np.mean(all_ssim)

print("\n" + "="*50)
print(f"FINETUNED MODEL RESULTS ({'Anomaly -> End' if EVAL_ANOMALY_ONLY else 'Full Sequence'})")
print("="*50)
print(f"{'Metric':<10} | {'Model Score':<15}")
print("-" * 50)
print(f"{'MSE':<10} | {avg_mse:.6f}")
print(f"{'SNR':<10} | {avg_snr:.4f} dB")
print(f"{'SSIM':<10} | {avg_ssim:.4f}")
print("="*50)

# --------------------
# Save Text Results
# --------------------
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
metrics_file = os.path.join(RESULTS_SAVE_DIR, "finetuned_mnist_eval_metrics.txt")

with open(metrics_file, "w") as f:
    f.write(f"Evaluation Mode: {'Anomaly -> End' if EVAL_ANOMALY_ONLY else 'Full Sequence'}\n")
    f.write("==================================================\n")
    f.write(f"{'Metric':<10} | {'Model Score':<15}\n")
    f.write("--------------------------------------------------\n")
    f.write(f"{'MSE':<10} | {avg_mse:.6f}\n")
    f.write(f"{'SNR':<10} | {avg_snr:.4f}\n")
    f.write(f"{'SSIM':<10} | {avg_ssim:.4f}\n")

print(f"Metrics saved to {metrics_file}")

# --------------------
# Plot predictions
# --------------------
plot_dir = os.path.join(RESULTS_SAVE_DIR, "finetuned_mnist_prediction_plots")
os.makedirs(plot_dir, exist_ok=True)

# Use the saved batch
X_true = plot_inputs
X_hat = plot_preds

print(f"Plotting {n_plot} sequences...")

for i in range(min(n_plot, X_true.shape[0])):
    fig = plt.figure(figsize=(nt, 4))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0.05, hspace=0.05)
    
    # Calculate specific metrics for this specific sequence (averaged over time)
    # Note: This is an approximation if batch shuffling was on, but here shuffle=False so it works.
    idx_start = i * len(time_indices)
    idx_end = idx_start + len(time_indices)
    
    # Check bounds safety
    if idx_end <= len(all_mse):
        seq_mse = np.mean(all_mse[idx_start:idx_end])
        seq_snr = np.mean(all_snr[idx_start:idx_end])
        seq_ssim = np.mean(all_ssim[idx_start:idx_end])
    else:
        seq_mse, seq_snr, seq_ssim = 0, 0, 0

    fig.suptitle(f"KITTI pretrained fintuned on MNIST Model Prediction (Sequence {i})", y=0.95)

    for t in range(nt):
        # --- PREDICTED (Top Row) ---
        ax_pred = plt.subplot(gs[0, t])
        ax_pred.imshow(X_hat[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax_pred.axis("off")
        
        if t == 0:
            ax_pred.set_title("Predicted", fontsize=10, loc='left')

        # --- ACTUAL (Bottom Row) ---
        ax_gt = plt.subplot(gs[1, t])
        ax_gt.imshow(X_true[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax_gt.axis("off")
        
        if t == 0:
            ax_gt.set_title("Actual", fontsize=10, loc='left')

    plt.savefig(os.path.join(plot_dir, f"sequence_{i}.png"), bbox_inches="tight")
    plt.close()

print(f"Saved finetuned MNIST prediction plots to {plot_dir}")