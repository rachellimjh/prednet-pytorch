import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Save plots without displaying
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from prednet import PredNet
from mnist_training.mnist_dataset import MovingMNISTDataset
from mnist_training import mnist_settings 

# --------------------
# Configuration
# --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_file = os.path.join(mnist_settings.KITTI_WEIGHTS, 'prednet_kitti_best.pth')

nt = 20
batch_size = 8
n_plot = 3  # How many sequence images to save

# Evaluation Mode
EVAL_ANOMALY_ONLY = mnist_settings.EVAL_ANOMALY_ONLY # Set True to only calculate metrics AFTER the anomaly starts
ANOMALY_T = nt // 2

# --------------------
# Model Params (Must match KITTI finetuning)
# --------------------
stack_sizes = (3, 48, 96, 192) # 3 channels (Adapted from KITTI)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# --------------------
# Load Model
# --------------------
model = PredNet(
    stack_sizes,
    R_stack_sizes,
    A_filt_sizes,
    Ahat_filt_sizes,
    R_filt_sizes,
    output_mode='prediction'
)

if os.path.exists(weights_file):
    model.load_state_dict(torch.load(weights_file, map_location=device))
    print(f"Weights loaded from {weights_file}")
else:
    print(f"ERROR: Weights file not found at {weights_file}")
    # exit()

model.to(device)
model.eval()

# --------------------
# Data Loader
# --------------------
test_dataset = MovingMNISTDataset(
    npz_file=os.path.join(mnist_settings.DATA_DIR, mnist_settings.NPZ_FILE), 
    nt=nt, 
    split="all"  # all
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# --------------------
# Evaluation Loop
# --------------------
print("Starting Evaluation (Calculating MSE, SNR, SSIM)...")

# Metric Storage
all_mse = []
all_snr = [] 
all_ssim = []

# Plotting Storage (First batch only)
plot_inputs = None
plot_preds = None

with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device)

        # 1. Channel Adaptation (1 -> 3)
        if inputs.shape[2] == 1:
            inputs = inputs.repeat(1, 1, 3, 1, 1)

        # 2. Normalization (Safety)
        if inputs.max() > 1.0:
            inputs = inputs.float() / 255.0

        # Run Model
        predictions = model(inputs)

        # Store first batch for plotting later
        if i == 0:
            plot_inputs = inputs.cpu().numpy()
            plot_preds = predictions.cpu().numpy()

        # Convert to CPU Numpy for Metric Calculation
        X_true = inputs.cpu().numpy()
        X_hat = predictions.cpu().numpy()

        # Iterate through batch
        for b in range(X_true.shape[0]):
            
            # --- MODIFIED LOGIC HERE ---
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
                mse_val = np.mean((gt - pred) ** 2)
                all_mse.append(mse_val)

                # --- 2. SNR ---
                p_signal = np.mean(gt ** 2)
                p_noise = mse_val + 1e-12 
                
                if p_signal == 0:
                     snr_val = 0.0 
                else:
                     snr_val = 10 * np.log10(p_signal / p_noise)
                
                all_snr.append(snr_val)

                # --- 3. SSIM ---
                ssim_val = ssim(gt, pred, data_range=1.0)
                all_ssim.append(ssim_val)
        
        # Optional: Print progress every 10 batches
        if (i+1) % 10 == 0:
            print(f"Processed {i+1} batches...")

# --------------------
# Aggregating Results
# --------------------
avg_mse = np.mean(all_mse)
avg_snr = np.mean(all_snr)
avg_ssim = np.mean(all_ssim)

mode_str = "Anomaly -> End" if EVAL_ANOMALY_ONLY else "Full Sequence"

print("\n" + "=" * 50)
print(f"FINAL RESULTS ({mode_str})")
print("=" * 50)
print("Model: KITTI")
print(f"  MSE:  {avg_mse:.4f}")
print(f"  SNR:  {avg_snr:.4f}")
print(f"  SSIM: {avg_ssim:.4f}")
print("-" * 50)

# Save Text Results (unified format)
os.makedirs(mnist_settings.RESULTS_SAVE_DIR, exist_ok=True)
txt_path = os.path.join(mnist_settings.RESULTS_SAVE_DIR, 'kitti_eval_metrics.txt')
with open(txt_path, 'w') as f:
    f.write(f"Evaluation Mode: {mode_str}\n")
    f.write("=" * 50 + "\n")
    f.write("Model: KITTI\n")
    f.write(f"MSE: {avg_mse:.6f}\n")
    f.write(f"SNR: {avg_snr:.4f}\n")
    f.write(f"SSIM: {avg_ssim:.4f}\n")
    f.write("-" * 50 + "\n")
print(f"Metrics saved to {txt_path}")

# --------------------
# Plotting
# --------------------
print(f"Plotting {n_plot} sequences...")
plot_save_dir = os.path.join(mnist_settings.RESULTS_SAVE_DIR, 'kitti_prediction_plots')
os.makedirs(plot_save_dir, exist_ok=True)

# Use the stored batch
X_true = plot_inputs
X_hat = plot_preds

for i in range(min(n_plot, X_true.shape[0])):
    fig = plt.figure(figsize=(nt, 4))
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0.05, hspace=0.05)
    
    # --- UPDATED TITLE ---
    fig.suptitle(f"KITTI Model Prediction (Sequence {i})", y=0.95)

    for t in range(nt):
        # --- PREDICTED (Top) ---
        ax_pred = plt.subplot(gs[0, t])
        ax_pred.imshow(X_hat[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax_pred.axis('off')
        if t == 0: ax_pred.set_title("Predicted", loc='left', fontsize=10)

        # --- ACTUAL (Bottom) ---
        ax_gt = plt.subplot(gs[1, t])
        ax_gt.imshow(X_true[i, t, 0], cmap="gray", vmin=0, vmax=1)
        ax_gt.axis('off')
        if t == 0: ax_gt.set_title("Actual", loc='left', fontsize=10)

    plt.savefig(os.path.join(plot_save_dir, f'seq_{i}.png'))
    plt.close()

print("Done.")