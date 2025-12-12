import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Save plots without displaying
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from prednet import PredNet
from data_utils import KittiDataset
from kitti_settings import *

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_best.pth')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

nt = 10
batch_size = 10 
n_plot = 20 # Number of sequences to plot

# Model Params (Must match training)
stack_sizes = (3, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)

# --- Load Model ---
# Note: output_mode='prediction' for evaluation
model = PredNet(stack_sizes, R_stack_sizes,
                A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                output_mode='prediction')

if os.path.exists(weights_file):
    model.load_state_dict(torch.load(weights_file, map_location=device))
    print("Weights loaded successfully.")
else:
    print("Weights file not found! Running with random weights.")

model.to(device)
model.eval()

# --- Data Loader ---
test_dataset = KittiDataset(test_file, test_sources, nt, sequence_start_mode='unique')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# --- Evaluation Loop ---
print("Evaluating...")
all_mse_model = []
all_mse_prev = []

X_test_all = []
X_hat_all = []

with torch.no_grad():
    for inputs in test_loader:
        # inputs: (Batch, Time, C, H, W)
        inputs_gpu = inputs.to(device)
        
        # Run Model
        # predictions: (Batch, Time, C, H, W)
        predictions = model(inputs_gpu)
        
        # Move to CPU for calculations
        X_test = inputs.numpy()
        X_hat = predictions.cpu().numpy()
        
        # Store for plotting later
        if len(X_test_all) < n_plot * nt: # Only store enough for plotting
             X_test_all.append(X_test)
             X_hat_all.append(X_hat)
        
        # Calculate MSE
        # Skip first frame (t=0) because prediction is meaningless there
        # Model MSE: difference between Actual(t) and Predicted(t)
        mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)
        all_mse_model.append(mse_model)
        
        # Previous Frame MSE (Baseline): difference between Actual(t) and Actual(t-1)
        # i.e., "What if I just guessed the image stays exactly the same?"
        mse_prev = np.mean((X_test[:, 1:] - X_test[:, :-1]) ** 2)
        all_mse_prev.append(mse_prev)

# --- Results ---
avg_mse_model = np.mean(all_mse_model)
avg_mse_prev = np.mean(all_mse_prev)

print(f"Model MSE: {avg_mse_model:.6f}")
print(f"Prev Frame MSE (Baseline): {avg_mse_prev:.6f}")

# Ensure directory exists
if not os.path.exists(RESULTS_SAVE_DIR):
    os.makedirs(RESULTS_SAVE_DIR)
    
with open(os.path.join(RESULTS_SAVE_DIR, 'prediction_scores.txt'), 'w') as f:
    f.write(f"Model MSE: {avg_mse_model}\n")
    f.write(f"Previous Frame MSE: {avg_mse_prev}\n")

# --- Plotting ---
print("Plotting predictions...")
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.makedirs(plot_save_dir)

# Flatten list of batches
X_test_all = np.concatenate(X_test_all, axis=0)[:n_plot]
X_hat_all = np.concatenate(X_hat_all, axis=0)[:n_plot]

aspect_ratio = float(X_hat_all.shape[3]) / X_hat_all.shape[4] # W / H
plt.figure(figsize = (nt, 2*aspect_ratio))

for i in range(n_plot):
    gs = gridspec.GridSpec(2, nt)
    gs.update(wspace=0., hspace=0.)
    
    for t in range(nt):
        # Actual
        plt.subplot(gs[t])
        # Permute (C, H, W) back to (H, W, C) for Matplotlib
        img_actual = np.transpose(X_test_all[i, t], (1, 2, 0))
        plt.imshow(img_actual, interpolation='none')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        if t==0: plt.ylabel('Actual', fontsize=10)

        # Predicted
        plt.subplot(gs[t + nt])
        img_pred = np.transpose(X_hat_all[i, t], (1, 2, 0))
        
        # Clip values to valid image range [0, 1] just in case
        img_pred = np.clip(img_pred, 0, 1)
        
        plt.imshow(img_pred, interpolation='none')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir + 'plot_' + str(i) + '.png')
    plt.clf()

print("Done.")