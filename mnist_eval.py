import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import your existing modules
from prednet import PredNet
from mnist_settings import *
from mnist_dataset import MovingMNISTDataset

def save_comparison_plot(inputs, predictions, idx, save_dir):
    """
    Saves a plot comparing Ground Truth vs. Prediction for a single sequence.
    inputs: (T, 1, H, W)
    predictions: (T, 1, H, W)
    """
    nt = inputs.shape[0]
    
    # Create a figure with 2 rows (Actual vs Pred) and 'nt' columns
    fig, axes = plt.subplots(2, nt, figsize=(nt * 1.5, 3.5))
    
    for t in range(nt):
        # --- Plot Actual ---
        # inputs is (T, 1, H, W), we need (H, W) for grayscale plot
        gt_frame = inputs[t, 0] 
        axes[0, t].imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
        axes[0, t].axis('off')
        if t == 0:
            axes[0, t].set_title("Actual (t=0)", fontsize=10)
        else:
            axes[0, t].set_title(f"t={t}", fontsize=10)

        # --- Plot Prediction ---
        pred_frame = predictions[t, 0]
        axes[1, t].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
        axes[1, t].axis('off')
        if t == 0:
            axes[1, t].set_title("Pred (Warm-up)", fontsize=10)
        else:
            axes[1, t].set_title(f"Pred t={t}", fontsize=10)

    plt.tight_layout()
    filename = os.path.join(save_dir, f"sample_{idx:03d}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization: {filename}")

# ----------------------------------------------------------------------
# Main Evaluation Function
# ----------------------------------------------------------------------
def evaluate():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    nt = 10  
    batch_size = 4
    
    # Define where to save images
    RESULTS_DIR = os.path.join(WEIGHTS_DIR, "results_plots")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Model ---
    model = PredNet(
        stack_sizes=(1, 16, 32, 64),
        R_stack_sizes=(1, 16, 32, 64),
        A_filt_sizes=(3, 3, 3),
        Ahat_filt_sizes=(3, 3, 3, 3),
        R_filt_sizes=(3, 3, 3, 3),
        output_mode="prediction" 
    ).to(device)

    weights_path = os.path.join(WEIGHTS_DIR, "prednet_mnist_best.pth")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Weights not found.")
        return

    model.eval()

    # --- Data ---
    test_dataset = MovingMNISTDataset(DATA_DIR, nt=nt, split="test") 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Loop ---
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            predictions = model(inputs)

            # --- Convert to CPU for plotting ---
            inputs_np = inputs.cpu().numpy()
            preds_np = predictions.cpu().numpy()

            # --- ONLY SAVE IMAGES FOR THE FIRST BATCH ---
            # (We don't want to save thousands of images)
            if batch_idx == 0:
                print("Saving visualization for the first batch...")
                for b in range(inputs.shape[0]): # Loop through the batch size (4)
                    save_comparison_plot(
                        inputs_np[b], 
                        preds_np[b], 
                        b, 
                        RESULTS_DIR
                    )

            # Iterate over batch
            for b in range(inputs.shape[0]):
                # Iterate over time steps
                # Start from t=1 because t=0 is the warm-up/initial frame
                for t in range(1, nt):
                    gt_frame = inputs_np[b, t, 0] # (H, W) - assuming grayscale
                    pred_frame = preds_np[b, t, 0]

                    # 1. MSE
                    mse = np.mean((gt_frame - pred_frame) ** 2)
                    total_mse += mse

                    # 2. PSNR
                    # data_range is 1.0 because images are float 0-1
                    val_psnr = psnr(gt_frame, pred_frame, data_range=1.0)
                    total_psnr += val_psnr

                    # 3. SSIM
                    val_ssim = ssim(gt_frame, pred_frame, data_range=1.0)
                    total_ssim += val_ssim

                    total_samples += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # --------------------
    # Final Results
    # --------------------
    avg_mse = total_mse / total_samples
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Samples Evaluated: {total_samples}")
    print(f"MSE  (Lower is better):  {avg_mse:.6f}")
    print(f"PSNR (Higher is better): {avg_psnr:.4f} dB")
    print(f"SSIM (Higher is better): {avg_ssim:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()