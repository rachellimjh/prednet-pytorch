"""
Teacher forcing horizon evaluation: measure how many ground-truth frames
are needed before the model generates stable predictions.

Computes both the overall summary metrics (averages) AND the
frame-by-frame metrics to visualize error spikes and recovery times.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from prednet import PredNet
from mnist_training.mnist_dataset import MovingMNISTDataset
from mnist_training import mnist_settings

# Horizons to evaluate: k = number of initial ground-truth frames
TEACHER_FORCING_HORIZONS = [1, 3, 5, 10]

def compute_metrics_for_horizon(
    model: PredNet,
    loader: DataLoader,
    nt: int,
    horizon: int,
    eval_anomaly_only: bool,
    anomaly_t: int,
) -> Dict[str, Any]:
    """
    Run the model with teacher-forcing horizon k and compute both frame-by-frame
    and summary MSE, SNR, SSIM.
    """
    device = next(model.parameters()).device
    model.extrap_start_time = horizon

    frame_mse = {t: [] for t in range(1, nt)}
    frame_snr = {t: [] for t in range(1, nt)}
    frame_ssim = {t: [] for t in range(1, nt)}

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)

            if inputs.dtype != torch.float32:
                inputs = inputs.float()
            if inputs.max() > 1.0:
                inputs = inputs / 255.0

            preds = model(inputs)

            X = inputs.cpu().numpy()  # (B, T, 1, H, W)
            X_hat = preds.cpu().numpy()

            B = X.shape[0]

            for b in range(B):
                for t in range(1, nt):
                    gt = X[b, t, 0]
                    pred = X_hat[b, t, 0]

                    mse_val = np.mean((gt - pred) ** 2)
                    p_signal = np.mean(gt ** 2)
                    p_noise = mse_val + 1e-12
                    snr_val = 10 * np.log10(p_signal / p_noise) if p_signal > 0 else 0.0
                    ssim_val = ssim(gt, pred, data_range=1.0)

                    frame_mse[t].append(mse_val)
                    frame_snr[t].append(snr_val)
                    frame_ssim[t].append(ssim_val)

    # Average across all batches for each specific frame
    avg_frame_mse = {t: float(np.mean(frame_mse[t])) for t in range(1, nt)}
    avg_frame_snr = {t: float(np.mean(frame_snr[t])) for t in range(1, nt)}
    avg_frame_ssim = {t: float(np.mean(frame_ssim[t])) for t in range(1, nt)}

    # Calculate the original summary metrics (averaging over the valid extrapolation window)
    t_start = max(anomaly_t, horizon) if eval_anomaly_only else max(1, horizon)
    valid_frames = [t for t in range(t_start, nt)]

    summary_mse = float(np.mean([avg_frame_mse[t] for t in valid_frames])) if valid_frames else float("nan")
    summary_snr = float(np.mean([avg_frame_snr[t] for t in valid_frames])) if valid_frames else float("nan")
    summary_ssim = float(np.mean([avg_frame_ssim[t] for t in valid_frames])) if valid_frames else float("nan")

    return {
        "horizon": horizon,
        "frame_mse": avg_frame_mse,
        "frame_snr": avg_frame_snr,
        "frame_ssim": avg_frame_ssim,
        "mse": summary_mse,
        "snr": summary_snr,
        "ssim": summary_ssim,
    }


def run_teacher_forcing_sweep(
    horizons: Optional[List[int]] = None,
    nt: int = 20,
    batch_size: int = 8,
) -> None:
    if horizons is None:
        horizons = list(TEACHER_FORCING_HORIZONS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_file = os.path.join(mnist_settings.WEIGHTS_DIR, mnist_settings.MNIST_MODEL)

    stack_sizes = (1, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)

    model = PredNet(
        stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
        output_mode="prediction", extrap_start_time=None,
    )

    if os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"  [TF] Loaded weights from {weights_file}")
    else:
        print(f"  [TF] WARNING: Weights not found at {weights_file}; using random weights.")

    model.to(device)
    model.eval()

    npz_path = os.path.join(mnist_settings.DATA_DIR, mnist_settings.NPZ_FILE)
    dataset = MovingMNISTDataset(npz_file=npz_path, nt=nt, split="all")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    eval_anomaly_only = getattr(mnist_settings, "EVAL_ANOMALY_ONLY", False)
    anomaly_t = nt // 2

    results: List[Dict[str, Any]] = []
    print(f"  [TF] Running sweep for k in {horizons}...")
    for k in horizons:
        metrics = compute_metrics_for_horizon(
            model=model, loader=loader, nt=nt, horizon=k,
            eval_anomaly_only=eval_anomaly_only, anomaly_t=anomaly_t,
        )
        results.append(metrics)

    out_dir = os.path.join(mnist_settings.RESULTS_SAVE_DIR, "teacher_forcing_horizon")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. THE ORIGINAL SUMMARY GRAPHS (Metrics vs Horizon)
    # ---------------------------------------------------------
    k_arr = np.array([r["horizon"] for r in results], dtype=float)
    mse_arr = np.array([r["mse"] for r in results], dtype=float)
    snr_arr = np.array([r["snr"] for r in results], dtype=float)
    ssim_arr = np.array([r["ssim"] for r in results], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(k_arr, mse_arr, marker="o", linewidth=2)
    axes[0].set_xlabel("Number of Ground-Truth Conditioning Frames")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_arr)

    axes[1].plot(k_arr, ssim_arr, marker="o", linewidth=2)
    axes[1].set_xlabel("Number of Ground-Truth Conditioning Frames")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(k_arr)

    axes[2].plot(k_arr, snr_arr, marker="o", linewidth=2)
    axes[2].set_xlabel("Number of Ground-Truth Conditioning Frames")
    axes[2].set_ylabel("SNR (dB)")
    axes[2].set_title("SNR")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(k_arr)

    plt.suptitle("Overall Adaptation Speed vs Teacher Forcing Horizon")
    plt.tight_layout()
    summary_plot_path = os.path.join(out_dir, "summary_teacher_forcing_horizon.png")
    plt.savefig(summary_plot_path)
    plt.close()
    print(f"  [TF] Saved {summary_plot_path}")

    # ---------------------------------------------------------
    # 2. THE NEW FRAME-BY-FRAME GRAPHS (For Anomaly Spikes)
    # ---------------------------------------------------------
    frames = list(range(1, nt))
    metrics_to_plot = [
        ("mse", "Mean Squared Error (MSE)", "frame_by_frame_mse.png"),
        ("ssim", "Structural Similarity Index (SSIM)", "frame_by_frame_ssim.png"),
        ("snr", "Signal-to-Noise Ratio (SNR)", "frame_by_frame_snr.png")
    ]

    for metric_key, ylabel, filename in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for r in results:
            k = r["horizon"]
            metric_vals = [r[f"frame_{metric_key}"][t] for t in frames]
            plt.plot(frames, metric_vals, marker="o", label=f"{k} Context Frames", linewidth=2)

        if eval_anomaly_only:
            plt.axvline(x=anomaly_t, color='black', linestyle='--', linewidth=2, label='Anomaly Introduced')

        plt.xlabel("Frame Number (Time)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} over time by Teacher Forcing Horizon")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(frames)
        plt.tight_layout()

        plot_path = os.path.join(out_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"  [TF] Saved {plot_path}")

def main() -> None:
    run_teacher_forcing_sweep(horizons=None)

if __name__ == "__main__":
    main()
