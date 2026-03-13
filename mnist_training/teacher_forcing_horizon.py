import os
from typing import List

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


def compute_metrics_for_horizon(
    model: PredNet,
    loader: DataLoader,
    nt: int,
    horizon: int,
    eval_anomaly_only: bool,
    anomaly_t: int,
) -> dict:
    """
    Run the model with a given teacher-forcing horizon (TTF) and compute
    aggregate MSE, SNR, SSIM.

    horizon (TTF): number of initial frames for which the model sees ground
    truth inputs. From t >= horizon, the model reuses its own predictions.
    """
    device = next(model.parameters()).device
    model.extrap_start_time = horizon

    all_mse: List[float] = []
    all_snr: List[float] = []
    all_ssim: List[float] = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)

            # Ensure float and 0–1 range
            if inputs.dtype != torch.float32:
                inputs = inputs.float()
            if inputs.max() > 1.0:
                inputs = inputs / 255.0

            # Run model with current extrap_start_time
            preds = model(inputs)

            X = inputs.cpu().numpy()  # (B, T, 1, H, W)
            X_hat = preds.cpu().numpy()

            B = X.shape[0]

            for b in range(B):
                if eval_anomaly_only:
                    t_start = max(anomaly_t, horizon)
                else:
                    # Always skip the very first frame when computing metrics
                    t_start = max(1, horizon)

                for t in range(t_start, nt):
                    gt = X[b, t, 0]
                    pred = X_hat[b, t, 0]

                    # MSE
                    mse_val = np.mean((gt - pred) ** 2)
                    all_mse.append(mse_val)

                    # SNR
                    p_signal = np.mean(gt ** 2)
                    p_noise = mse_val + 1e-12
                    if p_signal == 0:
                        snr_val = 0.0
                    else:
                        snr_val = 10 * np.log10(p_signal / p_noise)
                    all_snr.append(snr_val)

                    # SSIM
                    ssim_val = ssim(gt, pred, data_range=1.0)
                    all_ssim.append(ssim_val)

    return {
        "horizon": horizon,
        "mse": float(np.mean(all_mse)) if all_mse else float("nan"),
        "snr": float(np.mean(all_snr)) if all_snr else float("nan"),
        "ssim": float(np.mean(all_ssim)) if all_ssim else float("nan"),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Model setup (1‑channel MNIST model)
    # --------------------
    nt = 20
    batch_size = 8

    weights_file = os.path.join(mnist_settings.WEIGHTS_DIR, mnist_settings.MNIST_MODEL)

    stack_sizes = (1, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)

    model = PredNet(
        stack_sizes,
        R_stack_sizes,
        A_filt_sizes,
        Ahat_filt_sizes,
        R_filt_sizes,
        output_mode="prediction",
        extrap_start_time=None,
    )

    if os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"Loaded weights from {weights_file}")
    else:
        print(f"WARNING: Weights file not found at {weights_file}; using random weights.")

    model.to(device)
    model.eval()

    # --------------------
    # Data
    # --------------------
    dataset = MovingMNISTDataset(
        data_dir=mnist_settings.DATA_DIR,
        nt=nt,
        split="all",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    eval_anomaly_only = getattr(mnist_settings, "EVAL_ANOMALY_ONLY", False)
    anomaly_t = nt // 2

    # --------------------
    # Teacher forcing horizon sweep
    # --------------------
    horizons = list(range(0, nt))  # TTF = 0 .. nt-1
    results = []

    print("Running teacher forcing horizon sweep...")
    for h in horizons:
        print(f"  Horizon TTF = {h}")
        metrics = compute_metrics_for_horizon(
            model=model,
            loader=loader,
            nt=nt,
            horizon=h,
            eval_anomaly_only=eval_anomaly_only,
            anomaly_t=anomaly_t,
        )
        results.append(metrics)

    # --------------------
    # Save CSV
    # --------------------
    out_dir = os.path.join(mnist_settings.RESULTS_SAVE_DIR, "teacher_forcing_horizon")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "teacher_forcing_horizon.csv")
    with open(csv_path, "w") as f:
        f.write("horizon,mse,snr,ssim\n")
        for r in results:
            f.write(f"{r['horizon']},{r['mse']},{r['snr']},{r['ssim']}\n")

    print(f"Saved teacher forcing sweep metrics to {csv_path}")

    # --------------------
    # Plots
    # --------------------
    horizons_arr = np.array([r["horizon"] for r in results], dtype=float)
    mse_arr = np.array([r["mse"] for r in results], dtype=float)
    snr_arr = np.array([r["snr"] for r in results], dtype=float)
    ssim_arr = np.array([r["ssim"] for r in results], dtype=float)

    # Plot all three metrics in one figure for convenience
    plt.figure(figsize=(10, 6))
    plt.plot(horizons_arr, mse_arr, marker="o", label="MSE")
    plt.plot(horizons_arr, snr_arr, marker="o", label="SNR (dB)")
    plt.plot(horizons_arr, ssim_arr, marker="o", label="SSIM")
    plt.xlabel("Teacher Forcing Horizon TTF (frames)")
    plt.ylabel("Metric value")
    plt.title("Teacher Forcing Horizon Sweep (MNIST model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "teacher_forcing_horizon.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved teacher forcing sweep plot to {plot_path}")


if __name__ == "__main__":
    main()

