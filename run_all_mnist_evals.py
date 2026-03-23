import os

import numpy as np

from mnist_training import mnist_settings
from mnist_training.mnist_eval import run_eval as mnist_run_eval
from mnist_finetuning.finetuning_eval import run_eval as finetuned_run_eval
from kitti_pretraining.kitti_eval_mnist import run_eval as kitti_run_eval


# ---------------------------------------------------------------------------
# Teacher forcing horizon (k): list of k values to evaluate.
# For each k, first k frames use ground truth; after that the model uses
# its own predictions. Metrics are computed over frames after the window.
# Change this list to vary which k are run (e.g. [0, 1, 3, 5, 10] or [0, 2, 5]).
# ---------------------------------------------------------------------------
TEACHER_FORCING_HORIZONS = [1, 3, 5, 10, 15]

# Set to False to skip the teacher forcing sweep in run_for_condition.
RUN_TEACHER_FORCING_SWEEP = True

# ---------------------------------------------------------------------------
# Model variants. Each variant has its own weights_dir for Moving MNIST and
# KITTI finetuned. Edit these to match your setup.
# No need to edit mnist_settings.py or kitti_settings.py when using this script.
# ---------------------------------------------------------------------------
MODEL_VARIANTS = [
    {
        "name": "id_vertical",
        "type": "vertical",
        "moving_mnist": {
            "weights_dir": "models/mnist_id_vertical",
            "model_file": "prednet_mmnist_best.pth",
        },
        "kitti_finetuned": {
            "weights_dir": "models/mnist_id_vertical",
            "model_file": "prednet_kitti_to_mmnist_best.pth",
        },
        "kitti": {"kitti_weights_dir": "models"},
    },
    {
        "name": "id_digits",
        "type": "digits",
        "moving_mnist": {
            "weights_dir": "models/mnist_id_digits",
            "model_file": "prednet_mmnist_best.pth",
        },
        "kitti_finetuned": {
            "weights_dir": "models/mnist_id_digits",
            "model_file": "prednet_kitti_to_mmnist_best.pth",
        },
        "kitti": {"kitti_weights_dir": "models"},
    },
]


# List of conditions to evaluate in one run.
# results_dir is a subdir; full path = results/{variant_name}/{results_dir}
CONDITIONS = [
    {
        "name": "ID digits no anomaly",
        "type": "digits",
        "npz_file": "ID_NORMAL.npz",
        "results_dir": "id_digits_normal",
        "eval_anomaly_only": False,
    },
    {
        "name": "OOD digits no anomaly",
        "type": "digits",
        "npz_file": "OOD_NORMAL.npz",
        "results_dir": "ood_digits_normal",
        "eval_anomaly_only": False,
    },
    {
        "name": "OOD digits anomaly (disappear)",
        "type": "digits",
        "npz_file": "OOD_DISAPPEAR.npz",
        "results_dir": "ood_digits_disappear",
        "eval_anomaly_only": True,
    },
    {
        "name": "OOD digits anomaly (appear)",
        "type": "digits",
        "npz_file": "OOD_APPEAR.npz",
        "results_dir": "ood_digits_appear",
        "eval_anomaly_only": True,
    },
    {
        "name": "OOD digits anomaly (stick)",
        "type": "digits",
        "npz_file": "OOD_STICK.npz",
        "results_dir": "ood_digits_stick",
        "eval_anomaly_only": True,
    },
    {
        "name": "ID digits anomaly (disappear)",
        "type": "digits",
        "npz_file": "ID_DISAPPEAR.npz",
        "results_dir": "id_digits_disappear",
        "eval_anomaly_only": True,
    },
    {
        "name": "ID digits anomaly (appear)",
        "type": "digits",
        "npz_file": "ID_APPEAR.npz",
        "results_dir": "id_digits_appear",
        "eval_anomaly_only": True,
    },
    {
        "name": "ID digits anomaly (stick)",
        "type": "digits",
        "npz_file": "ID_STICK.npz",
        "results_dir": "id_digits_stick",
        "eval_anomaly_only": True,
    },

    {
        "name": "ID vertical no anomaly",
        "type": "vertical",
        "npz_file": "ID_NORMAL_VERTICAL.npz",
        "results_dir": "id_vertical_normal",
        "eval_anomaly_only": False,
    },
    {
        "name": "OOD horizontal no anomaly",
        "type": "vertical",
        "npz_file": "OOD_NORMAL_HORIZONTAL.npz",
        "results_dir": "ood_horizontal_normal",
        "eval_anomaly_only": False,
    },
    {
        "name": "OOD horizontal anomaly (disappear)",
        "type": "vertical",
        "npz_file": "OOD_DISAPPEAR_HORIZONTAL.npz",
        "results_dir": "ood_horizontal_disappear",
        "eval_anomaly_only": True,
    },
    {
        "name": "OOD horizontal anomaly (appear)",
        "type": "vertical",
        "npz_file": "OOD_APPEAR_HORIZONTAL.npz",
        "results_dir": "ood_horizontal_appear",
        "eval_anomaly_only": True,
    },
    {
        "name": "OOD horizontal anomaly (stick)",
        "type": "vertical",
        "npz_file": "OOD_STICK_HORIZONTAL.npz",
        "results_dir": "ood_horizontal_stick",
        "eval_anomaly_only": True,
    },
    {
        "name": "ID vertical anomaly (disappear)",
        "type": "vertical",
        "npz_file": "ID_DISAPPEAR_VERTICAL.npz",
        "results_dir": "id_vertical_disappear",
        "eval_anomaly_only": True,
    },
    {
        "name": "ID vertical anomaly (appear)",
        "type": "vertical",
        "npz_file": "ID_APPEAR_VERTICAL.npz",
        "results_dir": "id_vertical_appear",
        "eval_anomaly_only": True,
    },
    {
        "name": "ID vertical anomaly (stick)",
        "type": "vertical",
        "npz_file": "ID_STICK_VERTICAL.npz",
        "results_dir": "id_vertical_stick",
        "eval_anomaly_only": True,
    },
]


def _write_combined_metrics(results_dir: str, mode_str: str, model_results: list) -> None:
    """
    Write a single txt file containing metrics for all 4 baseline models
    plus standard deviation across them for each metric.

    model_results: list of (name, mse, snr, ssim, std_mse, std_snr, std_ssim) in order:
        1. Copy previous frame
        2. KITTI trained
        3. KITTI trained + finetuned on MNIST
        4. Trained on MNIST
    """
    mse_vals  = [r[1] for r in model_results]
    snr_vals  = [r[2] for r in model_results]
    ssim_vals = [r[3] for r in model_results]

    across_std_mse  = float(np.std(mse_vals))
    across_std_snr  = float(np.std(snr_vals))
    across_std_ssim = float(np.std(ssim_vals))

    out_path = os.path.join(results_dir, "all_baselines_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"Evaluation Mode: {mode_str}\n")
        f.write("=" * 50 + "\n")
        for name, mse, snr, ssim_val, s_mse, s_snr, s_ssim in model_results:
            f.write(f"Model: {name}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"MSE_STD: {s_mse:.4f}\n")
            f.write(f"SNR: {snr:.4f}\n")
            f.write(f"SNR_STD: {s_snr:.4f}\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
            f.write(f"SSIM_STD: {s_ssim:.4f}\n")
            f.write("-" * 50 + "\n")
        f.write("Standard Deviation across all 4 models:\n")
        f.write(f"MSE std:  {across_std_mse:.4f}\n")
        f.write(f"SNR std:  {across_std_snr:.4f}\n")
        f.write(f"SSIM std: {across_std_ssim:.4f}\n")
        f.write("=" * 50 + "\n")

    print(f"Combined baseline metrics saved to {out_path}")


def run_for_condition(variant: dict, cond: dict) -> None:
    """
    Run all MNIST-based evaluations (Moving MNIST, KITTI finetuned, KITTI raw)
    for a single variant + dataset condition, then write a combined summary.
    """
    variant_name = variant["name"]
    cond_name = cond["name"]
    npz_file = cond["npz_file"]
    results_subdir = cond["results_dir"]
    eval_anomaly_only = cond["eval_anomaly_only"]

    results_dir = os.path.join("results", variant_name, results_subdir)

    print("\n" + "=" * 80)
    print(f"Variant: {variant_name}  |  Condition: {cond_name}")
    print(f"  NPZ file:     {npz_file}")
    print(f"  Results dir:  {results_dir}")
    print(f"  Anomaly only: {eval_anomaly_only}")
    print("=" * 80)

    # Update shared settings for this condition (data + results).
    mnist_settings.NPZ_FILE = npz_file
    mnist_settings.RESULTS_SAVE_DIR = results_dir
    mnist_settings.EVAL_ANOMALY_ONLY = eval_anomaly_only

    os.makedirs(results_dir, exist_ok=True)

    mode_str = "Anomaly -> End" if eval_anomaly_only else "Full Sequence"

    # 1. Moving MNIST model (trained only on MNIST)
    cfg = variant["moving_mnist"]
    mnist_settings.WEIGHTS_DIR = cfg["weights_dir"]
    mnist_settings.MNIST_MODEL = cfg["model_file"]
    mnist_results = mnist_run_eval()
    mse_mnist, snr_mnist, ssim_mnist, std_mse_mnist, std_snr_mnist, std_ssim_mnist = mnist_results["mnist"]
    mse_prev, snr_prev, ssim_prev, std_mse_prev, std_snr_prev, std_ssim_prev = mnist_results["prev_frame"]

    # 1b. Teacher forcing horizon sweep: loop over k, store results, plot
    if RUN_TEACHER_FORCING_SWEEP:
        from mnist_training.teacher_forcing_horizon import run_teacher_forcing_sweep
        run_teacher_forcing_sweep(horizons=TEACHER_FORCING_HORIZONS)

    # 2. KITTI model finetuned on Moving MNIST
    cfg = variant["kitti_finetuned"]
    mnist_settings.WEIGHTS_DIR = cfg["weights_dir"]
    mnist_settings.MNIST_MODEL = cfg["model_file"]
    mse_finetuned, snr_finetuned, ssim_finetuned, std_mse_finetuned, std_snr_finetuned, std_ssim_finetuned = finetuned_run_eval()

    # 3. Raw KITTI model (no finetuning)
    cfg = variant["kitti"]
    mnist_settings.KITTI_WEIGHTS = cfg["kitti_weights_dir"]
    mse_kitti, snr_kitti, ssim_kitti, std_mse_kitti, std_snr_kitti, std_ssim_kitti = kitti_run_eval()

    # Collect all 4 baseline results and write combined summary.
    # Order: copy previous frame, KITTI trained, KITTI finetuned, trained on MNIST
    model_results = [
        ("Copy previous frame",                mse_prev,      snr_prev,      ssim_prev,      std_mse_prev,      std_snr_prev,      std_ssim_prev),
        ("KITTI trained",                      mse_kitti,     snr_kitti,     ssim_kitti,     std_mse_kitti,     std_snr_kitti,     std_ssim_kitti),
        ("KITTI trained + finetuned on MNIST", mse_finetuned, snr_finetuned, ssim_finetuned, std_mse_finetuned, std_snr_finetuned, std_ssim_finetuned),
        ("Trained on MNIST",                   mse_mnist,     snr_mnist,     ssim_mnist,     std_mse_mnist,     std_snr_mnist,     std_ssim_mnist),
    ]
    _write_combined_metrics(results_dir, mode_str, model_results)


def main() -> None:
    for variant in MODEL_VARIANTS:
        for cond in CONDITIONS:
            if cond["type"] != variant["type"]:
                continue
            run_for_condition(variant, cond)

    print("\nAll MNIST conditions evaluated for all variants.")


if __name__ == "__main__":
    main()
