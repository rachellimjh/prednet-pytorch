import importlib
import os

from mnist_training import mnist_settings


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
        "name": "OOD vertical no anomaly",
        "type": "vertical",
        "npz_file": "OOD_NORMAL_VERTICAL.npz",
        "results_dir": "ood_vertical_normal",
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


def run_for_condition(variant: dict, cond: dict) -> None:
    """
    Run all MNIST-based evaluations (Moving MNIST, KITTI finetuned, KITTI raw)
    for a single variant + dataset condition.
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

    # 1. Moving MNIST model (trained only on MNIST)
    cfg = variant["moving_mnist"]
    mnist_settings.WEIGHTS_DIR = cfg["weights_dir"]
    mnist_settings.MNIST_MODEL = cfg["model_file"]
    import mnist_training.mnist_eval as mnist_eval
    importlib.reload(mnist_eval)

    # 2. KITTI model finetuned on Moving MNIST
    cfg = variant["kitti_finetuned"]
    mnist_settings.WEIGHTS_DIR = cfg["weights_dir"]
    mnist_settings.MNIST_MODEL = cfg["model_file"]
    import mnist_finetuning.finetuning_eval as finetuning_eval
    importlib.reload(finetuning_eval)

    # 3. Raw KITTI model (no finetuning)
    cfg = variant["kitti"]
    mnist_settings.KITTI_WEIGHTS = cfg["kitti_weights_dir"]
    import kitti_pretraining.kitti_eval_mnist as kitti_eval_mnist
    importlib.reload(kitti_eval_mnist)


def main() -> None:
    for variant in MODEL_VARIANTS:
        for cond in CONDITIONS:
            if cond["type"] != variant["type"]:
                continue
            run_for_condition(variant, cond)

    print("\nAll MNIST conditions evaluated for all variants.")


if __name__ == "__main__":
    main()

