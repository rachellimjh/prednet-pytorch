import os
import re
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_ROOT = "results"

# Human-readable condition labels -> subdirectory names under results/
# You can change these directory names to point at any 4 result folders you like.
CONDITION_DIRS = {
    "ID no anomaly": "id_digits_normal",
    "OOD no anomaly": "ood_digits_normal",
    "ID anomaly": "id_digits_disappear",
    "OOD anomaly": "ood_digits_disappear",
}

# Model display names -> (metrics file name, model label inside that file)
MODEL_SPECS = {
    "Copy previous frame": ("mnist_eval_metrics.txt", "Copy previous frame"),
    "KITTI": ("kitti_eval_metrics.txt", "KITTI"),
    "KITTI + finetuned + Moving MNIST": (
        "finetuned_mnist_eval_metrics.txt",
        "KITTI + finetuned + Moving MNIST",
    ),
    "Moving MNIST": ("mnist_eval_metrics.txt", "Moving MNIST"),
}

METRICS = ["MSE", "SNR", "SSIM"]

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_floats(line: str) -> List[float]:
    return [float(x) for x in _FLOAT_RE.findall(line)]


def parse_metrics_file(path: str, fallback_model_label: str | None = None) -> Dict[str, Dict[str, float]]:
    """
    Parse a unified metrics file. Supports:
    - New format: "Model: NAME" blocks with MSE/SNR/SSIM
    - Old format: plain MSE/SNR/SSIM lines (no Model: line)
    Returns:
        {model_name: {metric_name: value}}
    """
    models: Dict[str, Dict[str, float]] = {}
    current_model: str | None = None
    orphan_metrics: Dict[str, float] = {}

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Model:"):
                current_model = line.split("Model:", 1)[1].strip()
                models[current_model] = {}
                continue

            for metric in METRICS:
                if line.startswith(metric):
                    nums = _extract_floats(line)
                    if nums:
                        if current_model is not None:
                            models[current_model][metric] = nums[0]
                        else:
                            orphan_metrics[metric] = nums[0]
                    break

    # Old format: no Model: line, just metrics at top level
    if not models and orphan_metrics and fallback_model_label:
        models[fallback_model_label] = orphan_metrics

    return models


def load_metrics_for_condition(condition_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load metrics for all models for a single condition directory.

    Returns:
        {model_name: {metric_name: value}}
    """
    condition_path = os.path.join(RESULTS_ROOT, condition_dir)
    if not os.path.isdir(condition_path):
        raise FileNotFoundError(f"Results directory not found: {condition_path}")

    result: Dict[str, Dict[str, float]] = {}

    for display_name, (filename, model_label) in MODEL_SPECS.items():
        metrics_path = os.path.join(condition_path, filename)
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(
                f"Expected metrics file '{filename}' for model '{display_name}' "
                f"not found in {condition_path}"
            )

        all_models = parse_metrics_file(metrics_path, fallback_model_label=model_label)
        if model_label not in all_models:
            raise KeyError(
                f"Model label '{model_label}' not found in metrics file {metrics_path}"
            )

        result[display_name] = all_models[model_label]

    return result


def collect_all_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Collect metrics for all conditions and models.

    Returns:
        {condition_label: {model_name: {metric_name: value}}}
    """
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond_label, cond_dir in CONDITION_DIRS.items():
        all_metrics[cond_label] = load_metrics_for_condition(cond_dir)
    return all_metrics


def plot_bar_charts(all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Create one bar chart per metric with:
      - X-axis: 4 conditions (ID/OOD x anomaly/no anomaly)
      - Bars within each group: 4 models
    """
    output_dir = os.path.join(RESULTS_ROOT, "summary_bar_plots")
    os.makedirs(output_dir, exist_ok=True)

    conditions = list(CONDITION_DIRS.keys())
    models = list(MODEL_SPECS.keys())

    num_conditions = len(conditions)
    num_models = len(models)

    x = list(range(num_conditions))
    bar_width = 0.18
    total_width = bar_width * num_models
    offsets = [i * bar_width - total_width / 2 + bar_width / 2 for i in range(num_models)]

    for metric in METRICS:
        plt.figure(figsize=(10, 6))

        for model_idx, model_name in enumerate(models):
            y_vals: List[float] = []
            for cond_label in conditions:
                try:
                    value = all_metrics[cond_label][model_name][metric]
                except KeyError:
                    value = float("nan")
                y_vals.append(value)

            bar_positions = [xi + offsets[model_idx] for xi in x]
            plt.bar(bar_positions, y_vals, width=bar_width, label=model_name)

        plt.xticks(x, conditions, rotation=20)
        plt.ylabel(metric)
        plt.title(f"{metric} across conditions and models")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{metric.lower()}_bar_plot.png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved {metric} bar plot to {out_path}")


def main() -> None:
    all_metrics = collect_all_metrics()
    plot_bar_charts(all_metrics)


if __name__ == "__main__":
    main()

