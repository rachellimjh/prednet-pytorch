import os
import re
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_ROOT = "results"

CONDITION_LABEL_MAP = {
    "id_digits_normal": "Seen digits",
    "ood_digits_normal": "Unseen digits",
    "id_vertical_normal": "Seen motion",
    "ood_vertical_normal": "Unseen motion",

    "id_digits_appearance": "Seen digits (appearance anomaly)",
    "id_digits_disappearance": "Seen digits (disappearance anomaly)",
    "id_digits_collision": "Seen digits (collision anomaly)",

    "ood_digits_appearance": "Unseen digits (appearance anomaly)",
    "ood_digits_disappearance": "Unseen digits (disappearance anomaly)",
    "ood_digits_collision": "Unseen digits (collision anomaly)",

    "id_vertical_appearance": "Seen motion (appearance anomaly)",
    "id_vertical_disappearance": "Seen motion (disappearance anomaly)",
    "id_vertical_collision": "Seen motion (collision anomaly)",

    "ood_vertical_appearance": "Unseen motion (appearance anomaly)",
    "ood_vertical_disappearance": "Unseen motion (disappearance anomaly)",
    "ood_vertical_collision": "Unseen motion (collision anomaly)",
}

# Conditions are discovered dynamically from RESULTS_ROOT; this mapping is kept
# for backwards compatibility but is not used directly.
CONDITION_DIRS: Dict[str, str] = {}

# Model display names -> (metrics file name, model label inside that file)
MODEL_SPECS = {
    "Copy previous frame": ("mnist_eval_metrics.txt", "Copy previous frame"),
    "KITTI": ("kitti_eval_metrics.txt", "KITTI"),
    "KITTI + finetuned on Moving MNIST": (
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


def discover_condition_groups() -> Dict[str, Dict[str, str]]:
    """
    Walk the results tree and find all directories that contain metrics for all
    models in MODEL_SPECS. Returns a mapping grouped by top-level variant
    directory (e.g., \"id_digits\", \"id_vertical\"):
        {variant: {condition_label: relative_dir_from_RESULTS_ROOT}}
    The condition label is the relative path without the variant prefix.
    """
    groups: Dict[str, Dict[str, str]] = {}
    required_files = {fname for (fname, _) in MODEL_SPECS.values()}

    for root, dirs, files in os.walk(RESULTS_ROOT):
        files_set = set(files)
        if not required_files.issubset(files_set):
            continue

        rel_path = os.path.relpath(root, RESULTS_ROOT)
        if rel_path == ".":
            continue

        parts = rel_path.split(os.sep, 1)
        variant = parts[0]
        leaf = parts[1] if len(parts) > 1 else ""

        # Use the leaf (e.g., \"id_digits_normal\") as label; if there's no
        # leaf, fall back to the variant name itself.
        label = leaf if leaf else variant

        if variant not in groups:
            groups[variant] = {}
        groups[variant][label] = rel_path

    return groups


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


def collect_all_metrics() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Collect metrics for all conditions and models.

    Returns:
        {variant: {condition_label: {model_name: {metric_name: value}}}}
    """
    all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    condition_groups = discover_condition_groups()
    if not condition_groups:
        raise RuntimeError(f"No condition directories with metrics found under {RESULTS_ROOT!r}")

    for variant, cond_map in sorted(condition_groups.items()):
        all_metrics[variant] = {}
        for cond_label, cond_dir in sorted(cond_map.items()):
            all_metrics[variant][cond_label] = load_metrics_for_condition(cond_dir)

    return all_metrics


def plot_bar_charts(all_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> None:
    """
    Create one bar chart per metric with:
      - X-axis: 4 conditions (ID/OOD x anomaly/no anomaly)
      - Bars within each group: 4 models
    """
    models = list(MODEL_SPECS.keys())
    num_models = len(models)

    for variant, cond_metrics in sorted(all_metrics.items()):
        conditions = list(cond_metrics.keys())
        num_conditions = len(conditions)

        if num_conditions == 0:
            continue

        output_dir = os.path.join(RESULTS_ROOT, variant, "summary_bar_plots")
        os.makedirs(output_dir, exist_ok=True)

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
                        value = cond_metrics[cond_label][model_name][metric]
                    except KeyError:
                        value = float("nan")
                    y_vals.append(value)

                bar_positions = [xi + offsets[model_idx] for xi in x]
              
                bars = plt.bar(bar_positions, y_vals, width=bar_width, label=model_name)
                for xpos, val in zip(bar_positions, y_vals):
                    if not (val != val):  # skip NaN
                        plt.text(xpos, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
            ##
            all_vals = []
            for cond_label in conditions:
                for model_name in models:
                    try:
                        all_vals.append(cond_metrics[cond_label][model_name][metric])
                    except KeyError:
                        pass

            if all_vals:
                y_min = min(all_vals)
                y_max = max(all_vals)
                margin = (y_max - y_min) * 0.2 if y_max != y_min else 0.01
                plt.ylim(y_min - margin, y_max + margin)
            ##
            labels = [CONDITION_LABEL_MAP.get(c, c) for c in conditions]
            plt.xticks(x, labels, rotation=25)
            plt.ylabel(metric)
            plt.title(f"{metric} across conditions and models ({variant})")
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

