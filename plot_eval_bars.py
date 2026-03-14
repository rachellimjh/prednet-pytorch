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
    "ood_horizontal_normal": "Unseen motion",

    "id_digits_appear": "Seen digits (appearance anomaly)",
    "id_digits_disappear": "Seen digits (disappearance anomaly)",
    "id_digits_stick": "Seen digits (collision anomaly)",

    "ood_digits_appear": "Unseen digits (appearance anomaly)",
    "ood_digits_disappear": "Unseen digits (disappearance anomaly)",
    "ood_digits_stick": "Unseen digits (collision anomaly)",

    "id_vertical_appear": "Seen motion (appearance anomaly)",
    "id_vertical_disappear": "Seen motion (disappearance anomaly)",
    "id_vertical_stick": "Seen motion (collision anomaly)",

    "ood_horizontal_appear": "Unseen motion (appearance anomaly)",
    "ood_horizontal_disappear": "Unseen motion (disappearance anomaly)",
    "ood_horizontal_stick": "Unseen motion (collision anomaly)",
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


def _group_condition(label: str):
    """Map raw condition folder name -> (base_group, anomaly_type)."""

    if "id_digits" in label:
        base = "Seen digits"
    elif "ood_digits" in label:
        base = "Unseen digits"
    elif "id_vertical" in label:
        base = "Seen motion"
    elif "ood_horizontal" in label:
        base = "Unseen motion"
    else:
        base = label

    # check longer strings first
    if "disappear" in label:
        anomaly = "Disappearance"
    elif "appear" in label:
        anomaly = "Appearance"
    elif "stick" in label:
        anomaly = "Collision"
    else:
        anomaly = "Normal"

    return base, anomaly


def plot_bar_charts(all_metrics):
    """
    Improved bar charts:
    - groups conditions logically
    - prevents number overlap
    - cleaner x-axis labels
    """

    models = list(MODEL_SPECS.keys())
    num_models = len(models)

    base_order = ["Seen digits", "Unseen digits", "Seen motion", "Unseen motion"]
    
    anomaly_order = ["Normal", "Appearance", "Disappearance", "Collision"]

    for variant, cond_metrics in sorted(all_metrics.items()):

        # group conditions
        from collections import defaultdict
        grouped = defaultdict(dict)

        for cond_label, data in cond_metrics.items():
            base, anomaly = _group_condition(cond_label)
            print(f"variant={variant}, cond_label={cond_label}, base={base}, anomaly={anomaly}")
            grouped[base][anomaly] = data

        output_dir = os.path.join(RESULTS_ROOT, variant, "summary_bar_plots")
        os.makedirs(output_dir, exist_ok=True)

        bar_width = 0.18

        for metric in METRICS:

            plt.figure(figsize=(17, 10))

            x_positions = []
            labels = []
            cond_lookup = []

            xpos = 0

            for base in base_order:
                if base not in grouped:
                    continue

                for anomaly in anomaly_order:
                    if anomaly not in grouped[base]:
                        continue

                    x_positions.append(xpos)
                    labels.append(f"{base}\n{anomaly}")
                    cond_lookup.append(grouped[base][anomaly])
                    xpos += 0.8

                xpos += 0.6  # gap between groups

            total_width = bar_width * num_models
            offsets = [
                i * bar_width - total_width / 2 + bar_width / 2
                for i in range(num_models)
            ]

            # collect all metric values (for y scaling)
            all_vals = []
            for cond in cond_lookup:
                for model in models:
                    if metric in cond[model]:
                        all_vals.append(cond[model][metric])

            y_min = min(all_vals)
            y_max = max(all_vals)
            margin = (y_max - y_min) * 0.05
            lower = y_min - margin
            upper = y_max + margin
            plt.ylim(lower, upper)

            for model_idx, model_name in enumerate(models):

                y_vals = []
                for cond in cond_lookup:
                    y_vals.append(cond[model_name].get(metric, float("nan")))

                bar_positions = [
                    xi + offsets[model_idx] for xi in x_positions
                ]

                bars = plt.bar(
                    bar_positions,
                    y_vals,
                    width=bar_width,
                    label=model_name,
                )

                # value labels (fixed overlap)
                for xpos_bar, val in zip(bar_positions, y_vals):
                    if not (val != val):  # skip NaN
                        plt.text(
                            xpos_bar,
                            val + margin * 0.05,
                            str(val),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            rotation=90,
                        )

            plt.xticks(x_positions, labels, rotation=0)
            plt.title(f"{metric}")
            plt.legend()

            plt.tight_layout()

            out_path = os.path.join(
                output_dir, f"{metric.lower()}_bar_plot.png"
            )
            plt.savefig(out_path)
            plt.close()

            print(f"Saved {metric} bar plot to {out_path}")

def main() -> None:
    all_metrics = collect_all_metrics()
    plot_bar_charts(all_metrics)


if __name__ == "__main__":
    main()

