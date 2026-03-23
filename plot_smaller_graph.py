import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

# Import your existing logic
from plot_eval_bars import collect_all_metrics, MODEL_SPECS, METRICS, METRIC_LABELS, _group_condition

RESULTS_ROOT = "results"
COMPARISON_DIR = os.path.join(RESULTS_ROOT, "comparisons")
os.makedirs(COMPARISON_DIR, exist_ok=True)


def _set_ylim_with_headroom(ax, all_vals, all_tops, headroom=0.10):
    """Set ylim so bars + error caps + rotated value labels all fit."""
    y_low = min(all_vals)
    y_high = max(all_tops)
    data_rng = (y_high - y_low) or 1.0
    ax.set_ylim(y_low - data_rng * 0.05, y_high + data_rng * headroom)



def plot_combined_comparisons(all_metrics):
    models = list(MODEL_SPECS.keys())
    bar_width = 0.15

    # --- Flatten: keep only Normal conditions (one entry per base group) ---
    flat_data = {}
    for variant, cond_metrics in all_metrics.items():
        for label, data in cond_metrics.items():
            base, anomaly = _group_condition(label)
            if anomaly == "Normal":
                flat_data[base] = data

    for metric in METRICS:
        std_key = f"{metric}_STD"

        # ----------------------------------------------------------------
        # PLOT 1: Seen vs Unseen Pairs (Digits & Motion side-by-side)
        # ----------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))

        pairs = [
            ("Digits Comparison", ["Seen digits", "Unseen digits"]),
            ("Motion Comparison", ["Seen motion", "Unseen motion"]),
        ]

        for idx, (title, group_labels) in enumerate(pairs):
            ax = ax1 if idx == 0 else ax2
            x = np.arange(len(group_labels))

            # Pre-compute y bounds including error bars
            all_vals_sc, all_tops_sc = [], []
            for mn in models:
                for lbl in group_labels:
                    v = flat_data.get(lbl, {}).get(mn, {}).get(metric, 0)
                    e = flat_data.get(lbl, {}).get(mn, {}).get(std_key, 0.0)
                    all_vals_sc.append(v)
                    all_tops_sc.append(v + e)
            _set_ylim_with_headroom(ax, all_vals_sc, all_tops_sc)

            for i, model_name in enumerate(models):
                vals = [flat_data.get(lbl, {}).get(model_name, {}).get(metric, 0) for lbl in group_labels]
                errs = [flat_data.get(lbl, {}).get(model_name, {}).get(std_key, 0.0) for lbl in group_labels]
                offset = (i - len(models) / 2) * bar_width + bar_width / 2
                ax.bar(x + offset, vals, bar_width, label=model_name,
                       yerr=errs, capsize=4,
                       error_kw={"elinewidth": 1.2, "capthick": 1.2})

            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, fontsize=14)
            ax.tick_params(axis='y', labelsize=13)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=16, labelpad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Single legend inside the right subplot
        ax2.legend(
            loc="best",
            fontsize=13,
            framealpha=0.9,
            edgecolor='gray',
        )

        plt.tight_layout()
        base_path = os.path.join(COMPARISON_DIR, f"combined_{metric}_pairs")
        plt.savefig(base_path + ".png", bbox_inches="tight", dpi=300)
        plt.savefig(base_path + ".pdf", bbox_inches="tight")
        plt.close()
        print(f"Saved combined_{metric}_pairs.png/.pdf")

        # ----------------------------------------------------------------
        # PLOT 2: Broad Seen vs Unseen
        # ----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(16, 8))

        categories = ["Overall Seen\n(0-4 / Vertical)", "Overall Unseen\n(5-9 / Horizontal)"]
        x_cat = np.arange(len(categories))

        # Pre-compute all model data to set ylim before drawing
        model_data = []
        for model_name in models:
            s_digit  = flat_data.get("Seen digits",   {}).get(model_name, {}).get(metric, 0)
            s_motion = flat_data.get("Seen motion",   {}).get(model_name, {}).get(metric, 0)
            u_digit  = flat_data.get("Unseen digits", {}).get(model_name, {}).get(metric, 0)
            u_motion = flat_data.get("Unseen motion", {}).get(model_name, {}).get(metric, 0)

            seen_avg   = (s_digit + s_motion) / 2
            unseen_avg = (u_digit + u_motion) / 2

            s_digit_std  = flat_data.get("Seen digits",   {}).get(model_name, {}).get(std_key, 0.0)
            s_motion_std = flat_data.get("Seen motion",   {}).get(model_name, {}).get(std_key, 0.0)
            u_digit_std  = flat_data.get("Unseen digits", {}).get(model_name, {}).get(std_key, 0.0)
            u_motion_std = flat_data.get("Unseen motion", {}).get(model_name, {}).get(std_key, 0.0)

            seen_std   = np.sqrt(s_digit_std**2 + s_motion_std**2) / 2
            unseen_std = np.sqrt(u_digit_std**2 + u_motion_std**2) / 2

            model_data.append((model_name, seen_avg, unseen_avg, seen_std, unseen_std))

        all_vals_b = [v for _, s, u, _, _ in model_data for v in (s, u)]
        all_tops_b = [v + e for _, s, u, ss, us in model_data for v, e in ((s, ss), (u, us))]
        _set_ylim_with_headroom(ax, all_vals_b, all_tops_b)

        for i, (model_name, seen_avg, unseen_avg, seen_std, unseen_std) in enumerate(model_data):
            vals = [seen_avg, unseen_avg]
            errs = [seen_std, unseen_std]
            offset = (i - len(models) / 2) * bar_width + bar_width / 2
            ax.bar(x_cat + offset, vals, bar_width, label=model_name,
                   yerr=errs, capsize=4,
                   error_kw={"elinewidth": 1.2, "capthick": 1.2})

        ax.set_xticks(x_cat)
        ax.set_xticklabels(categories, fontsize=14)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylabel(metric, fontsize=16, labelpad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(
            loc="best",
            fontsize=13,
            framealpha=0.9,
            edgecolor='gray',
        )

        plt.tight_layout()
        base_path = os.path.join(COMPARISON_DIR, f"broad_comparison_{metric}")
        plt.savefig(base_path + ".png", bbox_inches="tight", dpi=300)
        plt.savefig(base_path + ".pdf", bbox_inches="tight")
        plt.close()
        print(f"Saved broad_comparison_{metric}.png/.pdf")


def main():
    print("Collecting all metrics...")
    all_metrics = collect_all_metrics()
    print("Generating flattened comparison plots...")
    plot_combined_comparisons(all_metrics)
    print(f"Results saved to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
