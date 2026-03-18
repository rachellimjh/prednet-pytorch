import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

# Import your existing logic
from plot_eval_bars import collect_all_metrics, MODEL_SPECS, METRICS, _group_condition

RESULTS_ROOT = "results"
COMPARISON_DIR = os.path.join(RESULTS_ROOT, "comparisons")
os.makedirs(COMPARISON_DIR, exist_ok=True)

def add_value_labels(ax, spacing=5):
    """Add labels on top of each bar."""
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Don't label if value is 0 or NaN
        if y_value != 0 and not np.isnan(y_value):
            label = f"{y_value:.4f}"
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(0, spacing),
                textcoords="offset points",
                ha='center',
                va='bottom',
                # rotation=90,
                fontsize=8
            )

def plot_combined_comparisons(all_metrics):
    models = list(MODEL_SPECS.keys())
    bar_width = 0.15
    
    # --- STEP 1: Flatten the data ---
    # This combines all variants (digits and motion) into one dictionary
    flat_data = {}
    for variant, cond_metrics in all_metrics.items():
        for label, data in cond_metrics.items():
            base, anomaly = _group_condition(label)
            if anomaly == "Normal":
                flat_data[base] = data

    for metric in METRICS:
        # --- PLOT 1: Seen vs Unseen Pairs (Digits & Motion side-by-side) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        pairs = [
            ("Digits Comparison", ["Seen digits", "Unseen digits"]),
            ("Motion Comparison", ["Seen motion", "Unseen motion"])
        ]
        
        for idx, (title, group_labels) in enumerate(pairs):
            ax = ax1 if idx == 0 else ax2
            x = np.arange(len(group_labels))
            
            for i, model_name in enumerate(models):
                vals = [flat_data.get(lbl, {}).get(model_name, {}).get(metric, 0) for lbl in group_labels]
                offset = (i - len(models)/2) * bar_width + bar_width/2
                ax.bar(x + offset, vals, bar_width, label=model_name)
            
            ax.set_title(f"{title} ({metric})", pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(group_labels)
            add_value_labels(ax)

        plt.tight_layout()
        plt.savefig(os.path.join(COMPARISON_DIR, f"combined_{metric}_pairs.png"))
        plt.close()

        # --- PLOT 2: Broad Seen vs Unseen ---
        # Comparing {Seen Digits + Seen Motion} vs {Unseen Digits + Unseen Motion}
        plt.figure(figsize=(12, 8))
        categories = ["Overall Seen\n(0-4 / Vertical)", "Overall Unseen\n(5-9 / Horizontal)"]
        x_cat = np.arange(len(categories))

        ax = plt.gca()
        for i, model_name in enumerate(models):
            # Calculate averages across the two domains
            s_digit = flat_data.get("Seen digits", {}).get(model_name, {}).get(metric, 0)
            s_motion = flat_data.get("Seen motion", {}).get(model_name, {}).get(metric, 0)
            u_digit = flat_data.get("Unseen digits", {}).get(model_name, {}).get(metric, 0)
            u_motion = flat_data.get("Unseen motion", {}).get(model_name, {}).get(metric, 0)
            
            seen_avg = (s_digit + s_motion) / 2
            unseen_avg = (u_digit + u_motion) / 2
            
            vals = [seen_avg, unseen_avg]
            offset = (i - len(models)/2) * bar_width + bar_width/2
            ax.bar(x_cat + offset, vals, bar_width, label=model_name)

        plt.xticks(x_cat, categories)
        plt.title(f"Broad Comparison: Seen vs Unseen Categories ({metric})", pad=30)
        add_value_labels(ax)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.savefig(os.path.join(COMPARISON_DIR, f"broad_comparison_{metric}.png"))
        plt.close()

def main():
    print("Collecting all metrics...")
    all_metrics = collect_all_metrics()
    print("Generating flattened comparison plots...")
    plot_combined_comparisons(all_metrics)
    print(f"Results saved to: {COMPARISON_DIR}")

if __name__ == "__main__":
    main()