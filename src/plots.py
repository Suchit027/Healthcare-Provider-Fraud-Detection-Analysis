"""
plot_metrics.py
===============
Generates one bar chart per metric from evaluation_summary.csv.

Usage:
  python plot_metrics.py --csv evaluation_summary.csv [--out_dir ./plots]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Font / style (matches reference script) ──────────────────────────────────
plt.rcParams['font.weight']        = 'bold'
plt.rcParams['axes.labelweight']   = 'bold'
plt.rcParams['axes.titleweight']   = 'bold'


METRIC_LABELS = {
    "accuracy_mean":  "ACCURACY (%)",
    "precision_mean": "PRECISION (%)",
    "recall_mean":    "RECALL (%)",
    "f1_mean":        "F1-SCORE (%)",
    "pr_auc_mean":    "PR-AUC (%)",
}


def plot_metric(metric_col, metric_label, models, values, colors, out_dir):
    n_models = len(models)
    width    = 0.6

    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.2), 7))

    for j, (model, val, color) in enumerate(zip(models, values, colors)):
        ax.bar(x[j], val, width, color=color, label=model)
        ax.text(
            x[j], val + 0.3,
            f'{val:.2f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold', fontsize=9, rotation=15, ha='right')
    ax.set_ylabel(metric_label, fontweight='bold', fontsize=11)
    ax.set_title(metric_label, fontweight='bold', fontsize=13)

    # y-axis: give a little headroom above the tallest bar
    ax.set_ylim(0, min(max(values) * 1.15, 105))

    ax.legend(
        bbox_to_anchor=(1.02, 1), loc='upper left',
        fontsize=8, title='Models',
        title_fontsize=9,
    )

    plt.tight_layout()

    fname = metric_col.replace("_mean", "") + "_bar.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="evaluation_summary.csv",
                        help="evaluation_summary.csv")
    parser.add_argument("--out_dir", default="plots",
                        help="Directory to save chart PNGs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(r'D:\Suchit\Deep-Learning\evaluation_summary.csv')

    # Model names
    models = df["model"].tolist()
    n_models = len(models)

    # Assign one colour per model (same tab10 approach as reference script)
    cmap   = plt.cm.get_cmap("tab10", n_models)
    colors = [cmap(i) for i in range(n_models)]

    print(f"Loaded {len(df)} models from '{args.csv}'")
    print(f"Generating charts in '{args.out_dir}/'...\n")

    for metric_col, metric_label in METRIC_LABELS.items():
        if metric_col not in df.columns:
            print(f"  [SKIP] '{metric_col}' not found in CSV.")
            continue

        # Convert to percentage if values are in [0, 1]
        values = df[metric_col].values.copy().astype(float)
        if values.max() <= 1.0:
            values = values * 100

        plot_metric(metric_col, metric_label, models, values, colors, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()