#!/usr/bin/env python3
"""Utility to compile cached cross-fold metrics into a consolidated heatmap."""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


HEATMAP_METRIC_DEFS: List[Tuple[str, str]] = [
    ("train_corr", "Train corr"),
    ("test_corr", "Test corr"),
    ("train_p", "Train p-value"),
    ("test_p", "Test p-value"),
    ("total_loss", "Total loss"),
]


def _load_metric_records(pattern: str) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for file_path in sorted(glob.glob(pattern)):
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            metrics = payload.get("metrics", {})
            record = {
                "task_alpha": float(payload.get("task_alpha")),
                "bold_alpha": float(payload.get("bold_alpha")),
                "beta_alpha": float(payload.get("beta_alpha")),
                "rho": float(payload.get("rho")),
                "metrics": {key: (float(metrics[key]) if metrics.get(key) is not None else np.nan)
                             for key, _ in HEATMAP_METRIC_DEFS},
            }
            records.append(record)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Skipping {file_path}: {exc}")
    return records


def plot_metric_heatmap(combo_labels, metric_labels, metric_values, output_path, cmap="magma"):
    metric_array = np.asarray(metric_values, dtype=np.float64)
    if metric_array.size == 0:
        raise ValueError("No metric values available to plot.")

    finite_mask = np.isfinite(metric_array)
    if not np.any(finite_mask):
        raise ValueError("All metric values are NaN or infinite.")

    masked_values = np.ma.masked_invalid(metric_array)
    vmin = np.nanmin(metric_array)
    vmax = np.nanmax(metric_array)
    if np.isclose(vmin, vmax):
        # Avoid a degenerate color scale when all values are identical.
        vmin -= 0.5
        vmax += 0.5

    fig_width = max(9.0, min(0.6 * len(combo_labels), 24.0))
    fig_height = max(3.5, 0.6 * len(metric_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(masked_values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(combo_labels)))
    ax.set_xticklabels(combo_labels, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Alpha / rho combinations")
    ax.set_ylabel("Metric")
    ax.set_title("Cross-fold averages per hyperparameter set")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Metric value")

    def _format_display(label, value):
        if not np.isfinite(value):
            return "–"
        label_lower = label.lower()
        if "loss" in label_lower:
            if abs(value) >= 1e3 or abs(value) < 1e-2:
                return f"{value:.2e}"
            return f"{value:.3f}"
        if "p-value" in label_lower or label_lower.endswith("(p)"):
            return f"{value:.3f}"
        return f"{value:.3f}"

    for row_idx in range(metric_array.shape[0]):
        for col_idx in range(metric_array.shape[1]):
            display_text = _format_display(metric_labels[row_idx], metric_array[row_idx, col_idx])
            value = metric_array[row_idx, col_idx]
            if np.isfinite(value):
                normalized_val = norm(value)
                text_color = "#ffffff" if normalized_val > 0.6 else "#0a0a0a"
                ax.text(col_idx, row_idx, display_text, ha="center", va="center", color=text_color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a heatmap from cached metric summaries.")
    parser.add_argument("--subject", required=True, help="Subject identifier, e.g., 04")
    parser.add_argument("--session", required=True, help="Session identifier, e.g., 1")
    parser.add_argument("--cache-pattern", default=None,
                        help="Glob pattern for cache files. Defaults to crossfold_metrics_sub{subject}_ses{session}_*.json")
    parser.add_argument("--output", default=None, help="Path for the output PNG heatmap.")
    args = parser.parse_args()

    if args.cache_pattern is None:
        args.cache_pattern = f"crossfold_metrics_sub{args.subject}_ses{args.session}_*.json"
    records = _load_metric_records(args.cache_pattern)
    if not records:
        raise SystemExit(f"No cache files matched pattern '{args.cache_pattern}'.")

    # Sort records to keep columns deterministic.
    records.sort(key=lambda rec: (rec["task_alpha"], rec["bold_alpha"], rec["beta_alpha"], rec["rho"]))
    combo_labels = [f"task={rec['task_alpha']:g}\n bold={rec['bold_alpha']:g}\n beta={rec['beta_alpha']:g}\n rho={rec['rho']:g}"
                    for rec in records]
    metric_values = [[rec["metrics"][metric_key] for rec in records] for metric_key, _ in HEATMAP_METRIC_DEFS]
    metric_labels = [label for _, label in HEATMAP_METRIC_DEFS]
    if args.output is None:
        args.output = f"crossfold_metric_heatmap_sub{args.subject}_ses{args.session}_cached.png"

    plot_metric_heatmap(combo_labels, metric_labels, metric_values, args.output)


if __name__ == "__main__":
    main()
