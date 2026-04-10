"""
Generate publication-quality comparison plots across all 4 CARR modes.

Designed for ACL-format papers: white backgrounds, Times/serif fonts,
colorblind-friendly palette, consistent colors across all plots.

Reads history.json from each mode's output directory and produces:
  1. Training Loss curves
  2. Validation Perplexity comparison (bar)
  3. Routing Metrics over epochs (Entropy, CoV, Jaccard)
  4. Final Metrics bar chart comparison
  5. Before/After calibration comparison
  6. Per-layer routing heatmaps
  7. Summary table (image)
  8. Validation perplexity over epochs (line)

Usage:
    python scripts/plot_comparison.py                              # default dir
    python scripts/plot_comparison.py --results_dir ./carr_output  # custom dir
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  CONSISTENT COLOR PALETTE — used across ALL plots
# ══════════════════════════════════════════════════════════════════════
#  Colorblind-friendly (adapted from Tol's muted scheme)

MODE_ORDER = ["gate_only", "full_carr", "shared_expert"]

MODE_LABELS = {
    "gate_only":     "Gate-Only",
    "full_carr":     "Full CARR",
    "shared_expert": "CARR + Shared",
}

MODE_COLORS = {
    "gate_only":     "#CC6677",   # rose
    "full_carr":     "#4477AA",   # steel blue
    "shared_expert": "#228833",   # green
}

MODE_MARKERS = {
    "gate_only":     "^",
    "full_carr":     "o",
    "shared_expert": "D",
}

MODE_LINESTYLES = {
    "gate_only":     "-",
    "full_carr":     "-",
    "shared_expert": "-",
}

METRICS_LABELS = {
    "load_entropy": r"Load Entropy ($H$) $\uparrow$",
    "cov":          r"CoV $\downarrow$",
    "jaccard":      r"Jaccard Overlap $\downarrow$",
}


def setup_style():
    """Apply clean academic paper style — white bg, serif fonts."""
    plt.rcParams.update({
        # Background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # Borders & grid
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#cccccc",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",

        # Text & labels
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # Font — use serif (Times-compatible) for ACL papers
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Legend
        "legend.facecolor": "white",
        "legend.edgecolor": "#999999",
        "legend.framealpha": 0.9,

        # Figure quality
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Use TeX-like rendering
        "mathtext.fontset": "cm",
    })


def load_histories(results_dir):
    """Load history.json from each mode's subdirectory."""
    histories = {}
    for mode in MODE_ORDER:
        path = os.path.join(results_dir, mode, "history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[mode] = json.load(f)
            print(f"  ✓ Loaded: {path}")
        else:
            print(f"  ✗ Missing: {path} (skipped)")
    return histories


# ═══════════════════════════════════════════════════════════════════
#  Plot 1: Training Loss Curves
# ═══════════════════════════════════════════════════════════════════

def plot_training_loss(histories, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    has_data = False
    for mode in MODE_ORDER:
        if mode not in histories:
            continue
        losses = histories[mode].get("train_loss", [])
        if not losses:
            continue
        has_data = True
        steps = [e["step"] for e in losses]
        vals = [e["loss"] for e in losses]
        ax.plot(
            steps, vals,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            linewidth=1.6,
            linestyle=MODE_LINESTYLES[mode],
        )

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "1_training_loss.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 2: Perplexity Bar Chart
# ═══════════════════════════════════════════════════════════════════

def plot_perplexity_comparison(histories, output_dir):
    fig, ax = plt.subplots(figsize=(5, 3.5))

    modes_present = [m for m in MODE_ORDER if m in histories]
    labels = [MODE_LABELS[m] for m in modes_present]
    colors = [MODE_COLORS[m] for m in modes_present]

    ppls = []
    for mode in modes_present:
        h = histories[mode]
        fm = h.get("final_metrics", {})
        ppl = fm.get("perplexity", None)
        if ppl is None and h.get("val_perplexity"):
            ppl = h["val_perplexity"][-1]
        ppls.append(ppl if ppl is not None else 0)

    x = np.arange(len(modes_present))
    bars = ax.bar(x, ppls, color=colors, width=0.55, edgecolor="#333333", linewidth=0.6)

    for bar, ppl in zip(bars, ppls):
        if ppl > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ppls) * 0.02,
                f"{ppl:.1f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#111111",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Perplexity")
    ax.set_title("Final Perplexity (Post-Calibration)", fontweight="bold")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "2_perplexity_comparison.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 3: Routing Metrics Over Epochs
# ═══════════════════════════════════════════════════════════════════

def plot_routing_over_epochs(histories, output_dir):
    metric_keys = ["load_entropy", "cov", "jaccard"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    has_data = False
    for i, metric in enumerate(metric_keys):
        ax = axes[i]
        for mode in MODE_ORDER:
            if mode == "baseline" or mode not in histories:
                continue
            routing = histories[mode].get("routing_metrics", [])
            if not routing:
                continue
            has_data = True
            vals = [r.get(metric, 0) for r in routing]
            epochs = list(range(1, len(vals) + 1))
            ax.plot(
                epochs, vals,
                color=MODE_COLORS[mode],
                marker=MODE_MARKERS[mode],
                label=MODE_LABELS[mode],
                linewidth=1.6,
                markersize=6,
                markeredgecolor="#333333",
                markeredgewidth=0.5,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRICS_LABELS[metric])
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if i == 0:
            ax.legend(loc="best", framealpha=0.9)

    if not has_data:
        plt.close()
        return

    fig.suptitle("Routing Metrics Over Epochs", fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "3_routing_metrics_epochs.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 4: Final Metrics Grouped Bar Chart
# ═══════════════════════════════════════════════════════════════════

def plot_final_metrics_bars(histories, output_dir):
    metrics = ["perplexity", "load_entropy", "cov", "jaccard"]
    metric_labels = [
        r"Perplexity $\downarrow$",
        r"Entropy ($H$) $\uparrow$",
        r"CoV $\downarrow$",
        r"Jaccard $\downarrow$",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()

    modes_present = [m for m in MODE_ORDER if m in histories]

    for idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        labels_m = []
        vals = []
        colors = []

        for mode in modes_present:
            fm = histories[mode].get("final_metrics", {})
            v = fm.get(metric, None)
            if v is None:
                continue
            labels_m.append(MODE_LABELS[mode])
            vals.append(v)
            colors.append(MODE_COLORS[mode])

        if not vals:
            ax.set_visible(False)
            continue

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="#333333", linewidth=0.6)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#111111",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels_m, fontsize=8)
        ax.set_title(mlabel, fontweight="bold")
        ax.grid(True, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Final Metrics Comparison", fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(output_dir, "4_final_metrics_comparison.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 5: Before / After Calibration
# ═══════════════════════════════════════════════════════════════════

def plot_before_after(histories, output_dir):
    metrics = ["perplexity", "load_entropy", "cov", "jaccard"]
    metric_labels = [
        r"Perplexity",
        r"Entropy ($H$)",
        r"CoV",
        r"Jaccard",
    ]

    carr_modes = [m for m in MODE_ORDER if m in histories]
    if not carr_modes:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    before_color = "#AAAAAA"
    after_color_map = MODE_COLORS

    for idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        x_positions = np.arange(len(carr_modes))
        width = 0.32

        before_vals = []
        after_vals = []
        after_colors = []
        for mode in carr_modes:
            bm = histories[mode].get("baseline_metrics", {})
            fm = histories[mode].get("final_metrics", {})
            before_vals.append(bm.get(metric, 0) or 0)
            after_vals.append(fm.get(metric, 0) or 0)
            after_colors.append(MODE_COLORS[mode])

        ax.bar(
            x_positions - width / 2, before_vals, width,
            label="Before", color=before_color, edgecolor="#555555", linewidth=0.5,
        )
        for j, (xp, av, ac) in enumerate(zip(x_positions, after_vals, after_colors)):
            ax.bar(
                xp + width / 2, av, width,
                label="After" if j == 0 else None,
                color=ac, edgecolor="#333333", linewidth=0.5,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODE_LABELS[m] for m in carr_modes], fontsize=8)
        ax.set_title(mlabel, fontweight="bold")
        ax.grid(True, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Before vs. After Calibration", fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "5_before_after.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 6: Per-Layer Expert Usage Heatmaps
# ═══════════════════════════════════════════════════════════════════

def plot_per_layer_heatmaps(histories, output_dir):
    carr_modes = [m for m in MODE_ORDER if m != "baseline" and m in histories]
    modes_with_data = []

    for mode in carr_modes:
        routing = histories[mode].get("routing_metrics", [])
        if routing and "per_layer" in routing[-1]:
            modes_with_data.append(mode)

    if not modes_with_data:
        return

    fig, axes = plt.subplots(
        1, len(modes_with_data),
        figsize=(4.5 * len(modes_with_data), 5),
    )
    if len(modes_with_data) == 1:
        axes = [axes]

    for idx, mode in enumerate(modes_with_data):
        ax = axes[idx]
        per_layer = histories[mode]["routing_metrics"][-1]["per_layer"]

        layers = sorted(per_layer.keys(), key=lambda x: int(x))
        if not layers:
            continue

        num_experts = len(per_layer[layers[0]].get("expert_usage", []))
        heatmap_data = np.zeros((len(layers), num_experts))

        for i, lid in enumerate(layers):
            usage = per_layer[lid].get("expert_usage", [])
            for j, u in enumerate(usage):
                heatmap_data[i][j] = u

        # Normalize per layer
        row_sums = heatmap_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        heatmap_norm = heatmap_data / row_sums

        # Use a sequential colormap that prints well in B&W
        im = ax.imshow(
            heatmap_norm, aspect="auto", cmap="YlOrRd",
            interpolation="nearest", vmin=0,
        )
        ax.set_xlabel("Expert Index")
        if idx == 0:
            ax.set_ylabel("Layer")
        ax.set_title(MODE_LABELS[mode], fontweight="bold")
        ax.set_xticks(range(num_experts))
        # Show every 4th layer label to avoid clutter
        step = max(1, len(layers) // 8)
        ax.set_yticks(range(0, len(layers), step))
        ax.set_yticklabels([layers[i] for i in range(0, len(layers), step)])

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Usage", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Expert Usage Heatmaps (Final Epoch)", fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "6_per_layer_heatmaps.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 7: Summary Table (image for paper)
# ═══════════════════════════════════════════════════════════════════

def print_and_save_summary(histories, output_dir):
    modes_present = [m for m in MODE_ORDER if m in histories]

    header = f"  {'Mode':<20} {'PPL':>10} {'Entropy':>10} {'CoV':>10} {'Jaccard':>10} {'Time':>10}"
    print("\n" + "=" * 76)
    print("  FINAL COMPARISON SUMMARY")
    print("=" * 76)
    print(header)
    print("  " + "-" * 72)

    rows = []
    for mode in modes_present:
        fm = histories[mode].get("final_metrics", {})
        ppl = fm.get("perplexity")
        ent = fm.get("load_entropy")
        cov = fm.get("cov")
        jac = fm.get("jaccard")
        wt = histories[mode].get("wall_time_seconds", 0)

        row = {
            "mode": MODE_LABELS[mode],
            "ppl": f"{ppl:.2f}" if ppl else "---",
            "ent": f"{ent:.3f}" if ent else "---",
            "cov": f"{cov:.3f}" if cov else "---",
            "jac": f"{jac:.3f}" if jac else "---",
            "wt": f"{wt:.0f}s",
        }
        rows.append(row)
        print(f"  {row['mode']:<20} {row['ppl']:>10} {row['ent']:>10} {row['cov']:>10} {row['jac']:>10} {row['wt']:>10}")

    print("=" * 76)

    # Save as image
    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.45 * len(rows)))
    ax.axis("off")

    col_labels = [
        "Mode",
        r"PPL $\downarrow$",
        r"$H$ $\uparrow$",
        r"CoV $\downarrow$",
        r"Jaccard $\downarrow$",
        "Time",
    ]
    cell_data = [
        [r["mode"], r["ppl"], r["ent"], r["cov"], r["jac"], r["wt"]]
        for r in rows
    ]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.7)

    # Style — clean academic look
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(fontweight="bold", color="#111111")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#111111")
            # Color the mode name column with the mode color
            if col == 0:
                mode_name = modes_present[row - 1] if row - 1 < len(modes_present) else None
                if mode_name:
                    cell.set_text_props(
                        fontweight="bold",
                        color=MODE_COLORS.get(mode_name, "#111111"),
                    )

    path = os.path.join(output_dir, "7_summary_table.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # Also save as LaTeX table
    latex_path = os.path.join(output_dir, "summary_table.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Mode} & \\textbf{PPL} $\\downarrow$ & \\textbf{$H$} $\\uparrow$ & \\textbf{CoV} $\\downarrow$ & \\textbf{Jacc.} $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(f"{r['mode']} & {r['ppl']} & {r['ent']} & {r['cov']} & {r['jac']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{CARR 4-mode comparison results.}\n")
        f.write("\\label{tab:carr-comparison}\n")
        f.write("\\end{table}\n")
    print(f"  Saved: {latex_path}")


# ═══════════════════════════════════════════════════════════════════
#  Plot 8: Validation Perplexity Over Epochs
# ═══════════════════════════════════════════════════════════════════

def plot_val_perplexity_epochs(histories, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    has_data = False
    for mode in MODE_ORDER:
        if mode == "baseline" or mode not in histories:
            continue
        vppl = histories[mode].get("val_perplexity", [])
        if not vppl:
            continue
        has_data = True
        epochs = list(range(1, len(vppl) + 1))
        ax.plot(
            epochs, vppl,
            color=MODE_COLORS[mode],
            marker=MODE_MARKERS[mode],
            label=MODE_LABELS[mode],
            linewidth=1.8,
            markersize=7,
            markeredgecolor="#333333",
            markeredgewidth=0.5,
        )


    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Validation Perplexity Over Epochs", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(output_dir, "8_val_perplexity_epochs.png")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot CARR 4-mode comparison")
    parser.add_argument(
        "--results_dir", type=str, default="./carr_output",
        help="Directory containing mode subdirectories with history.json",
    )
    parser.add_argument(
        "--plots_dir", type=str, default=None,
        help="Output directory for plots (default: <results_dir>/plots)",
    )
    args = parser.parse_args()

    plots_dir = args.plots_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    setup_style()

    print("\n" + "=" * 60)
    print("  CARR 4-Mode Comparison — Plot Generator")
    print("=" * 60)
    print(f"\n  Results dir: {args.results_dir}")
    print(f"  Plots dir:   {plots_dir}\n")

    # Load histories
    print("Loading histories...")
    histories = load_histories(args.results_dir)

    if not histories:
        print("\n  ✗ No history files found! Run training first.")
        print(f"    Expected: {args.results_dir}/<mode>/history.json")
        sys.exit(1)

    print(f"\n  Found {len(histories)} mode(s): {', '.join(histories.keys())}\n")

    # Generate all plots
    print("Generating plots...\n")

    plot_training_loss(histories, plots_dir)
    plot_perplexity_comparison(histories, plots_dir)
    plot_routing_over_epochs(histories, plots_dir)
    plot_final_metrics_bars(histories, plots_dir)
    plot_before_after(histories, plots_dir)
    plot_per_layer_heatmaps(histories, plots_dir)
    plot_val_perplexity_epochs(histories, plots_dir)
    print_and_save_summary(histories, plots_dir)

    print(f"\n  ✓ All plots saved to: {plots_dir}")
    print(f"  ✓ LaTeX table: {os.path.join(plots_dir, 'summary_table.tex')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
