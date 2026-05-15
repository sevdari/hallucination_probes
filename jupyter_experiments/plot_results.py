#!/usr/bin/env python3
"""
Reproduce every plot from the Jupyter notebooks using pre-extracted CSV/npy data.

Reads from:  jupyter_experiments/extracted_data/
Writes to:   jupyter_experiments/extracted_data/figures/

Usage:
  python plot_results.py              # all plots
  python plot_results.py --only activations layers plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_DIR = Path(__file__).parent / "extracted_data"
FIG_DIR  = DATA_DIR / "figures"

LAYERS = [4, 10, 16, 20, 24, 26, 28, 30]
MODELS = ["apertus", "llama"]
COLORS = {"apertus": "#E07B39", "llama": "#3B82C4"}


def savefig(name: str):
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight")
    try:
        display = path.relative_to(Path.cwd())
    except ValueError:
        display = path
    print(f"  -> {display}")
    plt.close()


# ---------------------------------------------------------------------------
# 1.  activations.ipynb
# ---------------------------------------------------------------------------

def plot_activations():
    print("\n[activations] Generating plots ...")

    metrics_csv = DATA_DIR / "activation_metrics.csv"
    recomputed_csv = DATA_DIR / "activation_metrics_recomputed.csv"
    pca_csv = DATA_DIR / "activation_pca_scatter.csv"
    counts_csv = DATA_DIR / "activation_counts.csv"

    if recomputed_csv.exists():
        metrics_df = pd.read_csv(recomputed_csv)
    elif metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
    else:
        print("  activation_metrics.csv not found — run extract_data.py first.")
        return

    # Resolve token counts for plot titles
    if counts_csv.exists():
        counts_df = pd.read_csv(counts_csv)
        n_hall = int(counts_df["hallucinated_tokens"].iloc[0])
        n_supp = int(counts_df["supported_tokens"].iloc[0])
    else:
        n_hall = int(metrics_df["n_pos"].iloc[0])
        n_supp = int(metrics_df["n_neg"].iloc[0])

    dataset_split = "test"

    # --- Plot 1: Separation metrics line plots (7 metrics × 2 models) ---
    metric_cols = [
        "silhouette_pca10",
        "linear_probe_auc_pca20",
        "linear_probe_acc_pca20",
        "kmeans_ari_pca20",
        "kmeans_nmi_pca20",
        "fisher_ratio_hidden",
        "centroid_cosine_dist_hidden",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle(
        f"Clustering metrics of Llama and Apertus activations\n"
        f"number of hallucinated datapoints: {n_hall} | supported datapoints: {n_supp}"
        f"\n'{dataset_split}' dataset split",
        fontsize=15,
        y=0.995,
    )
    axes = axes.flatten()
    for ax, metric in zip(axes, metric_cols):
        sns.lineplot(
            data=metrics_df,
            x="layer",
            y=metric,
            hue="model",
            marker="o",
            ax=ax,
            palette=COLORS,
        )
        ax.set_title(metric)
        ax.set_xlabel("Layer")
        ax.set_xticks(LAYERS)
    for i in range(len(metric_cols), len(axes)):
        axes[i].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 1))
    savefig("activations_separation_metrics.png")

    # --- Plot 2: Activation L2 norm by layer (activations_2.ipynb) ---
    norms_csv = DATA_DIR / "activation_norms.csv"
    norm_df = pd.read_csv(norms_csv) if norms_csv.exists() else pd.DataFrame()

    if not norm_df.empty:
        MODEL_DISPLAY = {"apertus": "Apertus-8B-Instruct-2509", "llama": "Llama-3.1-8B-Instruct"}
        NOTE_COLOR = "#888888"

        fig, ax = plt.subplots(figsize=(8, 5.0))
        for model_key in ["apertus", "llama"]:
            cur = norm_df[norm_df["model"] == model_key]
            agg = cur.groupby("layer")[["norm_mean", "norm_std", "norm_p90"]].mean().reset_index()
            c = COLORS[model_key]
            ax.fill_between(agg["layer"], agg["norm_mean"] - agg["norm_std"],
                            agg["norm_mean"] + agg["norm_std"], color=c, alpha=0.18, zorder=1)
            ax.plot(agg["layer"], agg["norm_mean"], color=c, marker="o", linewidth=2.2,
                    markersize=7, label=MODEL_DISPLAY[model_key], zorder=3)
            ax.plot(agg["layer"], agg["norm_p90"], color=c, marker="s", linestyle="--",
                    linewidth=1.4, markersize=5, zorder=3, label="_nolegend_")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.2f}")
        )
        ax.set_xticks(norm_df["layer"].unique().tolist())
        ax.set_xlabel("Transformer layer", fontsize=11)
        ax.set_ylabel("‖h‖₂  (activations L2 norm, log scale)", fontsize=11)
        ax.grid(True, which="both", linestyle=":", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        handles, lbls = ax.get_legend_handles_labels()
        title_handle = mpatches.Patch(visible=False)
        legend = fig.legend(
            [title_handle] + handles, ["Model:"] + lbls,
            loc="upper center", ncol=len(lbls) + 1, frameon=False,
            bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=2,
            handletextpad=0.4, fontsize=10,
        )
        legend.get_texts()[0].set_fontweight("bold")
        fig.suptitle("Residual stream L2 norm by layer for test dataset split", fontsize=13, y=1.08)
        fig.text(0.5, -0.04,
                 "Solid line: mean ‖h‖₂; Dashed line: 90th percentile ‖h‖₂; Shaded band: ±1 std.",
                 ha="center", fontsize=9, color=NOTE_COLOR)
        fig.tight_layout(rect=(0, 0, 1, 1))
        savefig("activations_norms.png")
    else:
        print("  activation_norms.csv not found — skipping norm plot (re-run --run-activations).")

    # --- Plot 3: Silhouette + AUC heatmaps ---
    pivot_sil = metrics_df.pivot(index="layer", columns="model", values="silhouette_pca10")
    pivot_auc = metrics_df.pivot(index="layer", columns="model", values="linear_probe_auc_pca20")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sns.heatmap(pivot_sil, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=-0.1, vmax=0.4, ax=axes[0], linewidths=0.7, linecolor="#e0e0e0", cbar=False)
    axes[0].set_title("Silhouette score (PCA-10)\nhigher = better separated")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Layer")
    sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.45, vmax=0.85, ax=axes[1], linewidths=0.7, linecolor="#e0e0e0", cbar=True)
    axes[1].set_title("Linear probe AUC (PCA-20)\nhigher = more linearly separable")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("")
    fig.suptitle("Separation metrics by layer — Apertus vs. Llama", fontsize=13)
    savefig("activations_heatmaps.png")

    # --- Plot 4: PCA scatter for HIGHLIGHT_LAYERS (activations_2.ipynb style) ---
    if not pca_csv.exists():
        print("  activation_pca_scatter.csv not found — skipping PCA scatter plot.")
        print("  Re-run extract_data.py --run-activations to generate it.")
        return

    pca_df = pd.read_csv(pca_csv)
    HIGHLIGHT_LAYERS = [4, 16, 26, 30]
    MODEL_DISPLAY = {"apertus": "Apertus-8B-Instruct-2509", "llama": "Llama-3.1-8B-Instruct"}
    PALETTE = {"Hallucinated": "#e05c5c", "Supported": "#4a90d9"}
    NOTE_COLOR = "#888888"

    # Build norm lookup for axis labels
    norm_lookup = {}
    if not norm_df.empty:
        norm_lookup = (
            norm_df.groupby(["model", "layer"])["norm_mean"].mean().to_dict()
        )

    fig, axes = plt.subplots(2, len(HIGHLIGHT_LAYERS), figsize=(16, 7), sharex=False, sharey=False)
    for row_idx, model_key in enumerate(["apertus", "llama"]):
        for col_idx, layer in enumerate(HIGHLIGHT_LAYERS):
            ax = axes[row_idx][col_idx]
            cur = pca_df[(pca_df["layer"] == layer) & (pca_df["model"] == model_key)]
            if cur.empty:
                ax.set_visible(False)
                continue
            sns.scatterplot(data=cur, x="pc1", y="pc2", hue="label",
                            palette=PALETTE, alpha=0.5, s=10, ax=ax, legend=False)
            v1 = cur["pc1_var"].iloc[0]
            v2 = cur["pc2_var"].iloc[0]
            mean_norm = norm_lookup.get((model_key, layer), float("nan"))
            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{MODEL_DISPLAY[model_key]}\nPC2 ({v2:.1%} var.)", fontsize=9)
            else:
                ax.set_ylabel(f"PC2 ({v2:.1%} var.)", fontsize=8)
            norm_str = f"  ·  mean ‖h‖₂ = {mean_norm:,.0f}" if not np.isnan(mean_norm) else ""
            ax.set_xlabel(f"PC1 ({v1:.1%} var.){norm_str}", fontsize=8, color=NOTE_COLOR)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, linestyle=":", alpha=0.35)

    scatter_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e05c5c",
               markersize=8, label="Hallucinated"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a90d9",
               markersize=8, label="Supported"),
    ]
    title_handle = mpatches.Patch(visible=False)
    legend = fig.legend(
        [title_handle] + scatter_handles, ["Token label:"] + [h.get_label() for h in scatter_handles],
        loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.01), fontsize=9.5,
        handlelength=1.2, handletextpad=0.5, columnspacing=1.5,
    )
    legend.get_texts()[0].set_fontweight("bold")
    fig.suptitle(
        "PCA of residual stream activations — hallucinated (red) vs. supported (blue) tokens\n"
        "Apertus-8B-Instruct-2509 (top) · Llama-3.1-8B-Instruct (bottom) · independent axes per panel",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig("activations_pca_scatter.png")


# ---------------------------------------------------------------------------
# 2.  layers.ipynb
# ---------------------------------------------------------------------------

def plot_layers():
    print("\n[layers] Generating plots ...")

    long_csv = DATA_DIR / "layers_runs_long.csv"
    loss_csv = DATA_DIR / "layers_loss.csv"

    if not long_csv.exists():
        print("  layers_runs_long.csv not found — run extract_data.py first.")
        return

    df_long = pd.read_csv(long_csv)

    has_loss = loss_csv.exists()
    df_loss = pd.read_csv(loss_csv) if has_loss else pd.DataFrame()

    def _plot_layer_metrics(df, metric_family, test_model):
        subset = df[
            (df["metric_family"] == metric_family) & (df["test_model"] == test_model)
        ].copy()
        if subset.empty:
            print(f"  No data for metric_family={metric_family}, test_model={test_model} — skipping.")
            return

        agg = (
            subset.groupby(["model", "layer", "metric"])
            .agg(mean=("value", "mean"), std=("value", "std"))
            .reset_index()
        )
        agg["std"] = agg["std"].fillna(0.0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)

        for ax, metric_name in zip(axes, ["auc", "f1"]):
            metric_df = agg[agg["metric"] == metric_name]
            for model in MODELS:
                line_df = metric_df[metric_df["model"] == model].sort_values("layer")
                if line_df.empty:
                    continue
                x = line_df["layer"].to_numpy()
                y = line_df["mean"].to_numpy()
                yerr = line_df["std"].to_numpy()
                ax.plot(x, y, marker="o", linewidth=2.2, label=model.capitalize(), color=COLORS[model])
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, color=COLORS[model])

            ax.set_title(metric_name.upper(), fontsize=12)
            ax.set_xlabel("Probe training layer (activation layer)")
            ax.set_xticks(LAYERS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            if metric_name == "auc":
                ax.set_ylabel("Score")

        fig.suptitle(
            f"Probe performance by training layer ({test_model} test set)",
            fontsize=13,
            y=1.06,
        )

        handles, labels = axes[0].get_legend_handles_labels()
        title_handle = mpatches.Patch(visible=False)
        legend = fig.legend(
            [title_handle] + handles,
            ["Training set:"] + labels,
            loc="upper center",
            ncol=len(labels) + 1,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=1.2,
            handlelength=2,
            handletextpad=0.4,
        )
        legend.get_texts()[0].set_fontweight("bold")

        fig.tight_layout(rect=(0, 0, 1, 1))
        savefig(f"layers_performance_{metric_family}_{test_model}.png")

    def _plot_auc_r01_loss(df, df_loss, metric_family, test_model):
        eval_subset = df[
            (df["metric_family"] == metric_family)
            & (df["test_model"] == test_model)
            & (df["metric"].isin(["auc", "recall_at_0.1_fpr"]))
        ].copy()
        if eval_subset.empty:
            print(f"  No eval data for {metric_family}/{test_model} — skipping AUC+R@0.1+Loss plot.")
            return

        auc_agg = (
            eval_subset[eval_subset["metric"] == "auc"]
            .groupby(["model", "layer"])["value"]
            .agg(["mean", "std"]).reset_index()
        )
        auc_agg["std"] = auc_agg["std"].fillna(0.0)

        r01_agg = (
            eval_subset[eval_subset["metric"] == "recall_at_0.1_fpr"]
            .groupby(["model", "layer"])["value"]
            .agg(["mean", "std"]).reset_index()
        )
        r01_agg["std"] = r01_agg["std"].fillna(0.0)

        ncols = 2 if df_loss.empty else 3
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.8), sharex=True)
        panels = [(axes[0], auc_agg, "AUC", "Score"),
                  (axes[1], r01_agg, "Recall at 0.1 False Positive Rate", "Score")]
        if not df_loss.empty:
            loss_agg = (
                df_loss.groupby(["model", "layer"])["final_loss"]
                .agg(["mean", "std"]).reset_index()
            )
            loss_agg["std"] = loss_agg["std"].fillna(0.0)
            panels.append((axes[2], loss_agg, "Final training loss", "Cross-entropy loss (log scale)"))

        MODEL_DISPLAY = {"apertus": "Apertus-8B-Instruct-2509", "llama": "Llama-3.1-8B-Instruct"}
        for ax, agg, title, ylabel in panels:
            for model in MODELS:
                sub = agg[agg["model"] == model].sort_values("layer")
                if sub.empty:
                    continue
                x = sub["layer"].to_numpy()
                y = sub["mean"].to_numpy()
                yerr = sub["std"].to_numpy()
                ax.plot(x, y, marker="o", linewidth=2.2, label=MODEL_DISPLAY[model], color=COLORS[model])
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, color=COLORS[model])
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Probe training layer (activation layer)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(LAYERS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            if title == "Final training loss":
                ax.set_yscale("log")

        MODEL_DISPLAY_NAMES = {"apertus": "Apertus-8B-Instruct-2509", "llama": "Llama-3.1-8B-Instruct"}
        fig.suptitle(
            f"Baseline probe performance and training instability across layers on "
            f"{MODEL_DISPLAY_NAMES.get(test_model, test_model)} test set",
            fontsize=13, y=1.06,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        title_handle = mpatches.Patch(visible=False)
        legend = fig.legend(
            [title_handle] + handles, ["Training set:"] + labels,
            loc="upper center", ncol=len(labels) + 1, frameon=False,
            bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=2, handletextpad=0.4,
        )
        legend.get_texts()[0].set_fontweight("bold")
        fig.tight_layout(rect=(0, 0, 1, 1))
        savefig(f"layers_auc_r01_loss_{metric_family}_{test_model}.png")

    # Replicate exactly what the notebook calls
    _plot_layer_metrics(df_long, "all", "apertus")
    _plot_layer_metrics(df_long, "all", "llama")

    _plot_auc_r01_loss(df_long, df_loss, "all",  "apertus")
    _plot_auc_r01_loss(df_long, df_loss, "all",  "llama")


# ---------------------------------------------------------------------------
# 3.  plots.ipynb
# ---------------------------------------------------------------------------

def plot_plots():
    print("\n[plots] Generating plots ...")

    # --- Cross-model heatmaps ---
    for config_label, title_suffix in [
        ("no_lora", "no LoRA"),
        ("lora",    r"LoRA with $\lambda=0.5$ KL reg"),
    ]:
        csv_path = DATA_DIR / f"plots_cross_model_{config_label}_all_f1.csv"
        if not csv_path.exists():
            print(f"  {csv_path.name} not found — skipping {config_label} heatmap.")
            continue

        matrix_df = pd.read_csv(csv_path)
        data = matrix_df[["train_apertus", "train_llama"]].to_numpy(dtype=float)

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            data,
            annot=True,
            fmt=".4f",
            cmap="Greens",
            vmin=0.5,
            xticklabels=["Apertus", "Llama"],
            yticklabels=["Apertus", "Llama"],
            cbar_kws={"label": "F1"},
            square=True,
        )
        plt.title(
            f"Cross-Model Linear Probe Performance (F1)\n"
            f"for hallucination detection task; longform dataset; {title_suffix}",
            fontsize=12,
            pad=20,
        )
        plt.xlabel("Source of Train Data", fontsize=12)
        plt.ylabel("Source of Test Data", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        savefig(f"plots_cross_model_f1_{config_label}.png")

    # --- Training loss curves (no-LoRA vs LoRA, EMA smoothed) ---
    def ema_smooth(values, weight=0.95):
        smoothed, last = [], values[0]
        for v in values:
            last = last * weight + v * (1 - weight)
            smoothed.append(last)
        return smoothed

    lora_colors = {"no LoRA": "#E07B39", "with LoRA": "#3B82C4"}

    histories = {}
    for run_key in ["apertus_no_lora", "llama_no_lora", "apertus_lora", "llama_lora"]:
        csv = DATA_DIR / f"plots_training_history_{run_key}.csv"
        if csv.exists():
            histories[run_key] = pd.read_csv(csv)
        else:
            print(f"  {csv.name} not found — loss curve for {run_key} will be blank.")

    if any(k in histories for k in ["apertus_no_lora", "apertus_lora",
                                     "llama_no_lora",   "llama_lora"]):
        fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharey=False)

        data_pairs = [
            ("apertus_no_lora", "apertus_lora", "Apertus"),
            ("llama_no_lora",   "llama_lora",   "Llama"),
        ]

        for ax, (base_key, lora_key, name) in zip(axs, data_pairs):
            for label, key in [("no LoRA", base_key), ("with LoRA", lora_key)]:
                if key not in histories:
                    continue
                raw = histories[key]["train/loss"].dropna().to_numpy()
                steps = range(len(raw))
                color = lora_colors[label]
                ax.plot(steps, raw, color=color, alpha=0.2, linewidth=0.8)
                ax.plot(
                    steps,
                    ema_smooth(raw.tolist()),
                    color=color,
                    linewidth=2.2,
                    label=label,
                    linestyle="--" if "LoRA" in label else "-",
                )
            ax.set_title(f"Probe trained on {name} data", fontsize=12, pad=6)
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel("Cross-entropy Loss", fontsize=10)
            ax.legend(loc="upper right", framealpha=0.85, fontsize=10)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            r"$\bf{Cross\text{-}entropy\ Probe\ Loss\ During\ Training}$"
            + "\n for hallucination detection task; longform dataset",
            fontsize=12,
        )
        fig.text(
            0.5, -0.02,
            "Bold lines show EMA-smoothed loss (α=0.95); faint lines show raw per-step loss.",
            ha="center", fontsize=9, style="italic", color="gray",
        )
        fig.tight_layout()
        savefig("plots_training_loss_curves.png")


# ---------------------------------------------------------------------------
# 4.  improve_stability.ipynb
# ---------------------------------------------------------------------------

def plot_stability():
    print("\n[stability] Generating plots ...")

    long_csv = DATA_DIR / "stability_runs_long.csv"
    runs_csv = DATA_DIR / "stability_runs.csv"
    if not long_csv.exists():
        print("  stability_runs_long.csv not found — run extract_data.py first.")
        return

    # lr values like "1e-3" are scientific-notation strings; force str dtype so
    # pandas doesn't parse them as floats (which would break reindex(LRS) matching).
    df_long = pd.read_csv(long_csv, dtype={"lr": str})
    df_runs = pd.read_csv(runs_csv, dtype={"lr": str}) if runs_csv.exists() else pd.DataFrame()

    STABILITY_LAYERS = [10, 20, 26, 30]
    LRS = ["1e-3", "3e-4", "1e-4", "3e-5"]
    BASELINE_VARIANTS = ["apertus_layers", "llama_layers"]
    PRECISION_VARIANTS = {"bf16": ["ln_bf16", "bf16_only"], "fp32": ["ln_fp32", "fp32_only"]}
    FP32_COLOR = "#2563EB"
    BF16_COLOR = "#D97706"
    LLAMA_COLOR = "#111827"
    STYLE_MAP = {
        "llama_layers": {"label": "Llama baseline", "color": LLAMA_COLOR, "linestyle": "-",  "linewidth": 3.0, "zorder": 4},
        "bf16_only":    {"label": "No layer norm",  "color": BF16_COLOR,  "linestyle": "--", "linewidth": 2.4, "zorder": 2},
        "ln_bf16":      {"label": "Layer norm",     "color": BF16_COLOR,  "linestyle": "-",  "linewidth": 2.8, "zorder": 2},
        "fp32_only":    {"label": "No layer norm",  "color": FP32_COLOR,  "linestyle": "--", "linewidth": 2.4, "zorder": 2},
        "ln_fp32":      {"label": "Layer norm",     "color": FP32_COLOR,  "linestyle": "-",  "linewidth": 2.8, "zorder": 2},
    }
    METRIC_LABELS = {"auc": "AUC", "f1": "F1", "acc": "Accuracy", "recall_at_0.1_fpr": "R@0.1"}

    def _lr_sweep(df, metric_family, metric, test_model, precision, include_baseline=True):
        selected = PRECISION_VARIANTS[precision]
        sub = df[(df["metric_family"] == metric_family) & (df["metric"] == metric)
                 & (df["test_model"] == test_model)].copy()
        if sub.empty:
            return
        metric_label = METRIC_LABELS.get(metric, metric)
        main = sub[sub["variant"].isin(selected)]
        baselines = sub[sub["variant"].isin(BASELINE_VARIANTS)]

        main_agg = (main.groupby(["variant", "layer", "lr"], observed=True)
                    .agg(mean=("value", "mean"), std=("value", "std")).reset_index())
        main_agg["std"] = main_agg["std"].fillna(0.0)
        base_agg = (baselines.groupby(["variant", "layer"], observed=True)
                    .agg(mean=("value", "mean"), std=("value", "std")).reset_index())
        base_agg["std"] = base_agg["std"].fillna(0.0)

        xpos = list(range(len(LRS)))
        fig, axes = plt.subplots(1, len(STABILITY_LAYERS),
                                 figsize=(4.2 * len(STABILITY_LAYERS), 4.2),
                                 sharex=True, sharey=True)
        for col_idx, layer in enumerate(STABILITY_LAYERS):
            ax = axes[col_idx]
            d_main = main_agg[main_agg["layer"] == layer]
            d_base = base_agg[base_agg["layer"] == layer]
            for variant in selected:
                style = STYLE_MAP[variant]
                s = d_main[d_main["variant"] == variant].set_index("lr").reindex(LRS).reset_index()
                y = s["mean"].to_numpy()
                e = s["std"].to_numpy()
                if np.isnan(y).all():
                    continue
                ax.plot(xpos, y, color=style["color"], linestyle=style["linestyle"],
                        linewidth=style["linewidth"], marker="o", markersize=4.5, zorder=style["zorder"])
                ax.fill_between(xpos, y - e, y + e, color=style["color"], alpha=0.10, zorder=1)
            if include_baseline:
                b_llama = d_base[d_base["variant"] == "llama_layers"]
                if not b_llama.empty:
                    y0 = float(b_llama["mean"].iloc[0])
                    e0 = float(b_llama["std"].iloc[0])
                    st = STYLE_MAP["llama_layers"]
                    ax.plot(xpos, [y0] * len(LRS), color=st["color"],
                            linestyle=st["linestyle"], linewidth=st["linewidth"], zorder=st["zorder"])
                    ax.fill_between(xpos, [y0 - e0] * len(LRS), [y0 + e0] * len(LRS),
                                    color=st["color"], alpha=0.05)
            ax.set_title(f"Layer {layer}")
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            ax.set_xlabel("Learning rate")
            ax.set_xticks(xpos)
            ax.set_xticklabels(LRS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)

        prec_color = STYLE_MAP[selected[0]]["color"]
        legend_handles = [
            Line2D([0], [0], color=prec_color, linestyle="-",  linewidth=3.0, label="Layer norm (solid)"),
            Line2D([0], [0], color=prec_color, linestyle="--", linewidth=3.0, label="No layer norm (dashed)"),
        ]
        if include_baseline:
            legend_handles.append(Line2D([0], [0], color=LLAMA_COLOR, linestyle="-",
                                         linewidth=3.0, label="Llama baseline"))
        fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles),
                   frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(f"{precision.upper()} | test={test_model}, family={metric_family}, metric={metric_label}", y=1.10)
        fig.tight_layout()
        savefig(f"stability_lr_{precision}_{metric}_{test_model}.png")

    def _fp32_bf16_comparison(df, metric_family, metric, test_model):
        sub = df[(df["metric_family"] == metric_family) & (df["metric"] == metric)
                 & (df["test_model"] == test_model)
                 & (df["variant"].isin(["ln_fp32", "fp32_only", "ln_bf16", "bf16_only"]))].copy()
        if sub.empty:
            return
        metric_label = METRIC_LABELS.get(metric, metric)
        sub["norm_flag"] = sub["variant"].map(lambda v: "ln" if str(v).startswith("ln_") else "none")
        sub["dtype_flag"] = sub["probe_dtype"].map(
            lambda d: "fp32" if d == "float32" else ("bf16" if d == "bfloat16" else "other"))
        sub = sub[sub["dtype_flag"].isin(["fp32", "bf16"])]
        pair_keys = ["layer", "seed", "lr", "test_model", "metric_family", "metric", "norm_flag"]
        pair_pivot = (sub.groupby(pair_keys + ["dtype_flag"], dropna=False, observed=True)["value"]
                      .mean().unstack("dtype_flag").reset_index())
        if not {"fp32", "bf16"}.issubset(set(pair_pivot.columns)):
            print(f"  Skipping FP32/BF16 comparison for {metric}/{test_model} — no matched pairs.")
            return
        perf_pairs = pair_pivot.dropna(subset=["fp32", "bf16"]).copy()
        perf_pairs["delta"] = perf_pairs["fp32"] - perf_pairs["bf16"]
        perf_plot = (perf_pairs.groupby(["layer", "lr", "norm_flag"], dropna=False, observed=True)
                     .agg(mean_delta=("delta", "mean"), std_delta=("delta", "std")).reset_index())
        perf_plot["std_delta"] = perf_plot["std_delta"].fillna(0.0)

        layer_colors = {layer: plt.cm.tab10(i % 10) for i, layer in enumerate(STABILITY_LAYERS)}
        xpos = list(range(len(LRS)))
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True, sharey=True)
        for col_idx, (norm_flag, panel_title) in enumerate([("ln", "Layer norm"), ("none", "No layer norm")]):
            ax = axes[col_idx]
            d = perf_plot[perf_plot["norm_flag"] == norm_flag]
            for layer in STABILITY_LAYERS:
                s = d[d["layer"] == layer].set_index("lr").reindex(LRS).reset_index()
                y = s["mean_delta"].to_numpy()
                e = s["std_delta"].to_numpy()
                if np.isnan(y).all():
                    continue
                ax.plot(xpos, y, marker="o", linestyle="-", linewidth=2.0,
                        color=layer_colors[layer], label=f"Layer {layer}")
                ax.fill_between(xpos, y - e, y + e, color=layer_colors[layer], alpha=0.08)
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
            ax.set_title(panel_title)
            if col_idx == 0:
                ax.set_ylabel("fp32 − bf16 (+ = fp32 better)")
            ax.set_xlabel("Learning rate")
            ax.set_xticks(xpos)
            ax.set_xticklabels(LRS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
        handles, lbls = axes[0].get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, lbls) if not (l in seen or seen.add(l))]
        fig.legend([h for h, _ in uniq], [l for _, l in uniq],
                   loc="upper center", ncol=len(STABILITY_LAYERS), frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"FP32 vs BF16 delta | test={test_model}, family={metric_family}, metric={metric_label}", y=1.10)
        fig.tight_layout()
        savefig(f"stability_fp32_bf16_{metric}_{test_model}.png")

    def _precision_impact(df, df_runs_wide):
        color_by_precision = {"float32": "#2563EB", "bfloat16": "#D97706"}
        marker_by_precision = {"float32": "o", "bfloat16": "s"}
        precision_label = {"float32": "Probe dtype float32", "bfloat16": "Probe dtype bfloat16"}
        selected_variants = ["fp32_only", "bf16_only"]
        test_model = "apertus"
        metric_family = "all"

        metric_subset = df[
            (df["variant"].isin(selected_variants))
            & (df["metric_family"] == metric_family)
            & (df["test_model"] == test_model)
            & (df["metric"].isin(["auc", "recall_at_0.1_fpr"]))
            & (df["probe_dtype"].isin(["float32", "bfloat16"]))
            & (df["lr"] == "1e-3")
        ].copy()
        if metric_subset.empty:
            print("  _precision_impact: no matching rows — skipping.")
            return

        loss_subset = df_runs_wide[
            df_runs_wide["variant"].isin(selected_variants)
            & (df_runs_wide["lr"] == "1e-3")
            & df_runs_wide["probe_dtype"].isin(["float32", "bfloat16"])
        ][["probe_dtype", "layer", "final_loss"]].copy().dropna(subset=["final_loss"])

        agg_metrics = (
            metric_subset.groupby(["metric", "probe_dtype", "layer"], observed=True)
            .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
        )
        agg_metrics["std"] = agg_metrics["std"].fillna(0.0)

        agg_loss = pd.DataFrame()
        if not loss_subset.empty:
            agg_loss = (
                loss_subset.groupby(["probe_dtype", "layer"], observed=True)
                .agg(mean=("final_loss", "mean"), std=("final_loss", "std")).reset_index()
            )
            agg_loss["std"] = agg_loss["std"].fillna(0.0)

        fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharex=True)
        panel_specs = [
            ("auc", "AUC", "Score"),
            ("recall_at_0.1_fpr", "R@0.1", "Score"),
            ("final_loss", "Final training loss", "Cross-entropy loss (log scale)"),
        ]
        for ax, (metric_key, panel_title, y_label) in zip(axes, panel_specs):
            for probe_dtype in ["float32", "bfloat16"]:
                if metric_key == "final_loss":
                    if agg_loss.empty:
                        continue
                    s = agg_loss[agg_loss["probe_dtype"] == probe_dtype].set_index("layer").reindex(STABILITY_LAYERS).reset_index()
                else:
                    s = (agg_metrics[(agg_metrics["metric"] == metric_key) & (agg_metrics["probe_dtype"] == probe_dtype)]
                         .set_index("layer").reindex(STABILITY_LAYERS).reset_index())
                y = s["mean"].to_numpy()
                e = s["std"].to_numpy()
                if np.isnan(y).all():
                    continue
                ax.plot(STABILITY_LAYERS, y, color=color_by_precision[probe_dtype],
                        marker=marker_by_precision[probe_dtype], markersize=6.2, linewidth=2.4)
                ax.fill_between(STABILITY_LAYERS, y - e, y + e, color=color_by_precision[probe_dtype],
                                alpha=0.18, edgecolor=color_by_precision[probe_dtype], linewidth=0.8)
            ax.set_title(panel_title, fontsize=12)
            ax.set_xlabel("Probe training layer (activation layer)")
            ax.set_ylabel(y_label)
            ax.set_xticks(STABILITY_LAYERS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            if metric_key == "final_loss":
                ax.set_yscale("log")

        train_name = "Apertus-8B-Instruct-2509"
        test_name = {"apertus": "Apertus-8B-Instruct-2509", "llama": "Llama-3.1-8B-Instruct"}.get(test_model, test_model)
        fig.suptitle(
            "Precision Impact on Probe Performance and Stability Across Layers\n"
            f"Training activations: {train_name} | Evaluation set: {test_name}",
            y=0.98, fontsize=15,
        )
        legend_handles = [
            Line2D([0], [0], color=color_by_precision[d], marker=marker_by_precision[d],
                   linewidth=2.4, markersize=7, label=f"{precision_label[d]} ({train_name})")
            for d in ["float32", "bfloat16"]
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.90), fontsize=10)
        fig.text(0.5, 0.01,
                 "Aggregation: mean ± std across seeds (lr=1e-3, no layernorm) within each probe dtype.",
                 ha="center", fontsize=9, color="#6B7280", style="italic")
        fig.tight_layout(rect=[0, 0.04, 1, 0.87])
        savefig("stability_precision_impact_apertus.png")

    def _lr_impact(df, df_runs_wide):
        color_by_lr = {"1e-3": "#059669", "3e-4": "#3B82F6", "1e-4": "#F59E0B", "3e-5": "#EF4444"}
        marker_by_lr = {"1e-3": "o", "3e-4": "s", "1e-4": "^", "3e-5": "D"}
        lr_label = {"1e-3": "Learning rate 1e-3", "3e-4": "Learning rate 3e-4",
                    "1e-4": "Learning rate 1e-4", "3e-5": "Learning rate 3e-5"}
        selected_variants = ["bf16_only"]
        test_model = "apertus"
        metric_family = "all"

        metric_subset = df[
            (df["variant"].isin(selected_variants))
            & (df["metric_family"] == metric_family)
            & (df["test_model"] == test_model)
            & (df["metric"].isin(["auc", "recall_at_0.1_fpr"]))
            & (df["probe_dtype"] == "bfloat16")
            & (df["lr"].notna())
        ].copy()
        if metric_subset.empty:
            print("  _lr_impact: no matching rows — skipping.")
            return

        loss_subset = df_runs_wide[
            df_runs_wide["variant"].isin(selected_variants)
            & (df_runs_wide["probe_dtype"] == "bfloat16")
            & df_runs_wide["lr"].notna()
        ][["lr", "layer", "final_loss"]].copy().dropna(subset=["final_loss"])

        agg_metrics = (
            metric_subset.groupby(["metric", "lr", "layer"], observed=True)
            .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
        )
        agg_metrics["std"] = agg_metrics["std"].fillna(0.0)

        agg_loss = pd.DataFrame()
        if not loss_subset.empty:
            agg_loss = (
                loss_subset.groupby(["lr", "layer"], observed=True)
                .agg(mean=("final_loss", "mean"), std=("final_loss", "std")).reset_index()
            )
            agg_loss["std"] = agg_loss["std"].fillna(0.0)

        fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharex=True)
        panel_specs = [
            ("auc", "AUC", "Score"),
            ("recall_at_0.1_fpr", "R@0.1", "Score"),
            ("final_loss", "Final training loss", "Cross-entropy loss (log scale)"),
        ]
        for ax, (metric_key, panel_title, y_label) in zip(axes, panel_specs):
            for lr in LRS:
                if metric_key == "final_loss":
                    if agg_loss.empty:
                        continue
                    s = agg_loss[agg_loss["lr"] == lr].set_index("layer").reindex(STABILITY_LAYERS).reset_index()
                else:
                    s = (agg_metrics[(agg_metrics["metric"] == metric_key) & (agg_metrics["lr"] == lr)]
                         .set_index("layer").reindex(STABILITY_LAYERS).reset_index())
                y = s["mean"].to_numpy()
                e = s["std"].to_numpy()
                if np.isnan(y).all():
                    continue
                ax.plot(STABILITY_LAYERS, y, color=color_by_lr[lr], marker=marker_by_lr[lr],
                        markersize=6.2, linewidth=2.4)
                ax.fill_between(STABILITY_LAYERS, y - e, y + e, color=color_by_lr[lr],
                                alpha=0.18, edgecolor=color_by_lr[lr], linewidth=0.8)
            ax.set_title(panel_title, fontsize=12)
            ax.set_xlabel("Probe training layer (activation layer)")
            ax.set_ylabel(y_label)
            ax.set_xticks(STABILITY_LAYERS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            if metric_key == "final_loss":
                ax.set_yscale("log")

        train_name = "Apertus-8B-Instruct-2509"
        test_name = {"apertus": "Apertus-8B-Instruct-2509", "llama": "Llama-3.1-8B-Instruct"}.get(test_model, test_model)
        fig.suptitle(
            "Learning Rate Impact on Probe Performance and Stability Across Layers\n"
            f"Training activations: {train_name} (bfloat16) | Evaluation set: {test_name}",
            y=0.98, fontsize=15,
        )
        legend_handles = [
            Line2D([0], [0], color=color_by_lr[lr], marker=marker_by_lr[lr],
                   linewidth=2.4, markersize=7, label=lr_label[lr])
            for lr in LRS
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.90), fontsize=10)
        fig.text(0.5, 0.01,
                 "Aggregation: mean ± std across seeds (no layernorm, bfloat16) for each learning rate.",
                 ha="center", fontsize=9, color="#6B7280", style="italic")
        fig.tight_layout(rect=[0, 0.04, 1, 0.87])
        savefig("stability_lr_impact_apertus.png")

    # Mirror the notebook's actual calls
    _lr_sweep(df_long, "all", "recall_at_0.1_fpr", "apertus", "bf16")
    _lr_sweep(df_long, "all", "recall_at_0.1_fpr", "apertus", "fp32")
    _lr_sweep(df_long, "all", "auc",               "apertus", "bf16")
    _lr_sweep(df_long, "all", "auc",               "apertus", "fp32")
    _lr_sweep(df_long, "all", "auc",               "llama",   "fp32")
    _fp32_bf16_comparison(df_long, "all", "recall_at_0.1_fpr", "apertus")
    _precision_impact(df_long, df_runs)
    _lr_impact(df_long, df_runs)


# ---------------------------------------------------------------------------
# 5.  final_ablation.ipynb
# ---------------------------------------------------------------------------

def plot_final_ablation():
    print("\n[final_ablation] Generating plots ...")

    long_csv = DATA_DIR / "final_ablation_runs_long.csv"
    runs_csv = DATA_DIR / "final_ablation_runs.csv"
    if not long_csv.exists():
        print("  final_ablation_runs_long.csv not found — run extract_data.py first.")
        return

    df_metrics_long = pd.read_csv(long_csv)
    df_runs = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()

    ABL_LAYERS = [10, 20, 26, 30]
    TRAIN_MODELS = ["apertus", "llama"]
    COLOR_BY_MODEL = {"apertus": "#D97706", "llama": "#2563EB"}
    LINESTYLE_BY_NORM = {"none": "-", "layernorm": "--"}
    MARKER_BY_DTYPE = {"fp32": "o", "bf16": "s"}
    MODEL_DISPLAY = {"apertus": "Apertus-8B-Instruct-2509", "llama": "Llama-3.1-8B-Instruct"}
    METRIC_OPTIONS = {
        "r_at_0.1": {"kind": "metric", "metric": "recall_at_0.1_fpr", "label": "Recall at 0.1 FPR (R@0.1)"},
        "auc": {"kind": "metric", "metric": "auc", "label": "AUC"},
        "f1": {"kind": "metric", "metric": "f1", "label": "F1"},
        "final_loss": {"kind": "loss", "loss_key": "train/loss", "label": "Final training loss"},
    }

    def _variant_norm(variant):
        variant = str(variant)
        if variant.endswith("no_ln"):
            return "none"
        if variant.endswith("_ln"):
            return "layernorm"
        return "none"

    def _variant_lora_group(variant):
        return "lora" if str(variant).startswith("lora_") else "no_lora"

    def _build_legend_handles():
        return [
            Line2D([0], [0], color=COLOR_BY_MODEL["apertus"], linewidth=2.4,
                   label="Orange = Apertus-8B-Instruct-2509 runs"),
            Line2D([0], [0], color=COLOR_BY_MODEL["llama"], linewidth=2.4,
                   label="Blue = Llama-3.1-8B-Instruct runs"),
            Line2D([0], [0], color="#475569", marker=MARKER_BY_DTYPE["fp32"], linewidth=0,
                   markersize=7, label="Circle marker = probe dtype float32"),
            Line2D([0], [0], color="#475569", marker=MARKER_BY_DTYPE["bf16"], linewidth=0,
                   markersize=7, label="Square marker = probe dtype bfloat16"),
            Line2D([0], [0], color="#475569", linestyle="-", linewidth=2.2,
                   label="Solid line = no pre-head normalization"),
            Line2D([0], [0], color="#475569", linestyle="--", linewidth=2.2,
                   label="Dashed line = layernorm before probe head"),
            Line2D([0], [0], color=COLOR_BY_MODEL["apertus"], linestyle=":", linewidth=2.8,
                   label="Apertus baseline = bfloat16, no layernorm, no LoRA"),
            Line2D([0], [0], color=COLOR_BY_MODEL["llama"], linestyle=":", linewidth=2.8,
                   label="Llama baseline = bfloat16, no layernorm, no LoRA"),
        ]

    def _lora_vs_no_lora_by_layer(metric_option, train_model, test_model=None,
                                   metric_family="all", log_scale=False):
        spec = METRIC_OPTIONS[metric_option]
        train_display = MODEL_DISPLAY.get(train_model, train_model)
        test_display = MODEL_DISPLAY.get(test_model, test_model) if test_model else None

        if spec["kind"] == "metric":
            data = df_metrics_long[
                (df_metrics_long["run_kind"] == "final_ablation")
                & (df_metrics_long["train_model"] == train_model)
                & (df_metrics_long["metric"] == spec["metric"])
                & (df_metrics_long["metric_family"] == metric_family)
                & (df_metrics_long["test_model"] == test_model)
            ].copy()
        else:
            loss_key = spec["loss_key"]
            data = df_runs[
                (df_runs["run_kind"] == "final_ablation")
                & (df_runs["train_model"] == train_model)
            ][["variant", "probe_dtype_tag", "layer", loss_key]].copy()
            data = data.rename(columns={loss_key: "value"}).dropna(subset=["value"])

        if data.empty:
            print(f"  No data for metric_option={metric_option!r}, train={train_model} — skipping.")
            return

        data["norm_kind"] = data["variant"].map(_variant_norm)
        data["lora_group"] = data["variant"].map(_variant_lora_group)

        agg = (
            data.groupby(["lora_group", "variant", "probe_dtype_tag", "norm_kind", "layer"], observed=False)
            .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
        )
        agg["std"] = agg["std"].fillna(0.0)

        fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.9), sharex=True, sharey=True)
        for ax, (panel_key, panel_title) in zip(axes, [("no_lora", "No LoRA Adapters"),
                                                        ("lora", "LoRA Adapters on All Layers")]):
            panel = agg[agg["lora_group"] == panel_key]
            if panel.empty:
                ax.text(0.5, 0.5, "No runs available", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(panel_title)
                ax.set_xticks(ABL_LAYERS)
                ax.set_xlabel("Layer")
                ax.grid(True, alpha=0.25, linestyle=":")
                ax.spines[["top", "right"]].set_visible(False)
                if log_scale:
                    ax.set_yscale("log")
                continue

            for (variant, dtype_tag, norm_kind), sub in panel.groupby(
                    ["variant", "probe_dtype_tag", "norm_kind"], observed=False):
                s = sub.set_index("layer").reindex(ABL_LAYERS).reset_index()
                y = s["mean"].to_numpy()
                e = s["std"].to_numpy()
                if np.isnan(y).all():
                    continue
                ax.plot(ABL_LAYERS, y, color=COLOR_BY_MODEL[train_model],
                        linestyle=LINESTYLE_BY_NORM.get(norm_kind, "-"),
                        marker=MARKER_BY_DTYPE.get(dtype_tag, "o"),
                        markersize=6.5, linewidth=2.2)
                ax.fill_between(ABL_LAYERS, y - e, y + e, color=COLOR_BY_MODEL[train_model], alpha=0.10)

            for baseline_model in TRAIN_MODELS:
                base = df_runs[(df_runs["run_kind"] == "baseline")
                               & (df_runs["train_model"] == baseline_model)].copy()
                if base.empty:
                    continue
                if spec["kind"] == "metric":
                    col = f"train/longfact_test_{test_model}/{metric_family}_{spec['metric']}"
                    base = base[["layer", col]].rename(columns={col: "value"}).dropna(subset=["value"])
                else:
                    base = base[["layer", spec["loss_key"]]].rename(
                        columns={spec["loss_key"]: "value"}).dropna(subset=["value"])
                if base.empty:
                    continue
                b_agg = base.groupby("layer", observed=True).agg(
                    mean=("value", "mean"), std=("value", "std")).reset_index()
                b_agg["std"] = b_agg["std"].fillna(0.0)
                b = b_agg.set_index("layer").reindex(ABL_LAYERS).reset_index()
                by = b["mean"].to_numpy()
                be = b["std"].to_numpy()
                ax.plot(ABL_LAYERS, by, color=COLOR_BY_MODEL[baseline_model], linestyle=":", linewidth=2.8)
                ax.fill_between(ABL_LAYERS, by - be, by + be,
                                color=COLOR_BY_MODEL[baseline_model], alpha=0.06)

            ax.set_title(panel_title)
            ax.set_xticks(ABL_LAYERS)
            ax.set_xlabel("Layer")
            ax.set_ylabel(spec["label"])
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            if log_scale:
                ax.set_yscale("log")

        if spec["kind"] == "metric":
            subtitle = (f"Ablation runs trained on {train_display} | Evaluated on {test_display} | "
                        f"Metric family: {metric_family} | Metric: {spec['label']}")
        else:
            subtitle = f"Ablation runs trained on {train_display} | Metric: {spec['label']}"
        fig.suptitle(f"Final Probe Ablation Comparison Across Transformer Layers\n{subtitle}",
                     y=0.98, fontsize=16)
        fig.legend(handles=_build_legend_handles(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.89), ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.82])
        if spec["kind"] == "metric":
            savefig(f"final_ablation_lora_{metric_option}_{train_model}_{test_model}.png")
        else:
            savefig(f"final_ablation_lora_{metric_option}_{train_model}.png")

    def _baselines_vs_full_solution(test_model, metric_family="all"):
        color_by_model = {"llama": "#2563EB", "apertus": "#D97706"}
        linestyle_by_group = {"baseline": "-", "full_solution": "--"}
        marker_by_group = {"baseline": "o", "full_solution": "s"}

        full_solution = df_metrics_long[
            (df_metrics_long["run_kind"] == "final_ablation")
            & (df_metrics_long["variant"] == "lora_ln")
            & (df_metrics_long["probe_dtype_tag"] == "fp32")
            & (df_metrics_long["metric_family"] == metric_family)
            & (df_metrics_long["test_model"] == test_model)
            & (df_metrics_long["metric"].isin(["auc", "recall_at_0.1_fpr"]))
        ][["metric", "train_model", "layer", "seed", "value"]].copy()
        full_solution["group"] = "full_solution"

        baselines = df_metrics_long[
            (df_metrics_long["run_kind"] == "baseline")
            & (df_metrics_long["metric_family"] == metric_family)
            & (df_metrics_long["test_model"] == test_model)
            & (df_metrics_long["metric"].isin(["auc", "recall_at_0.1_fpr"]))
            & (df_metrics_long["train_model"].isin(["apertus", "llama"]))
        ][["metric", "train_model", "layer", "seed", "value"]].copy()
        baselines["group"] = "baseline"

        metric_subset = pd.concat([baselines, full_solution], ignore_index=True)
        if metric_subset.empty:
            print(f"  _baselines_vs_full_solution: no data for test_model={test_model} — skipping.")
            return

        agg_metrics = (
            metric_subset.groupby(["metric", "train_model", "group", "layer"], observed=False)
            .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
        )
        agg_metrics["std"] = agg_metrics["std"].fillna(0.0)

        loss_full = df_runs[
            (df_runs["run_kind"] == "final_ablation")
            & (df_runs["variant"] == "lora_ln")
            & (df_runs["probe_dtype_tag"] == "fp32")
        ][["train_model", "layer", "seed", "train/loss"]].rename(columns={"train/loss": "value"}).copy()
        loss_full["group"] = "full_solution"

        loss_base = df_runs[
            (df_runs["run_kind"] == "baseline")
            & (df_runs["train_model"].isin(["apertus", "llama"]))
        ][["train_model", "layer", "seed", "train/loss"]].rename(columns={"train/loss": "value"}).copy()
        loss_base["group"] = "baseline"

        loss_subset = pd.concat([loss_base, loss_full], ignore_index=True).dropna(subset=["value"])
        agg_loss = (
            loss_subset.groupby(["train_model", "group", "layer"], observed=False)
            .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
        )
        agg_loss["std"] = agg_loss["std"].fillna(0.0)

        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), sharex=True)
        plot_order = [("llama", "baseline"), ("apertus", "baseline"),
                      ("llama", "full_solution"), ("apertus", "full_solution")]
        for ax, (metric_key, panel_title, y_label) in zip(axes, [
            ("auc", "AUC", "Score"),
            ("recall_at_0.1_fpr", "R@0.1", "Score"),
            ("final_loss", "Final training loss", "Cross-entropy loss (log scale)"),
        ]):
            for train_model, group in plot_order:
                if metric_key == "final_loss":
                    s = (agg_loss[(agg_loss["train_model"] == train_model) & (agg_loss["group"] == group)]
                         .set_index("layer").reindex(ABL_LAYERS).reset_index())
                else:
                    s = (agg_metrics[(agg_metrics["metric"] == metric_key)
                                     & (agg_metrics["train_model"] == train_model)
                                     & (agg_metrics["group"] == group)]
                         .set_index("layer").reindex(ABL_LAYERS).reset_index())
                y = s["mean"].to_numpy()
                e = s["std"].to_numpy()
                if np.isnan(y).all():
                    continue
                ax.plot(ABL_LAYERS, y, color=color_by_model[train_model],
                        linestyle=linestyle_by_group[group], marker=marker_by_group[group],
                        markersize=6.3, linewidth=2.5)
                ax.fill_between(ABL_LAYERS, y - e, y + e, color=color_by_model[train_model],
                                alpha=0.14, edgecolor=color_by_model[train_model], linewidth=0.6)
            ax.set_title(panel_title, fontsize=12)
            ax.set_xlabel("Probe training layer (activation layer)")
            ax.set_ylabel(y_label)
            ax.set_xticks(ABL_LAYERS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            if metric_key == "final_loss":
                ax.set_yscale("log")

        test_display = MODEL_DISPLAY.get(test_model, test_model)
        fig.suptitle(
            "Baselines vs Full-Solution Probe Configuration Across Layers\n"
            f"Test set: {test_display} | Full-solution = layernorm + LoRA + fp32 + lr=3e-4",
            y=0.98, fontsize=15,
        )
        legend_handles = [
            Line2D([0], [0], color=color_by_model["llama"], linestyle="-", marker="o",
                   linewidth=2.5, markersize=7, label="Llama baseline (no layernorm, bfloat16)"),
            Line2D([0], [0], color=color_by_model["apertus"], linestyle="-", marker="o",
                   linewidth=2.5, markersize=7, label="Apertus baseline (no layernorm, bfloat16)"),
            Line2D([0], [0], color=color_by_model["llama"], linestyle="--", marker="s",
                   linewidth=2.5, markersize=7, label="Llama full-solution (layernorm + LoRA + fp32 + lr=3e-4)"),
            Line2D([0], [0], color=color_by_model["apertus"], linestyle="--", marker="s",
                   linewidth=2.5, markersize=7, label="Apertus full-solution (layernorm + LoRA + fp32 + lr=3e-4)"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.90), fontsize=10)
        fig.text(0.5, 0.01,
                 "Aggregation: mean ± std across seeds. Baselines are no layernorm + bfloat16; "
                 "full-solution combines layernorm + LoRA + fp32 + lr=3e-4.",
                 ha="center", fontsize=9, color="#6B7280", style="italic")
        fig.tight_layout(rect=[0, 0.04, 1, 0.86])
        savefig(f"final_ablation_baselines_vs_full_{test_model}.png")

    def _apertus_improvement(test_model, metric_family="all"):
        color_apertus = "#D97706"
        linestyle_by_group = {"baseline": "-", "full_solution": "--"}
        marker_by_group = {"baseline": "o", "full_solution": "s"}

        full_solution = df_metrics_long[
            (df_metrics_long["run_kind"] == "final_ablation")
            & (df_metrics_long["train_model"] == "apertus")
            & (df_metrics_long["variant"] == "lora_ln")
            & (df_metrics_long["probe_dtype_tag"] == "fp32")
            & (df_metrics_long["metric_family"] == metric_family)
            & (df_metrics_long["test_model"] == test_model)
            & (df_metrics_long["metric"] == "auc")
        ][["metric", "layer", "seed", "value"]].copy()
        full_solution["group"] = "full_solution"

        baseline = df_metrics_long[
            (df_metrics_long["run_kind"] == "baseline")
            & (df_metrics_long["train_model"] == "apertus")
            & (df_metrics_long["metric_family"] == metric_family)
            & (df_metrics_long["test_model"] == test_model)
            & (df_metrics_long["metric"] == "auc")
        ][["metric", "layer", "seed", "value"]].copy()
        baseline["group"] = "baseline"

        metric_subset = pd.concat([baseline, full_solution], ignore_index=True)
        if metric_subset.empty:
            print(f"  _apertus_improvement: no data for test_model={test_model} — skipping.")
            return

        agg_auc = (
            metric_subset.groupby(["group", "layer"], observed=False)
            .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
        )
        agg_auc["std"] = agg_auc["std"].fillna(0.0)

        loss_full = df_runs[
            (df_runs["run_kind"] == "final_ablation")
            & (df_runs["train_model"] == "apertus")
            & (df_runs["variant"] == "lora_ln")
            & (df_runs["probe_dtype_tag"] == "fp32")
        ][["layer", "seed", "train/loss"]].rename(columns={"train/loss": "value"}).copy()
        loss_full["group"] = "full_solution"

        loss_base = df_runs[
            (df_runs["run_kind"] == "baseline")
            & (df_runs["train_model"] == "apertus")
        ][["layer", "seed", "train/loss"]].rename(columns={"train/loss": "value"}).copy()
        loss_base["group"] = "baseline"

        loss_subset = pd.concat([loss_base, loss_full], ignore_index=True).dropna(subset=["value"])
        agg_loss = (
            loss_subset.groupby(["group", "layer"], observed=False)
            .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
        )
        agg_loss["std"] = agg_loss["std"].fillna(0.0)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), sharex=True)

        ax_auc = axes[0]
        for group in ["baseline", "full_solution"]:
            s = agg_auc[agg_auc["group"] == group].set_index("layer").reindex(ABL_LAYERS).reset_index()
            y = s["mean"].to_numpy()
            e = s["std"].to_numpy()
            if np.isnan(y).all():
                continue
            plot_kwargs = dict(color=color_apertus, linestyle=linestyle_by_group[group], linewidth=2.5)
            if group == "baseline":
                plot_kwargs.update(marker=marker_by_group[group], markersize=6.5)
            ax_auc.plot(ABL_LAYERS, y, **plot_kwargs)
            ax_auc.fill_between(ABL_LAYERS, y - e, y + e, color=color_apertus, alpha=0.14)
            for x, yv in zip(ABL_LAYERS, y):
                if not np.isnan(yv):
                    dy = 0.024 if group == "full_solution" else -0.030
                    ax_auc.text(x, yv + dy, f"{yv:.3f}", ha="center", va="center", fontsize=11)

        auc_base = agg_auc[agg_auc["group"] == "baseline"].set_index("layer").reindex(ABL_LAYERS)["mean"]
        auc_full = agg_auc[agg_auc["group"] == "full_solution"].set_index("layer").reindex(ABL_LAYERS)["mean"]
        for x in ABL_LAYERS:
            b = auc_base.get(x, float("nan"))
            f = auc_full.get(x, float("nan"))
            if pd.notna(b) and pd.notna(f) and f > b:
                ax_auc.annotate("", xy=(x, f), xytext=(x, b),
                                arrowprops=dict(arrowstyle="-|>", color="#16A34A", lw=2.3, mutation_scale=20))
                ax_auc.scatter([x], [f], marker="*", s=115, color="#16A34A", zorder=5)

        ax_auc.set_title("AUC", fontsize=12, pad=14)
        ax_auc.set_xlabel("Probe training layer")
        ax_auc.set_ylabel("AUC")
        ax_auc.set_xticks(ABL_LAYERS)
        ax_auc.grid(True, alpha=0.25, linestyle=":")
        ax_auc.spines[["top", "right"]].set_visible(False)

        ax_loss = axes[1]
        for group in ["baseline", "full_solution"]:
            s = agg_loss[agg_loss["group"] == group].set_index("layer").reindex(ABL_LAYERS).reset_index()
            y = s["mean"].to_numpy()
            e = s["std"].to_numpy()
            if np.isnan(y).all():
                continue
            ax_loss.plot(ABL_LAYERS, y, color=color_apertus, linestyle=linestyle_by_group[group],
                         marker=marker_by_group[group], markersize=6.5, linewidth=2.5)
            ax_loss.fill_between(ABL_LAYERS, y - e, y + e, color=color_apertus, alpha=0.14)

        ax_loss.set_title("Final training loss", fontsize=12, pad=14)
        ax_loss.set_xlabel("Probe training layer")
        ax_loss.set_ylabel("Cross-entropy loss (log scale)")
        ax_loss.set_xticks(ABL_LAYERS)
        ax_loss.set_yscale("log")
        ax_loss.grid(True, alpha=0.25, linestyle=":")
        ax_loss.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            "Hallucination probe performance on Apertus-8B-Instruct-2509: baseline vs after improvements",
            y=0.98, fontsize=14,
        )
        legend_handles = [
            Line2D([0], [0], color=color_apertus, linestyle="-", marker="o", linewidth=2.5, markersize=7,
                   label="Baseline (bfloat16, no layernorm, no LoRA)"),
            Line2D([0], [0], color=color_apertus, linestyle="--", linewidth=2.5,
                   label="After improvements (layernorm + LoRA + fp32 + lr=3e-4)"),
            Line2D([0], [0], color="#16A34A", marker="*", linewidth=0, markersize=10,
                   label="Green star + arrow = improved point"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.90), fontsize=9.5)
        fig.text(0.5, 0.01,
                 "Aggregation: mean ± std across seeds at each layer. Point labels show aggregated AUC values.",
                 ha="center", fontsize=9, color="#6B7280", style="italic")
        fig.tight_layout(rect=[0, 0.05, 1, 0.86])
        savefig(f"final_ablation_apertus_improvement_{test_model}.png")

    # Mirror the notebook's actual calls
    _lora_vs_no_lora_by_layer("r_at_0.1", "apertus", "apertus", "all")
    _lora_vs_no_lora_by_layer("auc", "llama", "llama", "all")
    _lora_vs_no_lora_by_layer("f1", "apertus", "llama", "all")
    _lora_vs_no_lora_by_layer("final_loss", "apertus", log_scale=True)
    _lora_vs_no_lora_by_layer("auc", "apertus", "llama", "all")
    _baselines_vs_full_solution("apertus", "all")
    _baselines_vs_full_solution("llama", "all")
    _apertus_improvement("apertus", "all")
    _apertus_improvement("llama", "all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = {
    "activations":     plot_activations,
    "layers":          plot_layers,
    "plots":           plot_plots,
    "stability":       plot_stability,
    "final_ablation":  plot_final_ablation,
}


def main():
    parser = argparse.ArgumentParser(description="Reproduce all notebook plots from extracted data.")
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing extracted CSV/npy files (default: jupyter_experiments/extracted_data/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write figures to (default: <data-dir>/figures/)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(SECTIONS),
        metavar="SECTION",
        help=f"Only run these sections: {list(SECTIONS.keys())}",
    )
    args = parser.parse_args()

    global DATA_DIR, FIG_DIR
    if args.data_dir is not None:
        DATA_DIR = args.data_dir.resolve()
    FIG_DIR = args.output_dir.resolve() if args.output_dir is not None else DATA_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sections = args.only if args.only else list(SECTIONS)

    print(f"Reading data from: {DATA_DIR.resolve()}")
    print(f"Saving figures to: {FIG_DIR.resolve()}")

    for name in sections:
        SECTIONS[name]()

    print(f"\nDone. Figures written:")
    for f in sorted(FIG_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
