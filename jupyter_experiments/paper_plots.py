#!/usr/bin/env python3
"""
Paper-ready plots for the hallucination probes paper.

Reads from:  <data-dir>/  (same CSVs produced by extract_data.py)
Writes to:   <output-dir>/  (default: <data-dir>/paper_figures/)

Usage:
  python paper_plots.py --data-dir /path/to/data
  python paper_plots.py --data-dir /path/to/data --output-dir /path/to/figures
  python paper_plots.py --data-dir /path/to/data --only precision_impact lr_impact
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

DATA_DIR = Path(__file__).parent / "extracted_data"
FIG_DIR  = DATA_DIR / "paper_figures"

ALL_LAYERS  = [4, 10, 16, 20, 24, 26, 28, 30]
ABL_LAYERS  = [10, 20, 26, 30]
LRS         = ["1e-3", "3e-4", "1e-4", "3e-5"]
MODEL_DISPLAY = {
    "apertus": "Apertus-8B-Instruct-2509",
    "llama":   "Llama-3.1-8B-Instruct",
}


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
# 1. Precision impact (improve_stability.ipynb → plot_precision_impact_auc_r01_loss)
# ---------------------------------------------------------------------------

def plot_precision_impact():
    print("\n[precision_impact] ...")
    long_csv = DATA_DIR / "stability_runs_long.csv"
    runs_csv = DATA_DIR / "stability_runs.csv"
    if not long_csv.exists():
        print("  stability_runs_long.csv missing — run extract_data.py first.")
        return

    df_long = pd.read_csv(long_csv, dtype={"lr": str})
    df_runs = pd.read_csv(runs_csv, dtype={"lr": str}) if runs_csv.exists() else pd.DataFrame()

    color_by_precision   = {"float32": "#2563EB", "bfloat16": "#D97706"}
    marker_by_precision  = {"float32": "o",       "bfloat16": "s"}
    precision_label      = {"float32": "Probe dtype float32", "bfloat16": "Probe dtype bfloat16"}
    selected_variants    = ["fp32_only", "bf16_only"]
    test_model           = "apertus"
    metric_family        = "all"

    metric_subset = df_long[
        df_long["variant"].isin(selected_variants)
        & (df_long["metric_family"] == metric_family)
        & (df_long["test_model"] == test_model)
        & df_long["metric"].isin(["auc", "recall_at_0.1_fpr"])
        & df_long["probe_dtype"].isin(["float32", "bfloat16"])
        & (df_long["lr"] == "1e-3")
    ].copy()
    if metric_subset.empty:
        print("  No data — skipping.")
        return

    loss_subset = pd.DataFrame()
    if not df_runs.empty:
        loss_subset = df_runs[
            df_runs["variant"].isin(selected_variants)
            & (df_runs["lr"] == "1e-3")
            & df_runs["probe_dtype"].isin(["float32", "bfloat16"])
        ][["probe_dtype", "layer", "final_loss"]].dropna(subset=["final_loss"])

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

    layers = ABL_LAYERS
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharex=True)
    for ax, (metric_key, panel_title, y_label) in zip(axes, [
        ("auc",               "AUC",                "Score"),
        ("recall_at_0.1_fpr", "R@0.1",              "Score"),
        ("final_loss",        "Final training loss", "Cross-entropy loss (log scale)"),
    ]):
        for probe_dtype in ["float32", "bfloat16"]:
            if metric_key == "final_loss":
                if agg_loss.empty:
                    continue
                s = (agg_loss[agg_loss["probe_dtype"] == probe_dtype]
                     .set_index("layer").reindex(layers).reset_index())
            else:
                s = (agg_metrics[(agg_metrics["metric"] == metric_key)
                                  & (agg_metrics["probe_dtype"] == probe_dtype)]
                     .set_index("layer").reindex(layers).reset_index())
            y = s["mean"].to_numpy()
            e = s["std"].to_numpy()
            if np.isnan(y).all():
                continue
            ax.plot(layers, y, color=color_by_precision[probe_dtype],
                    marker=marker_by_precision[probe_dtype], markersize=6.2, linewidth=2.4)
            ax.fill_between(layers, y - e, y + e, color=color_by_precision[probe_dtype],
                            alpha=0.18, edgecolor=color_by_precision[probe_dtype], linewidth=0.8)
        ax.set_title(panel_title, fontsize=12)
        ax.set_xlabel("Probe training layer (activation layer)")
        ax.set_ylabel(y_label)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        if metric_key == "final_loss":
            ax.set_yscale("log")

    train_name = MODEL_DISPLAY["apertus"]
    test_name  = MODEL_DISPLAY.get(test_model, test_model)
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
    savefig("paper_precision_impact.png")


# ---------------------------------------------------------------------------
# 2. Learning-rate impact (improve_stability.ipynb → plot_learning_rate_impact_auc_r01_loss)
# ---------------------------------------------------------------------------

def plot_lr_impact():
    print("\n[lr_impact] ...")
    long_csv = DATA_DIR / "stability_runs_long.csv"
    runs_csv = DATA_DIR / "stability_runs.csv"
    if not long_csv.exists():
        print("  stability_runs_long.csv missing — skipping.")
        return

    df_long = pd.read_csv(long_csv, dtype={"lr": str})
    df_runs = pd.read_csv(runs_csv, dtype={"lr": str}) if runs_csv.exists() else pd.DataFrame()

    color_by_lr  = {"1e-3": "#059669", "3e-4": "#3B82F6", "1e-4": "#F59E0B", "3e-5": "#EF4444"}
    marker_by_lr = {"1e-3": "o", "3e-4": "s", "1e-4": "^", "3e-5": "D"}
    lr_label     = {"1e-3": "Learning rate 1e-3", "3e-4": "Learning rate 3e-4",
                    "1e-4": "Learning rate 1e-4", "3e-5": "Learning rate 3e-5"}
    test_model   = "apertus"
    metric_family = "all"

    metric_subset = df_long[
        (df_long["variant"] == "bf16_only")
        & (df_long["metric_family"] == metric_family)
        & (df_long["test_model"] == test_model)
        & df_long["metric"].isin(["auc", "recall_at_0.1_fpr"])
        & (df_long["probe_dtype"] == "bfloat16")
        & df_long["lr"].notna()
    ].copy()
    if metric_subset.empty:
        print("  No data — skipping.")
        return

    loss_subset = pd.DataFrame()
    if not df_runs.empty:
        loss_subset = df_runs[
            (df_runs["variant"] == "bf16_only")
            & (df_runs["probe_dtype"] == "bfloat16")
            & df_runs["lr"].notna()
        ][["lr", "layer", "final_loss"]].dropna(subset=["final_loss"])

    layers = ABL_LAYERS
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
    for ax, (metric_key, panel_title, y_label) in zip(axes, [
        ("auc",               "AUC",                "Score"),
        ("recall_at_0.1_fpr", "R@0.1",              "Score"),
        ("final_loss",        "Final training loss", "Cross-entropy loss (log scale)"),
    ]):
        for lr in LRS:
            if metric_key == "final_loss":
                if agg_loss.empty:
                    continue
                s = agg_loss[agg_loss["lr"] == lr].set_index("layer").reindex(layers).reset_index()
            else:
                s = (agg_metrics[(agg_metrics["metric"] == metric_key) & (agg_metrics["lr"] == lr)]
                     .set_index("layer").reindex(layers).reset_index())
            y = s["mean"].to_numpy()
            e = s["std"].to_numpy()
            if np.isnan(y).all():
                continue
            ax.plot(layers, y, color=color_by_lr[lr], marker=marker_by_lr[lr],
                    markersize=6.2, linewidth=2.4)
            ax.fill_between(layers, y - e, y + e, color=color_by_lr[lr],
                            alpha=0.18, edgecolor=color_by_lr[lr], linewidth=0.8)
        ax.set_title(panel_title, fontsize=12)
        ax.set_xlabel("Probe training layer (activation layer)")
        ax.set_ylabel(y_label)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        if metric_key == "final_loss":
            ax.set_yscale("log")

    train_name = MODEL_DISPLAY["apertus"]
    test_name  = MODEL_DISPLAY.get(test_model, test_model)
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
    savefig("paper_lr_impact.png")


# ---------------------------------------------------------------------------
# 3. Cross-entropy loss trajectories  (plots.ipynb → training loss curves)
# ---------------------------------------------------------------------------

def plot_loss_trajectories():
    print("\n[loss_trajectories] ...")

    def ema_smooth(values, weight=0.95):
        smoothed, last = [], values[0]
        for v in values:
            last = last * weight + v * (1 - weight)
            smoothed.append(last)
        return smoothed

    RUN_KEYS = [
        ("apertus_no_lora", "apertus", "no LoRA"),
        ("llama_no_lora",   "llama",   "no LoRA"),
        ("apertus_lora",    "apertus", "with LoRA"),
        ("llama_lora",      "llama",   "with LoRA"),
    ]
    histories = {}
    for run_key, model, config in RUN_KEYS:
        csv = DATA_DIR / f"plots_training_history_{run_key}.csv"
        if csv.exists():
            histories[run_key] = pd.read_csv(csv)
        else:
            print(f"  {csv.name} not found — skipping {run_key}.")

    if not histories:
        print("  No training history CSVs found — skipping.")
        return

    color_by_model  = {"apertus": "#D97706", "llama": "#2563EB"}
    linestyle_by_config = {"no LoRA": "-", "with LoRA": "--"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=False)
    panel_data = [
        ("apertus", [("apertus_no_lora", "no LoRA"), ("apertus_lora", "with LoRA")]),
        ("llama",   [("llama_no_lora",   "no LoRA"), ("llama_lora",   "with LoRA")]),
    ]
    for ax, (model_key, run_specs) in zip(axes, panel_data):
        color = color_by_model[model_key]
        for run_key, config in run_specs:
            if run_key not in histories:
                continue
            raw = histories[run_key]["train/loss"].dropna().to_numpy()
            steps = np.arange(len(raw))
            ls = linestyle_by_config[config]
            ax.plot(steps, raw, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(steps, ema_smooth(raw.tolist()), color=color, linewidth=2.2,
                    linestyle=ls, label=config)
        ax.set_title(f"Probe trained on {MODEL_DISPLAY[model_key]} activations", fontsize=12, pad=6)
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_ylabel("Cross-entropy loss", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.85, fontsize=10)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    title_handles = [
        Line2D([0], [0], color=color_by_model["apertus"], linewidth=2.4,
               label=MODEL_DISPLAY["apertus"]),
        Line2D([0], [0], color=color_by_model["llama"], linewidth=2.4,
               label=MODEL_DISPLAY["llama"]),
    ]
    title_patch = mpatches.Patch(visible=False)
    legend = fig.legend(
        [title_patch] + title_handles, ["Activations from:"] + [h.get_label() for h in title_handles],
        loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04),
        columnspacing=1.2, handlelength=2, handletextpad=0.4, fontsize=10,
    )
    legend.get_texts()[0].set_fontweight("bold")
    fig.suptitle(
        "Cross-Entropy Probe Loss During Training\n"
        "Longform dataset; probes trained on activations from different models",
        fontsize=13, y=1.10,
    )
    fig.text(0.5, -0.02,
             "Bold lines: EMA-smoothed loss (α=0.95); faint lines: raw per-step loss.",
             ha="center", fontsize=9, style="italic", color="gray")
    fig.tight_layout()
    savefig("paper_loss_trajectories.png")


# ---------------------------------------------------------------------------
# 4. Baselines vs full-solution  (final_ablation.ipynb → plot_baselines_vs_full_solution)
# ---------------------------------------------------------------------------

def plot_baselines_vs_full(test_model="apertus", metric_family="all"):
    print(f"\n[baselines_vs_full test={test_model}] ...")
    long_csv = DATA_DIR / "final_ablation_runs_long.csv"
    runs_csv = DATA_DIR / "final_ablation_runs.csv"
    if not long_csv.exists():
        print("  final_ablation_runs_long.csv missing — skipping.")
        return

    df_metrics_long = pd.read_csv(long_csv)
    df_runs = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()

    color_by_model    = {"llama": "#2563EB", "apertus": "#D97706"}
    linestyle_by_group = {"baseline": "-",  "full_solution": "--"}
    marker_by_group    = {"baseline": "o",  "full_solution": "s"}

    full_solution = df_metrics_long[
        (df_metrics_long["run_kind"] == "final_ablation")
        & (df_metrics_long["variant"] == "lora_ln")
        & (df_metrics_long["probe_dtype_tag"] == "fp32")
        & (df_metrics_long["metric_family"] == metric_family)
        & (df_metrics_long["test_model"] == test_model)
        & df_metrics_long["metric"].isin(["auc", "recall_at_0.1_fpr"])
    ][["metric", "train_model", "layer", "seed", "value"]].copy()
    full_solution["group"] = "full_solution"

    baselines = df_metrics_long[
        (df_metrics_long["run_kind"] == "baseline")
        & (df_metrics_long["metric_family"] == metric_family)
        & (df_metrics_long["test_model"] == test_model)
        & df_metrics_long["metric"].isin(["auc", "recall_at_0.1_fpr"])
        & df_metrics_long["train_model"].isin(["apertus", "llama"])
    ][["metric", "train_model", "layer", "seed", "value"]].copy()
    baselines["group"] = "baseline"

    metric_subset = pd.concat([baselines, full_solution], ignore_index=True)
    if metric_subset.empty:
        print(f"  No data for test_model={test_model} — skipping.")
        return

    agg_metrics = (
        metric_subset.groupby(["metric", "train_model", "group", "layer"], observed=False)
        .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
    )
    agg_metrics["std"] = agg_metrics["std"].fillna(0.0)

    # Loss panels
    agg_loss = pd.DataFrame()
    if not df_runs.empty and "train/loss" in df_runs.columns:
        loss_full = df_runs[
            (df_runs["run_kind"] == "final_ablation")
            & (df_runs["variant"] == "lora_ln")
            & (df_runs["probe_dtype_tag"] == "fp32")
        ][["train_model", "layer", "seed", "train/loss"]].rename(columns={"train/loss": "value"}).copy()
        loss_full["group"] = "full_solution"

        loss_base = df_runs[
            (df_runs["run_kind"] == "baseline")
            & df_runs["train_model"].isin(["apertus", "llama"])
        ][["train_model", "layer", "seed", "train/loss"]].rename(columns={"train/loss": "value"}).copy()
        loss_base["group"] = "baseline"

        loss_subset = pd.concat([loss_base, loss_full], ignore_index=True).dropna(subset=["value"])
        if not loss_subset.empty:
            agg_loss = (
                loss_subset.groupby(["train_model", "group", "layer"], observed=False)
                .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
            )
            agg_loss["std"] = agg_loss["std"].fillna(0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), sharex=True)
    plot_order = [("llama", "baseline"), ("apertus", "baseline"),
                  ("llama", "full_solution"), ("apertus", "full_solution")]
    for ax, (metric_key, panel_title, y_label) in zip(axes, [
        ("auc",               "AUC",                "Score"),
        ("recall_at_0.1_fpr", "R@0.1",              "Score"),
        ("final_loss",        "Final training loss", "Cross-entropy loss (log scale)"),
    ]):
        if metric_key == "final_loss" and agg_loss.empty:
            ax.text(0.5, 0.5, "Loss data not available\n(re-run extract_data.py)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#9CA3AF")
            ax.set_title(panel_title, fontsize=12)
            ax.set_xlabel("Probe training layer (activation layer)")
            ax.set_xticks(ABL_LAYERS)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            continue

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
             "Aggregation: mean ± std across seeds. Baselines: no layernorm + bfloat16; "
             "full-solution: layernorm + LoRA + fp32 + lr=3e-4.",
             ha="center", fontsize=9, color="#6B7280", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    savefig(f"paper_baselines_vs_full_{test_model}.png")


# ---------------------------------------------------------------------------
# 5. Baseline instability  (layers.ipynb → _plot_auc_r01_loss)
# ---------------------------------------------------------------------------

def plot_baseline_instability(test_model="apertus", metric_family="all"):
    print(f"\n[baseline_instability test={test_model}] ...")
    long_csv = DATA_DIR / "layers_runs_long.csv"
    loss_csv = DATA_DIR / "layers_loss.csv"
    if not long_csv.exists():
        print("  layers_runs_long.csv missing — skipping.")
        return

    df_long = pd.read_csv(long_csv)
    df_loss = pd.read_csv(loss_csv) if loss_csv.exists() else pd.DataFrame()

    COLORS = {"apertus": "#D97706", "llama": "#2563EB"}

    eval_subset = df_long[
        (df_long["metric_family"] == metric_family)
        & (df_long["test_model"] == test_model)
        & df_long["metric"].isin(["auc", "recall_at_0.1_fpr"])
    ].copy()
    if eval_subset.empty:
        print(f"  No eval data for {metric_family}/{test_model} — skipping.")
        return

    auc_agg = (eval_subset[eval_subset["metric"] == "auc"]
               .groupby(["model", "layer"])["value"].agg(["mean", "std"]).reset_index())
    auc_agg["std"] = auc_agg["std"].fillna(0.0)

    r01_agg = (eval_subset[eval_subset["metric"] == "recall_at_0.1_fpr"]
               .groupby(["model", "layer"])["value"].agg(["mean", "std"]).reset_index())
    r01_agg["std"] = r01_agg["std"].fillna(0.0)

    ncols = 3 if not df_loss.empty else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.8), sharex=True)
    panels = [
        (axes[0], auc_agg, "AUC", "Score"),
        (axes[1], r01_agg, "Recall at 0.1 FPR", "Score"),
    ]
    if not df_loss.empty:
        loss_agg = (df_loss.groupby(["model", "layer"])["final_loss"]
                    .agg(["mean", "std"]).reset_index())
        loss_agg["std"] = loss_agg["std"].fillna(0.0)
        panels.append((axes[2], loss_agg, "Final training loss", "Cross-entropy loss (log scale)"))

    for ax, agg, title, ylabel in panels:
        for model in ["apertus", "llama"]:
            sub = agg[agg["model"] == model].sort_values("layer")
            if sub.empty:
                continue
            x = sub["layer"].to_numpy()
            y = sub["mean"].to_numpy()
            e = sub["std"].to_numpy()
            ax.plot(x, y, marker="o", linewidth=2.2,
                    label=MODEL_DISPLAY[model], color=COLORS[model])
            ax.fill_between(x, y - e, y + e, alpha=0.18, color=COLORS[model])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Probe training layer (activation layer)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ALL_LAYERS)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        if title == "Final training loss":
            ax.set_yscale("log")

    test_display = MODEL_DISPLAY.get(test_model, test_model)
    fig.suptitle(
        f"Baseline probe performance and training instability across layers\n"
        f"Evaluation set: {test_display}",
        fontsize=13, y=1.06,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    title_patch = mpatches.Patch(visible=False)
    legend = fig.legend(
        [title_patch] + handles, ["Training set:"] + labels,
        loc="upper center", ncol=len(labels) + 1, frameon=False,
        bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=2, handletextpad=0.4,
    )
    legend.get_texts()[0].set_fontweight("bold")
    fig.tight_layout(rect=(0, 0, 1, 1))
    savefig(f"paper_baseline_instability_{test_model}.png")


# ---------------------------------------------------------------------------
# 6. Baselines vs layernorm impact  (final_ablation.ipynb → no-LoRA panel)
# ---------------------------------------------------------------------------

def plot_baselines_vs_layernorm(train_model="apertus", test_model="apertus",
                                metric_family="all"):
    print(f"\n[baselines_vs_layernorm train={train_model} test={test_model}] ...")
    long_csv = DATA_DIR / "final_ablation_runs_long.csv"
    runs_csv = DATA_DIR / "final_ablation_runs.csv"
    if not long_csv.exists():
        print("  final_ablation_runs_long.csv missing — skipping.")
        return

    df_metrics_long = pd.read_csv(long_csv)
    df_runs = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame()

    COLOR_BY_MODEL   = {"apertus": "#D97706", "llama": "#2563EB"}
    LINESTYLE_BY_NORM = {"none": "-", "layernorm": "--"}
    MARKER_BY_DTYPE   = {"fp32": "o", "bf16": "s"}

    def _variant_norm(v):
        v = str(v)
        return "none" if v.endswith("no_ln") else ("layernorm" if v.endswith("_ln") else "none")

    # No-LoRA ablation runs only
    abl = df_metrics_long[
        (df_metrics_long["run_kind"] == "final_ablation")
        & (df_metrics_long["train_model"] == train_model)
        & df_metrics_long["metric"].isin(["auc", "recall_at_0.1_fpr"])
        & (df_metrics_long["metric_family"] == metric_family)
        & (df_metrics_long["test_model"] == test_model)
        & df_metrics_long["variant"].isin(["no_lora_no_ln", "no_lora_ln"])
    ].copy()
    if abl.empty:
        print(f"  No no-LoRA ablation rows — skipping.")
        return

    abl["norm_kind"] = abl["variant"].map(_variant_norm)
    agg = (
        abl.groupby(["metric", "variant", "probe_dtype_tag", "norm_kind", "layer"], observed=False)
        .agg(mean=("value", "mean"), std=("value", "std")).reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True, sharey=True)
    for ax, (metric_key, panel_title) in zip(axes, [
        ("auc",               "AUC"),
        ("recall_at_0.1_fpr", "R@0.1"),
    ]):
        sub_agg = agg[agg["metric"] == metric_key]
        for (variant, dtype_tag, norm_kind), sub in sub_agg.groupby(
                ["variant", "probe_dtype_tag", "norm_kind"], observed=False):
            s = sub.set_index("layer").reindex(ABL_LAYERS).reset_index()
            y = s["mean"].to_numpy()
            e = s["std"].to_numpy()
            if np.isnan(y).all():
                continue
            ax.plot(ABL_LAYERS, y,
                    color=COLOR_BY_MODEL[train_model],
                    linestyle=LINESTYLE_BY_NORM.get(norm_kind, "-"),
                    marker=MARKER_BY_DTYPE.get(dtype_tag, "o"),
                    markersize=6.5, linewidth=2.2)
            ax.fill_between(ABL_LAYERS, y - e, y + e,
                            color=COLOR_BY_MODEL[train_model], alpha=0.10)

        # Both baselines as dotted lines
        for baseline_model in ["apertus", "llama"]:
            if df_runs.empty:
                continue
            base = df_runs[(df_runs["run_kind"] == "baseline")
                           & (df_runs["train_model"] == baseline_model)].copy()
            if base.empty:
                continue
            col = f"train/longfact_test_{test_model}/{metric_family}_{metric_key}"
            base = base[["layer", col]].rename(columns={col: "value"}).dropna(subset=["value"])
            if base.empty:
                continue
            b_agg = base.groupby("layer").agg(mean=("value", "mean"), std=("value", "std")).reset_index()
            b_agg["std"] = b_agg["std"].fillna(0.0)
            b = b_agg.set_index("layer").reindex(ABL_LAYERS).reset_index()
            by, be = b["mean"].to_numpy(), b["std"].to_numpy()
            ax.plot(ABL_LAYERS, by, color=COLOR_BY_MODEL[baseline_model],
                    linestyle=":", linewidth=2.8)
            ax.fill_between(ABL_LAYERS, by - be, by + be,
                            color=COLOR_BY_MODEL[baseline_model], alpha=0.06)

        ax.set_title(panel_title, fontsize=12)
        ax.set_xticks(ABL_LAYERS)
        ax.set_xlabel("Layer")
        ax.set_ylabel(panel_title)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)

    train_display = MODEL_DISPLAY.get(train_model, train_model)
    test_display  = MODEL_DISPLAY.get(test_model,  test_model)
    fig.suptitle(
        "Baselines vs LayerNorm Impact on Probe Performance (No LoRA)\n"
        f"Train: {train_display} | Test: {test_display} | Family: {metric_family}",
        y=0.98, fontsize=14,
    )
    legend_handles = [
        Line2D([0], [0], color=COLOR_BY_MODEL["apertus"], linewidth=2.4,
               label="Orange = Apertus runs"),
        Line2D([0], [0], color=COLOR_BY_MODEL["llama"], linewidth=2.4,
               label="Blue = Llama runs"),
        Line2D([0], [0], color="#475569", marker="o", linewidth=0, markersize=7,
               label="Circle = float32"),
        Line2D([0], [0], color="#475569", marker="s", linewidth=0, markersize=7,
               label="Square = bfloat16"),
        Line2D([0], [0], color="#475569", linestyle="-",  linewidth=2.2,
               label="Solid = no layernorm"),
        Line2D([0], [0], color="#475569", linestyle="--", linewidth=2.2,
               label="Dashed = layernorm"),
        Line2D([0], [0], color=COLOR_BY_MODEL["apertus"], linestyle=":", linewidth=2.8,
               label="Apertus baseline"),
        Line2D([0], [0], color=COLOR_BY_MODEL["llama"], linestyle=":", linewidth=2.8,
               label="Llama baseline"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.89), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    savefig(f"paper_baselines_vs_layernorm_{train_model}_{test_model}.png")


# ---------------------------------------------------------------------------
# 7. Residual stream L2 norm by layer  (activations_2.ipynb)
# ---------------------------------------------------------------------------

def plot_l2_norm():
    print("\n[l2_norm] ...")
    norms_csv = DATA_DIR / "activation_norms.csv"
    if not norms_csv.exists():
        print("  activation_norms.csv missing — run extract_data.py --run-activations.")
        return

    norm_df = pd.read_csv(norms_csv)
    COLORS = {"apertus": "#D97706", "llama": "#2563EB"}
    MODEL_LABEL = {
        "apertus": MODEL_DISPLAY["apertus"],
        "llama":   MODEL_DISPLAY["llama"],
    }
    NOTE_COLOR = "#888888"

    layers = sorted(norm_df["layer"].unique().tolist())
    fig, ax = plt.subplots(figsize=(8, 5.0))
    for model_key in ["apertus", "llama"]:
        cur = norm_df[norm_df["model"] == model_key]
        agg = cur.groupby("layer")[["norm_mean", "norm_std", "norm_p90"]].mean().reset_index()
        c = COLORS[model_key]
        ax.fill_between(agg["layer"], agg["norm_mean"] - agg["norm_std"],
                        agg["norm_mean"] + agg["norm_std"], color=c, alpha=0.18, zorder=1)
        ax.plot(agg["layer"], agg["norm_mean"], color=c, marker="o", linewidth=2.2,
                markersize=7, label=MODEL_LABEL[model_key], zorder=3)
        ax.plot(agg["layer"], agg["norm_p90"], color=c, marker="s", linestyle="--",
                linewidth=1.4, markersize=5, zorder=3, label="_nolegend_")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.2f}")
    )
    ax.set_xticks(layers)
    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("‖h‖₂  (residual stream L2 norm, log scale)", fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    handles, lbls = ax.get_legend_handles_labels()
    title_patch = mpatches.Patch(visible=False)
    legend = fig.legend(
        [title_patch] + handles, ["Model:"] + lbls,
        loc="upper center", ncol=len(lbls) + 1, frameon=False,
        bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=2,
        handletextpad=0.4, fontsize=10,
    )
    legend.get_texts()[0].set_fontweight("bold")
    fig.suptitle("Residual stream L2 norm by layer — test dataset split",
                 fontsize=13, y=1.08)
    fig.text(0.5, -0.04,
             "Solid line: mean ‖h‖₂; Dashed line: 90th percentile ‖h‖₂; Shaded band: ±1 std.",
             ha="center", fontsize=9, color=NOTE_COLOR)
    fig.tight_layout(rect=(0, 0, 1, 1))
    savefig("paper_l2_norm.png")


# ---------------------------------------------------------------------------
# 8. PCA scatter  (activations_2.ipynb)
# ---------------------------------------------------------------------------

def plot_pca_scatter():
    print("\n[pca_scatter] ...")
    pca_csv = DATA_DIR / "activation_pca_scatter.csv"
    norms_csv = DATA_DIR / "activation_norms.csv"
    if not pca_csv.exists():
        print("  activation_pca_scatter.csv missing — run extract_data.py --run-activations.")
        return

    pca_df = pd.read_csv(pca_csv)
    norm_df = pd.read_csv(norms_csv) if norms_csv.exists() else pd.DataFrame()

    HIGHLIGHT_LAYERS = [4, 16, 26, 30]
    PALETTE = {"Hallucinated": "#e05c5c", "Supported": "#4a90d9"}
    NOTE_COLOR = "#888888"

    norm_lookup = {}
    if not norm_df.empty:
        norm_lookup = norm_df.groupby(["model", "layer"])["norm_mean"].mean().to_dict()

    fig, axes = plt.subplots(2, len(HIGHLIGHT_LAYERS), figsize=(16, 7),
                             sharex=False, sharey=False)
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
    title_patch = mpatches.Patch(visible=False)
    legend = fig.legend(
        [title_patch] + scatter_handles, ["Token label:"] + [h.get_label() for h in scatter_handles],
        loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.01), fontsize=9.5,
        handlelength=1.2, handletextpad=0.5, columnspacing=1.5,
    )
    legend.get_texts()[0].set_fontweight("bold")
    fig.suptitle(
        "PCA of residual stream activations — hallucinated (red) vs. supported (blue) tokens\n"
        f"{MODEL_DISPLAY['apertus']} (top) · {MODEL_DISPLAY['llama']} (bottom) · independent axes per panel",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    savefig("paper_pca_scatter.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = {
    "precision_impact":        plot_precision_impact,
    "lr_impact":               plot_lr_impact,
    "loss_trajectories":       plot_loss_trajectories,
    "baselines_vs_full":       lambda: [plot_baselines_vs_full("apertus"),
                                        plot_baselines_vs_full("llama")],
    "baseline_instability":    lambda: plot_baseline_instability("apertus"),
    "baselines_vs_layernorm":  lambda: [plot_baselines_vs_layernorm("apertus", "apertus"),
                                        plot_baselines_vs_layernorm("apertus", "llama")],
    "l2_norm":                 plot_l2_norm,
    "pca_scatter":             plot_pca_scatter,
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready plots from extracted data.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing extracted CSVs (default: jupyter_experiments/extracted_data/)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write figures to (default: <data-dir>/paper_figures/)")
    parser.add_argument("--only", nargs="+", choices=list(SECTIONS), metavar="SECTION",
                        help=f"Only run these sections: {list(SECTIONS.keys())}")
    args = parser.parse_args()

    global DATA_DIR, FIG_DIR
    if args.data_dir is not None:
        DATA_DIR = args.data_dir.resolve()
    FIG_DIR = args.output_dir.resolve() if args.output_dir is not None else DATA_DIR / "paper_figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sections = args.only if args.only else list(SECTIONS)
    print(f"Reading data from: {DATA_DIR}")
    print(f"Saving figures to: {FIG_DIR}")

    for name in sections:
        fn = SECTIONS[name]
        result = fn()
        if result is not None and hasattr(result, "__iter__"):
            list(result)

    print(f"\nDone. Figures written:")
    for f in sorted(FIG_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
