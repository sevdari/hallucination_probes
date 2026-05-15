#!/usr/bin/env python3
"""
Extract all experimental data from the Jupyter notebooks and save to CSV/numpy.

Covers:
  - activations.ipynb   → activation separation metrics (precomputed + optional re-run)
  - layers.ipynb        → W&B layer-wise probe performance across seeds
  - plots.ipynb         → W&B summary + training history for no-LoRA vs LoRA runs
  - class_inbalance.ipynb → token-level class balance statistics (precomputed)
  - dataset_properties.ipynb → HuggingFace dataset statistics

Usage:
  python extract_data.py                      # all sections
  python extract_data.py --skip-wandb         # skip W&B (no credentials needed)
  python extract_data.py --skip-hf            # skip HuggingFace dataset fetching
  python extract_data.py --run-activations    # re-run full activation collection (GPU required)
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Output directory (overridable via --output-dir)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "extracted_data"


# ---------------------------------------------------------------------------
# 1. Activation metrics  (activations.ipynb)
# ---------------------------------------------------------------------------

def extract_activation_metrics_precomputed():
    """
    Save the metrics_df and counts_df that were computed and printed in
    activations.ipynb.  Values are taken verbatim from the cell output.
    """
    print("\n[activations] Saving precomputed metrics ...")

    metrics_data = {
        "model": [
            "apertus", "apertus", "apertus", "apertus",
            "apertus", "apertus", "apertus", "apertus",
            "llama",   "llama",   "llama",   "llama",
            "llama",   "llama",   "llama",   "llama",
        ],
        "layer": [4, 10, 16, 20, 24, 26, 28, 30,
                  4, 10, 16, 20, 24, 26, 28, 30],
        "n_pos":  [30000.0] * 16,
        "n_neg":  [30000.0] * 16,
        "pos_neg_ratio_used": [1.0] * 16,
        "silhouette_pca10": [
            0.027785, 0.050628, 0.068458, 0.062924, 0.075170, 0.077452, 0.071655, 0.079076,
            0.039833, 0.046509, 0.051183, 0.052317, 0.054845, 0.054691, 0.051397, 0.044381,
        ],
        "centroid_l2_hidden": [
            6.960910, 60.554287, 359.963135, 807.219238,
            1957.342163, 2513.031738, 3102.258301, 4566.378418,
            0.550055, 1.116571, 1.910677, 2.732390,
            4.205286, 5.136626, 6.009090, 6.553996,
        ],
        "centroid_cosine_dist_hidden": [
            0.001167, 0.007587, 0.017713, 0.015108, 0.008300, 0.007442, 0.006754, 0.011487,
            0.041147, 0.049757, 0.054243, 0.058177, 0.040536, 0.032821, 0.026035, 0.025124,
        ],
        "fisher_ratio_hidden": [
            51.634626, 101.555378, 124.007178, 90.454609,
            133.135047, 141.317170, 138.026321, 169.635740,
            52.189599, 61.441534, 65.578954, 60.838476,
            69.869934, 76.684798, 74.556515, 57.550653,
        ],
        "linear_probe_auc_pca20": [
            0.847235, 0.883875, 0.882948, 0.882949, 0.867817, 0.864106, 0.864774, 0.856180,
            0.860388, 0.884164, 0.888672, 0.881289, 0.873833, 0.875293, 0.866394, 0.865732,
        ],
        "linear_probe_acc_pca20": [
            0.773056, 0.804444, 0.810389, 0.805889, 0.794167, 0.788889, 0.790500, 0.784000,
            0.780222, 0.810611, 0.814111, 0.806389, 0.803500, 0.799056, 0.787167, 0.788611,
        ],
        "kmeans_ari_pca20": [
            0.010044, 0.004038, 0.215933, 0.182714, 0.081575, 0.076220, 0.097978, 0.074736,
            0.029066, 0.152893, 0.122229, 0.040681, 0.030891, 0.035669, 0.036454, 0.035706,
        ],
        "kmeans_nmi_pca20": [
            0.007270, 0.003126, 0.164322, 0.136818, 0.091620, 0.086207, 0.098888, 0.079042,
            0.021836, 0.114467, 0.092787, 0.030825, 0.025822, 0.029749, 0.029853, 0.027357,
        ],
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(OUTPUT_DIR / "activation_metrics.csv", index=False)
    print(f"  Saved {len(metrics_df)} rows → extracted_data/activation_metrics.csv")

    # Per-layer delta (Apertus - Llama) for each metric
    delta_rows = []
    metric_cols = [
        "silhouette_pca10", "linear_probe_auc_pca20", "linear_probe_acc_pca20",
        "kmeans_ari_pca20", "kmeans_nmi_pca20", "fisher_ratio_hidden",
        "centroid_cosine_dist_hidden",
    ]
    for layer in [4, 10, 16, 20, 24, 26, 28, 30]:
        row = {"layer": layer}
        a = metrics_df[(metrics_df.model == "apertus") & (metrics_df.layer == layer)].iloc[0]
        l = metrics_df[(metrics_df.model == "llama")   & (metrics_df.layer == layer)].iloc[0]
        for col in metric_cols:
            row[f"delta_{col}"] = a[col] - l[col]
        delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUTPUT_DIR / "activation_metrics_delta_apertus_minus_llama.csv", index=False)
    print(f"  Saved {len(delta_df)} rows → extracted_data/activation_metrics_delta_apertus_minus_llama.csv")

    # Model-level averages
    avg_df = metrics_df.groupby("model")[metric_cols].mean().reset_index()
    avg_df.to_csv(OUTPUT_DIR / "activation_metrics_model_averages.csv", index=False)
    print(f"  Saved averages → extracted_data/activation_metrics_model_averages.csv")

    # Best layer per model per metric
    best_rows = []
    for metric in metric_cols:
        idx = metrics_df.groupby("model")[metric].idxmax()
        cur = metrics_df.loc[idx, ["model", "layer", metric]].copy()
        cur["metric"] = metric
        best_rows.append(cur)
    best_df = pd.concat(best_rows, ignore_index=True)
    best_df.to_csv(OUTPUT_DIR / "activation_best_layer_per_metric.csv", index=False)
    print(f"  Saved best-layer table → extracted_data/activation_best_layer_per_metric.csv")

    return metrics_df


def run_activation_collection():
    """
    Re-run the full activation collection from activations.ipynb.
    Requires GPU, HuggingFace model access, and ~16 GB VRAM per model.
    """
    print("\n[activations] Running full activation collection (this requires a GPU) ...")
    sys.path.insert(0, str(ROOT))
    import random
    import torch
    import importlib
    import utils.activation_analysis as activation_analysis
    activation_analysis = importlib.reload(activation_analysis)

    from datasets import disable_progress_bar
    disable_progress_bar()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    TARGET_LAYERS = [4, 10, 16, 20, 24, 26, 28, 30]
    MAX_TOKENS_PER_LABEL = 30000
    MAX_COMPLETION_LENGTH = 1536
    MAX_SAMPLES = 3000

    DatasetSpec = activation_analysis.DatasetSpec
    CollectionSpec = activation_analysis.CollectionSpec
    build_metrics_table_for_ratio = activation_analysis.build_metrics_table_for_ratio
    build_pca_plot_frame = activation_analysis.build_pca_plot_frame
    collect_multilayer_activations_for_model = activation_analysis.collect_multilayer_activations_for_model
    equalize_sweep_results_for_fair_comparison = activation_analysis.equalize_sweep_results_for_fair_comparison
    summarize_collection_counts = activation_analysis.summarize_collection_counts

    MODEL_SPECS = {
        "apertus": {
            "subset": "Apertus_8B_Instruct_2509",
            "model_name": "swiss-ai/Apertus-8B-Instruct-2509",
        },
        "llama": {
            "subset": "Meta_Llama_3.1_8B_Instruct",
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        },
    }

    dataset_spec = DatasetSpec(
        hf_repo="tkwiecinski/longfact-test-split",
        split="test",
        max_length=MAX_COMPLETION_LENGTH,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )
    collection_spec = CollectionSpec(
        layers=TARGET_LAYERS,
        max_tokens_per_label=MAX_TOKENS_PER_LABEL,
        pos_neg_ratio=1.0,
        enforce_ratio_on_finalize=True,
    )

    sweep_results = {}
    for model_key in ["apertus", "llama"]:
        spec = MODEL_SPECS[model_key]
        sweep_results[model_key] = collect_multilayer_activations_for_model(
            model_key=model_key,
            model_name=spec["model_name"],
            subset=spec["subset"],
            dataset_spec=dataset_spec,
            collection_spec=collection_spec,
        )

    fair_sweep = equalize_sweep_results_for_fair_comparison(
        sweep_results, TARGET_LAYERS, pos_neg_ratio=1.0, seed=SEED
    )

    # Save raw activation arrays as float16 (~234 MB/file vs ~469 MB for float32)
    act_dir = OUTPUT_DIR / "activations"
    act_dir.mkdir(exist_ok=True)
    for model_key, model_res in fair_sweep.items():
        for layer in TARGET_LAYERS:
            pos = np.array(model_res["per_layer"][layer]["positive"], dtype=np.float16)
            neg = np.array(model_res["per_layer"][layer]["negative"], dtype=np.float16)
            np.save(act_dir / f"{model_key}_layer{layer}_hallucinated.npy", pos)
            np.save(act_dir / f"{model_key}_layer{layer}_supported.npy", neg)
            print(f"  Saved activations: {model_key} layer {layer}: {pos.shape} float16")

    # Activation L2 norm stats (needed by activations_2.ipynb)
    norm_rows = []
    for model_key, model_res in fair_sweep.items():
        for layer in TARGET_LAYERS:
            for label, split_key in [("hallucinated", "positive"), ("supported", "negative")]:
                arr = np.array(model_res["per_layer"][layer][split_key], dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1)
                norm_rows.append({
                    "model": model_key, "layer": layer, "label": label,
                    "norm_mean": float(norms.mean()),
                    "norm_std":  float(norms.std()),
                    "norm_p90":  float(np.percentile(norms, 90)),
                    "n_tokens":  len(norms),
                })
    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(OUTPUT_DIR / "activation_norms.csv", index=False)
    print(f"  Saved activation L2 norm stats → extracted_data/activation_norms.csv")

    # Save metrics — use sample_total=3000 to keep silhouette O(n²) tractable;
    # the full 60k-sample version would take hours (silhouette is O(n²)).
    # Raw arrays are saved above if you need to recompute with more samples.
    METRICS_SAMPLE_TOTAL = 3000
    print(f"  Computing separation metrics (sample_total={METRICS_SAMPLE_TOTAL} per layer) ...")
    metrics_df = build_metrics_table_for_ratio(
        fair_sweep, TARGET_LAYERS,
        pos_neg_ratio=1.0,
        sample_total=METRICS_SAMPLE_TOTAL,
        seed=SEED,
    )
    metrics_df.to_csv(OUTPUT_DIR / "activation_metrics_recomputed.csv", index=False)
    counts_df = summarize_collection_counts(fair_sweep, TARGET_LAYERS)
    counts_df.to_csv(OUTPUT_DIR / "activation_counts.csv", index=False)

    # Save PCA scatter data (needed for scatter plots — 700 points per label per layer/model)
    pca_df = build_pca_plot_frame(fair_sweep, TARGET_LAYERS, max_points_per_label=700, seed=SEED)
    pca_df.to_csv(OUTPUT_DIR / "activation_pca_scatter.csv", index=False)
    print(f"  Saved PCA scatter frame: {len(pca_df)} rows → extracted_data/activation_pca_scatter.csv")

    print("  Full activation collection complete.")
    return metrics_df


# ---------------------------------------------------------------------------
# 2. Layer-wise W&B run data  (layers.ipynb)
# ---------------------------------------------------------------------------

def extract_wandb_layers():
    """
    Fetch all 48 no-LoRA runs from W&B (2 models × 8 layers × 3 seeds) and save:
      - layers_runs.csv          : per-run metadata + all summary metrics
      - layers_runs_long.csv     : tidy/melted format
      - layers_loss.csv          : final training loss per run
      - layers_agg_mean_std.csv  : mean±std per (model, layer, metric)
    """
    print("\n[layers] Fetching W&B run data ...")
    try:
        import wandb
    except ImportError:
        print("  wandb not installed. Run: pip install wandb")
        return

    import re
    from typing import Optional

    api = wandb.Api()
    ENTITY = "ethz-lsai-25"
    PROJECT = "hallucination-probes"
    RUN_PATH = f"{ENTITY}/{PROJECT}"

    MODELS = ["apertus", "llama"]
    LAYERS = [4, 10, 16, 20, 24, 26, 28, 30]
    SEEDS = [42, 43, 44]
    TEST_MODELS = ["apertus", "llama"]
    METRIC_FAMILIES = ["all", "span", "span_max"]
    METRICS = ["auc", "f1", "acc", "recall_at_0.1_fpr"]

    RUN_NAME_PATTERN = re.compile(
        r"^(apertus|llama)_no_lora_long_form_layer(4|10|16|20|24|26|28|30)_seed(42|43|44)$"
    )

    def _extract_probe_id(run) -> Optional[str]:
        name = run.name or ""
        if RUN_NAME_PATTERN.match(name):
            return name
        cfg = run.config or {}
        if isinstance(cfg.get("probe_config"), dict):
            nested = cfg["probe_config"].get("probe_id")
            if isinstance(nested, str) and RUN_NAME_PATTERN.match(nested):
                return nested
        for key in ["probe_config.probe_id", "probe_config/probe_id", "probe_id"]:
            value = cfg.get(key)
            if isinstance(value, str) and RUN_NAME_PATTERN.match(value):
                return value
        return None

    def _safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Server-side pre-filter: only fetch runs whose display name matches the pattern.
    # This avoids paginating through every run in the project.
    wandb_filter = {
        "display_name": {"$regex": r"^(apertus|llama)_no_lora_long_form_layer\d+_seed\d+$"}
    }

    rows = []
    for run in api.runs(RUN_PATH, filters=wandb_filter):
        probe_id = _extract_probe_id(run)
        if probe_id is None:
            continue
        match = RUN_NAME_PATTERN.match(probe_id)
        if match is None:
            continue
        model, layer, seed = match.groups()
        summary = run.summary._json_dict
        row = {
            "run_id":     run.id,
            "run_name":   run.name,
            "probe_id":   probe_id,
            "state":      run.state,
            "created_at": pd.to_datetime(run.created_at, utc=True),
            "model":      model,
            "layer":      int(layer),
            "seed":       int(seed),
            # train/loss is auto-tracked in the W&B summary as the last logged value;
            # grab it here so we don't need a second round-trip per run below.
            "final_loss": _safe_float(summary.get("train/loss")),
        }
        for test_model in TEST_MODELS:
            for family in METRIC_FAMILIES:
                for metric in METRICS:
                    key = f"train/longfact_test_{test_model}/{family}_{metric}"
                    row[key] = _safe_float(summary.get(key))
        rows.append(row)

    df_runs = pd.DataFrame(rows)
    if df_runs.empty:
        print("  No matching runs found.")
        return

    state_rank = {"finished": 0, "running": 1, "queued": 2, "failed": 3, "crashed": 4}
    df_runs["_state_rank"] = df_runs["state"].map(state_rank).fillna(99)
    df_runs = (
        df_runs.sort_values(["_state_rank", "created_at"], ascending=[True, False])
        .drop_duplicates(["model", "layer", "seed"], keep="first")
        .drop(columns=["_state_rank"])
        .sort_values(["model", "layer", "seed"])
        .reset_index(drop=True)
    )
    df_runs.to_csv(OUTPUT_DIR / "layers_runs.csv", index=False)
    print(f"  Saved {len(df_runs)} runs → extracted_data/layers_runs.csv")

    # Tidy long format
    metric_columns = [
        f"train/longfact_test_{test_model}/{family}_{metric}"
        for test_model in TEST_MODELS
        for family in METRIC_FAMILIES
        for metric in METRICS
    ]
    df_long = df_runs.melt(
        id_vars=["run_id", "probe_id", "state", "created_at", "model", "layer", "seed"],
        value_vars=metric_columns,
        var_name="metric_key",
        value_name="value",
    )
    parsed = df_long["metric_key"].str.extract(
        r"train/longfact_test_(apertus|llama)/(all|span|span_max)_(auc|f1|acc|recall_at_0\.1_fpr)"
    )
    parsed.columns = ["test_model", "metric_family", "metric"]
    df_long = pd.concat([df_long, parsed], axis=1)
    df_long = df_long.dropna(subset=["value", "test_model", "metric_family", "metric"]).copy()
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.dropna(subset=["value"]).reset_index(drop=True)
    df_long.to_csv(OUTPUT_DIR / "layers_runs_long.csv", index=False)
    print(f"  Saved long-format → extracted_data/layers_runs_long.csv")

    # Final training loss — already in summary, no extra API calls needed.
    # For any runs where it's missing, fall back to fetching history in parallel.
    loss_cols = ["run_id", "probe_id", "model", "layer", "seed", "final_loss"]
    df_loss = df_runs[loss_cols].copy()

    missing = df_loss[df_loss["final_loss"].isna()]
    if not missing.empty:
        print(f"  {len(missing)} runs missing train/loss in summary — fetching histories in parallel ...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_loss(run_id):
            run_obj = api.run(f"{RUN_PATH}/{run_id}")
            hist = run_obj.history(keys=["train/loss"], pandas=True)
            if not hist.empty:
                return run_id, hist["train/loss"].dropna().iloc[-1]
            return run_id, np.nan

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_loss, rid): rid for rid in missing["run_id"]}
            for fut in as_completed(futures):
                run_id, loss = fut.result()
                df_loss.loc[df_loss["run_id"] == run_id, "final_loss"] = loss
    else:
        print(f"  All {len(df_loss)} final losses read from summary (no extra API calls).")

    df_loss.to_csv(OUTPUT_DIR / "layers_loss.csv", index=False)
    print(f"  Saved final losses → extracted_data/layers_loss.csv")

    # Aggregated mean±std per (model, layer)
    agg_rows = []
    val_cols = [c for c in df_runs.columns if c.startswith("train/")]
    for (model, layer), grp in df_runs.groupby(["model", "layer"]):
        row = {"model": model, "layer": layer}
        for col in val_cols:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"{col}_std"]  = vals.std()  if len(vals) > 1 else np.nan
        # loss
        loss_vals = df_loss[(df_loss.model == model) & (df_loss.layer == layer)]["final_loss"].dropna()
        row["final_loss_mean"] = loss_vals.mean() if len(loss_vals) > 0 else np.nan
        row["final_loss_std"]  = loss_vals.std()  if len(loss_vals) > 1 else np.nan
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUTPUT_DIR / "layers_aggregated.csv", index=False)
    print(f"  Saved aggregated stats → extracted_data/layers_aggregated.csv")

    return df_runs, df_long, df_loss


# ---------------------------------------------------------------------------
# 3. Plots W&B data  (plots.ipynb)
# ---------------------------------------------------------------------------

# Run IDs from the notebook
PLOT_RUNS = {
    "apertus_no_lora": "ynankpjz",
    "llama_no_lora":   "rct5pows",
    "apertus_lora":    "payanx0q",   # LoRA λ_KL=0.5
    "llama_lora":      "fsr8yp0j",
}


def extract_wandb_plots():
    """
    Fetch run summaries and full training histories for the 4 runs used in plots.ipynb:
      - plots_summaries.csv        : all summary metrics for all 4 runs
      - plots_cross_model_f1.csv   : 2×2 cross-model F1 matrix (no-LoRA and LoRA)
      - plots_training_history_<run_key>.csv : step-level training loss per run
    """
    print("\n[plots] Fetching W&B run summaries and training histories ...")
    try:
        import wandb
    except ImportError:
        print("  wandb not installed.")
        return

    api = wandb.Api()
    ENTITY = "ethz-lsai-25"
    PROJECT = "hallucination-probes"
    RUN_PATH = f"{ENTITY}/{PROJECT}"

    summary_rows = []
    for run_key, run_id in PLOT_RUNS.items():
        run = api.run(f"{RUN_PATH}/{run_id}")
        summary = {k: v for k, v in run.summary._json_dict.items() if not k.startswith("_")}
        summary["run_key"] = run_key
        summary["run_id"]  = run_id
        summary["run_name"] = run.name
        summary["state"]   = run.state
        summary_rows.append(summary)

    summaries_df = pd.DataFrame(summary_rows)
    summaries_df.to_csv(OUTPUT_DIR / "plots_summaries.csv", index=False)
    print(f"  Saved run summaries → extracted_data/plots_summaries.csv")

    # 2×2 cross-model matrices for F1, AUC, accuracy
    model_names = ["apertus", "llama"]
    for config_label, run_keys in [
        ("no_lora", ["apertus_no_lora", "llama_no_lora"]),
        ("lora",    ["apertus_lora",    "llama_lora"]),
    ]:
        for metric_name in ["all_f1", "all_auc", "span_max_auc", "span_max_f1"]:
            matrix_rows = []
            for test_model in model_names:
                matrix_row = {"test_model": test_model}
                for train_model, run_key_suffix in zip(model_names, run_keys):
                    run_row = summaries_df[summaries_df["run_key"] == run_key_suffix]
                    col = f"train/longfact_test_{test_model}/{metric_name}"
                    val = run_row[col].values[0] if col in run_row.columns else np.nan
                    matrix_row[f"train_{train_model}"] = val
                matrix_rows.append(matrix_row)
            matrix_df = pd.DataFrame(matrix_rows)
            fname = f"plots_cross_model_{config_label}_{metric_name}.csv"
            matrix_df.to_csv(OUTPUT_DIR / fname, index=False)
        print(f"  Saved cross-model matrices for {config_label} → extracted_data/plots_cross_model_{config_label}_*.csv")

    # Training histories (loss curves) — fetch all 4 runs in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_history(run_key, run_id):
        run = api.run(f"{RUN_PATH}/{run_id}")
        hist = run.history(keys=["train/loss", "_step"], pandas=True)
        return run_key, hist

    print(f"  Fetching training histories for {len(PLOT_RUNS)} runs in parallel ...")
    with ThreadPoolExecutor(max_workers=len(PLOT_RUNS)) as pool:
        futures = {pool.submit(_fetch_history, k, v): k for k, v in PLOT_RUNS.items()}
        for fut in as_completed(futures):
            run_key, hist = fut.result()
            if not hist.empty:
                hist = hist.dropna(subset=["train/loss"]).reset_index(drop=True)
                hist["run_key"] = run_key
                hist.to_csv(OUTPUT_DIR / f"plots_training_history_{run_key}.csv", index=False)
                print(f"    {run_key}: {len(hist)} steps → plots_training_history_{run_key}.csv")
            else:
                print(f"    {run_key}: no history found.")

    return summaries_df


# ---------------------------------------------------------------------------
# 4. Stability sweep W&B data  (improve_stability.ipynb)
# ---------------------------------------------------------------------------

def extract_stability_runs():
    """
    Fetch Apertus stability-sweep runs (lr × variant × layer × seed) plus
    baseline runs (apertus/llama no-LoRA long-form) and save:
      - stability_runs.csv       : per-run metadata + summary metrics
      - stability_runs_long.csv  : tidy long format
    """
    print("\n[stability] Fetching W&B stability sweep runs ...")
    try:
        import wandb
    except ImportError:
        print("  wandb not installed.")
        return

    import re

    api = wandb.Api()
    ENTITY = "ethz-lsai-25"
    PROJECT = "hallucination-probes"
    RUN_PATH = f"{ENTITY}/{PROJECT}"

    LAYERS = [10, 20, 26, 30]
    SEEDS = [42, 43, 44]
    LRS = ["1e-3", "3e-4", "1e-4", "3e-5"]
    TEST_MODELS = ["apertus", "llama"]
    METRIC_FAMILIES = ["all", "span", "span_max"]
    METRICS = ["auc", "f1", "acc", "recall_at_0.1_fpr"]
    VARIANTS = ["ln_fp32", "fp32_only", "ln_bf16", "bf16_only"]

    VARIANT_META = {
        "apertus_layers": {"probe_dtype": "bfloat16", "normalize_before_head": "none", "train_model": "apertus"},
        "llama_layers":   {"probe_dtype": "bfloat16", "normalize_before_head": "none", "train_model": "llama"},
        "ln_fp32":  {"probe_dtype": "float32",  "normalize_before_head": "layernorm", "train_model": "apertus"},
        "fp32_only":{"probe_dtype": "float32",  "normalize_before_head": "none",      "train_model": "apertus"},
        "ln_bf16":  {"probe_dtype": "bfloat16", "normalize_before_head": "layernorm", "train_model": "apertus"},
        "bf16_only":{"probe_dtype": "bfloat16", "normalize_before_head": "none",      "train_model": "apertus"},
    }
    LR_TAG_TO_VALUE = {"1em3": "1e-3", "3em4": "3e-4", "1em4": "1e-4", "3em5": "3e-5"}

    RUN_PATTERN = re.compile(
        r"^apertus_no_lora_(ln_fp32|fp32_only|ln_bf16|bf16_only)_layer(10|20|26|30)_seed(42|43|44)_lr(1em3|3em4|1em4|3em5)$"
    )
    BASELINE_PATTERN = re.compile(
        r"^(apertus|llama)_no_lora_long_form_layer(10|20|26|30)_seed(42|43|44)$"
    )

    def _extract_id(run):
        name = run.name or ""
        if RUN_PATTERN.match(name) or BASELINE_PATTERN.match(name):
            return name
        cfg = run.config or {}
        if isinstance(cfg.get("probe_config"), dict):
            v = cfg["probe_config"].get("probe_id", "")
            if RUN_PATTERN.match(v) or BASELINE_PATTERN.match(v):
                return v
        for k in ["probe_config.probe_id", "probe_config/probe_id", "probe_id"]:
            v = cfg.get(k, "")
            if isinstance(v, str) and (RUN_PATTERN.match(v) or BASELINE_PATTERN.match(v)):
                return v
        return None

    def _safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def _canon_dtype(v):
        if v is None: return None
        k = str(v).strip().lower()
        if k in {"fp32", "float32", "torch.float32"}: return "float32"
        if k in {"bf16", "bfloat16", "torch.bfloat16"}: return "bfloat16"
        return k

    def _canon_norm(v):
        return str(v).strip().lower() if v is not None else None

    def _cfg_get(cfg, key, *, nested_parent=None, nested_child=None):
        v = cfg.get(key)
        if v is not None: return v
        if nested_parent and nested_child:
            p = cfg.get(nested_parent)
            if isinstance(p, dict): return p.get(nested_child)
        return None

    def _extract_final_loss(summary):
        for k in ["train/loss", "train/final_loss", "loss", "final_loss"]:
            v = _safe_float(summary.get(k))
            if v is not None: return v
        return None

    wandb_filter = {"display_name": {"$regex": r"^(apertus_no_lora_(ln_fp32|fp32_only|ln_bf16|bf16_only)|apertus_no_lora_long_form|llama_no_lora_long_form)"}}
    rows = []
    for run in api.runs(RUN_PATH, filters=wandb_filter):
        probe_id = _extract_id(run)
        if not probe_id:
            continue
        m = RUN_PATTERN.match(probe_id)
        b = BASELINE_PATTERN.match(probe_id) if not m else None
        if not m and not b:
            continue

        if m:
            variant, layer, seed, lr_tag = m.groups()
            lr = LR_TAG_TO_VALUE[lr_tag]
        else:
            baseline_model, layer, seed = b.groups()
            variant = f"{baseline_model}_layers"
            lr = None

        meta = VARIANT_META[variant]
        cfg = run.config or {}
        summary = run.summary._json_dict

        actual_probe_dtype = _canon_dtype(
            _cfg_get(cfg, "probe_config.probe_dtype", nested_parent="probe_config", nested_child="probe_dtype")
            or _cfg_get(cfg, "probe_config/probe_dtype") or _cfg_get(cfg, "probe_dtype")
        ) or _canon_dtype(meta["probe_dtype"])
        actual_norm = _canon_norm(
            _cfg_get(cfg, "probe_config.normalize_before_head", nested_parent="probe_config", nested_child="normalize_before_head")
            or _cfg_get(cfg, "probe_config/normalize_before_head") or _cfg_get(cfg, "normalize_before_head")
        ) or _canon_norm(meta["normalize_before_head"])

        row = {
            "run_id": run.id, "run_name": run.name, "probe_id": probe_id,
            "state": run.state,
            "created_at": pd.to_datetime(run.created_at, utc=True),
            "train_model": meta["train_model"],
            "variant": variant, "layer": int(layer), "seed": int(seed), "lr": lr,
            "probe_dtype": actual_probe_dtype,
            "normalize_before_head": actual_norm,
            "final_loss": _extract_final_loss(summary),
        }
        for tm in TEST_MODELS:
            for fam in METRIC_FAMILIES:
                for metric in METRICS:
                    key = f"train/longfact_test_{tm}/{fam}_{metric}"
                    row[key] = _safe_float(summary.get(key))
        rows.append(row)

    df_runs = pd.DataFrame(rows)
    if df_runs.empty:
        print("  No matching stability runs found.")
        return

    state_rank = {"finished": 0, "running": 1, "queued": 2, "failed": 3, "crashed": 4}
    df_runs["_state_rank"] = df_runs["state"].map(state_rank).fillna(99)
    df_runs = (
        df_runs.sort_values(["_state_rank", "created_at"], ascending=[True, False])
        .drop_duplicates(["variant", "layer", "seed", "lr"], keep="first")
        .drop(columns=["_state_rank"])
        .sort_values(["variant", "layer", "seed", "lr"])
        .reset_index(drop=True)
    )
    df_runs.to_csv(OUTPUT_DIR / "stability_runs.csv", index=False)
    print(f"  Saved {len(df_runs)} runs → extracted_data/stability_runs.csv")

    metric_columns = [
        f"train/longfact_test_{tm}/{fam}_{m}"
        for tm in TEST_MODELS for fam in METRIC_FAMILIES for m in METRICS
    ]
    id_cols = ["run_id", "probe_id", "state", "created_at", "train_model",
               "variant", "layer", "seed", "lr", "probe_dtype", "normalize_before_head", "final_loss"]
    df_long = df_runs.melt(id_vars=id_cols, value_vars=metric_columns,
                           var_name="metric_key", value_name="value")
    parsed = df_long["metric_key"].str.extract(
        r"train/longfact_test_(apertus|llama)/(all|span|span_max)_(auc|f1|acc|recall_at_0\.1_fpr)"
    )
    parsed.columns = ["test_model", "metric_family", "metric"]
    df_long = pd.concat([df_long, parsed], axis=1)
    df_long = df_long.dropna(subset=["value", "test_model", "metric_family", "metric"]).copy()
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.dropna(subset=["value"]).reset_index(drop=True)
    df_long.to_csv(OUTPUT_DIR / "stability_runs_long.csv", index=False)
    print(f"  Saved long-format → extracted_data/stability_runs_long.csv")
    return df_runs, df_long


# ---------------------------------------------------------------------------
# 5. Final ablation W&B data  (final_ablation.ipynb)
# ---------------------------------------------------------------------------

def extract_final_ablation_runs():
    """
    Fetch final ablation runs (LoRA × LayerNorm × dtype × model × layer × seed)
    plus no-LoRA baselines and save:
      - final_ablation_runs.csv      : per-run metadata + summary metrics
      - final_ablation_runs_long.csv : tidy long format
    """
    print("\n[final_ablation] Fetching W&B final ablation runs ...")
    try:
        import wandb
    except ImportError:
        print("  wandb not installed.")
        return

    import re

    api = wandb.Api()
    ENTITY = "ethz-lsai-25"
    PROJECT = "hallucination-probes"
    RUN_PATH = f"{ENTITY}/{PROJECT}"

    LAYERS = [10, 20, 26, 30]
    SEEDS = [42, 43, 44]
    TEST_MODELS = ["apertus", "llama"]
    METRIC_FAMILIES = ["all", "span", "span_max"]
    METRICS = ["auc", "f1", "acc", "recall_at_0.1_fpr"]
    LOSS_KEYS = ["train/loss", "train/lm_loss", "train/kl_loss"]

    FINAL_VARIANT_META = {
        "no_lora_no_ln": {"normalize_before_head": "none",      "lora_layers": "none"},
        "no_lora_ln":    {"normalize_before_head": "layernorm", "lora_layers": "none"},
        "lora_no_ln":    {"normalize_before_head": "none",      "lora_layers": "all"},
        "lora_ln":       {"normalize_before_head": "layernorm", "lora_layers": "all"},
    }
    DTYPE_TAG_TO_FULL = {"fp32": "float32", "bf16": "bfloat16"}

    FINAL_PATTERN = re.compile(
        r"^(apertus|llama)_seed_ablation_(no_lora_no_ln|no_lora_ln|lora_no_ln|lora_ln)_(fp32|bf16)_layer(10|20|26|30)_seed(42|43|44)$"
    )
    BASELINE_PATTERN = re.compile(
        r"^(apertus|llama)_no_lora_long_form_layer(10|20|26|30)_seed(42|43|44)$"
    )

    def _safe_float(x):
        try: return float(x)
        except (TypeError, ValueError): return None

    def _canon_dtype(v):
        if v is None: return None
        k = str(v).strip().lower()
        if k in {"fp32", "float32", "torch.float32"}: return "float32"
        if k in {"bf16", "bfloat16", "torch.bfloat16"}: return "bfloat16"
        return k

    def _canon_norm(v):
        return str(v).strip().lower() if v is not None else None

    def _cfg_get(cfg, key, *, nested_parent=None, nested_child=None):
        v = cfg.get(key)
        if v is not None: return v
        if nested_parent and nested_child:
            p = cfg.get(nested_parent)
            if isinstance(p, dict): return p.get(nested_child)
        return None

    def _extract_id(run):
        cfg = run.config or {}
        for text in [run.name or ""] + [
            str(cfg.get(k, "")) for k in
            ["probe_config.probe_id", "probe_config/probe_id", "probe_id"]
        ]:
            text = text.strip()
            if FINAL_PATTERN.match(text) or BASELINE_PATTERN.match(text):
                return text
        cfg = run.config or {}
        if isinstance(cfg.get("probe_config"), dict):
            v = str(cfg["probe_config"].get("probe_id", "")).strip()
            if FINAL_PATTERN.match(v) or BASELINE_PATTERN.match(v):
                return v
        return None

    wandb_filter = {"display_name": {"$regex": r"^(apertus|llama)_(seed_ablation_|no_lora_long_form_)"}}
    rows = []
    for run in api.runs(RUN_PATH, filters=wandb_filter):
        selected_id = _extract_id(run)
        if not selected_id:
            continue
        m_final = FINAL_PATTERN.match(selected_id)
        m_base = BASELINE_PATTERN.match(selected_id)
        if not m_final and not m_base:
            continue

        cfg = run.config or {}
        summary = run.summary._json_dict

        if m_final:
            train_model, variant, dtype_tag, layer, seed = m_final.groups()
            run_kind = "final_ablation"
            expected_probe_dtype = DTYPE_TAG_TO_FULL[dtype_tag]
            expected_norm = _canon_norm(FINAL_VARIANT_META[variant]["normalize_before_head"])
        else:
            baseline_model, layer, seed = m_base.groups()
            train_model = baseline_model
            variant = "llama_baseline" if baseline_model == "llama" else "apertus_bf16_baseline"
            dtype_tag = "bf16"
            run_kind = "baseline"
            expected_probe_dtype = "bfloat16"
            expected_norm = None

        actual_probe_dtype = _canon_dtype(
            _cfg_get(cfg, "probe_config.probe_dtype", nested_parent="probe_config", nested_child="probe_dtype")
            or _cfg_get(cfg, "probe_config/probe_dtype") or _cfg_get(cfg, "probe_dtype")
        ) or expected_probe_dtype
        actual_norm = _canon_norm(
            _cfg_get(cfg, "probe_config.normalize_before_head", nested_parent="probe_config", nested_child="normalize_before_head")
            or _cfg_get(cfg, "probe_config/normalize_before_head") or _cfg_get(cfg, "normalize_before_head")
        )

        row = {
            "run_id": run.id, "run_name": run.name, "selected_id": selected_id,
            "state": run.state,
            "created_at": pd.to_datetime(run.created_at, utc=True),
            "run_kind": run_kind, "train_model": train_model,
            "variant": variant, "probe_dtype_tag": dtype_tag,
            "layer": int(layer), "seed": int(seed),
            "probe_dtype": actual_probe_dtype,
            "normalize_before_head": actual_norm,
            "probe_dtype_expected": expected_probe_dtype,
            "normalize_before_head_expected": expected_norm,
        }
        for k in LOSS_KEYS:
            row[k] = _safe_float(summary.get(k))
        for tm in TEST_MODELS:
            for fam in METRIC_FAMILIES:
                for metric in METRICS:
                    key = f"train/longfact_test_{tm}/{fam}_{metric}"
                    row[key] = _safe_float(summary.get(key))
        rows.append(row)

    df_runs = pd.DataFrame(rows)
    if df_runs.empty:
        print("  No matching final ablation runs found.")
        return

    state_rank = {"finished": 0, "running": 1, "queued": 2, "failed": 3, "crashed": 4}
    key_cols = ["run_kind", "train_model", "variant", "probe_dtype_tag", "layer", "seed"]
    metric_cols_all = [
        f"train/longfact_test_{tm}/{fam}_{m}"
        for tm in TEST_MODELS for fam in METRIC_FAMILIES for m in METRICS
    ]
    coalesce_cols = ["probe_dtype", "normalize_before_head"] + LOSS_KEYS + metric_cols_all

    df_runs["_state_rank"] = df_runs["state"].map(state_rank).fillna(99)
    df_runs["_has_loss"] = df_runs[LOSS_KEYS[0]].notna().astype(int)
    df_runs = df_runs.sort_values(
        key_cols + ["_state_rank", "_has_loss", "created_at"],
        ascending=[True] * len(key_cols) + [True, False, False],
    )

    # Backfill: for each (key) group, coalesce non-null values across all matching runs,
    # then keep one representative row. Mirrors the notebook's loss-backfill logic.
    loss_backfilled = 0
    merged_rows = []
    for _, group in df_runs.groupby(key_cols, sort=False):
        base = group.iloc[0].copy()
        had_missing_loss = pd.isna(base.get(LOSS_KEYS[0]))
        for col in coalesce_cols:
            if col not in group.columns:
                continue
            if pd.isna(base.get(col)):
                non_na = group[col].dropna()
                if not non_na.empty:
                    base[col] = non_na.iloc[0]
        if had_missing_loss and pd.notna(base.get(LOSS_KEYS[0])):
            loss_backfilled += 1
        merged_rows.append(base)

    df_runs = (
        pd.DataFrame(merged_rows)
        .drop(columns=["_state_rank", "_has_loss"], errors="ignore")
        .sort_values(["run_kind", "train_model", "variant", "layer", "seed"])
        .reset_index(drop=True)
    )
    print(f"  Loss backfilled from matching runs: {loss_backfilled}")
    print(f"  Rows with train/loss: {df_runs[LOSS_KEYS[0]].notna().sum()}/{len(df_runs)}")
    df_runs.to_csv(OUTPUT_DIR / "final_ablation_runs.csv", index=False)
    print(f"  Saved {len(df_runs)} runs → extracted_data/final_ablation_runs.csv")

    metric_columns = [
        f"train/longfact_test_{tm}/{fam}_{m}"
        for tm in TEST_MODELS for fam in METRIC_FAMILIES for m in METRICS
    ]
    id_cols = ["run_id", "run_name", "selected_id", "state", "created_at", "run_kind",
               "train_model", "variant", "probe_dtype_tag", "layer", "seed",
               "probe_dtype", "normalize_before_head",
               "probe_dtype_expected", "normalize_before_head_expected"] + LOSS_KEYS
    id_cols = [c for c in id_cols if c in df_runs.columns]
    df_long = df_runs.melt(id_vars=id_cols, value_vars=metric_columns,
                           var_name="metric_key", value_name="value")
    parsed = df_long["metric_key"].str.extract(
        r"train/longfact_test_(apertus|llama)/(all|span|span_max)_(auc|f1|acc|recall_at_0\.1_fpr)"
    )
    parsed.columns = ["test_model", "metric_family", "metric"]
    df_long = pd.concat([df_long, parsed], axis=1)
    df_long = df_long.dropna(subset=["value", "test_model", "metric_family", "metric"]).copy()
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.dropna(subset=["value"]).reset_index(drop=True)
    df_long.to_csv(OUTPUT_DIR / "final_ablation_runs_long.csv", index=False)
    print(f"  Saved long-format → extracted_data/final_ablation_runs_long.csv")
    return df_runs, df_long


# ---------------------------------------------------------------------------
# 6. Class balance statistics  (class_inbalance.ipynb)
# ---------------------------------------------------------------------------

def extract_class_balance_precomputed():
    """
    Save the token-level class balance statistics that were printed in
    class_inbalance.ipynb.  Values taken verbatim from notebook output.
    """
    print("\n[class_balance] Saving precomputed class balance statistics ...")

    stats = {
        "model": ["apertus", "llama"],
        "dataset_split": ["train", "train"],
        "hf_repo": [
            "tkwiecinski/longfact-test-split",
            "tkwiecinski/longfact-test-split",
        ],
        "subset": [
            "Apertus_8B_Instruct_2509",
            "Meta_Llama_3.1_8B_Instruct",
        ],
        "max_length": [1536, 1536],
        "positive_tokens":  [383343, 484040],
        "negative_tokens":  [12501121, 14509964],
        "ignored_tokens":   [14742032, 12591020],
        "total_tokens":     [27626496, 27585024],
        "labeled_tokens":   [12884464, 14994004],
        "hallucination_rate": [0.0298, 0.0323],
        "recommended_pos_weight": [32.6108, 29.9768],
    }

    df = pd.DataFrame(stats)
    df.to_csv(OUTPUT_DIR / "class_balance.csv", index=False)
    print(f"  Saved → extracted_data/class_balance.csv")
    return df


def run_class_balance_computation():
    """
    Re-run the class balance computation from class_inbalance.ipynb.
    Requires HuggingFace model access and tokenizer downloads.
    """
    print("\n[class_balance] Re-running class balance computation ...")
    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer
    from probe.dataset import create_probing_dataset, TokenizedProbingDatasetConfig

    results = []
    for model_key, model_name, subset, hf_repo in [
        ("apertus", "swiss-ai/Apertus-8B-Instruct-2509",     "Apertus_8B_Instruct_2509",    "tkwiecinski/longfact-test-split"),
        ("llama",   "meta-llama/Meta-Llama-3.1-8B-Instruct", "Meta_Llama_3.1_8B_Instruct",  "tkwiecinski/longfact-test-split"),
    ]:
        print(f"  Loading tokenizer: {model_key} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        cfg = TokenizedProbingDatasetConfig(
            dataset_id=f"longfact_train_{model_key}",
            hf_repo=hf_repo,
            subset=subset,
            split="train",
            max_length=1536,
            pos_weight=1.0,
            neg_weight=1.0,
            default_ignore=False,
            shuffle=False,
            seed=42,
            process_on_the_fly=False,
        )
        ds = create_probing_dataset(cfg, tokenizer)

        pos_tokens = neg_tokens = ignored_tokens = total_tokens = 0
        for i in range(len(ds)):
            labels = ds[i]["classification_labels"]
            pos_tokens     += (labels == 1.0).sum().item()
            neg_tokens     += (labels == 0.0).sum().item()
            ignored_tokens += (labels == -100.0).sum().item()
            total_tokens   += len(labels)

        labeled = pos_tokens + neg_tokens
        results.append({
            "model":                 model_key,
            "dataset_split":         "train",
            "hf_repo":               hf_repo,
            "subset":                subset,
            "max_length":            1536,
            "positive_tokens":       pos_tokens,
            "negative_tokens":       neg_tokens,
            "ignored_tokens":        ignored_tokens,
            "total_tokens":          total_tokens,
            "labeled_tokens":        labeled,
            "hallucination_rate":    pos_tokens / labeled if labeled > 0 else 0,
            "recommended_pos_weight": neg_tokens / max(pos_tokens, 1),
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "class_balance_recomputed.csv", index=False)
    print(f"  Saved → extracted_data/class_balance_recomputed.csv")
    return df


# ---------------------------------------------------------------------------
# 5. Dataset properties  (dataset_properties.ipynb)
# ---------------------------------------------------------------------------

# Mirrors dataset_properties.ipynb: both our dataset and the original
_LABEL_MAP = {
    "Not Supported": 1.0,
    "NS": 1.0,
    "Insufficient Information": 1.0,
    "Supported": 0.0,
    "S": 0.0,
    "N/A": -100.0,
    None: -100.0,
}

_OUR_REPO = "tkwiecinski/longfact-test-split"
_OUR_SUBSETS = ["Apertus_8B_Instruct_2509", "Meta_Llama_3.1_8B_Instruct"]
_ORIG_REPO = "obalcells/longfact-augmented-annotations"
_ORIG_SUBSETS = [
    "Mistral-Small-24B-Instruct-2501",
    "Qwen2.5-7B-Instruct",
    "gemma-2-9b-it",
    "Llama-3.3-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
]


def _parse_longfact_row(row: dict) -> dict:
    """Extract prompt, completion, and annotations from a raw HF row."""
    import hashlib

    conversation = row.get("conversation")
    prompt = completion = None

    if isinstance(conversation, list) and len(conversation) >= 2:
        u = conversation[-2]
        a = conversation[-1]
        if isinstance(u, dict):
            prompt = u.get("content")
        if isinstance(a, dict):
            completion = a.get("content")

    if completion is None:
        completion = row.get("completion")
    if prompt is None and isinstance(conversation, list) and conversation and isinstance(conversation[0], dict):
        prompt = conversation[0].get("content")

    raw_anns = row.get("annotations") or row.get("verified_entities") or []
    source = "annotations" if row.get("annotations") is not None else (
        "verified_entities" if row.get("verified_entities") is not None else "none"
    )

    annotations = []
    for entity in (raw_anns if isinstance(raw_anns, list) else []):
        if not isinstance(entity, dict):
            continue
        if source == "annotations":
            span = entity.get("span")
            index = entity.get("index")
        elif source == "verified_entities":
            span = entity.get("text")
            index = entity.get("idx")
        else:
            span = entity.get("span") or entity.get("text")
            index = entity.get("index", entity.get("idx"))
        label = entity.get("label")
        annotations.append({
            "span": span,
            "index": index,
            "label": label,
            "label_scalar": _LABEL_MAP.get(label),
        })

    prompt_hash = None
    if isinstance(prompt, str):
        prompt_hash = __import__("hashlib").md5(prompt.strip().encode()).hexdigest()

    return {
        "prompt": prompt,
        "completion": completion,
        "prompt_hash": prompt_hash,
        "annotation_source": source,
        "annotations": annotations,
    }


def _analyze_longfact_dataset(ds, repo: str, subset: str, split: str):
    """
    Per-row and per-span analysis mirroring dataset_properties.ipynb.
    Returns (rows_df, spans_df, issues_df).
    """
    rows, spans, issues = [], [], []

    for row_id, row in enumerate(ds):
        parsed = _parse_longfact_row(row)
        completion = parsed["completion"]
        annotations = parsed["annotations"]

        n_valid = n_invalid = 0
        for ann_id, ann in enumerate(annotations):
            span = ann["span"]
            index = ann["index"]
            label = ann["label"]
            label_scalar = ann["label_scalar"]

            issue = None
            matched_at_index = False
            found_anywhere = False

            if not isinstance(span, str) or not span:
                issue = "empty_or_nonstring_span"
            elif not isinstance(completion, str):
                issue = "completion_missing"
            elif isinstance(index, int):
                if index < 0 or index >= len(completion):
                    issue = "index_out_of_bounds"
                elif completion[index: index + len(span)] == span:
                    matched_at_index = True
                elif span in completion:
                    found_anywhere = True
                    issue = "span_not_matching_declared_index_but_found_elsewhere"
                else:
                    issue = "span_not_found_in_completion"
            else:
                if isinstance(completion, str) and span in completion:
                    found_anywhere = True
                    issue = "missing_or_nonint_index"
                else:
                    issue = "missing_or_nonint_index_and_span_not_found"

            if label_scalar is None and label not in _LABEL_MAP:
                issue = "unknown_label" if issue is None else f"{issue}|unknown_label"

            if issue is None:
                n_valid += 1
            else:
                n_invalid += 1
                issues.append({
                    "repo": repo, "subset": subset, "split": split,
                    "row_id": row_id, "ann_id": ann_id,
                    "issue": issue, "label": label, "index": index,
                    "span_preview": (span[:120] + "...") if isinstance(span, str) and len(span) > 120 else span,
                })

            spans.append({
                "repo": repo, "subset": subset, "split": split,
                "row_id": row_id, "ann_id": ann_id,
                "label_raw": label,
                "label_scalar": label_scalar if label_scalar is not None else float("nan"),
                "index": index if isinstance(index, int) else float("nan"),
                "span_len_chars": len(span) if isinstance(span, str) else float("nan"),
                "matched_at_index": matched_at_index,
                "found_anywhere": found_anywhere,
                "is_valid": issue is None,
            })

        rows.append({
            "repo": repo, "subset": subset, "split": split,
            "row_id": row_id,
            "prompt_hash": parsed["prompt_hash"],
            "prompt_len": len(parsed["prompt"]) if isinstance(parsed["prompt"], str) else float("nan"),
            "completion_len": len(completion) if isinstance(completion, str) else float("nan"),
            "annotation_source": parsed["annotation_source"],
            "n_annotations": len(annotations),
            "n_valid_spans": n_valid,
            "n_invalid_spans": n_invalid,
        })

    return pd.DataFrame(rows), pd.DataFrame(spans), pd.DataFrame(issues)


def _aggregate_dataset_summary(rows_df: pd.DataFrame, spans_df: pd.DataFrame, issues_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (repo, subset, split), grp_rows in rows_df.groupby(["repo", "subset", "split"]):
        grp_spans = spans_df[
            (spans_df.repo == repo) & (spans_df.subset == subset) & (spans_df.split == split)
        ]
        grp_issues = issues_df[
            (issues_df.repo == repo) & (issues_df.subset == subset) & (issues_df.split == split)
        ] if not issues_df.empty else pd.DataFrame()

        n_sp = len(grp_spans)
        n_valid = int(grp_spans["is_valid"].sum()) if n_sp else 0

        out.append({
            "repo": repo, "subset": subset, "split": split,
            "rows": len(grp_rows),
            "rows_empty_annotations": int((grp_rows["n_annotations"] == 0).sum()),
            "rows_no_valid_spans": int((grp_rows["n_valid_spans"] == 0).sum()),
            "rows_any_invalid_spans": int((grp_rows["n_invalid_spans"] > 0).sum()),
            "spans_total": n_sp,
            "spans_per_row": (n_sp / len(grp_rows)) if len(grp_rows) else float("nan"),
            "spans_invalid": n_sp - n_valid,
            "spans_invalid_rate": ((n_sp - n_valid) / n_sp) if n_sp else float("nan"),
            "label_pos(1.0)": int((grp_spans["label_scalar"] == 1.0).sum()) if n_sp else 0,
            "label_neg(0.0)": int((grp_spans["label_scalar"] == 0.0).sum()) if n_sp else 0,
            "label_ignore(-100)": int((grp_spans["label_scalar"] == -100.0).sum()) if n_sp else 0,
            "label_unknown": int(grp_spans["label_scalar"].isna().sum()) if n_sp else 0,
            "completion_len_mean": float(grp_rows["completion_len"].mean()),
            "completion_len_median": float(grp_rows["completion_len"].median()),
            "span_len_mean": float(grp_spans["span_len_chars"].mean()) if n_sp else float("nan"),
            "span_len_median": float(grp_spans["span_len_chars"].median()) if n_sp else float("nan"),
            "span_len_p90": float(grp_spans["span_len_chars"].quantile(0.9)) if n_sp else float("nan"),
            "top_issue": grp_issues["issue"].value_counts().index[0] if len(grp_issues) else None,
            "top_issue_count": int(grp_issues["issue"].value_counts().iloc[0]) if len(grp_issues) else 0,
        })
    return pd.DataFrame(out).sort_values(["repo", "subset", "split"]).reset_index(drop=True)


def extract_dataset_properties():
    """
    Analyze both our dataset and the original one and save:
      - dataset_properties_rows.csv      : per-row statistics
      - dataset_properties_spans.csv     : per-span statistics
      - dataset_properties_issues.csv    : invalid-span records
      - dataset_properties_summary.csv   : aggregate summary table
    Mirrors the analysis in dataset_properties.ipynb.
    """
    print("\n[dataset_properties] Fetching HuggingFace dataset statistics ...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not installed. Run: pip install datasets")
        return

    targets = (
        [(_OUR_REPO, s, sp) for s in _OUR_SUBSETS for sp in ["train", "test"]] +
        [(_ORIG_REPO, s, sp) for s in _ORIG_SUBSETS for sp in ["train", "test"]]
    )

    rows_parts, spans_parts, issues_parts = [], [], []
    for repo, subset, split in targets:
        try:
            ds = load_dataset(repo, subset, split=split)
            print(f"  Loaded {repo} | {subset} | {split}: {len(ds)} rows")
        except Exception as e:
            print(f"  Could not load {repo} | {subset} | {split}: {e}")
            continue
        r, s, i = _analyze_longfact_dataset(ds, repo=repo, subset=subset, split=split)
        rows_parts.append(r)
        spans_parts.append(s)
        issues_parts.append(i)

    if not rows_parts:
        print("  No datasets loaded.")
        return

    rows_df = pd.concat(rows_parts, ignore_index=True)
    spans_df = pd.concat(spans_parts, ignore_index=True)
    issues_df = pd.concat(issues_parts, ignore_index=True) if issues_parts else pd.DataFrame()
    summary_df = _aggregate_dataset_summary(rows_df, spans_df, issues_df)

    rows_df.to_csv(OUTPUT_DIR / "dataset_properties_rows.csv", index=False)
    print(f"  Saved {len(rows_df)} row-level entries → extracted_data/dataset_properties_rows.csv")

    spans_df.to_csv(OUTPUT_DIR / "dataset_properties_spans.csv", index=False)
    print(f"  Saved {len(spans_df)} span-level entries → extracted_data/dataset_properties_spans.csv")

    if not issues_df.empty:
        issues_df.to_csv(OUTPUT_DIR / "dataset_properties_issues.csv", index=False)
        print(f"  Saved {len(issues_df)} issue records → extracted_data/dataset_properties_issues.csv")

    summary_df.to_csv(OUTPUT_DIR / "dataset_properties_summary.csv", index=False)
    print(f"  Saved summary ({len(summary_df)} rows) → extracted_data/dataset_properties_summary.csv")

    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Jupyter experiment data to CSV/numpy.")
    parser.add_argument("--output-dir",        type=Path, default=None,
                        help="Directory to write all output files (default: jupyter_experiments/extracted_data/)")
    parser.add_argument("--skip-activations",    action="store_true", help="Skip all activation steps (no GPU needed)")
    parser.add_argument("--skip-wandb",          action="store_true", help="Skip ALL W&B extraction (layers + plots + stability + final_ablation)")
    parser.add_argument("--skip-layers",         action="store_true", help="Skip layers W&B extraction")
    parser.add_argument("--skip-plots",          action="store_true", help="Skip plots W&B extraction")
    parser.add_argument("--skip-stability",      action="store_true", help="Skip stability sweep W&B extraction")
    parser.add_argument("--skip-final-ablation", action="store_true", help="Skip final ablation W&B extraction")
    parser.add_argument("--skip-hf",             action="store_true", help="Skip HuggingFace dataset fetching")
    parser.add_argument("--run-activations",     action="store_true", help="Re-run full activation collection (GPU required)")
    parser.add_argument("--run-class-balance",   action="store_true", help="Re-run class balance computation (model download required)")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving all output to: {OUTPUT_DIR.resolve()}")

    # --- activations.ipynb ---
    if args.skip_activations:
        print("\n[activations] Skipped (--skip-activations).")
    elif args.run_activations:
        run_activation_collection()
    else:
        extract_activation_metrics_precomputed()

    # --- layers.ipynb ---
    if not args.skip_wandb and not args.skip_layers:
        extract_wandb_layers()
    else:
        print("\n[layers] Skipped.")

    # --- plots.ipynb ---
    if not args.skip_wandb and not args.skip_plots:
        extract_wandb_plots()
    else:
        print("\n[plots] Skipped.")

    # --- improve_stability.ipynb ---
    if not args.skip_wandb and not args.skip_stability:
        extract_stability_runs()
    else:
        print("\n[stability] Skipped.")

    # --- final_ablation.ipynb ---
    if not args.skip_wandb and not args.skip_final_ablation:
        extract_final_ablation_runs()
    else:
        print("\n[final_ablation] Skipped.")

    # --- class_inbalance.ipynb ---
    if args.run_class_balance:
        run_class_balance_computation()
    else:
        extract_class_balance_precomputed()

    # --- dataset_properties.ipynb ---
    if not args.skip_hf:
        extract_dataset_properties()
    else:
        print("\n[dataset_properties] Skipped (--skip-hf).")

    print(f"\nDone. All files saved to: {OUTPUT_DIR.resolve()}")
    print("\nFiles written:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        unit = "B"
        if size > 1_000_000:
            size /= 1_000_000; unit = "MB"
        elif size > 1_000:
            size /= 1_000; unit = "KB"
        print(f"  {f.name:<55} {size:6.1f} {unit}")


if __name__ == "__main__":
    main()
