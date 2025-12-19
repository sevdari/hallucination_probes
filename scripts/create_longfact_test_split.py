#!/usr/bin/env python3
"""Create a stratified 90/10 train/test split from a Hugging Face dataset and push to the Hub.

This script is designed for the case where you have multiple *subsets/configs*
constructed from the same underlying datapoints (e.g., ablations). In that case,
it is critical that the exact same prompts land in the same split across all
configs.

How it works:
- We define a *prompt group key* using the `conversation` field: the ordered list
    of message contents where role == "user".
- We create the train/test split at the prompt-group level (never splitting a
    prompt group across partitions).
- We stratify prompt-groups by `topic` and `difficulty`.
- We compute the split mapping once from a reference config, then apply that
    mapping to all other configs.

Mismatch handling:
- If another config has *extra* prompt-groups (not present in the reference),
  those groups are assigned to `train`.
- If a prompt-group key exists in both configs but the canonical user text differs
  (unexpected), that group is forced to `train` in that config.

Default input:
    repo:   tymciurymciu/longfact-annotated
    config: Meta_Llama_3.1_8B_Instruct
    split:  train

Default output:
    repo:   tymciurymciu/longfact-test-split
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, load_dataset


DEFAULT_INPUT_REPO = "tymciurymciu/longfact-annotated"
DEFAULT_INPUT_CONFIG = "Meta_Llama_3.1_8B_Instruct"
DEFAULT_INPUT_SPLIT = "train"

DEFAULT_OUTPUT_REPO = "tymciurymciu/longfact-test-split"


def _require_columns(dataset: Dataset, columns: List[str]) -> None:
    missing = [c for c in columns if c not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {dataset.column_names}"
        )


def _make_strata(dataset: Dataset, topic_col: str, difficulty_col: str) -> List[str]:
    # Using `str()` ensures consistent keys for e.g. ints/enums.
    topics = dataset[topic_col]
    difficulties = dataset[difficulty_col]
    return [f"{str(t)}||{str(d)}" for t, d in zip(topics, difficulties)]


def _extract_user_contents(conversation: Sequence[dict]) -> List[str]:
    if not isinstance(conversation, (list, tuple)):
        raise TypeError(f"conversation must be a list of messages, got {type(conversation)}")

    user_contents: List[str] = []
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            content = msg.get("content")
            if content is None:
                continue
            user_contents.append(str(content))

    if not user_contents:
        raise ValueError("conversation contains no user messages (role=='user')")

    return user_contents


def _prompt_group_key_from_user_contents(user_contents: Sequence[str]) -> str:
    # Stable key for potentially long prompts; ensures identical prompts map identically.
    joined = "\n\n".join([c.strip() for c in user_contents])
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _prompt_group_keys(dataset: Dataset, conversation_col: str) -> Tuple[List[str], Dict[str, str]]:
    """Return per-row group keys and a mapping key->canonical_user_text (for debugging)."""
    keys: List[str] = []
    canonical: Dict[str, str] = {}
    for conv in dataset[conversation_col]:
        user_contents = _extract_user_contents(conv)
        joined = "\n\n".join([c.strip() for c in user_contents])
        key = _prompt_group_key_from_user_contents(user_contents)
        keys.append(key)
        # Store one representative prompt text per key.
        canonical.setdefault(key, joined)
    return keys, canonical


def _build_groups(
    prompt_keys: Sequence[str],
    strata: Sequence[str],
) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """Return (group_to_indices, group_to_stratum)."""
    if len(prompt_keys) != len(strata):
        raise ValueError("prompt_keys and strata must have the same length")

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    group_to_stratum: Dict[str, str] = {}

    for idx, (g, s) in enumerate(zip(prompt_keys, strata)):
        group_to_indices[g].append(idx)
        if g in group_to_stratum and group_to_stratum[g] != s:
            # If this happens, stratification is ambiguous; we keep the first and continue.
            # This should not happen in a clean dataset.
            continue
        group_to_stratum[g] = s

    return dict(group_to_indices), group_to_stratum


def stratified_group_split(
    group_to_indices: Dict[str, List[int]],
    group_to_stratum: Dict[str, str],
    test_size: float,
    seed: int,
) -> Dict[str, str]:
    """Return mapping: prompt_group_key -> {"train"|"test"}.

    Split happens at the group level (all rows for a prompt group stay together).
    Stratification is approximate, performed by stratum on *row counts*.
    """

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")

    rng = random.Random(seed)

    # Organize groups by stratum.
    groups_by_stratum: Dict[str, List[str]] = defaultdict(list)
    for group_key, stratum in group_to_stratum.items():
        groups_by_stratum[stratum].append(group_key)

    # Deterministic shuffle per stratum.
    for s in groups_by_stratum:
        rng.shuffle(groups_by_stratum[s])

    total_rows = sum(len(idxs) for idxs in group_to_indices.values())
    desired_test_rows = int(round(total_rows * test_size))

    split: Dict[str, str] = {g: "train" for g in group_to_indices.keys()}

    # Initial allocation per stratum.
    test_rows = 0
    for s, groups in groups_by_stratum.items():
        if len(groups) <= 1:
            # With <=1 group in a stratum, keep it in train to avoid empty-train strata.
            continue

        stratum_rows = sum(len(group_to_indices[g]) for g in groups)
        desired_stratum_test_rows = int(round(stratum_rows * test_size))
        desired_stratum_test_rows = min(desired_stratum_test_rows, stratum_rows - 1)
        desired_stratum_test_rows = max(desired_stratum_test_rows, 0)

        current = 0
        for g in groups:
            if current >= desired_stratum_test_rows:
                break
            # Don't allow the stratum to have no train rows.
            if stratum_rows - (current + len(group_to_indices[g])) <= 0:
                continue
            split[g] = "test"
            current += len(group_to_indices[g])

        test_rows += current

    # Global adjustment towards desired_test_rows by moving whole groups.
    # We only move within strata that keep at least 1 train row.
    def _can_move_train_to_test(group_key: str) -> bool:
        s = group_to_stratum[group_key]
        groups = groups_by_stratum[s]
        # Would moving this group make stratum have 0 train rows?
        stratum_rows = sum(len(group_to_indices[g]) for g in groups)
        current_test = sum(len(group_to_indices[g]) for g in groups if split[g] == "test")
        return (split[group_key] == "train") and (stratum_rows - (current_test + len(group_to_indices[group_key])) > 0)

    def _can_move_test_to_train(group_key: str) -> bool:
        return split[group_key] == "test"

    # Try to increase test rows.
    if test_rows < desired_test_rows:
        need = desired_test_rows - test_rows
        candidates = [g for g in split.keys() if _can_move_train_to_test(g)]
        rng.shuffle(candidates)
        for g in candidates:
            if need <= 0:
                break
            split[g] = "test"
            need -= len(group_to_indices[g])

    # Try to decrease test rows.
    test_rows = sum(len(group_to_indices[g]) for g, v in split.items() if v == "test")
    if test_rows > desired_test_rows:
        extra = test_rows - desired_test_rows
        candidates = [g for g in split.keys() if _can_move_test_to_train(g)]
        rng.shuffle(candidates)
        for g in candidates:
            if extra <= 0:
                break
            split[g] = "train"
            extra -= len(group_to_indices[g])

    # Final safety: ensure every stratum has at least 1 train row.
    for s, groups in groups_by_stratum.items():
        stratum_rows = sum(len(group_to_indices[g]) for g in groups)
        if stratum_rows == 0:
            continue
        train_rows = sum(len(group_to_indices[g]) for g in groups if split[g] == "train")
        if train_rows == 0:
            # Move the smallest test group back to train.
            test_groups = [g for g in groups if split[g] == "test"]
            if not test_groups:
                continue
            smallest = min(test_groups, key=lambda g: len(group_to_indices[g]))
            split[smallest] = "train"

    return split


def _indices_from_group_split(prompt_keys: Sequence[str], group_split: Dict[str, str]) -> Tuple[List[int], List[int]]:
    train_idx: List[int] = []
    test_idx: List[int] = []
    for idx, g in enumerate(prompt_keys):
        # Default unknown prompt groups to train. This can happen when processing
        # configs that contain extra prompts not present in the reference.
        part = group_split.get(g, "train")
        if part == "train":
            train_idx.append(idx)
        elif part == "test":
            test_idx.append(idx)
        else:
            raise ValueError(f"Invalid split value for group {g}: {part}")
    return train_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/test split and push to Hugging Face")
    parser.add_argument("--input-repo", default=DEFAULT_INPUT_REPO)
    parser.add_argument(
        "--input-config",
        default=DEFAULT_INPUT_CONFIG,
        help="Single input config (subset). Use --input-configs for multiple.",
    )
    parser.add_argument(
        "--input-configs",
        nargs="+",
        default=None,
        help="One or more input configs (subsets) to split and push.",
    )
    parser.add_argument(
        "--reference-config",
        default=None,
        help="Config used to compute the canonical split mapping. Defaults to first in --input-configs (or --input-config).",
    )
    parser.add_argument("--input-split", default=DEFAULT_INPUT_SPLIT)

    parser.add_argument("--output-repo", default=DEFAULT_OUTPUT_REPO)
    parser.add_argument(
        "--output-config",
        default=None,
        help="Output config name to use for ALL pushes. If omitted, uses each input config name.",
    )

    parser.add_argument("--topic-col", default="topic")
    parser.add_argument("--difficulty-col", default="difficulty")
    parser.add_argument("--conversation-col", default="conversation")

    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--private", action="store_true", help="Push as a private dataset")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do everything except push to the Hub (prints summary only)",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_WRITE_TOKEN",
        help="Environment variable name that contains a HF write token",
    )

    args = parser.parse_args()

    input_configs = args.input_configs if args.input_configs else [args.input_config]
    reference_config = args.reference_config or input_configs[0]
    if reference_config not in input_configs:
        raise ValueError("--reference-config must be one of --input-configs")

    print(f"Reference config: {reference_config}")
    print(f"Input configs: {input_configs}")

    print(f"Loading reference dataset: {args.input_repo} (config={reference_config}, split={args.input_split})")
    ref_ds = load_dataset(args.input_repo, reference_config, split=args.input_split)

    _require_columns(ref_ds, [args.topic_col, args.difficulty_col, args.conversation_col])

    ref_strata = _make_strata(ref_ds, args.topic_col, args.difficulty_col)
    ref_prompt_keys, ref_prompt_text = _prompt_group_keys(ref_ds, args.conversation_col)
    ref_group_to_indices, ref_group_to_stratum = _build_groups(ref_prompt_keys, ref_strata)

    group_split = stratified_group_split(
        ref_group_to_indices,
        ref_group_to_stratum,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Now apply mapping to each config.
    split_summaries: List[Tuple[str, int, int, int]] = []
    prepared: List[Tuple[str, Dataset, Dataset]] = []

    for cfg in input_configs:
        print(f"\nLoading dataset: {args.input_repo} (config={cfg}, split={args.input_split})")
        ds = load_dataset(args.input_repo, cfg, split=args.input_split)
        _require_columns(ds, [args.topic_col, args.difficulty_col, args.conversation_col])

        strata = _make_strata(ds, args.topic_col, args.difficulty_col)
        prompt_keys, prompt_text = _prompt_group_keys(ds, args.conversation_col)

        # Compare prompt groups vs reference, but do not fail hard.
        # Extra/mismatched prompts in non-reference configs should go to TRAIN.
        ref_groups = set(ref_group_to_indices.keys())
        this_groups = set(prompt_text.keys())

        missing = ref_groups - this_groups
        extra = this_groups - ref_groups
        if missing or extra:
            print(
                "WARNING: Prompt-group mismatch vs reference; defaulting unknown groups to TRAIN. "
                f"config={cfg} missing={len(missing)} extra={len(extra)}"
            )

        # Additional safety: ensure canonical user text matches for keys present in both.
        # If it doesn't (unexpected), force that key to TRAIN for this config.
        shared = ref_groups & this_groups
        mismatched = [k for k in shared if ref_prompt_text.get(k) != prompt_text.get(k)]
        if mismatched:
            sample = mismatched[0]
            ref_preview = (ref_prompt_text.get(sample) or "")[:200]
            cur_preview = (prompt_text.get(sample) or "")[:200]
            print(
                "WARNING: User prompt text mismatch for at least one shared prompt-group key; "
                f"forcing mismatches to TRAIN for this config. Example key={sample}\nref={ref_preview}\ncur={cur_preview}"
            )

        effective_split = dict(group_split)
        for k in extra:
            effective_split[k] = "train"
        for k in mismatched:
            effective_split[k] = "train"

        train_idx, test_idx = _indices_from_group_split(prompt_keys, effective_split)

        train_ds = ds.select(train_idx)
        test_ds = ds.select(test_idx)
        prepared.append((cfg, train_ds, test_ds))
        split_summaries.append((cfg, len(ds), len(train_ds), len(test_ds)))

    print("\nSplit summary (per config):")
    for cfg, total, n_train, n_test in split_summaries:
        print(f"  {cfg}: total={total} train={n_train} ({n_train / max(1, total):.3%}) test={n_test} ({n_test / max(1, total):.3%})")

    if args.no_push:
        print("--no-push set; skipping push_to_hub.")
        return

    token = os.environ.get(args.hf_token_env)
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token. Set {args.hf_token_env} to a write token, "
            "or run with --no-push to only create the split locally."
        )

    print(f"Pushing to Hugging Face: {args.output_repo}")
    # Push both splits for each config_name.
    for cfg, train_ds, test_ds in prepared:
        out_config = args.output_config or cfg
        print(f"  - config={out_config}: pushing train/test")
        train_ds.push_to_hub(
            args.output_repo,
            split="train",
            config_name=out_config,
            private=args.private,
            token=token,
        )
        test_ds.push_to_hub(
            args.output_repo,
            split="test",
            config_name=out_config,
            private=args.private,
            token=token,
        )

    print("Done.")


if __name__ == "__main__":
    main()
