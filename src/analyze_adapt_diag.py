#!/usr/bin/env python3
"""Paired base-vs-D3 critic degradation analysis for the single-point diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.critic_model import LABEL_STATES, LABEL_TO_ID, POLICY_IDS
from src.eval_critic import binary_violated_f1, multiclass_ece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_labels", required=True)
    parser.add_argument("--adapted_labels", required=True)
    parser.add_argument("--base_logits", required=True)
    parser.add_argument("--adapted_logits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def load_variant(labels_path: str, logits_path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels_rows = read_jsonl(labels_path)
    logits_rows = read_jsonl(logits_path)
    if [row["id"] for row in labels_rows] != [row["id"] for row in logits_rows]:
        raise ValueError("teacher labels and critic logits are not ID-aligned")
    labels = np.asarray(
        [[LABEL_TO_ID[row["labels"][policy]] for policy in POLICY_IDS] for row in labels_rows]
    )
    logits = np.asarray(
        [[row["logits"][policy] for policy in POLICY_IDS] for row in logits_rows],
        dtype=np.float64,
    )
    return [row["id"] for row in labels_rows], labels, softmax(logits)


def policy_metrics(labels: np.ndarray, probabilities: np.ndarray, index: int) -> dict[str, float]:
    truth = labels[:, index]
    predictions = probabilities[:, index].argmax(axis=-1)
    applicable = truth != LABEL_TO_ID["not_applicable"]
    violated = truth == LABEL_TO_ID["violated"]
    satisfied = truth == LABEL_TO_ID["satisfied"]
    both_classes = violated.any() and satisfied.any()
    return {
        "ece": multiclass_ece(probabilities[:, index], truth, n_bins=15),
        "violated_f1": (
            binary_violated_f1(truth[applicable], predictions[applicable])
            if both_classes else math.nan
        ),
        "fn_rate": (
            float(np.mean(predictions[violated] != LABEL_TO_ID["violated"]))
            if violated.any() else math.nan
        ),
        "fp_rate": (
            float(np.mean(predictions[satisfied] == LABEL_TO_ID["violated"]))
            if satisfied.any() else math.nan
        ),
        "n_satisfied": int(satisfied.sum()),
        "n_violated": int(violated.sum()),
        "n_na": int((~applicable).sum()),
    }


def percentile(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [math.nan, math.nan]


def signed_cv(values: np.ndarray) -> float:
    mean = float(values.mean())
    return math.nan if mean <= 0 else float(values.std(ddof=0) / mean)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    base_ids, base_labels, base_probs = load_variant(args.base_labels, args.base_logits)
    adapted_ids, adapted_labels, adapted_probs = load_variant(
        args.adapted_labels, args.adapted_logits
    )
    if base_ids != adapted_ids or len(base_ids) != 1000:
        raise ValueError("base/adapted variants must share the same 1000 ordered prompt IDs")

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, len(base_ids), size=(args.bootstrap, len(base_ids)))
    result = {"per_policy": {}}
    point_deltas = {metric: [] for metric in ("ece", "violated_f1", "fn_rate", "fp_rate")}
    bootstrap_deltas = {
        metric: np.empty((args.bootstrap, len(POLICY_IDS)), dtype=np.float64)
        for metric in point_deltas
    }
    for policy_index, policy in enumerate(POLICY_IDS):
        base = policy_metrics(base_labels, base_probs, policy_index)
        adapted = policy_metrics(adapted_labels, adapted_probs, policy_index)
        policy_result = {"base": base, "adapted": adapted, "delta": {}, "bootstrap_95ci": {}}
        base_boot = {metric: np.empty(args.bootstrap) for metric in point_deltas}
        adapted_boot = {metric: np.empty(args.bootstrap) for metric in point_deltas}
        for replicate, indices in enumerate(bootstrap_indices):
            b = policy_metrics(base_labels[indices], base_probs[indices], policy_index)
            a = policy_metrics(adapted_labels[indices], adapted_probs[indices], policy_index)
            for metric in point_deltas:
                base_boot[metric][replicate] = b[metric]
                adapted_boot[metric][replicate] = a[metric]
                bootstrap_deltas[metric][replicate, policy_index] = a[metric] - b[metric]
        for metric in point_deltas:
            delta = adapted[metric] - base[metric]
            point_deltas[metric].append(delta)
            policy_result["delta"][metric] = delta
            policy_result["bootstrap_95ci"][f"base_{metric}"] = percentile(base_boot[metric])
            policy_result["bootstrap_95ci"][f"adapted_{metric}"] = percentile(adapted_boot[metric])
            policy_result["bootstrap_95ci"][f"delta_{metric}"] = percentile(
                bootstrap_deltas[metric][:, policy_index]
            )
        result["per_policy"][policy] = policy_result

    delta_ece = np.asarray(point_deltas["ece"])
    delta_cv_boot = np.asarray(
        [signed_cv(bootstrap_deltas["ece"][replicate]) for replicate in range(args.bootstrap)]
    )
    directional = np.asarray(point_deltas["fn_rate"]) - np.asarray(point_deltas["fp_rate"])
    directional_boot = np.nanmean(
        bootstrap_deltas["fn_rate"] - bootstrap_deltas["fp_rate"], axis=1
    )
    result.update(
        {
            "n_paired_prompts": len(base_ids),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "delta_ece": {
                "per_policy": dict(zip(POLICY_IDS, delta_ece.tolist())),
                "mean": float(delta_ece.mean()),
                "std": float(delta_ece.std(ddof=0)),
                "signed_cv": signed_cv(delta_ece),
                "signed_cv_95ci": percentile(delta_cv_boot),
                "base_original_f1_cv_reference": 0.08056771993372937,
            },
            "fn_asymmetry_direction": {
                "mean_delta_fn": float(np.nanmean(point_deltas["fn_rate"])),
                "mean_delta_fp": float(np.nanmean(point_deltas["fp_rate"])),
                "mean_delta_fn_minus_delta_fp": float(np.nanmean(directional)),
                "mean_delta_fn_minus_delta_fp_95ci": percentile(directional_boot),
                "policies_delta_fn_gt_delta_fp": int(np.sum(directional > 0)),
                "note": "diagnostic direction only; not a G2 verdict",
            },
        }
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_result = json_safe(result)
    output.write_text(json.dumps(safe_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"delta_ece": safe_result["delta_ece"], "fn": safe_result["fn_asymmetry_direction"]}, indent=2))


if __name__ == "__main__":
    main()
