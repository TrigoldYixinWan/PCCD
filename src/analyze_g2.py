#!/usr/bin/env python3
"""Compute the locked, paired full-grid G2 metrics and gate verdict."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic_model import LABEL_TO_ID, POLICY_IDS
from src.eval_critic import binary_violated_f1, multiclass_ece

METRICS = ("ece", "violated_f1", "fn_rate", "fp_rate")


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


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
        [[LABEL_TO_ID[row["labels"][policy]] for policy in POLICY_IDS] for row in labels_rows],
        dtype=np.int8,
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
            if both_classes
            else math.nan
        ),
        "fn_rate": (
            float(np.mean(predictions[violated] != LABEL_TO_ID["violated"]))
            if violated.any()
            else math.nan
        ),
        "fp_rate": (
            float(np.mean(predictions[satisfied] == LABEL_TO_ID["violated"]))
            if satisfied.any()
            else math.nan
        ),
        "n_satisfied": int(satisfied.sum()),
        "n_violated": int(violated.sum()),
        "n_na": int((~applicable).sum()),
    }


def percentile(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [math.nan, math.nan]


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_variant(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("variant must be NAME,LABELS,LOGITS,KL_JSON")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_labels", required=True)
    parser.add_argument("--base_logits", required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum_violated", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    names = [spec[0] for spec in args.variant]
    if len(names) != len(set(names)) or not {"D3_control", "D5", "D6"} <= set(names):
        raise ValueError("variants must be unique and include D3_control, D5, and D6")
    base_ids, base_labels, base_probabilities = load_variant(args.base_labels, args.base_logits)
    if not base_ids:
        raise ValueError("empty base evaluation set")
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, len(base_ids), size=(args.bootstrap, len(base_ids)), dtype=np.int32
    )
    base_point = [
        policy_metrics(base_labels, base_probabilities, index)
        for index in range(len(POLICY_IDS))
    ]
    result = {
        "definition": {
            "delta": "adapted minus D0 on identical fixed prompts",
            "ece": "15-bin top-class 3-way ECE including N/A",
            "violated_f1": "violated-positive; teacher N/A excluded",
            "fn": "P(pred != violated | teacher violated)",
            "fp": "P(pred = violated | teacher satisfied)",
            "adequately_powered": f"base and adapted teacher-violated support >= {args.minimum_violated}",
            "gate_rule_frozen_before_run": "G2 PASS iff at least one of D5 or D6 passes both (a) and (b); both point verdicts are reported",
        },
        "n_paired_prompts": len(base_ids),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "base": {policy: base_point[i] for i, policy in enumerate(POLICY_IDS)},
        "variants": {},
    }

    for name, labels_path, logits_path, kl_path in args.variant:
        adapted_ids, adapted_labels, adapted_probabilities = load_variant(labels_path, logits_path)
        if adapted_ids != base_ids:
            raise ValueError(f"{name} is not exactly paired with D0")
        kl = json.loads(Path(kl_path).read_text(encoding="utf-8"))
        point_deltas = {metric: np.full(len(POLICY_IDS), np.nan) for metric in METRICS}
        boot_deltas = {
            metric: np.full((args.bootstrap, len(POLICY_IDS)), np.nan, dtype=np.float64)
            for metric in METRICS
        }
        per_policy = {}
        powered = []
        for policy_index, policy in enumerate(POLICY_IDS):
            base = base_point[policy_index]
            adapted = policy_metrics(adapted_labels, adapted_probabilities, policy_index)
            is_powered = (
                base["n_violated"] >= args.minimum_violated
                and adapted["n_violated"] >= args.minimum_violated
            )
            if is_powered:
                powered.append(policy)
            policy_result = {
                "base": base,
                "adapted": adapted,
                "adequately_powered": is_powered,
                "underpowered_reason": None if is_powered else "teacher-violated support <30 in base or adapted",
                "delta": {},
                "delta_95ci": {},
            }
            for metric in METRICS:
                point_deltas[metric][policy_index] = adapted[metric] - base[metric]
            for replicate, indices in enumerate(bootstrap_indices):
                base_boot = policy_metrics(
                    base_labels[indices], base_probabilities[indices], policy_index
                )
                adapted_boot = policy_metrics(
                    adapted_labels[indices], adapted_probabilities[indices], policy_index
                )
                for metric in METRICS:
                    boot_deltas[metric][replicate, policy_index] = (
                        adapted_boot[metric] - base_boot[metric]
                    )
            for metric in METRICS:
                policy_result["delta"][metric] = point_deltas[metric][policy_index]
                policy_result["delta_95ci"][metric] = percentile(
                    boot_deltas[metric][:, policy_index]
                )
            per_policy[policy] = policy_result

        delta_ece = point_deltas["ece"]
        mean_ece_boot = np.nanmean(boot_deltas["ece"], axis=1)
        rms_boot = np.sqrt(np.nanmean(np.square(boot_deltas["ece"]), axis=1))
        sd_boot = np.nanstd(boot_deltas["ece"], axis=1, ddof=0)
        powered_indices = [POLICY_IDS.index(policy) for policy in powered]
        if powered_indices:
            direction = point_deltas["fn_rate"][powered_indices] - point_deltas["fp_rate"][powered_indices]
            direction_boot = np.nanmean(
                boot_deltas["fn_rate"][:, powered_indices]
                - boot_deltas["fp_rate"][:, powered_indices],
                axis=1,
            )
            asymmetry = float(np.nanmean(direction))
            asymmetry_ci = percentile(direction_boot)
            positive_count = int(np.sum(direction > 0))
        else:
            asymmetry = math.nan
            asymmetry_ci = [math.nan, math.nan]
            positive_count = 0
        mean_delta_ece = float(np.mean(delta_ece))
        mean_delta_ece_ci = percentile(mean_ece_boot)
        g2a = mean_delta_ece > 0 and mean_delta_ece_ci[0] > 0
        g2b = asymmetry > 0 and asymmetry_ci[0] > 0
        variant_result = {
            "kl": kl,
            "powered_policies": powered,
            "underpowered_policies": [p for p in POLICY_IDS if p not in powered],
            "per_policy": per_policy,
            "aggregate": {
                "mean_delta_ece": mean_delta_ece,
                "mean_delta_ece_95ci": mean_delta_ece_ci,
                "rms_delta_ece": float(np.sqrt(np.mean(np.square(delta_ece)))),
                "rms_delta_ece_95ci": percentile(rms_boot),
                "sd_delta_ece": float(np.std(delta_ece, ddof=0)),
                "sd_delta_ece_95ci": percentile(sd_boot),
                "mean_delta_fn_minus_delta_fp_powered": asymmetry,
                "mean_delta_fn_minus_delta_fp_95ci": asymmetry_ci,
                "policies_delta_fn_gt_delta_fp": positive_count,
                "powered_policy_count": len(powered),
            },
            "g2a_degradation_exists": g2a,
            "g2b_fn_asymmetry": g2b,
            "point_verdict": "PASS" if g2a and g2b else "FAIL",
        }
        result["variants"][name] = variant_result

    primary = {name: result["variants"][name] for name in ("D5", "D6")}
    passing = [name for name, value in primary.items() if value["point_verdict"] == "PASS"]
    result["g2_gate"] = {
        "evaluated_points": ["D5", "D6"],
        "passing_points": passing,
        "criterion_a": "mean per-policy Delta_ECE >0 with 95% CI lower bound >0",
        "criterion_b": "powered-policy mean(Delta_FN-Delta_FP) >0 with 95% CI lower bound >0",
        "verdict": "PASS" if passing else "FAIL",
        "control": {
            "D3_control_mean_delta_fn_minus_delta_fp": result["variants"]["D3_control"]["aggregate"]["mean_delta_fn_minus_delta_fp_powered"],
            "expected": "negative/no FN asymmetry",
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_result = safe(result)
    output.write_text(json.dumps(safe_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(safe_result["g2_gate"], indent=2))
    for name in names:
        aggregate = safe_result["variants"][name]["aggregate"]
        print(name, json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
