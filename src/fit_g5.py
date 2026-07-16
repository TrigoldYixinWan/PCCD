#!/usr/bin/env python3
"""Locked G5/P7 low-shot target-aware recalibration analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic_model import POLICY_IDS
from src.eval_critic import binary_violated_f1, multiclass_ece
from src.fit_g3 import load_labels_logits, softmax
from src.fit_g4 import (
    LOCKED_D0_MANIFEST_SHA256,
    fit_temperature,
    holm,
    to_jsonable,
    violated_auroc,
)


SEED = 20260722
BUDGETS = (50, 100, 200, 500)
METHODS = ("source_T", "target_global_T", "target_per_policy_T", "hierarchical_shrinkage")
METRICS = ("ece", "f1", "auroc")
LOCKED_G4_TEMPERATURES_SHA256 = "908d61608f22297227fd1a09bc119c8fe6a9d4a34338749758e875297e1b68c3"
TAU_LOW = math.log(0.05)
TAU_HIGH = math.log(20.0)
CURVATURE_FLOOR = 1e-8
_BOOT_DATA: dict | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen split manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, filename = line.split(maxsplit=1)
            result[str(Path(filename.lstrip("* ")).resolve())] = digest
    return result


def validate_split_hashes(hash_manifest: Path, paths: list[Path]) -> dict[str, str]:
    entries = parse_sha256_manifest(hash_manifest)
    result = {}
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in entries:
            raise ValueError(f"split manifest not frozen in hash file: {path}")
        digest = sha256(path)
        if digest != entries[resolved]:
            raise ValueError(f"split manifest hash changed: {path}")
        result[str(path)] = digest
    return result


def expected_partition(ids: list[str]) -> tuple[list[str], list[str]]:
    sorted_ids = np.asarray(sorted(ids), dtype=object)
    order = np.random.default_rng(SEED).permutation(len(sorted_ids))
    permuted = sorted_ids[order].tolist()
    return permuted[:1000], permuted[1000:]


def prepare_splits(args: argparse.Namespace) -> None:
    ids, _, _ = load_labels_logits(args.labels, args.logits)
    if len(ids) != 3000 or len(ids) != len(set(ids)):
        raise ValueError("locked D5 source must have exactly 3,000 unique IDs")
    calib_ids, test_ids = expected_partition(ids)
    common = {
        "schema": "pccd.g5.target_split.v1",
        "seed": SEED,
        "algorithm": "lexicographic ID sort; numpy default_rng(20260722).permutation",
        "source_labels": str(args.labels.resolve()),
        "source_labels_sha256": sha256(args.labels),
        "source_logits": str(args.logits.resolve()),
        "source_logits_sha256": sha256(args.logits),
    }
    write_manifest(args.calib_manifest, {
        **common,
        "split": "TARGET_CALIB",
        "n_ids": len(calib_ids),
        "nested_budget_prefixes": list(BUDGETS),
        "ids": calib_ids,
    })
    write_manifest(args.test_manifest, {
        **common,
        "split": "TARGET_TEST",
        "n_ids": len(test_ids),
        "ids": test_ids,
    })
    print(json.dumps({
        "calib_manifest": str(args.calib_manifest),
        "calib_sha256": sha256(args.calib_manifest),
        "test_manifest": str(args.test_manifest),
        "test_sha256": sha256(args.test_manifest),
    }, indent=2))


def summed_nll_curvature(logits: np.ndarray, labels: np.ndarray, tau: float) -> float:
    """Observed curvature of summed three-way NLL with respect to log T."""
    scaled = logits / math.exp(tau)
    probabilities = softmax(scaled)
    expectation = np.sum(probabilities * scaled, axis=1)
    second_moment = np.sum(probabilities * scaled**2, axis=1)
    variance = np.maximum(second_moment - expectation**2, 0.0)
    correct_scaled_logit = scaled[np.arange(len(labels)), labels]
    curvature = float(np.sum(variance + expectation - correct_scaled_logit))
    return max(curvature, CURVATURE_FLOOR)


def fit_all_methods(labels: np.ndarray, logits: np.ndarray, source_temperatures: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    flat_fit = fit_temperature(logits.reshape(-1, 3), labels.reshape(-1), diagnostics=False)
    tau_global = float(flat_fit["tau"])
    temperature_global = float(flat_fit["temperature"])
    tau_policy = np.empty(len(POLICY_IDS), dtype=np.float64)
    temperatures_policy = np.empty(len(POLICY_IDS), dtype=np.float64)
    variances = np.empty(len(POLICY_IDS), dtype=np.float64)
    policy_success = []
    for p in range(len(POLICY_IDS)):
        fitted = fit_temperature(logits[:, p], labels[:, p], diagnostics=False)
        tau_policy[p] = fitted["tau"]
        temperatures_policy[p] = fitted["temperature"]
        curvature = summed_nll_curvature(logits[:, p], labels[:, p], tau_policy[p])
        variances[p] = 1.0 / curvature
        policy_success.append(bool(fitted["success"]))
    between = max(float(np.var(tau_policy, ddof=1) - np.mean(variances)), 0.0)
    weights = between / (between + variances)
    tau_shrunk = np.clip(weights * tau_policy + (1.0 - weights) * tau_global, TAU_LOW, TAU_HIGH)
    temperatures = {
        "source_T": source_temperatures.copy(),
        "target_global_T": np.full(len(POLICY_IDS), temperature_global),
        "target_per_policy_T": temperatures_policy,
        "hierarchical_shrinkage": np.exp(tau_shrunk),
    }
    diagnostics = {
        "global_success": bool(flat_fit["success"]),
        "policy_success": policy_success,
        "tau_global": tau_global,
        "tau_policy": tau_policy,
        "curvature_variance": variances,
        "between_policy_variance_s2": between,
        "shrinkage_weight": weights,
    }
    return temperatures, diagnostics


def metric_vector(labels: np.ndarray, logits: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = softmax(logits / temperature)
    predictions = probabilities.argmax(axis=1)
    applicable = labels != 2
    violated = labels == 1
    satisfied = labels == 0
    f1 = (
        binary_violated_f1(labels[applicable], predictions[applicable])
        if violated.any() and satisfied.any()
        else math.nan
    )
    return np.asarray([
        multiclass_ece(probabilities, labels, 15),
        f1,
        violated_auroc(labels, probabilities),
    ])


def _bootstrap_worker(task: tuple[int, int, int]) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if _BOOT_DATA is None:
        raise RuntimeError("bootstrap worker was not initialized")
    budget, start, stop = task
    calib_labels = _BOOT_DATA["calib_labels"][:budget]
    calib_logits = _BOOT_DATA["calib_logits"][:budget]
    test_labels = _BOOT_DATA["test_labels"]
    test_logits = _BOOT_DATA["test_logits"]
    source_temperatures = _BOOT_DATA["source_temperatures"]
    calib_indices = _BOOT_DATA["calib_indices"][budget]
    test_indices = _BOOT_DATA["test_indices"]
    raw = np.empty((stop - start, len(POLICY_IDS), len(METRICS)), dtype=np.float64)
    method_metrics = np.empty(
        (stop - start, len(METHODS), len(POLICY_IDS), len(METRICS)), dtype=np.float64
    )
    method_temperatures = np.empty(
        (stop - start, len(METHODS), len(POLICY_IDS)), dtype=np.float64
    )
    fit_failures = 0
    bound_hits = 0
    for local, replicate in enumerate(range(start, stop)):
        crows = calib_indices[replicate]
        temperatures, diagnostics = fit_all_methods(
            calib_labels[crows], calib_logits[crows], source_temperatures
        )
        fit_failures += int(not diagnostics["global_success"])
        fit_failures += sum(not status for status in diagnostics["policy_success"])
        trows = test_indices[replicate]
        for p in range(len(POLICY_IDS)):
            raw[local, p] = metric_vector(test_labels[trows, p], test_logits[trows, p], 1.0)
            for m, method in enumerate(METHODS):
                temperature = temperatures[method][p]
                method_temperatures[local, m, p] = temperature
                bound_hits += int(abs(temperature - 0.05) <= 1e-6 or abs(temperature - 20.0) <= 1e-6)
                method_metrics[local, m, p] = metric_vector(
                    test_labels[trows, p], test_logits[trows, p], temperature
                )
    return budget, start, stop, raw, method_metrics, method_temperatures, fit_failures, bound_hits


def ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [math.nan, math.nan]


def support(labels: np.ndarray) -> dict[str, int]:
    return {
        "satisfied": int(np.sum(labels == 0)),
        "violated": int(np.sum(labels == 1)),
        "not_applicable": int(np.sum(labels == 2)),
    }


def load_and_validate_splits(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    validate_split_hashes(args.split_hash_manifest, [args.calib_manifest, args.test_manifest])
    calib = json.loads(args.calib_manifest.read_text(encoding="utf-8"))
    test = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    ids, _, _ = load_labels_logits(args.labels, args.logits)
    expected_calib, expected_test = expected_partition(ids)
    if calib["ids"] != expected_calib or test["ids"] != expected_test:
        raise ValueError("split IDs do not reproduce the locked deterministic partition")
    if set(calib["ids"]) & set(test["ids"]) or set(calib["ids"]) | set(test["ids"]) != set(ids):
        raise ValueError("target calib/test are not a disjoint exhaustive partition")
    for manifest in (calib, test):
        if manifest["source_labels_sha256"] != sha256(args.labels):
            raise ValueError("D5 label hash differs from split manifest")
        if manifest["source_logits_sha256"] != sha256(args.logits):
            raise ValueError("D5 logit hash differs from split manifest")
    return calib["ids"], test["ids"]


def run_fit(args: argparse.Namespace) -> None:
    if args.bootstrap != 10_000 or args.seed != SEED:
        raise ValueError("locked G5 protocol requires 10,000 replicates and seed 20260722")
    if sha256(args.checkpoint_manifest) != LOCKED_D0_MANIFEST_SHA256:
        raise ValueError("frozen D0 manifest changed")
    if sha256(args.source_temperatures) != LOCKED_G4_TEMPERATURES_SHA256:
        raise ValueError("frozen G4 source temperatures changed")
    calib_ids, test_ids = load_and_validate_splits(args)
    ids, labels, logits = load_labels_logits(args.labels, args.logits)
    row = {item_id: index for index, item_id in enumerate(ids)}
    calib_rows = np.asarray([row[item_id] for item_id in calib_ids])
    test_rows = np.asarray([row[item_id] for item_id in test_ids])
    calib_labels, calib_logits = labels[calib_rows], logits[calib_rows]
    test_labels, test_logits = labels[test_rows], logits[test_rows]
    source_payload = json.loads(args.source_temperatures.read_text(encoding="utf-8"))
    source_t = np.asarray(
        [source_payload["per_policy"][policy]["temperature"] for policy in POLICY_IDS],
        dtype=np.float64,
    )

    observed_raw = np.empty((len(POLICY_IDS), len(METRICS)), dtype=np.float64)
    observed = np.empty(
        (len(BUDGETS), len(METHODS), len(POLICY_IDS), len(METRICS)), dtype=np.float64
    )
    observed_temperatures = np.empty(
        (len(BUDGETS), len(METHODS), len(POLICY_IDS)), dtype=np.float64
    )
    fit_diagnostics = {}
    for p in range(len(POLICY_IDS)):
        observed_raw[p] = metric_vector(test_labels[:, p], test_logits[:, p], 1.0)
    for b_index, budget in enumerate(BUDGETS):
        temperatures, diagnostics = fit_all_methods(
            calib_labels[:budget], calib_logits[:budget], source_t
        )
        fit_diagnostics[str(budget)] = diagnostics
        for m, method in enumerate(METHODS):
            observed_temperatures[b_index, m] = temperatures[method]
            for p in range(len(POLICY_IDS)):
                observed[b_index, m, p] = metric_vector(
                    test_labels[:, p], test_logits[:, p], temperatures[method][p]
                )

    rng = np.random.default_rng(args.seed)
    calib_indices = {
        budget: rng.integers(0, budget, size=(args.bootstrap, budget), dtype=np.int32)
        for budget in BUDGETS
    }
    test_indices = rng.integers(
        0, len(test_ids), size=(args.bootstrap, len(test_ids)), dtype=np.int32
    )
    global _BOOT_DATA
    _BOOT_DATA = {
        "calib_labels": calib_labels,
        "calib_logits": calib_logits,
        "test_labels": test_labels,
        "test_logits": test_logits,
        "source_temperatures": source_t,
        "calib_indices": calib_indices,
        "test_indices": test_indices,
    }
    chunks_per_budget = max(1, math.ceil(args.jobs / len(BUDGETS)))
    boundaries = np.linspace(0, args.bootstrap, chunks_per_budget + 1, dtype=int)
    tasks = [
        (budget, int(boundaries[c]), int(boundaries[c + 1]))
        for budget in BUDGETS
        for c in range(chunks_per_budget)
        if boundaries[c] < boundaries[c + 1]
    ]
    if args.jobs == 1:
        worker_results = [_bootstrap_worker(task) for task in tasks]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=min(args.jobs, len(tasks))) as pool:
            worker_results = pool.map(_bootstrap_worker, tasks)
    boot_raw = np.empty((len(BUDGETS), args.bootstrap, len(POLICY_IDS), len(METRICS)))
    boot = np.empty(
        (len(BUDGETS), args.bootstrap, len(METHODS), len(POLICY_IDS), len(METRICS))
    )
    boot_temperatures = np.empty(
        (len(BUDGETS), args.bootstrap, len(METHODS), len(POLICY_IDS))
    )
    failures = bound_hits = 0
    for budget, start, stop, raw, metrics, temperatures, failed, bounds in worker_results:
        b = BUDGETS.index(budget)
        boot_raw[b, start:stop] = raw
        boot[b, start:stop] = metrics
        boot_temperatures[b, start:stop] = temperatures
        failures += failed
        bound_hits += bounds

    results = {}
    verdict_conditions = {}
    for b, budget in enumerate(BUDGETS):
        budget_result = {
            "calib_support": {
                policy: support(calib_labels[:budget, p])
                for p, policy in enumerate(POLICY_IDS)
            },
            "methods": {},
            "structure_benefit": {},
        }
        for m, method in enumerate(METHODS):
            per_policy = {}
            recovery_boot = boot[b, :, m, :, 0] - boot[b, :, METHODS.index("source_T"), :, 0]
            # Positive values below mean the named method improves over source-T.
            recovery_boot = -recovery_boot
            reduction_boot = boot_raw[b, :, :, 0] - boot[b, :, m, :, 0]
            f1_change_boot = boot[b, :, m, :, 1] - boot_raw[b, :, :, 1]
            auroc_change_boot = boot[b, :, m, :, 2] - boot_raw[b, :, :, 2]
            p_values = {}
            for p, policy in enumerate(POLICY_IDS):
                recovery = observed[b, METHODS.index("source_T"), p, 0] - observed[b, m, p, 0]
                p_values[policy] = float(
                    (1 + np.sum(recovery_boot[:, p] <= 0)) / (args.bootstrap + 1)
                )
                per_policy[policy] = {
                    "temperature": observed_temperatures[b, m, p],
                    "temperature_95ci": ci(boot_temperatures[b, :, m, p]),
                    "ece": observed[b, m, p, 0],
                    "ece_95ci": ci(boot[b, :, m, p, 0]),
                    "recovery_vs_source_T": recovery,
                    "recovery_vs_source_T_95ci": ci(recovery_boot[:, p]),
                    "reduction_vs_raw": observed_raw[p, 0] - observed[b, m, p, 0],
                    "reduction_vs_raw_95ci": ci(reduction_boot[:, p]),
                    "f1": observed[b, m, p, 1],
                    "f1_change_vs_raw": observed[b, m, p, 1] - observed_raw[p, 1],
                    "f1_change_95ci": ci(f1_change_boot[:, p]),
                    "auroc": observed[b, m, p, 2],
                    "auroc_change_vs_raw": observed[b, m, p, 2] - observed_raw[p, 2],
                    "auroc_change_95ci": ci(auroc_change_boot[:, p]),
                }
            adjusted = holm(p_values)
            for policy in POLICY_IDS:
                per_policy[policy]["recovery_p_one_sided"] = p_values[policy]
                per_policy[policy]["recovery_p_holm"] = adjusted[policy]
            budget_result["methods"][method] = {
                "per_policy": per_policy,
                "aggregate": {
                    "mean_ece": float(observed[b, m, :, 0].mean()),
                    "mean_ece_95ci": ci(boot[b, :, m, :, 0].mean(axis=1)),
                    "mean_recovery_vs_source_T": float(
                        observed[b, METHODS.index("source_T"), :, 0].mean()
                        - observed[b, m, :, 0].mean()
                    ),
                    "mean_recovery_vs_source_T_95ci": ci(recovery_boot.mean(axis=1)),
                    "mean_reduction_vs_raw": float(observed_raw[:, 0].mean() - observed[b, m, :, 0].mean()),
                    "mean_reduction_vs_raw_95ci": ci(reduction_boot.mean(axis=1)),
                    "mean_f1_change_vs_raw": float((observed[b, m, :, 1] - observed_raw[:, 1]).mean()),
                    "mean_f1_change_95ci": ci(f1_change_boot.mean(axis=1)),
                    "mean_auroc_change_vs_raw": float(np.nanmean(observed[b, m, :, 2] - observed_raw[:, 2])),
                    "mean_auroc_change_95ci": ci(np.nanmean(auroc_change_boot, axis=1)),
                },
            }
        global_index = METHODS.index("target_global_T")
        for structured in ("target_per_policy_T", "hierarchical_shrinkage"):
            m = METHODS.index(structured)
            difference = observed[b, global_index, :, 0].mean() - observed[b, m, :, 0].mean()
            difference_boot = (
                boot[b, :, global_index, :, 0].mean(axis=1)
                - boot[b, :, m, :, 0].mean(axis=1)
            )
            budget_result["structure_benefit"][structured] = {
                "global_minus_structured_mean_ece": float(difference),
                "95ci": ci(difference_boot),
            }
        pp = budget_result["methods"]["target_per_policy_T"]["aggregate"]
        conditions = {
            "mean_reduction_vs_raw_ci_lower_gt_0": pp["mean_reduction_vs_raw"] > 0
            and pp["mean_reduction_vs_raw_95ci"][0] > 0,
            "mean_ece_ci_upper_le_0_05": pp["mean_ece_95ci"][1] <= 0.05,
            "f1_noninferior": pp["mean_f1_change_95ci"][0] >= -0.005,
            "auroc_noninferior": pp["mean_auroc_change_95ci"][0] >= -0.01,
        }
        conditions["recovery_achieved"] = all(conditions.values())
        verdict_conditions[str(budget)] = conditions
        results[str(budget)] = budget_result

    successful = [budget for budget in BUDGETS if verdict_conditions[str(budget)]["recovery_achieved"]]
    b_star = min(successful) if successful else None
    structure_pass = False
    structure_method = None
    if b_star is not None:
        for method, value in results[str(b_star)]["structure_benefit"].items():
            if value["global_minus_structured_mean_ece"] > 0 and value["95ci"][0] > 0:
                structure_pass = True
                structure_method = method
                break
    if b_star is None:
        verdict = "NEGATIVE"
    elif b_star <= 200 and structure_pass:
        verdict = "SUPPORTED"
    else:
        verdict = "PARTIAL"

    output = {
        "protocol": {
            "preregistration": "reports/PREREG_G5.md (LOCKED 2026-07-16)",
            "point": "D5",
            "budgets": list(BUDGETS),
            "methods": list(METHODS),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "fit_failures": failures,
            "temperature_bound_hits_across_bootstrap_cells": bound_hits,
            "input_sha256": {
                "labels": sha256(args.labels),
                "logits": sha256(args.logits),
                "source_temperatures": sha256(args.source_temperatures),
                "checkpoint_manifest": sha256(args.checkpoint_manifest),
                "calib_manifest": sha256(args.calib_manifest),
                "test_manifest": sha256(args.test_manifest),
                "split_hash_manifest": sha256(args.split_hash_manifest),
            },
            "split": {
                "target_calib_n": len(calib_ids),
                "target_test_n": len(test_ids),
                "overlap": len(set(calib_ids) & set(test_ids)),
            },
        },
        "raw_target_test": {
            policy: {
                "ece": observed_raw[p, 0],
                "f1": observed_raw[p, 1],
                "auroc": observed_raw[p, 2],
                "support": support(test_labels[:, p]),
            }
            for p, policy in enumerate(POLICY_IDS)
        },
        "fit_diagnostics": fit_diagnostics,
        "budgets": results,
        "verdict": {
            "budget_conditions": verdict_conditions,
            "b_star": b_star,
            "structure_benefit_pass": structure_pass,
            "structure_method": structure_method,
            "P7": verdict,
            "frozen_G4_unchanged": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(output), indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for method in METHODS:
        means = [results[str(b)]["methods"][method]["aggregate"]["mean_ece"] for b in BUDGETS]
        intervals = [results[str(b)]["methods"][method]["aggregate"]["mean_ece_95ci"] for b in BUDGETS]
        # Percentile intervals need not contain the original point estimate;
        # clip only the plotted error-bar lengths while retaining the exact
        # bootstrap endpoints in the JSON result.
        low = [max(0.0, mean - interval[0]) for mean, interval in zip(means, intervals)]
        high = [max(0.0, interval[1] - mean) for mean, interval in zip(means, intervals)]
        axis.errorbar(BUDGETS, means, yerr=np.asarray([low, high]), marker="o", capsize=3, label=method)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="P1 anchor ceiling")
    axis.set_xscale("log")
    axis.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
    axis.set_xlabel("Target calibration labels (prompts)")
    axis.set_ylabel("TARGET-TEST mean ECE")
    axis.set_title(f"P7 low-shot target-aware recalibration: {verdict}")
    axis.legend(fontsize=8)
    figure.tight_layout()
    for path in (args.plot, args.summary_plot):
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
    plt.close(figure)

    print(json.dumps(to_jsonable(output["verdict"]), indent=2))
    print(f"result={args.out} plot={args.plot}")


def parse_args() -> argparse.Namespace:
    root = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--labels", type=Path, default=root / "g2" / "D5_teacher.jsonl")
    prepare.add_argument("--logits", type=Path, default=root / "g2" / "D5_logits.jsonl")
    prepare.add_argument("--calib_manifest", type=Path, default=root / "results" / "g5_target_calib_ids.json")
    prepare.add_argument("--test_manifest", type=Path, default=root / "results" / "g5_target_test_ids.json")

    fit = subparsers.add_parser("fit")
    fit.add_argument("--labels", type=Path, default=root / "g2" / "D5_teacher.jsonl")
    fit.add_argument("--logits", type=Path, default=root / "g2" / "D5_logits.jsonl")
    fit.add_argument("--calib_manifest", type=Path, default=root / "results" / "g5_target_calib_ids.json")
    fit.add_argument("--test_manifest", type=Path, default=root / "results" / "g5_target_test_ids.json")
    fit.add_argument("--split_hash_manifest", type=Path, default=root / "results" / "g5_split_manifests.sha256")
    fit.add_argument("--source_temperatures", type=Path, default=root / "results" / "g4_temperatures.json")
    fit.add_argument("--checkpoint_manifest", type=Path, default=root / "critic" / "d0.sha256")
    fit.add_argument("--out", type=Path, default=root / "results" / "g5_lowshot.json")
    fit.add_argument("--plot", type=Path, default=root / "results" / "g5_learning_curve.png")
    fit.add_argument("--summary_plot", type=Path, default=Path("reports/figures/day8_g5_learning_curve.png"))
    fit.add_argument("--bootstrap", type=int, default=10_000)
    fit.add_argument("--seed", type=int, default=SEED)
    fit.add_argument("--jobs", type=int, default=min(80, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_splits(args)
    else:
        run_fit(args)


if __name__ == "__main__":
    main()
