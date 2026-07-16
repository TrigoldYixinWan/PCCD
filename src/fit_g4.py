#!/usr/bin/env python3
"""Run the locked G4 source-calibration temperature-transfer analysis.

Ten scalar temperatures are fit once on Day-2 calib logits and then applied
unchanged to frozen D0-critic logits on D0--D6.  The teacher and critic model are
never invoked by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic_model import LABEL_TO_ID, POLICY_IDS
from src.eval_critic import adaptive_multiclass_ece, binary_violated_f1, multiclass_ece


POINTS = ("D0", "D1", "D2", "D3_control", "D4", "D5", "D6")
CONFIRMED = ("D2", "D3_control", "D4", "D5")
BOOTSTRAP_SEED = 20260721
LOCKED_D0_MANIFEST_SHA256 = "c64e6b74eb00a88ad50c65df50ecc81fcb5369897aef0231658f9e9bf28553a1"
TAU_BOUNDS = (math.log(0.05), math.log(20.0))
METRIC_NAMES = (
    "raw_ece", "scaled_ece", "raw_adaptive_ece", "scaled_adaptive_ece",
    "raw_f1", "scaled_f1", "raw_auroc", "scaled_auroc",
)
_BOOT_DATA: dict | None = None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def load_labels_logits(labels_path: Path, logits_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    label_rows = read_jsonl(labels_path)
    logit_rows = read_jsonl(logits_path)
    ids = [row["id"] for row in label_rows]
    if ids != [row["id"] for row in logit_rows]:
        raise ValueError(f"ID mismatch: {labels_path} vs {logits_path}")
    labels = np.asarray(
        [[LABEL_TO_ID[row["labels"][policy]] for policy in POLICY_IDS] for row in label_rows],
        dtype=np.int8,
    )
    logits = np.asarray(
        [[row["logits"][policy] for policy in POLICY_IDS] for row in logit_rows],
        dtype=np.float64,
    )
    return ids, labels, logits


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, diagnostics: bool = True) -> dict:
    result = minimize_scalar(
        lambda tau: nll(logits, labels, math.exp(float(tau))),
        method="bounded",
        bounds=TAU_BOUNDS,
        options={"xatol": 1e-8},
    )
    tau = float(result.x)
    temperature = math.exp(tau)
    tolerance = 1e-6
    output = {
        "temperature": temperature,
        "tau": tau,
        "success": bool(result.success),
        "status": int(result.status) if hasattr(result, "status") else None,
        "message": str(result.message),
        "nfev": int(result.nfev),
        "at_lower_bound": abs(tau - TAU_BOUNDS[0]) <= tolerance,
        "at_upper_bound": abs(tau - TAU_BOUNDS[1]) <= tolerance,
    }
    if diagnostics:
        output["nll_raw"] = nll(logits, labels, 1.0)
        output["nll_scaled"] = nll(logits, labels, temperature)
    return output


def violated_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    applicable = labels != LABEL_TO_ID["not_applicable"]
    truth = labels[applicable] == LABEL_TO_ID["violated"]
    scores = probabilities[applicable, LABEL_TO_ID["violated"]]
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def policy_metrics(labels: np.ndarray, logits: np.ndarray, temperature: float) -> dict[str, float]:
    raw_probabilities = softmax(logits)
    scaled_probabilities = softmax(logits / temperature)
    raw_prediction = raw_probabilities.argmax(axis=1)
    scaled_prediction = scaled_probabilities.argmax(axis=1)
    applicable = labels != LABEL_TO_ID["not_applicable"]
    violated = labels == LABEL_TO_ID["violated"]
    satisfied = labels == LABEL_TO_ID["satisfied"]
    both = violated.any() and satisfied.any()
    return {
        "raw_ece": multiclass_ece(raw_probabilities, labels, 15),
        "scaled_ece": multiclass_ece(scaled_probabilities, labels, 15),
        "raw_adaptive_ece": adaptive_multiclass_ece(raw_probabilities, labels, 15),
        "scaled_adaptive_ece": adaptive_multiclass_ece(scaled_probabilities, labels, 15),
        "raw_f1": binary_violated_f1(labels[applicable], raw_prediction[applicable]) if both else math.nan,
        "scaled_f1": binary_violated_f1(labels[applicable], scaled_prediction[applicable]) if both else math.nan,
        "raw_auroc": violated_auroc(labels, raw_probabilities),
        "scaled_auroc": violated_auroc(labels, scaled_probabilities),
        "argmax_identical": bool(np.array_equal(raw_prediction, scaled_prediction)),
        "n_satisfied": int(satisfied.sum()),
        "n_violated": int(violated.sum()),
        "n_na": int((~applicable).sum()),
    }


def _bootstrap_worker(task: tuple[int, int, int]) -> tuple[int, int, int, np.ndarray, np.ndarray, int, int]:
    if _BOOT_DATA is None:
        raise RuntimeError("bootstrap worker was not initialized")
    policy, start, stop = task
    calib_logits = _BOOT_DATA["calib_logits"][:, policy]
    calib_labels = _BOOT_DATA["calib_labels"][:, policy]
    eval_logits = _BOOT_DATA["eval_logits"][:, :, policy]
    eval_labels = _BOOT_DATA["eval_labels"][:, :, policy]
    calib_indices = _BOOT_DATA["calib_indices"]
    eval_indices = _BOOT_DATA["eval_indices"]
    values = np.empty((stop - start, len(POINTS), len(METRIC_NAMES)), dtype=np.float64)
    temperatures = np.empty(stop - start, dtype=np.float64)
    failures = 0
    bound_hits = 0
    for local, replicate in enumerate(range(start, stop)):
        calibration_rows = calib_indices[replicate]
        fitted = fit_temperature(
            calib_logits[calibration_rows], calib_labels[calibration_rows], diagnostics=False
        )
        temperature = fitted["temperature"]
        temperatures[local] = temperature
        failures += int(not fitted["success"])
        bound_hits += int(fitted["at_lower_bound"] or fitted["at_upper_bound"])
        rows = eval_indices[replicate]
        for d in range(len(POINTS)):
            metrics = policy_metrics(eval_labels[d, rows], eval_logits[d, rows], temperature)
            values[local, d] = [metrics[name] for name in METRIC_NAMES]
    return policy, start, stop, values, temperatures, failures, bound_hits


def ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [math.nan, math.nan]


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (m - rank) * p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[list[float], list[float]]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for index in range(n_bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == n_bins - 1 else confidence < edges[index + 1]
        )
        if np.any(selected):
            xs.append(float(confidence[selected].mean()))
            ys.append(float(correct[selected].mean()))
    return xs, ys


def parse_args() -> argparse.Namespace:
    root = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_labels", type=Path, default=root / "labels" / "calib.jsonl")
    parser.add_argument("--calib_logits", type=Path, default=root / "results" / "d0_calib_logits.jsonl")
    parser.add_argument("--g2_dir", type=Path, default=root / "g2")
    parser.add_argument("--checkpoint_manifest", type=Path, default=root / "critic" / "d0.sha256")
    parser.add_argument("--temperatures_out", type=Path, default=root / "results" / "g4_temperatures.json")
    parser.add_argument("--out", type=Path, default=root / "results" / "g4_recalibration.json")
    parser.add_argument("--figures_dir", type=Path, default=root / "results" / "g4_figures")
    parser.add_argument("--summary_plot", type=Path, default=Path("reports/figures/day7_g4_recovery.png"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--jobs", type=int, default=min(40, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != BOOTSTRAP_SEED:
        raise ValueError(f"locked seed is {BOOTSTRAP_SEED}")
    if args.bootstrap != 10_000:
        raise ValueError("locked analysis requires 10,000 bootstrap replicates")
    observed_manifest_hash = sha256(args.checkpoint_manifest)
    if observed_manifest_hash != LOCKED_D0_MANIFEST_SHA256:
        raise ValueError(
            f"frozen D0 manifest changed: {observed_manifest_hash} != {LOCKED_D0_MANIFEST_SHA256}"
        )

    paths: dict[str, Path] = {
        "calib_labels": args.calib_labels,
        "calib_logits": args.calib_logits,
        "checkpoint_manifest": args.checkpoint_manifest,
    }
    calib_ids, calib_labels, calib_logits = load_labels_logits(args.calib_labels, args.calib_logits)
    if len(calib_ids) != 1000:
        raise ValueError(f"locked calib split must contain 1,000 items, got {len(calib_ids)}")

    eval_labels, eval_logits, eval_ids = [], [], None
    for point in POINTS:
        label_path = args.g2_dir / f"{point}_teacher.jsonl"
        logit_path = args.g2_dir / f"{point}_logits.jsonl"
        paths[f"{point}_labels"] = label_path
        paths[f"{point}_logits"] = logit_path
        ids, labels, logits = load_labels_logits(label_path, logit_path)
        if eval_ids is None:
            eval_ids = ids
        elif ids != eval_ids:
            raise ValueError(f"{point} IDs differ from D0")
        eval_labels.append(labels)
        eval_logits.append(logits)
    assert eval_ids is not None
    if len(eval_ids) != 3000:
        raise ValueError(f"locked G2 evaluation set must contain 3,000 items, got {len(eval_ids)}")
    eval_labels_array = np.asarray(eval_labels)
    eval_logits_array = np.asarray(eval_logits)

    fits, temperatures = {}, np.empty(len(POLICY_IDS), dtype=np.float64)
    for p, policy in enumerate(POLICY_IDS):
        fitted = fit_temperature(calib_logits[:, p], calib_labels[:, p])
        fits[policy] = fitted
        temperatures[p] = fitted["temperature"]

    observed = np.empty((len(POINTS), len(POLICY_IDS), len(METRIC_NAMES)), dtype=np.float64)
    supports: dict[str, dict[str, dict[str, int]]] = {}
    argmax_all_identical = True
    for d, point in enumerate(POINTS):
        supports[point] = {}
        for p, policy in enumerate(POLICY_IDS):
            metrics = policy_metrics(eval_labels_array[d, :, p], eval_logits_array[d, :, p], temperatures[p])
            observed[d, p] = [metrics[name] for name in METRIC_NAMES]
            supports[point][policy] = {
                "satisfied": metrics["n_satisfied"],
                "violated": metrics["n_violated"],
                "not_applicable": metrics["n_na"],
            }
            argmax_all_identical &= metrics["argmax_identical"]

    rng = np.random.default_rng(args.seed)
    calib_indices = rng.integers(0, len(calib_ids), size=(args.bootstrap, len(calib_ids)), dtype=np.int32)
    eval_indices_boot = rng.integers(0, len(eval_ids), size=(args.bootstrap, len(eval_ids)), dtype=np.int32)
    global _BOOT_DATA
    _BOOT_DATA = {
        "calib_logits": calib_logits,
        "calib_labels": calib_labels,
        "eval_logits": eval_logits_array,
        "eval_labels": eval_labels_array,
        "calib_indices": calib_indices,
        "eval_indices": eval_indices_boot,
    }
    chunks_per_policy = max(1, math.ceil(args.jobs / len(POLICY_IDS)))
    boundaries = np.linspace(0, args.bootstrap, chunks_per_policy + 1, dtype=int)
    tasks = [
        (p, int(boundaries[c]), int(boundaries[c + 1]))
        for p in range(len(POLICY_IDS))
        for c in range(chunks_per_policy)
        if boundaries[c] < boundaries[c + 1]
    ]
    if args.jobs == 1:
        worker_results = [_bootstrap_worker(task) for task in tasks]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=min(args.jobs, len(tasks))) as pool:
            worker_results = pool.map(_bootstrap_worker, tasks)

    boot = np.empty((args.bootstrap, len(POINTS), len(POLICY_IDS), len(METRIC_NAMES)), dtype=np.float64)
    temperature_boot = np.empty((args.bootstrap, len(POLICY_IDS)), dtype=np.float64)
    failures = bound_hits = 0
    for policy, start, stop, values, temps, failed, bounds in worker_results:
        boot[start:stop, :, policy] = values
        temperature_boot[start:stop, policy] = temps
        failures += failed
        bound_hits += bounds

    metric_index = {name: i for i, name in enumerate(METRIC_NAMES)}
    raw_ece = observed[:, :, metric_index["raw_ece"]]
    scaled_ece = observed[:, :, metric_index["scaled_ece"]]
    recovery = np.abs(raw_ece - raw_ece[0]) - np.abs(scaled_ece - scaled_ece[0])
    absolute_gain = raw_ece - scaled_ece
    boot_raw_ece = boot[:, :, :, metric_index["raw_ece"]]
    boot_scaled_ece = boot[:, :, :, metric_index["scaled_ece"]]
    recovery_boot = np.abs(boot_raw_ece - boot_raw_ece[:, :1]) - np.abs(
        boot_scaled_ece - boot_scaled_ece[:, :1]
    )
    absolute_gain_boot = boot_raw_ece - boot_scaled_ece

    per_point: dict[str, dict] = {}
    for d, point in enumerate(POINTS):
        per_policy: dict[str, dict] = {}
        powered_indices = []
        for p, policy in enumerate(POLICY_IDS):
            powered = supports[point][policy]["violated"] >= 30
            if powered:
                powered_indices.append(p)
            policy_result = {
                "temperature": temperatures[p],
                "support": supports[point][policy],
                "discrimination_powered": powered,
                "raw_ece": raw_ece[d, p],
                "scaled_ece": scaled_ece[d, p],
                "raw_ece_95ci": ci(boot_raw_ece[:, d, p]),
                "scaled_ece_95ci": ci(boot_scaled_ece[:, d, p]),
                "raw_adaptive_ece": observed[d, p, metric_index["raw_adaptive_ece"]],
                "scaled_adaptive_ece": observed[d, p, metric_index["scaled_adaptive_ece"]],
                "raw_adaptive_ece_95ci": ci(boot[:, d, p, metric_index["raw_adaptive_ece"]]),
                "scaled_adaptive_ece_95ci": ci(boot[:, d, p, metric_index["scaled_adaptive_ece"]]),
                "recovery": recovery[d, p],
                "recovery_95ci": ci(recovery_boot[:, d, p]),
                "absolute_gain": absolute_gain[d, p],
                "absolute_gain_95ci": ci(absolute_gain_boot[:, d, p]),
                "raw_f1": observed[d, p, metric_index["raw_f1"]],
                "scaled_f1": observed[d, p, metric_index["scaled_f1"]],
                "f1_change": observed[d, p, metric_index["scaled_f1"]] - observed[d, p, metric_index["raw_f1"]],
                "f1_change_95ci": ci(boot[:, d, p, metric_index["scaled_f1"]] - boot[:, d, p, metric_index["raw_f1"]]),
                "raw_auroc": observed[d, p, metric_index["raw_auroc"]],
                "scaled_auroc": observed[d, p, metric_index["scaled_auroc"]],
                "auroc_change": observed[d, p, metric_index["scaled_auroc"]] - observed[d, p, metric_index["raw_auroc"]],
                "auroc_change_95ci": ci(boot[:, d, p, metric_index["scaled_auroc"]] - boot[:, d, p, metric_index["raw_auroc"]]),
            }
            per_policy[policy] = policy_result
        mean_recovery_boot = recovery_boot[:, d].mean(axis=1)
        mean_gain_boot = absolute_gain_boot[:, d].mean(axis=1)
        f1_change = observed[d, :, metric_index["scaled_f1"]] - observed[d, :, metric_index["raw_f1"]]
        auroc_change = observed[d, :, metric_index["scaled_auroc"]] - observed[d, :, metric_index["raw_auroc"]]
        f1_boot = boot[:, d, :, metric_index["scaled_f1"]] - boot[:, d, :, metric_index["raw_f1"]]
        auroc_boot = boot[:, d, :, metric_index["scaled_auroc"]] - boot[:, d, :, metric_index["raw_auroc"]]
        per_point[point] = {
            "per_policy": per_policy,
            "aggregate": {
                "mean_recovery": float(recovery[d].mean()),
                "mean_recovery_95ci": ci(mean_recovery_boot),
                "mean_absolute_gain": float(absolute_gain[d].mean()),
                "mean_absolute_gain_95ci": ci(mean_gain_boot),
                "policies_positive_recovery": int(np.sum(recovery[d] > 0)),
                "powered_policies": [POLICY_IDS[p] for p in powered_indices],
                "mean_f1_change_powered": float(np.nanmean(f1_change[powered_indices])) if powered_indices else math.nan,
                "mean_f1_change_95ci": ci(np.nanmean(f1_boot[:, powered_indices], axis=1)) if powered_indices else [math.nan, math.nan],
                "mean_auroc_change_powered": float(np.nanmean(auroc_change[powered_indices])) if powered_indices else math.nan,
                "mean_auroc_change_95ci": ci(np.nanmean(auroc_boot[:, powered_indices], axis=1)) if powered_indices else [math.nan, math.nan],
            },
        }

    d5 = per_point["D5"]["aggregate"]
    d5_index = POINTS.index("D5")
    d5_auroc_changes = observed[d5_index, :, metric_index["scaled_auroc"]] - observed[d5_index, :, metric_index["raw_auroc"]]
    d5_calibration = (
        d5["mean_recovery"] > 0
        and d5["mean_recovery_95ci"][0] > 0
        and d5["mean_absolute_gain"] > 0
        and d5["mean_absolute_gain_95ci"][0] > 0
        and d5["policies_positive_recovery"] >= 7
    )
    d5_discrimination = (
        d5["mean_f1_change_95ci"][0] >= -0.005
        and argmax_all_identical
        and d5["mean_auroc_change_95ci"][0] >= -0.01
        and bool(np.all(d5_auroc_changes >= -0.02))
    )
    confirmed_indices = [POINTS.index(point) for point in CONFIRMED]
    pooled_recovery = float(recovery[confirmed_indices].mean())
    pooled_gain = float(absolute_gain[confirmed_indices].mean())
    pooled_recovery_boot = recovery_boot[:, confirmed_indices].mean(axis=(1, 2))
    pooled_gain_boot = absolute_gain_boot[:, confirmed_indices].mean(axis=(1, 2))
    generalization = (
        ci(pooled_recovery_boot)[0] > 0
        and ci(pooled_gain_boot)[0] > 0
        and all(per_point[point]["aggregate"]["mean_recovery"] > 0 for point in CONFIRMED)
    )
    verdict = "FAIL"
    if d5_calibration and d5_discrimination:
        verdict = "PASS" if generalization else "PARTIAL"

    d5_p = {
        policy: float((1 + np.sum(recovery_boot[:, d5_index, p] <= 0)) / (args.bootstrap + 1))
        for p, policy in enumerate(POLICY_IDS)
    }
    d5_p_holm = holm(d5_p)
    for policy in POLICY_IDS:
        per_point["D5"]["per_policy"][policy]["recovery_p_one_sided"] = d5_p[policy]
        per_point["D5"]["per_policy"][policy]["recovery_p_holm"] = d5_p_holm[policy]

    input_hashes = {key: sha256(path) for key, path in paths.items()}
    temperature_result = {
        "protocol": {
            "preregistration": "reports/PREREG_G4.md (LOCKED 2026-07-16)",
            "parameterization": "T_p=exp(tau_p), one scalar per policy",
            "tau_bounds": list(TAU_BOUNDS),
            "temperature_bounds": [0.05, 20.0],
            "optimizer": "scipy.optimize.minimize_scalar(method=bounded)",
            "optimizer_xatol": 1e-8,
            "scipy_version": scipy.__version__,
            "python_version": platform.python_version(),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "bootstrap_optimizer_failures": failures,
            "bootstrap_bound_hits": bound_hits,
            "input_sha256": input_hashes,
        },
        "per_policy": {
            policy: {
                **fits[policy],
                "temperature_95ci": ci(temperature_boot[:, p]),
            }
            for p, policy in enumerate(POLICY_IDS)
        },
    }
    recalibration_result = {
        "protocol": temperature_result["protocol"],
        "definition": {
            "ece": "15-bin top-class 3-way ECE including N/A",
            "recovery": "|ECE_raw_d-ECE_raw_D0|-|ECE_scaled_d-ECE_scaled_D0|",
            "absolute_gain": "ECE_raw_d-ECE_scaled_d",
            "temperature_fit": "Day-2 calib only; same ten temperatures applied D0-D6",
        },
        "scaled_vs_raw_d0": {
            policy: {
                "raw_ece": raw_ece[0, p],
                "scaled_ece": scaled_ece[0, p],
                "change": scaled_ece[0, p] - raw_ece[0, p],
                "change_95ci": ci(boot_scaled_ece[:, 0, p] - boot_raw_ece[:, 0, p]),
            }
            for p, policy in enumerate(POLICY_IDS)
        },
        "points": per_point,
        "confirmed_degradation_generalization": {
            "points": list(CONFIRMED),
            "pooled_mean_recovery": pooled_recovery,
            "pooled_mean_recovery_95ci": ci(pooled_recovery_boot),
            "pooled_mean_absolute_gain": pooled_gain,
            "pooled_mean_absolute_gain_95ci": ci(pooled_gain_boot),
            "every_point_positive_mean_recovery": all(
                per_point[point]["aggregate"]["mean_recovery"] > 0 for point in CONFIRMED
            ),
            "passes": generalization,
        },
        "gate": {
            "d5_calibration_recovery_pass": d5_calibration,
            "d5_discrimination_preservation_pass": d5_discrimination,
            "generalization_pass": generalization,
            "argmax_exactly_identical_all_points_policies": argmax_all_identical,
            "verdict": verdict,
        },
    }

    args.temperatures_out.parent.mkdir(parents=True, exist_ok=True)
    args.temperatures_out.write_text(json.dumps(to_jsonable(temperature_result), indent=2) + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(recalibration_result), indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for point in ("D0", "D5"):
        d = POINTS.index(point)
        figure, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
        for p, (axis, policy) in enumerate(zip(axes.flat, POLICY_IDS)):
            raw_probs = softmax(eval_logits_array[d, :, p])
            scaled_probs = softmax(eval_logits_array[d, :, p] / temperatures[p])
            raw_x, raw_y = reliability_bins(raw_probs, eval_labels_array[d, :, p])
            scaled_x, scaled_y = reliability_bins(scaled_probs, eval_labels_array[d, :, p])
            axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
            axis.plot(raw_x, raw_y, marker="o", label="raw", alpha=0.8)
            axis.plot(scaled_x, scaled_y, marker="s", label="scaled", alpha=0.8)
            axis.set_title(policy)
            axis.grid(alpha=0.2)
        axes.flat[0].legend()
        figure.supxlabel("Mean confidence")
        figure.supylabel("Empirical accuracy")
        figure.suptitle(f"G4 {point}: raw vs source-calib temperature-scaled reliability")
        figure.tight_layout()
        figure.savefig(args.figures_dir / f"{point}_reliability.png", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    positions = np.arange(len(POINTS))
    mean_recovery = np.asarray([per_point[point]["aggregate"]["mean_recovery"] for point in POINTS])
    mean_gain = np.asarray([per_point[point]["aggregate"]["mean_absolute_gain"] for point in POINTS])
    axes[0].bar(positions, mean_recovery)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(positions, POINTS, rotation=30)
    axes[0].set_ylabel("Mean recovery")
    axes[0].set_title("Gap recovery vs correspondingly scaled D0")
    axes[1].bar(positions, mean_gain, color="tab:orange")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, POINTS, rotation=30)
    axes[1].set_ylabel("Mean absolute ECE gain")
    axes[1].set_title(f"Absolute calibration gain (G4 {verdict})")
    figure.tight_layout()
    args.summary_plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.summary_plot, dpi=180)
    figure.savefig(args.figures_dir / "g4_recovery_summary.png", dpi=180)
    plt.close(figure)

    print(json.dumps(to_jsonable(recalibration_result["gate"]), indent=2))
    print(json.dumps(to_jsonable(recalibration_result["confirmed_degradation_generalization"]), indent=2))
    print(f"temperatures={args.temperatures_out} results={args.out} figures={args.figures_dir}")


if __name__ == "__main__":
    main()
