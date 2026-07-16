#!/usr/bin/env python3
"""P8 structured recalibration with a paired two-stage bootstrap.

The primary method is the published Structured Matrix Scaling implementation
from ``probmetrics==1.3.0``.  Structured Vector Scaling is secondary and the
P7 per-policy temperature fit is re-estimated on every calibration resample.

This module deliberately accepts frozen split manifests instead of constructing
splits.  In ``confirmation`` mode it requires 500 target-calibration prompts and
3,500 untouched test prompts.  ``development`` mode exists only to run the same
software on the already-consumed P7 split; it can never produce a confirmatory
P8 verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic_model import POLICY_IDS
from src.eval_critic import binary_violated_f1, multiclass_ece
from src.fit_g3 import read_jsonl, softmax
from src.fit_g4 import fit_temperature, to_jsonable, violated_auroc
from src.policy_defs import LABEL_STATES


REQUIRED_PROBMETRICS_VERSION = "1.3.0"
SEED = 20260724
BUDGETS = (50, 100, 200, 500)
PRIMARY_BUDGET = 500
METHODS = ("P7_per_policy_T", "SMS", "SVS")
METRICS = ("ece", "f1", "auroc", "nll", "brier")
METRIC_INDEX = {name: index for index, name in enumerate(METRICS)}
EXPECTED_CONFIRM_CALIB = 500
EXPECTED_CONFIRM_TEST = 3500
EXPECTED_DEVELOPMENT_CALIB = 1000
EXPECTED_DEVELOPMENT_TEST = 2000
_BOOT_DATA: dict | None = None
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_STATES)}


class ProbmetricsDependencyError(RuntimeError):
    """Raised before fitting when the exact published dependency is unavailable."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, filename = line.split(maxsplit=1)
            artifact = Path(filename.lstrip("* "))
            if not artifact.is_absolute():
                artifact = path.parent / artifact
            entries[str(artifact.resolve())] = digest
    return entries


def validate_manifest_hashes(hash_manifest: Path, paths: list[Path]) -> dict[str, str]:
    if not hash_manifest.exists():
        raise FileNotFoundError(f"frozen split hash manifest is missing: {hash_manifest}")
    entries = parse_sha256_manifest(hash_manifest)
    result = {}
    for path in paths:
        resolved = str(path.resolve())
        observed = sha256(path)
        if entries.get(resolved) != observed:
            raise ValueError(f"split manifest is absent from hash file or changed: {path}")
        result[str(path)] = observed
    return result


def probmetrics_status() -> dict:
    """Return an import/version report without raising or fitting anything."""
    try:
        version = importlib.metadata.version("probmetrics")
    except importlib.metadata.PackageNotFoundError:
        return {
            "available": False,
            "version": None,
            "required_version": REQUIRED_PROBMETRICS_VERSION,
            "error": "probmetrics is not installed",
            "install": f"python -m pip install probmetrics=={REQUIRED_PROBMETRICS_VERSION}",
        }
    if version != REQUIRED_PROBMETRICS_VERSION:
        return {
            "available": False,
            "version": version,
            "required_version": REQUIRED_PROBMETRICS_VERSION,
            "error": "probmetrics version mismatch",
            "install": f"python -m pip install --upgrade probmetrics=={REQUIRED_PROBMETRICS_VERSION}",
        }
    try:
        from probmetrics.calibrators import SMSCalibrator, SVSCalibrator
    except Exception as exc:  # dependency/import errors need a concise preflight report
        return {
            "available": False,
            "version": version,
            "required_version": REQUIRED_PROBMETRICS_VERSION,
            "error": f"probmetrics import failed: {type(exc).__name__}: {exc}",
            "install": f"python -m pip install --upgrade probmetrics=={REQUIRED_PROBMETRICS_VERSION}",
        }
    expected = {
        "penalty": "ridge",
        "rho": 1.0,
        "tau": 1.0,
        "lambda_intercept": 1.0,
        "lambda_diagonal": 1.0,
        "opt": "bfgs",
        "max_iter": 500,
        "tol": 1e-5,
        "print_init_info": True,
    }
    sms_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SMSCalibrator).parameters.items()
        if name != "self"
    }
    svs_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SVSCalibrator).parameters.items()
        if name != "self"
    }
    expected_sms = {**expected, "lambda_off_diagonal": 1.0}
    defaults_ok = all(sms_defaults.get(key) == value for key, value in expected_sms.items())
    defaults_ok &= all(svs_defaults.get(key) == value for key, value in expected.items())
    return {
        "available": bool(defaults_ok),
        "version": version,
        "required_version": REQUIRED_PROBMETRICS_VERSION,
        "defaults_verified": bool(defaults_ok),
        "SMS_constructor": str(inspect.signature(SMSCalibrator)),
        "SVS_constructor": str(inspect.signature(SVSCalibrator)),
        "SMS_defaults": {key: sms_defaults.get(key) for key in expected_sms},
        "SVS_defaults": {key: svs_defaults.get(key) for key in expected},
        "internal_preprocessing": "ts-mix (implemented by probmetrics 1.3.0)",
        "higher_iteration_retry_available_for_primary_bfgs": False,
        "retry_note": "probmetrics 1.3.0 documents max_iter/tol as ignored by the BFGS implementation",
        "error": None if defaults_ok else "published SMS/SVS constructor defaults differ",
    }


@lru_cache(maxsize=1)
def require_probmetrics() -> tuple[type, type, dict]:
    status = probmetrics_status()
    if not status["available"]:
        raise ProbmetricsDependencyError(
            f"P8 requires the exact published dependency probmetrics=="
            f"{REQUIRED_PROBMETRICS_VERSION}; {status['error']}. "
            f"Run: {status.get('install', 'install the pinned package')}"
        )
    from probmetrics.calibrators import SMSCalibrator, SVSCalibrator

    return SMSCalibrator, SVSCalibrator, status


def _new_structured_calibrator(
    method: str, factory: Callable[[str], object] | None = None
) -> object:
    if factory is not None:
        return factory(method)
    sms, svs, _ = require_probmetrics()
    if method == "SMS":
        return sms()  # exact probmetrics 1.3.0 defaults; do not tune kwargs
    if method == "SVS":
        return svs()  # exact probmetrics 1.3.0 defaults; do not tune kwargs
    raise ValueError(method)


def metric_vector(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != (len(labels), 3):
        raise ValueError("probabilities must have shape [n, 3]")
    valid = labels >= 0
    labels = labels[valid]
    probabilities = probabilities[valid]
    if not len(labels):
        return np.full(len(METRICS), np.nan, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("calibrated probabilities are non-finite or negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("calibrated probabilities do not sum to one")
    predictions = probabilities.argmax(axis=1)
    applicable = labels != 2
    has_violated = bool(np.any(labels == 1))
    has_satisfied = bool(np.any(labels == 0))
    f1 = (
        binary_violated_f1(labels[applicable], predictions[applicable])
        if has_violated and has_satisfied
        else math.nan
    )
    clipped = np.clip(probabilities, 1e-15, 1.0)
    nll = float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))
    one_hot = np.eye(3, dtype=np.float64)[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return np.asarray(
        [
            multiclass_ece(probabilities, labels, 15),
            f1,
            violated_auroc(labels, probabilities),
            nll,
            brier,
        ],
        dtype=np.float64,
    )


def support(labels: np.ndarray) -> dict[str, int]:
    return {
        "satisfied": int(np.sum(labels == 0)),
        "violated": int(np.sum(labels == 1)),
        "not_applicable": int(np.sum(labels == 2)),
        "missing": int(np.sum(labels < 0)),
    }


def ci(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [math.nan, math.nan]
    return np.quantile(finite, [0.025, 0.975]).tolist()


def _safe_nanmean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def _strict_mean(values: np.ndarray, axis: int) -> np.ndarray:
    """Average fixed criteria only when every criterion is finite."""
    values = np.asarray(values, dtype=np.float64)
    result = np.mean(values, axis=axis)
    invalid = np.any(~np.isfinite(values), axis=axis)
    return np.where(invalid, np.nan, result)


def _fit_models(
    labels: np.ndarray,
    logits: np.ndarray,
    factory: Callable[[str], object] | None = None,
) -> tuple[dict[str, list], dict[str, list[str]]]:
    raw_probabilities = softmax(logits)
    models: dict[str, list] = {method: [] for method in METHODS}
    failures: dict[str, list[str]] = {method: [] for method in METHODS}
    for policy_index, policy in enumerate(POLICY_IDS):
        valid = labels[:, policy_index] >= 0
        policy_labels = labels[valid, policy_index]
        policy_logits = logits[valid, policy_index]
        policy_probabilities = raw_probabilities[valid, policy_index]
        if not len(policy_labels):
            for method in METHODS:
                failures[method].append(f"{policy}: no valid calibration labels")
                models[method].append(None)
            continue
        fitted_t = fit_temperature(
            policy_logits, policy_labels, diagnostics=False
        )
        if not fitted_t["success"]:
            failures["P7_per_policy_T"].append(f"{policy}: {fitted_t['message']}")
            models["P7_per_policy_T"].append(None)
        else:
            models["P7_per_policy_T"].append(float(fitted_t["temperature"]))
        for method in ("SMS", "SVS"):
            try:
                calibrator = _new_structured_calibrator(method, factory)
                calibrator.fit(policy_probabilities, policy_labels)
                probe = np.asarray(
                    calibrator.predict_proba(policy_probabilities[: min(3, len(policy_labels))]),
                    dtype=np.float64,
                )
                if probe.shape != (min(3, len(policy_labels)), 3) or not np.all(np.isfinite(probe)):
                    raise ValueError("non-finite or wrong-shaped predict_proba output")
                models[method].append(calibrator)
            except Exception as exc:
                failures[method].append(f"{policy}: {type(exc).__name__}: {exc}")
                models[method].append(None)
    return models, failures


def _predict_method(
    method: str,
    model: object | None,
    logits: np.ndarray,
    raw_probabilities: np.ndarray,
) -> np.ndarray | None:
    if model is None:
        return None
    if method == "P7_per_policy_T":
        return softmax(logits / float(model))
    return np.asarray(model.predict_proba(raw_probabilities), dtype=np.float64)


def _model_diagnostics(models: dict[str, list]) -> dict:
    result: dict[str, dict] = {method: {} for method in METHODS}
    for policy_index, policy in enumerate(POLICY_IDS):
        temperature = models["P7_per_policy_T"][policy_index]
        result["P7_per_policy_T"][policy] = {
            "temperature": temperature,
        }
        for method in ("SMS", "SVS"):
            calibrator = models[method][policy_index]
            payload = {}
            if calibrator is not None:
                for name in ("W_", "v_", "b_"):
                    if hasattr(calibrator, name):
                        payload[name] = np.asarray(getattr(calibrator, name))
            result[method][policy] = payload
    return result


def _evaluate_models(
    labels: np.ndarray,
    logits: np.ndarray,
    models: dict[str, list],
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.empty((len(POLICY_IDS), len(METRICS)), dtype=np.float64)
    methods = np.full(
        (len(METHODS), len(POLICY_IDS), len(METRICS)), np.nan, dtype=np.float64
    )
    raw_probabilities = softmax(logits)
    for policy_index in range(len(POLICY_IDS)):
        raw[policy_index] = metric_vector(
            labels[:, policy_index], raw_probabilities[:, policy_index]
        )
        for method_index, method in enumerate(METHODS):
            probabilities = _predict_method(
                method,
                models[method][policy_index],
                logits[:, policy_index],
                raw_probabilities[:, policy_index],
            )
            if probabilities is not None:
                methods[method_index, policy_index] = metric_vector(
                    labels[:, policy_index], probabilities
                )
    return raw, methods


def _bootstrap_worker(
    task: tuple[int, int]
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[str]]:
    if _BOOT_DATA is None:
        raise RuntimeError("bootstrap worker was not initialized")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    start, stop = task
    budgets = _BOOT_DATA["budgets"]
    labels_calib = _BOOT_DATA["calib_labels"]
    logits_calib = _BOOT_DATA["calib_logits"]
    labels_test = _BOOT_DATA["test_labels"]
    logits_test = _BOOT_DATA["test_logits"]
    seed = _BOOT_DATA["seed"]
    factory = _BOOT_DATA.get("factory")
    n_local = stop - start
    raw = np.empty((n_local, len(POLICY_IDS), len(METRICS)), dtype=np.float64)
    methods = np.full(
        (n_local, len(budgets), len(METHODS), len(POLICY_IDS), len(METRICS)),
        np.nan,
        dtype=np.float64,
    )
    temperatures = np.full(
        (n_local, len(budgets), len(POLICY_IDS)), np.nan, dtype=np.float64
    )
    failure_counts: Counter = Counter()
    failure_examples: list[str] = []
    for local_index, replicate in enumerate(range(start, stop)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        test_rows = rng.integers(
            0, len(labels_test), size=len(labels_test), dtype=np.int32
        )
        test_raw, _ = _evaluate_models(
            labels_test[test_rows],
            logits_test[test_rows],
            {method: [None] * len(POLICY_IDS) for method in METHODS},
        )
        raw[local_index] = test_raw
        for budget_index, budget in enumerate(budgets):
            calib_rows = rng.integers(0, budget, size=budget, dtype=np.int32)
            fitted, failures = _fit_models(
                labels_calib[:budget][calib_rows],
                logits_calib[:budget][calib_rows],
                factory,
            )
            for method in METHODS:
                for message in failures[method]:
                    policy = message.split(":", maxsplit=1)[0]
                    failure_counts[f"{budget}:{method}"] += 1
                    failure_counts[f"{budget}:{method}:{policy}"] += 1
                if failures[method] and len(failure_examples) < 20:
                    failure_examples.extend(
                        f"replicate={replicate} budget={budget} {message}"
                        for message in failures[method][
                            : max(0, 20 - len(failure_examples))
                        ]
                    )
            for policy_index, value in enumerate(fitted["P7_per_policy_T"]):
                if value is not None:
                    temperatures[local_index, budget_index, policy_index] = float(value)
            _, evaluated = _evaluate_models(
                labels_test[test_rows], logits_test[test_rows], fitted
            )
            methods[local_index, budget_index] = evaluated
    return (
        start,
        stop,
        raw,
        methods,
        temperatures,
        dict(failure_counts),
        failure_examples[:20],
    )


def _initialize_bootstrap_worker(data: dict) -> None:
    global _BOOT_DATA
    _BOOT_DATA = data


def simultaneous_auroc_interval(
    observed_delta: np.ndarray, bootstrap_delta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Two-sided 95% studentized max-|t| simultaneous intervals."""
    observed_delta = np.asarray(observed_delta, dtype=np.float64)
    bootstrap_delta = np.asarray(bootstrap_delta, dtype=np.float64)
    if bootstrap_delta.ndim != 2 or bootstrap_delta.shape[1] != len(observed_delta):
        raise ValueError("bootstrap AUROC deltas must have shape [replicates, policies]")
    usable_rows = np.all(np.isfinite(bootstrap_delta), axis=1)
    if not np.all(np.isfinite(observed_delta)) or usable_rows.sum() < 2:
        empty = np.full_like(observed_delta, np.nan)
        return empty, empty.copy(), math.nan
    samples = bootstrap_delta[usable_rows]
    standard_error = samples.std(axis=0, ddof=1)
    usable_policies = standard_error > 1e-15
    standardized = np.zeros_like(samples)
    standardized[:, usable_policies] = (
        samples[:, usable_policies] - observed_delta[usable_policies]
    ) / standard_error[usable_policies]
    critical = float(np.quantile(np.max(np.abs(standardized), axis=1), 0.95))
    half_width = critical * standard_error
    return observed_delta - half_width, observed_delta + half_width, critical


def classify_verdict(
    *,
    mode: str,
    p2_status: str,
    evaluable: bool,
    conditions: dict[str, bool],
    harm: bool,
) -> tuple[str, str]:
    """Return one of the mutually exclusive locked P8 outcome categories."""
    if mode != "confirmation":
        return "DEVELOPMENT_ONLY", "old P7 artifacts are development evidence only"
    if p2_status != "PASS":
        return "NOT_REACHED", "P8 is tested only if P2-C and the base anchor pass"
    if not evaluable:
        return "NON_EVALUABLE", "dependency, fit, support, or numerical checks failed"
    if harm:
        return (
            "CONTRADICTED_OR_HARM",
            "raw calibration benefit is contradicted or discrimination harm is identified",
        )
    if all(conditions.values()):
        return "SUCCESS", "all primary SMS recovery and non-inferiority conditions pass"
    partial_keys = (
        "sms_reduction_vs_raw_ci_lower_gt_0",
        "mean_f1_ci_lower_ge_minus_0_005",
        "mean_auroc_ci_lower_ge_minus_0_01",
        "all_policy_simultaneous_auroc_lower_ge_minus_0_02",
    )
    if all(conditions[key] for key in partial_keys):
        return (
            "PARTIAL_SUPPORT",
            "SMS improves raw calibration without identified discrimination harm, but "
            "the absolute ceiling or paired advantage over P7 temperature is not established",
        )
    return "NOT_ESTABLISHED", "the primary SMS benefit is neither established nor contradicted"


def _aggregate(values: np.ndarray) -> np.ndarray:
    return _safe_nanmean(values, axis=-1)


def _method_summary(
    method_index: int,
    observed: np.ndarray,
    bootstrap: np.ndarray,
    observed_raw: np.ndarray,
    bootstrap_raw: np.ndarray,
    observed_p7: np.ndarray,
    bootstrap_p7: np.ndarray,
) -> dict:
    method_name = METHODS[method_index]
    per_policy = {}
    for policy_index, policy in enumerate(POLICY_IDS):
        metrics = {}
        for metric_index, metric in enumerate(METRICS):
            metrics[metric] = float(observed[policy_index, metric_index])
            metrics[f"{metric}_95ci"] = ci(bootstrap[:, policy_index, metric_index])
        metrics["ece_reduction_vs_raw"] = float(
            observed_raw[policy_index, METRIC_INDEX["ece"]]
            - observed[policy_index, METRIC_INDEX["ece"]]
        )
        metrics["ece_reduction_vs_raw_95ci"] = ci(
            bootstrap_raw[:, policy_index, METRIC_INDEX["ece"]]
            - bootstrap[:, policy_index, METRIC_INDEX["ece"]]
        )
        metrics["ece_improvement_vs_P7_T"] = float(
            observed_p7[policy_index, METRIC_INDEX["ece"]]
            - observed[policy_index, METRIC_INDEX["ece"]]
        )
        metrics["ece_improvement_vs_P7_T_95ci"] = ci(
            bootstrap_p7[:, policy_index, METRIC_INDEX["ece"]]
            - bootstrap[:, policy_index, METRIC_INDEX["ece"]]
        )
        for metric in ("f1", "auroc"):
            metric_index = METRIC_INDEX[metric]
            metrics[f"{metric}_change_vs_raw"] = float(
                observed[policy_index, metric_index]
                - observed_raw[policy_index, metric_index]
            )
            metrics[f"{metric}_change_vs_raw_95ci"] = ci(
                bootstrap[:, policy_index, metric_index]
                - bootstrap_raw[:, policy_index, metric_index]
            )
        per_policy[policy] = metrics
    aggregate = {}
    for metric_index, metric in enumerate(METRICS):
        aggregate[f"mean_{metric}"] = float(
            _strict_mean(observed[:, metric_index][None, :], axis=1)[0]
        )
        aggregate[f"mean_{metric}_95ci"] = ci(
            _strict_mean(bootstrap[:, :, metric_index], axis=1)
        )
    ece = METRIC_INDEX["ece"]
    raw_reduction = observed_raw[:, ece] - observed[:, ece]
    raw_reduction_boot = bootstrap_raw[:, :, ece] - bootstrap[:, :, ece]
    p7_improvement = observed_p7[:, ece] - observed[:, ece]
    p7_improvement_boot = bootstrap_p7[:, :, ece] - bootstrap[:, :, ece]
    aggregate.update(
        {
            "mean_ece_reduction_vs_raw": float(
                _strict_mean(raw_reduction[None, :], axis=1)[0]
            ),
            "mean_ece_reduction_vs_raw_95ci": ci(
                _strict_mean(raw_reduction_boot, axis=1)
            ),
            "mean_ece_improvement_vs_P7_T": float(
                _strict_mean(p7_improvement[None, :], axis=1)[0]
            ),
            "mean_ece_improvement_vs_P7_T_95ci": ci(
                _strict_mean(p7_improvement_boot, axis=1)
            ),
        }
    )
    for metric in ("f1", "auroc", "nll", "brier"):
        metric_index = METRIC_INDEX[metric]
        sign = 1.0 if metric in ("f1", "auroc") else -1.0
        delta = sign * (observed[:, metric_index] - observed_raw[:, metric_index])
        delta_boot = sign * (
            bootstrap[:, :, metric_index] - bootstrap_raw[:, :, metric_index]
        )
        name = "change" if metric in ("f1", "auroc") else "improvement"
        aggregate[f"mean_{metric}_{name}_vs_raw"] = float(
            _strict_mean(delta[None, :], axis=1)[0]
        )
        aggregate[f"mean_{metric}_{name}_vs_raw_95ci"] = ci(
            _strict_mean(delta_boot, axis=1)
        )
    return {"method": method_name, "per_policy": per_policy, "aggregate": aggregate}


def analyze_arrays(
    calib_labels: np.ndarray,
    calib_logits: np.ndarray,
    test_labels: np.ndarray,
    test_logits: np.ndarray,
    *,
    budgets: tuple[int, ...] = BUDGETS,
    bootstrap_replicates: int = 10_000,
    seed: int = SEED,
    jobs: int = 1,
    mode: str = "confirmation",
    p2_status: str = "UNKNOWN",
    factory: Callable[[str], object] | None = None,
    reference_integrity: bool = True,
    reference_integrity_details: dict | None = None,
) -> dict:
    """Core numerical analysis, exposed separately for deterministic CPU tests."""
    budgets = tuple(int(value) for value in budgets)
    if not budgets or PRIMARY_BUDGET not in budgets:
        raise ValueError("the fixed primary budget 500 must be included")
    if max(budgets) > len(calib_labels):
        raise ValueError("a requested budget exceeds target-calibration size")
    expected_shape = (len(calib_labels), len(POLICY_IDS))
    if calib_labels.shape != expected_shape or calib_logits.shape != (*expected_shape, 3):
        raise ValueError("calibration labels/logits have wrong shape")
    expected_test = (len(test_labels), len(POLICY_IDS))
    if test_labels.shape != expected_test or test_logits.shape != (*expected_test, 3):
        raise ValueError("test labels/logits have wrong shape")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    observed_raw, _ = _evaluate_models(
        test_labels,
        test_logits,
        {method: [None] * len(POLICY_IDS) for method in METHODS},
    )
    observed = np.full(
        (len(budgets), len(METHODS), len(POLICY_IDS), len(METRICS)),
        np.nan,
        dtype=np.float64,
    )
    observed_temperatures = np.full(
        (len(budgets), len(POLICY_IDS)), np.nan, dtype=np.float64
    )
    observed_failures: dict[str, dict[str, list[str]]] = {}
    point_diagnostics: dict[str, dict] = {}
    calib_nll: dict[str, dict] = {}
    for budget_index, budget in enumerate(budgets):
        models, failures = _fit_models(
            calib_labels[:budget], calib_logits[:budget], factory
        )
        observed_failures[str(budget)] = failures
        point_diagnostics[str(budget)] = _model_diagnostics(models)
        for policy_index, temperature in enumerate(models["P7_per_policy_T"]):
            if temperature is not None:
                observed_temperatures[budget_index, policy_index] = float(temperature)
        _, evaluated = _evaluate_models(test_labels, test_logits, models)
        observed[budget_index] = evaluated
        _, evaluated_calib = _evaluate_models(
            calib_labels[:budget], calib_logits[:budget], models
        )
        calib_nll[str(budget)] = {
            method: {
                policy: float(evaluated_calib[m, p, METRIC_INDEX["nll"]])
                for p, policy in enumerate(POLICY_IDS)
            }
            for m, method in enumerate(METHODS)
        }

    global _BOOT_DATA
    bootstrap_data = {
        "budgets": budgets,
        "calib_labels": calib_labels,
        "calib_logits": calib_logits,
        "test_labels": test_labels,
        "test_logits": test_logits,
        "seed": seed,
        "factory": factory,
    }
    _BOOT_DATA = bootstrap_data
    n_tasks = min(max(1, jobs * 4), bootstrap_replicates)
    boundaries = np.linspace(0, bootstrap_replicates, n_tasks + 1, dtype=int)
    tasks = [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(n_tasks)
        if boundaries[index] < boundaries[index + 1]
    ]
    bootstrap_start_method = "single_process"
    if jobs == 1:
        pieces = [_bootstrap_worker(task) for task in tasks]
    else:
        if factory is not None:
            raise ValueError("custom calibrator factories are supported only with jobs=1")
        available = mp.get_all_start_methods()
        method = "forkserver" if "forkserver" in available else "spawn"
        bootstrap_start_method = method
        if method == "forkserver":
            mp.set_forkserver_preload(["src.fit_g6"])
        with mp.get_context(method).Pool(
            processes=min(max(1, jobs), len(tasks)),
            initializer=_initialize_bootstrap_worker,
            initargs=(bootstrap_data,),
        ) as pool:
            pieces = pool.map(_bootstrap_worker, tasks)
    boot_raw = np.empty(
        (bootstrap_replicates, len(POLICY_IDS), len(METRICS)), dtype=np.float64
    )
    boot = np.empty(
        (
            bootstrap_replicates,
            len(budgets),
            len(METHODS),
            len(POLICY_IDS),
            len(METRICS),
        ),
        dtype=np.float64,
    )
    boot_temperatures = np.empty(
        (bootstrap_replicates, len(budgets), len(POLICY_IDS)), dtype=np.float64
    )
    bootstrap_failures: Counter = Counter()
    failure_examples: list[str] = []
    for start, stop, raw, methods, temperatures, failures, examples in pieces:
        boot_raw[start:stop] = raw
        boot[start:stop] = methods
        boot_temperatures[start:stop] = temperatures
        bootstrap_failures.update(failures)
        failure_examples.extend(examples[: max(0, 20 - len(failure_examples))])

    budgets_output = {}
    for budget_index, budget in enumerate(budgets):
        p7_index = METHODS.index("P7_per_policy_T")
        methods_output = {}
        for method_index, method in enumerate(METHODS):
            methods_output[method] = _method_summary(
                method_index,
                observed[budget_index, method_index],
                boot[:, budget_index, method_index],
                observed_raw,
                boot_raw,
                observed[budget_index, p7_index],
                boot[:, budget_index, p7_index],
            )
            if method == "P7_per_policy_T":
                for policy_index, policy in enumerate(POLICY_IDS):
                    methods_output[method]["per_policy"][policy]["temperature"] = float(
                        observed_temperatures[budget_index, policy_index]
                    )
                    methods_output[method]["per_policy"][policy]["temperature_95ci"] = ci(
                        boot_temperatures[:, budget_index, policy_index]
                    )
            for policy_index, policy in enumerate(POLICY_IDS):
                train_nll = calib_nll[str(budget)][method][policy]
                test_nll = methods_output[method]["per_policy"][policy]["nll"]
                methods_output[method]["per_policy"][policy]["target_calib_fit_nll"] = train_nll
                methods_output[method]["per_policy"][policy]["test_minus_calib_nll_gap"] = (
                    test_nll - train_nll
                )
        budgets_output[str(budget)] = {
            "role": "PRIMARY" if budget == PRIMARY_BUDGET else "SECONDARY",
            "calib_support": {
                policy: support(calib_labels[:budget, policy_index])
                for policy_index, policy in enumerate(POLICY_IDS)
            },
            "methods": methods_output,
            "point_fit_failures": observed_failures[str(budget)],
            "point_fit_parameters": point_diagnostics[str(budget)],
        }

    primary_index = budgets.index(PRIMARY_BUDGET)
    sms_index = METHODS.index("SMS")
    p7_index = METHODS.index("P7_per_policy_T")
    ece_index = METRIC_INDEX["ece"]
    f1_index = METRIC_INDEX["f1"]
    auroc_index = METRIC_INDEX["auroc"]
    sms = observed[primary_index, sms_index]
    sms_boot = boot[:, primary_index, sms_index]
    p7 = observed[primary_index, p7_index]
    p7_boot = boot[:, primary_index, p7_index]
    reduction_boot = _strict_mean(
        boot_raw[:, :, ece_index] - sms_boot[:, :, ece_index], axis=1
    )
    improvement_boot = _strict_mean(
        p7_boot[:, :, ece_index] - sms_boot[:, :, ece_index], axis=1
    )
    sms_ece_boot = _strict_mean(sms_boot[:, :, ece_index], axis=1)
    reduction_ci = ci(reduction_boot)
    improvement_ci = ci(improvement_boot)
    sms_ece_ci = ci(sms_ece_boot)
    f1_change = sms[:, f1_index] - observed_raw[:, f1_index]
    f1_change_boot = sms_boot[:, :, f1_index] - boot_raw[:, :, f1_index]
    auroc_change = sms[:, auroc_index] - observed_raw[:, auroc_index]
    auroc_change_boot = sms_boot[:, :, auroc_index] - boot_raw[:, :, auroc_index]
    mean_f1_boot = _strict_mean(f1_change_boot, axis=1)
    mean_auroc_boot = _strict_mean(auroc_change_boot, axis=1)
    mean_f1_ci = ci(mean_f1_boot)
    mean_auroc_ci = ci(mean_auroc_boot)
    simultaneous_lower, simultaneous_upper, simultaneous_critical = simultaneous_auroc_interval(
        auroc_change, auroc_change_boot
    )
    conditions = {
        "sms_reduction_vs_raw_ci_lower_gt_0": bool(reduction_ci[0] > 0.0),
        "sms_mean_ece_ci_upper_le_0_05": bool(sms_ece_ci[1] <= 0.05),
        "sms_improvement_vs_P7_T_ci_lower_gt_0": bool(improvement_ci[0] > 0.0),
        "mean_f1_ci_lower_ge_minus_0_005": bool(mean_f1_ci[0] >= -0.005),
        "mean_auroc_ci_lower_ge_minus_0_01": bool(mean_auroc_ci[0] >= -0.01),
        "all_policy_simultaneous_auroc_lower_ge_minus_0_02": bool(
            np.all(np.isfinite(simultaneous_lower))
            and np.min(simultaneous_lower) >= -0.02
        ),
    }
    primary_methods = ("P7_per_policy_T", "SMS")
    primary_point_failures = sum(
        len(observed_failures[str(PRIMARY_BUDGET)][method])
        for method in primary_methods
    )
    primary_bootstrap_failures = sum(
        bootstrap_failures.get(f"{PRIMARY_BUDGET}:{method}", 0)
        for method in primary_methods
    )
    primary_bootstrap_fit_count = (
        bootstrap_replicates * len(POLICY_IDS) * len(primary_methods)
    )
    primary_failure_rate = (
        primary_bootstrap_failures / primary_bootstrap_fit_count
        if primary_bootstrap_fit_count
        else math.nan
    )
    per_policy_failure_rates = {
        method: {
            policy: (
                bootstrap_failures.get(
                    f"{PRIMARY_BUDGET}:{method}:{policy}", 0
                )
                / bootstrap_replicates
            )
            for policy in POLICY_IDS
        }
        for method in primary_methods
    }
    primary_fit_guard = bool(
        primary_point_failures == 0
        and primary_failure_rate <= 0.01
        and all(
            rate <= 0.05
            for method_rates in per_policy_failure_rates.values()
            for rate in method_rates.values()
        )
    )
    primary_finite = bool(
        np.all(np.isfinite(sms))
        and np.all(np.isfinite(p7))
        and np.all(np.isfinite(observed_raw))
    )
    support_ok = bool(
        all(
            np.any(test_labels[:, policy_index] == 0)
            and np.any(test_labels[:, policy_index] == 1)
            for policy_index in range(len(POLICY_IDS))
        )
    )
    complete_primary_replicates = {
        "raw_reduction": int(np.isfinite(reduction_boot).sum()),
        "increment_over_temperature": int(np.isfinite(improvement_boot).sum()),
        "SMS_mean_ECE": int(np.isfinite(sms_ece_boot).sum()),
        "mean_F1_change": int(np.isfinite(mean_f1_boot).sum()),
        "mean_AUROC_change": int(np.isfinite(mean_auroc_boot).sum()),
    }
    valid_primary_replicates = bool(
        all(count >= 2 for count in complete_primary_replicates.values())
    )
    evaluable = bool(
        reference_integrity
        and primary_fit_guard
        and primary_finite
        and support_ok
        and valid_primary_replicates
        and np.all(np.isfinite(simultaneous_lower))
        and np.all(np.isfinite(simultaneous_upper))
    )
    # CONTRADICTED_OR_HARM is reserved for an identified reversal or failed
    # utility guardrail, not merely for absence of positive evidence.
    harm = bool(
        ci(reduction_boot)[1] <= 0.0
        or mean_f1_ci[1] < -0.005
        or mean_auroc_ci[1] < -0.01
        or (
            np.all(np.isfinite(simultaneous_upper))
            and np.min(simultaneous_upper) < -0.02
        )
    )
    verdict, reason = classify_verdict(
        mode=mode,
        p2_status=p2_status,
        evaluable=evaluable,
        conditions=conditions,
        harm=harm,
    )
    verdict_tags = []
    if verdict == "PARTIAL_SUPPORT":
        if not conditions["sms_mean_ece_ci_upper_le_0_05"]:
            verdict_tags.append("TOLERANCE_NOT_ESTABLISHED")
        if not conditions["sms_improvement_vs_P7_T_ci_lower_gt_0"]:
            verdict_tags.append("NO_INCREMENT_OVER_TEMPERATURE")
    return {
        "protocol": {
            "mode": mode,
            "reference_integrity": reference_integrity_details
            or {"pass": reference_integrity},
            "probmetrics": (
                probmetrics_status()
                if factory is None
                else {"test_factory": True, "required_version": REQUIRED_PROBMETRICS_VERSION}
            ),
            "methods": list(METHODS),
            "metrics": list(METRICS),
            "budgets": list(budgets),
            "primary_budget": PRIMARY_BUDGET,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "bootstrap_rng": "PCG64 SeedSequence([seed, replicate]); test then ascending-budget calibration draws",
            "multiprocessing_start_method": bootstrap_start_method,
            "paired_two_stage": True,
            "shared_resamples_across_methods": True,
            "refit_all_methods_each_calibration_resample": True,
            "SMS_configuration": "SMSCalibrator() exact 1.3.0 defaults",
            "SVS_configuration": "SVSCalibrator() exact 1.3.0 defaults",
            "P7_baseline": "per-policy temperature refit using src.fit_g4.fit_temperature",
            "brier_definition": "mean sum_c (p_c - onehot_c)^2",
            "simultaneous_auroc": "two-sided 95% studentized paired max-|t| intervals",
        },
        "raw_test": {
            policy: {
                **{
                    metric: float(observed_raw[policy_index, metric_index])
                    for metric_index, metric in enumerate(METRICS)
                },
                **{
                    f"{metric}_95ci": ci(boot_raw[:, policy_index, metric_index])
                    for metric_index, metric in enumerate(METRICS)
                },
                "support": support(test_labels[:, policy_index]),
            }
            for policy_index, policy in enumerate(POLICY_IDS)
        },
        "budgets": budgets_output,
        "fit_health": {
            "observed_failures": observed_failures,
            "bootstrap_failure_counts": dict(bootstrap_failures),
            "bootstrap_failure_examples": failure_examples[:20],
            "primary_methods": list(primary_methods),
            "primary_point_failure_count": primary_point_failures,
            "primary_bootstrap_failure_count": primary_bootstrap_failures,
            "primary_bootstrap_fit_count": primary_bootstrap_fit_count,
            "primary_bootstrap_failure_rate": primary_failure_rate,
            "per_policy_bootstrap_failure_rate": per_policy_failure_rates,
            "locked_overall_failure_ceiling": 0.01,
            "locked_per_policy_failure_ceiling": 0.05,
            "primary_fit_guard_pass": primary_fit_guard,
            "test_support_guard_pass": support_ok,
            "valid_primary_replicates_guard_pass": valid_primary_replicates,
            "complete_primary_replicates": complete_primary_replicates,
        },
        "verdict": {
            "P2_base_anchor_status": p2_status,
            "primary_budget": PRIMARY_BUDGET,
            "conditions": conditions,
            "primary_statistics": {
                "SMS_mean_ece_95ci": sms_ece_ci,
                "SMS_reduction_vs_raw_95ci": reduction_ci,
                "SMS_improvement_vs_P7_T_95ci": improvement_ci,
                "SMS_mean_f1_change_vs_raw_95ci": mean_f1_ci,
                "SMS_mean_auroc_change_vs_raw_95ci": mean_auroc_ci,
                "SMS_policy_auroc_change": {
                    policy: float(auroc_change[index])
                    for index, policy in enumerate(POLICY_IDS)
                },
                "SMS_policy_simultaneous_auroc_lower": {
                    policy: float(simultaneous_lower[index])
                    for index, policy in enumerate(POLICY_IDS)
                },
                "SMS_policy_simultaneous_auroc_upper": {
                    policy: float(simultaneous_upper[index])
                    for index, policy in enumerate(POLICY_IDS)
                },
                "simultaneous_critical_value": simultaneous_critical,
            },
            "evaluable": evaluable,
            "harm_detected": harm,
            "P8": verdict,
            "reason": reason,
            "tags": verdict_tags,
            "mutually_exclusive_categories": [
                "NOT_REACHED",
                "NON_EVALUABLE",
                "SUCCESS",
                "CONTRADICTED_OR_HARM",
                "PARTIAL_SUPPORT",
                "NOT_ESTABLISHED",
            ],
            "development_only_category": "DEVELOPMENT_ONLY",
            "frozen_P4_P7_unchanged": True,
        },
    }


def _manifest_ids(path: Path) -> tuple[list[str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids):
        raise ValueError(f"manifest has no valid ordered ID list: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate IDs: {path}")
    return ids, payload


def load_labels_logits_with_missing(
    labels_path: Path, logits_path: Path
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    label_rows = read_jsonl(labels_path)
    logit_rows = read_jsonl(logits_path)
    label_ids = [str(row.get("id")) for row in label_rows]
    logit_ids = [str(row.get("id")) for row in logit_rows]
    if (
        not label_ids
        or label_ids != logit_ids
        or len(label_ids) != len(set(label_ids))
    ):
        raise ValueError("labels/logits must have identical, non-empty, unique ID order")
    labels = np.full((len(label_rows), len(POLICY_IDS)), -1, dtype=np.int8)
    logits = np.empty((len(logit_rows), len(POLICY_IDS), 3), dtype=np.float64)
    strict_successes = 0
    for row_index, (label_row, logit_row) in enumerate(zip(label_rows, logit_rows)):
        raw_labels = label_row.get("labels")
        strict = bool(
            label_row.get("parse_ok", raw_labels is not None)
            and isinstance(raw_labels, dict)
            and set(raw_labels) == set(POLICY_IDS)
            and all(raw_labels.get(policy) in LABEL_TO_ID for policy in POLICY_IDS)
        )
        strict_successes += int(strict)
        raw_logits = logit_row.get("logits")
        if not isinstance(raw_logits, dict) or set(raw_logits) != set(POLICY_IDS):
            raise ValueError(f"logit row {row_index} lacks exactly ten policy heads")
        for policy_index, policy in enumerate(POLICY_IDS):
            state = raw_labels.get(policy) if isinstance(raw_labels, dict) else None
            if state in LABEL_TO_ID:
                labels[row_index, policy_index] = LABEL_TO_ID[state]
            values = np.asarray(raw_logits[policy], dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid logits for {policy} at row {row_index}")
            logits[row_index, policy_index] = values
    missing_rates = {
        policy: float(np.mean(labels[:, policy_index] < 0))
        for policy_index, policy in enumerate(POLICY_IDS)
    }
    integrity = {
        "strict_ten_key_success_rate": strict_successes / len(label_rows),
        "missing_rate_by_policy": missing_rates,
        "thresholds": {
            "strict_ten_key_success_rate_min": 0.99,
            "per_policy_missing_rate_max": 0.01,
        },
    }
    integrity["pass"] = bool(
        integrity["strict_ten_key_success_rate"] >= 0.99
        and all(rate <= 0.01 for rate in missing_rates.values())
    )
    return label_ids, labels, logits, integrity


def load_frozen_inputs(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    frozen_paths = [args.calib_manifest, args.test_manifest]
    if args.mode == "confirmation":
        frozen_paths.extend([args.labels, args.logits])
    validate_manifest_hashes(
        args.split_hash_manifest, frozen_paths
    )
    calib_ids, calib_manifest = _manifest_ids(args.calib_manifest)
    test_ids, test_manifest = _manifest_ids(args.test_manifest)
    if set(calib_ids) & set(test_ids):
        raise ValueError("target-calibration and target-test IDs overlap")
    ids, labels, logits, reference_integrity = load_labels_logits_with_missing(
        args.labels, args.logits
    )
    if len(ids) != len(set(ids)):
        raise ValueError("labels/logits contain duplicate IDs")
    if set(calib_ids) | set(test_ids) != set(ids):
        raise ValueError("frozen split manifests are not an exhaustive partition of inputs")
    if args.mode == "confirmation":
        expected = (EXPECTED_CONFIRM_CALIB, EXPECTED_CONFIRM_TEST)
    else:
        expected = (EXPECTED_DEVELOPMENT_CALIB, EXPECTED_DEVELOPMENT_TEST)
    if (len(calib_ids), len(test_ids)) != expected:
        raise ValueError(
            f"{args.mode} mode requires calib/test sizes {expected}, got "
            f"{(len(calib_ids), len(test_ids))}"
        )
    for payload in (calib_manifest, test_manifest):
        if payload.get("source_labels_sha256") not in (None, sha256(args.labels)):
            raise ValueError("label artifact differs from frozen split manifest")
        if payload.get("source_logits_sha256") not in (None, sha256(args.logits)):
            raise ValueError("logit artifact differs from frozen split manifest")
    row = {item_id: index for index, item_id in enumerate(ids)}
    calib_rows = np.asarray([row[item_id] for item_id in calib_ids], dtype=np.int64)
    test_rows = np.asarray([row[item_id] for item_id in test_ids], dtype=np.int64)
    hashes = {
        "labels": sha256(args.labels),
        "logits": sha256(args.logits),
        "calib_manifest": sha256(args.calib_manifest),
        "test_manifest": sha256(args.test_manifest),
        "split_hash_manifest": sha256(args.split_hash_manifest),
    }
    return (
        labels[calib_rows],
        logits[calib_rows],
        labels[test_rows],
        logits[test_rows],
        hashes,
        reference_integrity,
    )


def run_fit(args: argparse.Namespace) -> None:
    if args.bootstrap != 10_000 or args.seed != SEED:
        raise ValueError("P8 requires 10,000 bootstrap replicates and seed 20260724")
    if args.mode == "confirmation" and args.p2_status == "UNKNOWN":
        raise ValueError("confirmation mode requires an explicit --p2_status PASS or FAIL")
    if args.out.exists() or (args.plot is not None and args.plot.exists()):
        raise FileExistsError("refusing to overwrite a P8 output artifact")
    if args.mode == "confirmation" and args.p2_status != "PASS":
        result = {
            "protocol": {
                "mode": args.mode,
                "P2_base_anchor_status": args.p2_status,
                "target_inputs_read": False,
                "frozen_P4_P7_unchanged": True,
            },
            "verdict": {
                "P8": "NOT_REACHED",
                "reason": "P8 is tested only if P2-C and the base anchor pass",
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["verdict"], indent=2))
        return
    _, _, dependency = require_probmetrics()
    (
        calib_labels,
        calib_logits,
        test_labels,
        test_logits,
        hashes,
        reference_integrity,
    ) = load_frozen_inputs(args)
    result = analyze_arrays(
        calib_labels,
        calib_logits,
        test_labels,
        test_logits,
        budgets=BUDGETS,
        bootstrap_replicates=args.bootstrap,
        seed=args.seed,
        jobs=args.jobs,
        mode=args.mode,
        p2_status=args.p2_status,
        reference_integrity=bool(reference_integrity["pass"]),
        reference_integrity_details=reference_integrity,
    )
    result["protocol"]["probmetrics"] = dependency
    result["protocol"]["input_sha256"] = hashes
    result["protocol"]["split"] = {
        "target_calib_n": len(calib_labels),
        "target_test_n": len(test_labels),
        "overlap": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        for method in METHODS:
            means = [
                result["budgets"][str(budget)]["methods"][method]["aggregate"]["mean_ece"]
                for budget in BUDGETS
            ]
            intervals = [
                result["budgets"][str(budget)]["methods"][method]["aggregate"]["mean_ece_95ci"]
                for budget in BUDGETS
            ]
            low = [max(0.0, mean - interval[0]) for mean, interval in zip(means, intervals)]
            high = [max(0.0, interval[1] - mean) for mean, interval in zip(means, intervals)]
            axis.errorbar(
                BUDGETS,
                means,
                yerr=np.asarray([low, high]),
                marker="o",
                capsize=3,
                label=method,
            )
        axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="base anchor ceiling")
        axis.set_xscale("log")
        axis.set_xticks(BUDGETS, [str(value) for value in BUDGETS])
        axis.set_xlabel("Target calibration prompts")
        axis.set_ylabel("Target-test mean ECE")
        axis.set_title(f"P8 structured recalibration: {result['verdict']['P8']}")
        axis.legend(fontsize=8)
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=180)
        plt.close(figure)
    print(json.dumps(to_jsonable(result["verdict"]), indent=2))
    print(f"result={args.out}")


def resolve_paths(args: argparse.Namespace) -> None:
    root = Path(os.environ.get("PCCD_OUT", "outputs"))
    if args.mode == "development":
        defaults = {
            "labels": root / "g2" / "D5_teacher.jsonl",
            "logits": root / "g2" / "D5_logits.jsonl",
            "calib_manifest": root / "results" / "g5_target_calib_ids.json",
            "test_manifest": root / "results" / "g5_target_test_ids.json",
            "split_hash_manifest": root / "results" / "g5_split_manifests.sha256",
            "out": root / "results" / "g6_development.json",
            "plot": root / "results" / "g6_development.png",
        }
    else:
        defaults = {
            "labels": root / "confirmation" / "D5_seed20260723_teacher.jsonl",
            "logits": root / "confirmation" / "D5_seed20260723_logits.jsonl",
            "calib_manifest": root / "confirmation" / "confirmation_target_calib_ids.json",
            "test_manifest": root / "confirmation" / "confirmation_test_ids.json",
            "split_hash_manifest": root / "confirmation" / "confirmation_preunseal.sha256",
            "out": root / "results" / "g6_confirmation.json",
            "plot": root / "results" / "g6_confirmation.png",
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-dependency")
    fit = subparsers.add_parser("fit")
    fit.add_argument("--mode", choices=("confirmation", "development"), default="confirmation")
    fit.add_argument("--labels", type=Path)
    fit.add_argument("--logits", type=Path)
    fit.add_argument("--calib_manifest", type=Path)
    fit.add_argument("--test_manifest", type=Path)
    fit.add_argument("--split_hash_manifest", type=Path)
    fit.add_argument("--out", type=Path)
    fit.add_argument("--plot", type=Path)
    fit.add_argument("--p2_status", choices=("PASS", "FAIL", "UNKNOWN"), default="UNKNOWN")
    fit.add_argument("--bootstrap", type=int, default=10_000)
    fit.add_argument("--seed", type=int, default=SEED)
    fit.add_argument("--jobs", type=int, default=min(80, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check-dependency":
        status = probmetrics_status()
        print(json.dumps(status, indent=2))
        if not status["available"]:
            raise SystemExit(2)
        return
    resolve_paths(args)
    run_fit(args)


if __name__ == "__main__":
    main()
