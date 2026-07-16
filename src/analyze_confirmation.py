#!/usr/bin/env python3
"""Locked lockbox analysis for the P2/P3 confirmation experiment.

This module deliberately has no calibration-split input.  It reads only the
frozen CONFIRM-TEST manifest plus aligned D0/new-D5 (and optionally old-D5)
teacher labels and frozen-critic logits.  The primary analysis is the new D5
seed; the old D5 adapter on the same prompts is permanently secondary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.policy_defs import LABEL_STATES, POLICY_IDS

LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_STATES)}
ID_TO_LABEL = dict(enumerate(LABEL_STATES))


BOOTSTRAP_SEED = 20260733
LOCKED_BOOTSTRAP = 10_000
LOCKED_TEST_COUNT = 3_500
N_BINS = 15
PINV_RCOND = 1e-12
MATERIAL_SD_FLOOR = 0.01
MATERIAL_MEAN_FLOOR = 0.01
BASE_ECE_CEILING = 0.05
_BOOT_DATA: dict[str, Any] | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    ids: list[str]
    labels: np.ndarray  # [items, policies]
    logits: np.ndarray  # [items, policies, 3]
    probabilities: np.ndarray  # [items, policies, 3]
    strict_json_success_rate: float
    missing_rate_by_policy: dict[str, float]

    @property
    def reference_integrity_pass(self) -> bool:
        return bool(
            self.strict_json_success_rate >= 0.99
            and all(rate <= 0.01 for rate in self.missing_rate_by_policy.values())
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=-1, keepdims=True)


def _resolve_manifest_path(raw: str, manifest: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def load_hash_manifest(path: Path) -> dict[Path, str]:
    """Read either a sha256sum file or a JSON file map/list."""

    text = path.read_text(encoding="utf-8")
    entries: dict[Path, str] = {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if payload is None:
        for line in text.splitlines():
            if not line.strip():
                continue
            pieces = line.strip().split(maxsplit=1)
            if len(pieces) != 2 or len(pieces[0]) != 64:
                raise ValueError(f"invalid sha256sum line: {line!r}")
            raw_path = pieces[1].lstrip("*")
            entries[_resolve_manifest_path(raw_path, path)] = pieces[0].lower()
        return entries

    if not isinstance(payload, dict):
        raise ValueError("hash manifest JSON must be an object")
    raw_entries: Any = payload.get("files", payload)
    if isinstance(raw_entries, dict):
        iterable = []
        for raw_path, value in raw_entries.items():
            if isinstance(value, str):
                iterable.append((raw_path, value))
            elif isinstance(value, dict):
                iterable.append((value.get("path", raw_path), value.get("sha256")))
            else:
                raise ValueError("invalid hash-manifest value")
    elif isinstance(raw_entries, list):
        iterable = [(item.get("path"), item.get("sha256")) for item in raw_entries]
    else:
        raise ValueError("hash manifest 'files' must be an object or list")
    for raw_path, digest in iterable:
        if not isinstance(raw_path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("each hash entry requires path and 64-character sha256")
        entries[_resolve_manifest_path(raw_path, path)] = digest.lower()
    return entries


def verify_hashes(required: list[Path], hash_manifest: Path) -> dict[str, str]:
    entries = load_hash_manifest(hash_manifest)
    observed: dict[str, str] = {}
    for original in required:
        path = original.resolve()
        if path not in entries:
            raise ValueError(f"required input absent from hash manifest: {path}")
        digest = sha256_file(path)
        if digest != entries[path]:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {entries[path]}, observed {digest}"
            )
        observed[str(path)] = digest
    return observed


def _family_from_record(record: dict) -> str | None:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    for container in (record, meta):
        for key in ("family_id", "query_family_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def referenced_prompt_artifact(test_manifest: Path) -> Path | None:
    payload = json.loads(test_manifest.read_text(encoding="utf-8"))
    raw = payload.get("prompt_artifact") if isinstance(payload, dict) else None
    if not isinstance(raw, str) or not raw:
        return None
    return _resolve_manifest_path(raw, test_manifest)


def load_test_manifest(path: Path, expected_count: int) -> tuple[list[str], list[str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("test manifest must be a JSON object")
    role = str(payload.get("role", payload.get("split", ""))).upper().replace("_", "-")
    if role not in {"CONFIRM-TEST", "P8-CONFIRM-TEST"}:
        raise ValueError("manifest role must be CONFIRM-TEST or P8-CONFIRM-TEST")
    if payload.get("frozen_before_outcomes") is not True:
        raise ValueError("test manifest must assert frozen_before_outcomes=true")
    if int(payload.get("overlap_with_target_calib", -1)) != 0:
        raise ValueError("test manifest must assert zero TARGET-CALIB overlap")

    if isinstance(payload.get("items"), list):
        ids = [str(item["id"]) for item in payload["items"]]
        families = [_family_from_record(item) for item in payload["items"]]
    else:
        ids = [str(item) for item in payload.get("ids", [])]
        raw_families = payload.get("family_ids", payload.get("query_family_ids"))
        if isinstance(raw_families, dict):
            families = [str(raw_families[item]) for item in ids]
        elif isinstance(raw_families, list):
            families = [str(item) for item in raw_families]
        else:
            artifact = referenced_prompt_artifact(path)
            if artifact is None or not artifact.is_file():
                raise ValueError(
                    "test manifest requires aligned family IDs or a readable prompt_artifact"
                )
            expected_prompt_hash = payload.get("prompt_artifact_sha256")
            if not isinstance(expected_prompt_hash, str) or sha256_file(artifact) != expected_prompt_hash:
                raise ValueError("prompt artifact does not match test-manifest SHA-256")
            prompt_rows = read_jsonl(artifact)
            prompt_by_id = {str(row.get("id")): row for row in prompt_rows}
            if len(prompt_by_id) != len(prompt_rows) or any(item not in prompt_by_id for item in ids):
                raise ValueError("prompt artifact lacks unique records for every test ID")
            families = [_family_from_record(prompt_by_id[item]) for item in ids]
    if not ids or len(ids) != len(families):
        raise ValueError("test IDs and family IDs must be non-empty and aligned")
    if expected_count > 0 and len(ids) != expected_count:
        raise ValueError(f"expected {expected_count} test rows, observed {len(ids)}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate test IDs")
    # The locked sampler selects one prompt per query family.  Enforcing that here
    # makes row bootstrap exactly query-family bootstrap and prevents pseudo-replication.
    if any(not isinstance(item, str) or not item for item in families) or len(families) != len(set(families)):
        raise ValueError("family IDs must be non-empty and unique (one row per query family)")
    return ids, [str(item) for item in families], payload


def load_source_edges(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source-bin file must be a JSON object")
    role = str(payload.get("role", payload.get("split", ""))).upper().replace("_", "-")
    if role not in {"SOURCE-BASE-CALIB", "BASE-CALIB"}:
        raise ValueError("fixed bin edges must declare source-base-calib provenance")
    raw = payload.get("edges", payload.get("per_policy"))
    if not isinstance(raw, dict) or set(raw) != set(POLICY_IDS):
        raise ValueError("source-bin file must contain exactly the ten policies")
    result: dict[str, np.ndarray] = {}
    for policy in POLICY_IDS:
        values = raw[policy]
        if isinstance(values, dict):
            values = values.get("edges")
        edges = np.asarray(values, dtype=np.float64)
        if (
            edges.shape != (N_BINS + 1,)
            or not np.all(np.isfinite(edges))
            or not np.isclose(edges[0], 0.0)
            or not np.isclose(edges[-1], 1.0)
            or np.any(np.diff(edges) < 0)
        ):
            raise ValueError(f"invalid fixed source edges for {policy}")
        result[policy] = edges
    return result


def load_variant(name: str, labels_path: Path, logits_path: Path, expected_ids: list[str]) -> Variant:
    label_rows = read_jsonl(labels_path)
    logit_rows = read_jsonl(logits_path)
    label_ids = [str(row.get("id")) for row in label_rows]
    logit_ids = [str(row.get("id")) for row in logit_rows]
    if label_ids != expected_ids or logit_ids != expected_ids:
        raise ValueError(f"{name} labels/logits are not exactly test-manifest aligned")
    labels = np.full((len(expected_ids), len(POLICY_IDS)), -1, dtype=np.int8)
    logits = np.empty((len(expected_ids), len(POLICY_IDS), 3), dtype=np.float64)
    strict_successes = 0
    for row_index, (label_row, logit_row) in enumerate(zip(label_rows, logit_rows)):
        if set(logit_row.get("logits", {})) != set(POLICY_IDS):
            raise ValueError(f"{name} logit row {row_index} lacks strict ten-policy schema")
        raw_labels = label_row.get("labels")
        strict = bool(
            label_row.get("parse_ok", raw_labels is not None)
            and isinstance(raw_labels, dict)
            and set(raw_labels) == set(POLICY_IDS)
            and all(raw_labels.get(policy) in LABEL_TO_ID for policy in POLICY_IDS)
        )
        strict_successes += int(strict)
        for policy_index, policy in enumerate(POLICY_IDS):
            state = raw_labels.get(policy) if isinstance(raw_labels, dict) else None
            if state in LABEL_TO_ID:
                labels[row_index, policy_index] = LABEL_TO_ID[state]
            values = np.asarray(logit_row["logits"][policy], dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} invalid logits for {policy} at row {row_index}")
            logits[row_index, policy_index] = values
    missing = {
        policy: float(np.mean(labels[:, policy_index] < 0))
        for policy_index, policy in enumerate(POLICY_IDS)
    }
    return Variant(
        name,
        expected_ids,
        labels,
        logits,
        softmax(logits),
        strict_successes / len(expected_ids),
        missing,
    )


def percentile_ci(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [math.nan, math.nan]
    return np.quantile(finite, [0.025, 0.975]).tolist()


def fixed_bin_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    if len(labels) == 0 or weights.shape != (len(labels),) or np.any(weights < 0):
        return math.nan
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return math.nan
    bins = np.searchsorted(edges[1:-1], confidence, side="right")
    result = 0.0
    for index in range(len(edges) - 1):
        selected = bins == index
        mass = float(weights[selected].sum())
        if mass > 0:
            accuracy = float(np.average(correct[selected], weights=weights[selected]))
            mean_confidence = float(np.average(confidence[selected], weights=weights[selected]))
            result += (mass / total) * abs(accuracy - mean_confidence)
    return float(result)


def multiclass_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Frozen top-class ECE, reproduced without importing the torch evaluator."""

    return fixed_bin_ece(
        probabilities,
        labels,
        np.linspace(0.0, 1.0, n_bins + 1),
    )


def adaptive_multiclass_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ordered = np.argsort(confidence)
    result = 0.0
    for selected in np.array_split(ordered, min(n_bins, len(ordered))):
        if len(selected):
            result += (len(selected) / len(labels)) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(result)


def binary_ece(scores: np.ndarray, truth: np.ndarray, n_bins: int = N_BINS) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.searchsorted(edges[1:-1], scores, side="right")
    result = 0.0
    for index in range(n_bins):
        selected = bins == index
        if np.any(selected):
            result += (selected.sum() / len(scores)) * abs(
                float(truth[selected].mean()) - float(scores[selected].mean())
            )
    return float(result)


def classwise_ece(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = {
        ID_TO_LABEL[index]: binary_ece(probabilities[:, index], labels == index)
        for index in range(3)
    }
    values["macro"] = float(np.mean(list(values.values())))
    return values


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    target = np.eye(3, dtype=np.float64)[labels]
    return float(np.mean(np.sum(np.square(probabilities - target), axis=1)))


def logistic_calibration(scores: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    if not np.any(truth == 0) or not np.any(truth == 1):
        return {"intercept": math.nan, "slope": math.nan, "converged": False, "reason": "one_class"}
    x = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1.0))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * x
        loss = float(np.sum(np.logaddexp(0.0, eta) - truth * eta))
        residual = expit(eta) - truth
        gradient = np.asarray([residual.sum(), np.dot(residual, x)], dtype=np.float64)
        return loss, gradient

    fitted = minimize(
        lambda theta: objective(theta)[0],
        np.asarray([0.0, 1.0]),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-10.0, 10.0)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    finite = bool(np.all(np.isfinite(fitted.x)) and math.isfinite(float(fitted.fun)))
    return {
        "intercept": float(fitted.x[0]) if finite else math.nan,
        "slope": float(fitted.x[1]) if finite else math.nan,
        "converged": bool(fitted.success and finite),
        "reason": None if fitted.success and finite else str(fitted.message),
    }


def calibration_parameters(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, dict]:
    return {
        ID_TO_LABEL[index]: logistic_calibration(probabilities[:, index], labels == index)
        for index in range(3)
    }


def histogram_overlap(left: np.ndarray, right: np.ndarray, bins: int = 50) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    left_mass = np.histogram(left, bins=edges)[0].astype(np.float64)
    right_mass = np.histogram(right, bins=edges)[0].astype(np.float64)
    left_mass /= left_mass.sum()
    right_mass /= right_mass.sum()
    return float(np.minimum(left_mass, right_mass).sum())


def support_overlap(base: np.ndarray, adapted: np.ndarray) -> dict[str, float]:
    result = {
        ID_TO_LABEL[index]: histogram_overlap(base[:, index], adapted[:, index])
        for index in range(3)
    }
    result["top_confidence"] = histogram_overlap(base.max(axis=1), adapted.max(axis=1))
    return result


def prevalence_weights(source_labels: np.ndarray, target_labels: np.ndarray) -> tuple[np.ndarray, float]:
    source_counts = np.bincount(source_labels, minlength=3).astype(np.float64)
    target_counts = np.bincount(target_labels, minlength=3).astype(np.float64)
    source_prevalence = source_counts / source_counts.sum()
    target_prevalence = target_counts / target_counts.sum()
    if np.any((source_prevalence > 0) & (target_prevalence == 0)):
        return np.full(len(target_labels), np.nan), math.nan
    ratios = np.divide(
        source_prevalence,
        target_prevalence,
        out=np.zeros(3, dtype=np.float64),
        where=target_prevalence > 0,
    )
    weights = ratios[target_labels]
    denominator = float(np.square(weights).sum())
    ess = float(weights.sum() ** 2 / denominator) if denominator > 0 else math.nan
    return weights, ess


def standardized_deltas(
    base_labels: np.ndarray,
    base_probabilities: np.ndarray,
    adapted_labels: np.ndarray,
    adapted_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    target_to_base = np.full(len(POLICY_IDS), np.nan, dtype=np.float64)
    base_to_target = np.full(len(POLICY_IDS), np.nan, dtype=np.float64)
    diagnostics: list[dict[str, float]] = []
    equal_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    for policy_index in range(len(POLICY_IDS)):
        base_valid = base_labels[:, policy_index] >= 0
        adapted_valid = adapted_labels[:, policy_index] >= 0
        base_y = base_labels[base_valid, policy_index]
        adapted_y = adapted_labels[adapted_valid, policy_index]
        base_p = base_probabilities[base_valid, policy_index]
        adapted_p = adapted_probabilities[adapted_valid, policy_index]
        if not len(base_y) or not len(adapted_y):
            diagnostics.append(
                {"target_to_base_ess": math.nan, "base_to_target_ess": math.nan}
            )
            continue
        adapted_weights, adapted_ess = prevalence_weights(base_y, adapted_y)
        base_weights, base_ess = prevalence_weights(adapted_y, base_y)
        if np.all(np.isfinite(adapted_weights)):
            target_to_base[policy_index] = fixed_bin_ece(
                adapted_p, adapted_y, equal_edges, adapted_weights
            ) - multiclass_ece(base_p, base_y, N_BINS)
        if np.all(np.isfinite(base_weights)):
            base_to_target[policy_index] = multiclass_ece(
                adapted_p, adapted_y, N_BINS
            ) - fixed_bin_ece(
                base_p, base_y, equal_edges, base_weights
            )
        diagnostics.append({"target_to_base_ess": adapted_ess, "base_to_target_ess": base_ess})
    return target_to_base, base_to_target, diagnostics


def strict_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if np.all(np.isfinite(values)) else math.nan


def detailed_point_metrics(
    base: Variant, adapted: Variant, source_edges: dict[str, np.ndarray]
) -> dict[str, Any]:
    per_policy: dict[str, Any] = {}
    standard_target, standard_base, prevalence_diagnostics = standardized_deltas(
        base.labels, base.probabilities, adapted.labels, adapted.probabilities
    )
    for index, policy in enumerate(POLICY_IDS):
        base_valid = base.labels[:, index] >= 0
        adapted_valid = adapted.labels[:, index] >= 0
        base_y = base.labels[base_valid, index]
        adapted_y = adapted.labels[adapted_valid, index]
        base_p = base.probabilities[base_valid, index]
        adapted_p = adapted.probabilities[adapted_valid, index]
        base_classwise = classwise_ece(base_p, base_y)
        adapted_classwise = classwise_ece(adapted_p, adapted_y)
        base_calibration = calibration_parameters(base_p, base_y)
        adapted_calibration = calibration_parameters(adapted_p, adapted_y)
        per_policy[policy] = {
            "support": {
                "D0": {
                    **{ID_TO_LABEL[c]: int(np.sum(base_y == c)) for c in range(3)},
                    "missing": int(np.sum(~base_valid)),
                },
                adapted.name: {
                    **{ID_TO_LABEL[c]: int(np.sum(adapted_y == c)) for c in range(3)},
                    "missing": int(np.sum(~adapted_valid)),
                },
            },
            "primary_ece": {
                "D0": multiclass_ece(base_p, base_y, N_BINS),
                adapted.name: multiclass_ece(adapted_p, adapted_y, N_BINS),
            },
            "adaptive_ece": {
                "D0": adaptive_multiclass_ece(base_p, base_y, N_BINS),
                adapted.name: adaptive_multiclass_ece(adapted_p, adapted_y, N_BINS),
                "delta": adaptive_multiclass_ece(adapted_p, adapted_y, N_BINS)
                - adaptive_multiclass_ece(base_p, base_y, N_BINS),
            },
            "fixed_source_bin_ece": {
                "D0": fixed_bin_ece(base_p, base_y, source_edges[policy]),
                adapted.name: fixed_bin_ece(adapted_p, adapted_y, source_edges[policy]),
                "delta": fixed_bin_ece(adapted_p, adapted_y, source_edges[policy])
                - fixed_bin_ece(base_p, base_y, source_edges[policy]),
            },
            "classwise_ece": {"D0": base_classwise, adapted.name: adapted_classwise},
            "violated_ece": {
                "D0": base_classwise["violated"],
                adapted.name: adapted_classwise["violated"],
                "delta": adapted_classwise["violated"] - base_classwise["violated"],
            },
            "nll": {
                "D0": multiclass_nll(base_p, base_y),
                adapted.name: multiclass_nll(adapted_p, adapted_y),
                "delta": multiclass_nll(adapted_p, adapted_y) - multiclass_nll(base_p, base_y),
            },
            "brier": {
                "D0": multiclass_brier(base_p, base_y),
                adapted.name: multiclass_brier(adapted_p, adapted_y),
                "delta": multiclass_brier(adapted_p, adapted_y) - multiclass_brier(base_p, base_y),
            },
            "calibration_intercept_slope": {
                "D0": base_calibration,
                adapted.name: adapted_calibration,
            },
            "probability_support_overlap": support_overlap(base_p, adapted_p),
            "prevalence_standardization": {
                "target_to_D0_delta_ece": standard_target[index],
                "D0_to_target_delta_ece": standard_base[index],
                **prevalence_diagnostics[index],
            },
        }
    return {
        "per_policy": per_policy,
        "prevalence_standardization_point": {
            "target_to_D0_mean_delta_ece": strict_mean(standard_target),
            "D0_to_target_mean_delta_ece": strict_mean(standard_base),
        },
    }


def compute_ece_vector(variant: Variant, indices: np.ndarray) -> np.ndarray:
    result = []
    for policy in range(len(POLICY_IDS)):
        selected_labels = variant.labels[indices, policy]
        valid = selected_labels >= 0
        result.append(
            multiclass_ece(
                variant.probabilities[indices[valid], policy],
                selected_labels[valid],
                N_BINS,
            )
            if np.any(valid)
            else math.nan
        )
    return np.asarray(result, dtype=np.float64)


def _bootstrap_worker(task: tuple[int, int]) -> tuple[int, int, np.ndarray, dict, dict, dict]:
    if _BOOT_DATA is None:
        raise RuntimeError("bootstrap worker was not initialized")
    start, stop = task
    base: Variant = _BOOT_DATA["base"]
    variants: dict[str, Variant] = _BOOT_DATA["variants"]
    bootstrap_indices: np.ndarray = _BOOT_DATA["indices"]
    count = stop - start
    base_boot = np.empty((count, len(POLICY_IDS)), dtype=np.float64)
    variant_boot = {
        name: np.empty((count, len(POLICY_IDS)), dtype=np.float64) for name in variants
    }
    standard_target_boot = {name: np.full(count, np.nan) for name in variants}
    standard_base_boot = {name: np.full(count, np.nan) for name in variants}
    for offset, replicate in enumerate(range(start, stop)):
        indices = bootstrap_indices[replicate]
        base_boot[offset] = compute_ece_vector(base, indices)
        for name, variant in variants.items():
            variant_boot[name][offset] = compute_ece_vector(variant, indices)
            target, reverse, _ = standardized_deltas(
                base.labels[indices],
                base.probabilities[indices],
                variant.labels[indices],
                variant.probabilities[indices],
            )
            standard_target_boot[name][offset] = strict_mean(target)
            standard_base_boot[name][offset] = strict_mean(reverse)
    return start, stop, base_boot, variant_boot, standard_target_boot, standard_base_boot


def bootstrap_primary(
    base: Variant,
    variants: dict[str, Variant],
    replicates: int,
    seed: int,
    jobs: int = 1,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(base.ids)
    # The int32 matrix is 100 MB at the locked 10,000 x 2,500 run and is
    # shared copy-on-write by forked Linux workers.  It also proves every
    # method sees the exact same paired family resamples.
    bootstrap_indices = rng.integers(0, n, size=(replicates, n), dtype=np.int32)
    global _BOOT_DATA
    _BOOT_DATA = {"base": base, "variants": variants, "indices": bootstrap_indices}
    jobs = max(1, min(int(jobs), replicates))
    if os.name == "nt" and jobs > 1:
        raise ValueError("parallel bootstrap requires POSIX fork; use --jobs 1 on Windows")
    chunks = min(replicates, jobs * 4)
    boundaries = np.linspace(0, replicates, chunks + 1, dtype=int)
    tasks = [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(chunks)
        if boundaries[index] < boundaries[index + 1]
    ]
    if jobs == 1:
        pieces = [_bootstrap_worker(task) for task in tasks]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=min(jobs, len(tasks))) as pool:
            pieces = pool.map(_bootstrap_worker, tasks)
    base_boot = np.empty((replicates, len(POLICY_IDS)), dtype=np.float64)
    variant_boot = {
        name: np.empty((replicates, len(POLICY_IDS)), dtype=np.float64)
        for name in variants
    }
    standard_target_boot = {name: np.full(replicates, np.nan) for name in variants}
    standard_base_boot = {name: np.full(replicates, np.nan) for name in variants}
    for start, stop, base_piece, variant_piece, target_piece, reverse_piece in pieces:
        base_boot[start:stop] = base_piece
        for name in variants:
            variant_boot[name][start:stop] = variant_piece[name]
            standard_target_boot[name][start:stop] = target_piece[name]
            standard_base_boot[name][start:stop] = reverse_piece[name]
    _BOOT_DATA = None
    return {
        "base_ece": base_boot,
        "variant_ece": variant_boot,
        "target_to_base": standard_target_boot,
        "base_to_target": standard_base_boot,
    }


def p3_omnibus(delta: np.ndarray, delta_boot: np.ndarray) -> dict[str, Any]:
    contrast = helmert(len(POLICY_IDS), full=False).astype(np.float64)
    covariance = np.cov(delta_boot, rowvar=False, ddof=1)
    contrast_covariance = contrast @ covariance @ contrast.T
    singular_values = np.linalg.svd(contrast_covariance, compute_uv=False)
    largest_singular = float(singular_values[0]) if len(singular_values) else 0.0
    rank = int(
        np.sum(
            singular_values
            > (largest_singular * PINV_RCOND)
        )
    )
    if rank < len(POLICY_IDS) - 1:
        return {
            "evaluable": False,
            "reason": f"contrast covariance rank {rank} < 9",
            "rank": rank,
            "rcond": PINV_RCOND,
        }
    inverse = np.linalg.pinv(contrast_covariance, rcond=PINV_RCOND)
    observed_contrast = contrast @ delta
    statistic = float(observed_contrast @ inverse @ observed_contrast)
    centered = (delta_boot - delta) @ contrast.T
    bootstrap_statistics = np.einsum("bi,ij,bj->b", centered, inverse, centered)
    p_value = float((1 + np.sum(bootstrap_statistics >= statistic)) / (len(delta_boot) + 1))

    centered_effect = delta - delta.mean()
    centered_boot = delta_boot - delta_boot.mean(axis=1, keepdims=True)
    standard_error = centered_boot.std(axis=0, ddof=1)
    usable = standard_error > 1e-15
    standardized = np.zeros_like(centered_boot)
    standardized[:, usable] = (
        centered_boot[:, usable] - centered_effect[usable]
    ) / standard_error[usable]
    critical = float(np.quantile(np.max(np.abs(standardized[:, usable]), axis=1), 0.95))
    simultaneous = {
        policy: [
            float(centered_effect[index] - critical * standard_error[index]),
            float(centered_effect[index] + critical * standard_error[index]),
        ]
        for index, policy in enumerate(POLICY_IDS)
    }
    return {
        "evaluable": True,
        "null": "all ten Delta-ECE values share one common domain shift",
        "helmert_shape": list(contrast.shape),
        "rank": rank,
        "rcond": PINV_RCOND,
        "wald_statistic": statistic,
        "recentered_bootstrap_p": p_value,
        "omnibus_significant": p_value < 0.05,
        "centered_effect": {policy: centered_effect[i] for i, policy in enumerate(POLICY_IDS)},
        "max_t_critical_95": critical,
        "simultaneous_centered_95ci": simultaneous,
    }


def mutually_exclusive_verdict(
    integrity_evaluable: bool,
    base_anchor: bool,
    p2_ci: list[float],
    p3: dict[str, Any],
) -> str:
    if not integrity_evaluable:
        return "NON_EVALUABLE"
    if not p3.get("evaluable", False):
        return "NON_EVALUABLE"
    if not base_anchor:
        return "BASE_ANCHOR_NOT_REPLICATED"
    if p2_ci[1] <= 0:
        return "P2_CONTRADICTED"
    if p2_ci[0] <= 0:
        return "CORE_NOT_ESTABLISHED"
    if not p3["omnibus_significant"]:
        return "P2_ONLY"
    return "P2_P3_CONFIRMED"


def assemble_variant_result(
    base: Variant,
    adapted: Variant,
    base_boot: np.ndarray,
    adapted_boot: np.ndarray,
    standard_target_boot: np.ndarray,
    standard_base_boot: np.ndarray,
    source_edges: dict[str, np.ndarray],
    primary: bool,
    integrity_evaluable: bool,
) -> dict[str, Any]:
    base_point = compute_ece_vector(base, np.arange(len(base.ids)))
    adapted_point = compute_ece_vector(adapted, np.arange(len(adapted.ids)))
    delta = adapted_point - base_point
    delta_boot = adapted_boot - base_boot
    mean_delta_boot = delta_boot.mean(axis=1)
    base_mean_boot = base_boot.mean(axis=1)
    p2_ci = percentile_ci(mean_delta_boot)
    base_ci = percentile_ci(base_mean_boot)
    p3 = p3_omnibus(delta, delta_boot)
    p2_material = bool(p2_ci[0] > MATERIAL_MEAN_FLOOR)
    sd_point = float(np.std(delta, ddof=0))
    sd_boot = np.std(delta_boot, axis=1, ddof=0)
    sd_ci = percentile_ci(sd_boot)
    material = bool(sd_ci[0] > MATERIAL_SD_FLOOR)
    base_anchor = bool(base_ci[1] <= BASE_ECE_CEILING)
    statistical_verdict = mutually_exclusive_verdict(
        integrity_evaluable, base_anchor, p2_ci, p3
    )
    details = detailed_point_metrics(base, adapted, source_edges)
    prevalence_support_ok = all(
        details["per_policy"][policy]["prevalence_standardization"][key] >= 100
        for policy in POLICY_IDS
        for key in ("target_to_base_ess", "base_to_target_ess")
    )
    for index, policy in enumerate(POLICY_IDS):
        details["per_policy"][policy]["primary_ece"].update(
            {
                "delta": delta[index],
                "delta_95ci": percentile_ci(delta_boot[:, index]),
            }
        )
    details["prevalence_standardization_bootstrap"] = {
        "target_to_D0_mean_delta_ece_95ci": percentile_ci(standard_target_boot),
        "target_to_D0_valid_replicates": int(np.isfinite(standard_target_boot).sum()),
        "D0_to_target_mean_delta_ece_95ci": percentile_ci(standard_base_boot),
        "D0_to_target_valid_replicates": int(np.isfinite(standard_base_boot).sum()),
        "all_point_ess_ge_100": prevalence_support_ok,
        "robust_tag": bool(
            prevalence_support_ok
            and np.isfinite(standard_target_boot).all()
            and np.isfinite(standard_base_boot).all()
            and percentile_ci(standard_target_boot)[0] > 0
            and percentile_ci(standard_base_boot)[0] > 0
        ),
    }
    result = {
        "role": "PRIMARY_NEW_D5" if primary else "SECONDARY_OLD_D5_NO_GATE",
        "primary_gate_eligible": primary,
        "reference_integrity_evaluable": integrity_evaluable,
        "P2": {
            "mean_delta_ece": float(delta.mean()),
            "mean_delta_ece_95ci": p2_ci,
            "lower_ci_gt_zero": bool(p2_ci[0] > 0),
            "material_floor": MATERIAL_MEAN_FLOOR,
            "material_lower_ci_gt_0_01": p2_material,
            "material_tag": "MATERIAL_GE_0.01" if p2_material else "MATERIAL_LT_0.01",
            "material_tag_affects_confirmatory_gate": False,
            "D0_mean_ece": float(base_point.mean()),
            "D0_mean_ece_95ci": base_ci,
            "D0_anchor_upper_le_0_05": base_anchor,
        },
        "P3": {
            **p3,
            "sd_delta_ece": sd_point,
            "sd_delta_ece_95ci": sd_ci,
            "material_floor": MATERIAL_SD_FLOOR,
            "material_lower_ci_gt_0_01": material,
            "material_tag": "MATERIAL_GE_0.01" if material else "MATERIAL_LT_0.01",
            "material_tag_affects_confirmatory_gate": False,
        },
        "statistical_verdict": statistical_verdict if primary else "SECONDARY_NO_VERDICT",
        "metrics": details,
    }
    return result


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != BOOTSTRAP_SEED:
        raise ValueError(f"locked bootstrap seed is {BOOTSTRAP_SEED}")
    if args.bootstrap < 20:
        raise ValueError("bootstrap must be at least 20 (10,000 in the locked run)")
    old_pair = (args.old_d5_labels is None, args.old_d5_logits is None)
    if old_pair[0] != old_pair[1]:
        raise ValueError("old-D5 labels and logits must be supplied together")

    required = [
        args.test_manifest,
        args.source_bin_edges,
        args.d0_labels,
        args.d0_logits,
        args.new_d5_labels,
        args.new_d5_logits,
    ]
    prompt_artifact = referenced_prompt_artifact(args.test_manifest)
    if prompt_artifact is not None:
        required.append(prompt_artifact)
    if args.old_d5_labels is not None:
        required.extend([args.old_d5_labels, args.old_d5_logits])
    input_hashes = verify_hashes(required, args.hash_manifest)
    ids, families, test_manifest = load_test_manifest(args.test_manifest, args.expected_test_count)
    source_edges = load_source_edges(args.source_bin_edges)
    base = load_variant("D0", args.d0_labels, args.d0_logits, ids)
    new_d5 = load_variant("new_D5", args.new_d5_labels, args.new_d5_logits, ids)
    variants = {"new_D5": new_d5}
    if args.old_d5_labels is not None:
        variants["old_D5"] = load_variant(
            "old_D5", args.old_d5_labels, args.old_d5_logits, ids
        )
    boot = bootstrap_primary(base, variants, args.bootstrap, args.seed, getattr(args, "jobs", 1))
    primary = assemble_variant_result(
        base,
        new_d5,
        boot["base_ece"],
        boot["variant_ece"]["new_D5"],
        boot["target_to_base"]["new_D5"],
        boot["base_to_target"]["new_D5"],
        source_edges,
        primary=True,
        integrity_evaluable=base.reference_integrity_pass
        and new_d5.reference_integrity_pass,
    )
    secondary: dict[str, Any] = {}
    if "old_D5" in variants:
        secondary["old_D5"] = assemble_variant_result(
            base,
            variants["old_D5"],
            boot["base_ece"],
            boot["variant_ece"]["old_D5"],
            boot["target_to_base"]["old_D5"],
            boot["base_to_target"]["old_D5"],
            source_edges,
            primary=False,
            integrity_evaluable=base.reference_integrity_pass
            and variants["old_D5"].reference_integrity_pass,
        )
        new_delta = np.asarray(
            [
                primary["metrics"]["per_policy"][policy]["primary_ece"]["delta"]
                for policy in POLICY_IDS
            ],
            dtype=np.float64,
        )
        old_delta = np.asarray(
            [
                secondary["old_D5"]["metrics"]["per_policy"][policy]["primary_ece"]["delta"]
                for policy in POLICY_IDS
            ],
            dtype=np.float64,
        )
        correlation_boot = np.asarray(
            [
                spearmanr(
                    boot["variant_ece"]["new_D5"][index] - boot["base_ece"][index],
                    boot["variant_ece"]["old_D5"][index] - boot["base_ece"][index],
                ).statistic
                for index in range(args.bootstrap)
            ],
            dtype=np.float64,
        )
        secondary["old_vs_new_delta_ece_spearman"] = {
            "rho": float(spearmanr(new_delta, old_delta).statistic),
            "family_bootstrap_95ci": percentile_ci(correlation_boot),
            "finite_bootstrap_replicates": int(np.isfinite(correlation_boot).sum()),
            "role": "SECONDARY_NO_GATE",
        }
    locked_run = bool(
        args.bootstrap == LOCKED_BOOTSTRAP
        and len(ids) == LOCKED_TEST_COUNT
        and args.expected_test_count == LOCKED_TEST_COUNT
    )
    return {
        "protocol": {
            "analysis": "new lockbox P2/P3 confirmation",
            "primary": "D0 versus new-seed D5 on CONFIRM-TEST",
            "old_D5": "secondary only; cannot affect verdict",
            "test_only": True,
            "calibration_split_read": False,
            "n_query_families": len(set(families)),
            "n_rows": len(ids),
            "one_row_per_family": True,
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "production_locked_run": locked_run,
            "primary_ece": "15-bin equal-width top-class 3-way ECE including N/A",
            "criteria_fixed_not_resampled": True,
            "input_sha256": input_hashes,
            "hash_manifest": str(args.hash_manifest.resolve()),
            "test_manifest_role": test_manifest.get("role", test_manifest.get("split")),
            "reference_integrity": {
                name: {
                    "strict_ten_key_success_rate": variant.strict_json_success_rate,
                    "missing_rate_by_policy": variant.missing_rate_by_policy,
                    "thresholds": {
                        "strict_ten_key_success_rate_min": 0.99,
                        "per_policy_missing_rate_max": 0.01,
                    },
                    "pass": variant.reference_integrity_pass,
                }
                for name, variant in {"D0": base, **variants}.items()
            },
        },
        "primary_new_D5": primary,
        "secondary": secondary,
        "reported_verdict": primary["statistical_verdict"] if locked_run else "TEST_MODE_ONLY",
        "test_mode_statistical_verdict": primary["statistical_verdict"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manifest", type=Path, required=True)
    parser.add_argument("--hash_manifest", type=Path, required=True)
    parser.add_argument("--source_bin_edges", type=Path, required=True)
    parser.add_argument("--d0_labels", type=Path, required=True)
    parser.add_argument("--d0_logits", type=Path, required=True)
    parser.add_argument("--new_d5_labels", type=Path, required=True)
    parser.add_argument("--new_d5_logits", type=Path, required=True)
    parser.add_argument("--old_d5_labels", type=Path)
    parser.add_argument("--old_d5_logits", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected_test_count", type=int, default=LOCKED_TEST_COUNT)
    parser.add_argument("--bootstrap", type=int, default=LOCKED_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1 if os.name == "nt" else min(80, os.cpu_count() or 1),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite confirmation output: {args.out}")
    result = to_jsonable(run_analysis(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "reported_verdict": result["reported_verdict"],
        "statistical_verdict": result["test_mode_statistical_verdict"],
        "P2": result["primary_new_D5"]["P2"],
        "P3": {
            key: result["primary_new_D5"]["P3"].get(key)
            for key in (
                "recentered_bootstrap_p",
                "sd_delta_ece",
                "sd_delta_ece_95ci",
                "material_lower_ci_gt_0_01",
            )
        },
    }, indent=2))


if __name__ == "__main__":
    main()
