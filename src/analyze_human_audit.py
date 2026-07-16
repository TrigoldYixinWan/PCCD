#!/usr/bin/env python3
"""Analyze the locked blinded human construct-validity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


POLICY_IDS = ("H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2")
DOMAINS = ("D0", "new_D5")
LABELS = ("satisfied", "violated", "not_applicable")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
LOCKED_BOOTSTRAP = 10_000
LOCKED_SEED = 20260726


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            audit_id = row.get("audit_id")
            if not isinstance(audit_id, str) or not audit_id or audit_id in seen:
                raise ValueError(f"{path}:{line_number}: missing or duplicate audit_id")
            seen.add(audit_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def percentile_ci(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [math.nan, math.nan]
    return [float(value) for value in np.quantile(finite, [0.025, 0.975])]


def helmert_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size - 1, size), dtype=np.float64)
    for row in range(size - 1):
        scale = math.sqrt((row + 1) * (row + 2))
        matrix[row, : row + 1] = 1.0 / scale
        matrix[row, row + 1] = -(row + 1) / scale
    return matrix


def weighted_confusion(
    truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.float64)
    np.add.at(matrix, (truth, prediction), weights)
    return matrix


def cohen_kappa(confusion: np.ndarray) -> float:
    total = float(confusion.sum())
    if total <= 0:
        return math.nan
    observed = float(np.trace(confusion) / total)
    expected = float(
        np.dot(confusion.sum(axis=1), confusion.sum(axis=0)) / (total * total)
    )
    if expected >= 1.0:
        return math.nan
    return float((observed - expected) / (1.0 - expected))


def weighted_rate(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    return math.nan if denominator <= 0 else float(np.dot(values, weights) / denominator)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_and_join(completed_path: Path, private_path: Path) -> list[dict]:
    completed = load_jsonl(completed_path)
    private = load_jsonl(private_path)
    if len(completed) != 800 or len(private) != 800:
        raise ValueError("locked human audit requires exactly 800 completed/private rows")
    completed_by_id = {row["audit_id"]: row for row in completed}
    private_by_id = {row["audit_id"]: row for row in private}
    if set(completed_by_id) != set(private_by_id):
        raise ValueError("completed/private audit ID sets differ")

    joined: list[dict] = []
    counts: Counter[tuple[str, str]] = Counter()
    for audit_id in sorted(completed_by_id):
        blind, hidden = completed_by_id[audit_id], private_by_id[audit_id]
        label_a = blind.get("annotator_1")
        label_b = blind.get("annotator_2")
        human = blind.get("adjudicated")
        reference = hidden.get("reference_state")
        if label_a not in LABEL_TO_ID or label_b not in LABEL_TO_ID or human not in LABEL_TO_ID:
            raise ValueError(f"{audit_id}: incomplete or invalid human annotation")
        if label_a == label_b and human != label_a:
            raise ValueError(f"{audit_id}: agreement row changed during adjudication")
        domain, policy = hidden.get("domain"), hidden.get("policy_id")
        if domain not in DOMAINS or policy not in POLICY_IDS or reference not in LABEL_TO_ID:
            raise ValueError(f"{audit_id}: invalid private domain/policy/reference state")
        weight = hidden.get("inverse_probability_weight")
        family_id = hidden.get("family_id")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{audit_id}: invalid inverse probability weight")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"{audit_id}: invalid family_id")
        counts[(domain, policy)] += 1
        joined.append(
            {
                "audit_id": audit_id,
                "domain": domain,
                "policy": policy,
                "family_id": family_id,
                "weight": float(weight),
                "annotator_a": LABEL_TO_ID[label_a],
                "annotator_b": LABEL_TO_ID[label_b],
                "human": LABEL_TO_ID[human],
                "reference": LABEL_TO_ID[reference],
            }
        )
    expected = {(domain, policy): 40 for domain in DOMAINS for policy in POLICY_IDS}
    if dict(counts) != expected:
        raise ValueError(f"locked 40-cell domain-policy allocation changed: {dict(counts)}")
    return joined


def analyze(
    joined: list[dict],
    *,
    bootstrap: int = LOCKED_BOOTSTRAP,
    seed: int = LOCKED_SEED,
) -> dict:
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    n = len(joined)
    domains = np.array([DOMAINS.index(row["domain"]) for row in joined], dtype=np.int64)
    policies = np.array([POLICY_IDS.index(row["policy"]) for row in joined], dtype=np.int64)
    cell_index = domains * len(POLICY_IDS) + policies
    weights = np.array([row["weight"] for row in joined], dtype=np.float64)
    annotator_a = np.array([row["annotator_a"] for row in joined], dtype=np.int64)
    annotator_b = np.array([row["annotator_b"] for row in joined], dtype=np.int64)
    human = np.array([row["human"] for row in joined], dtype=np.int64)
    reference = np.array([row["reference"] for row in joined], dtype=np.int64)
    mismatch = (human != reference).astype(np.float64)

    families = sorted({row["family_id"] for row in joined})
    family_to_id = {family: index for index, family in enumerate(families)}
    family_codes = np.array(
        [family_to_id[row["family_id"]] for row in joined], dtype=np.int64
    )

    numerator = np.bincount(
        cell_index, weights=weights * mismatch, minlength=2 * len(POLICY_IDS)
    ).reshape(2, len(POLICY_IDS))
    denominator = np.bincount(
        cell_index, weights=weights, minlength=2 * len(POLICY_IDS)
    ).reshape(2, len(POLICY_IDS))
    if np.any(denominator <= 0):
        raise ValueError("non-positive weighted domain-policy denominator")
    rates = numerator / denominator
    differences = rates[1] - rates[0]
    mean_difference = float(differences.mean())

    rng = np.random.default_rng(seed)
    boot_rates = np.full((bootstrap, 2, len(POLICY_IDS)), np.nan, dtype=np.float64)
    family_count = len(families)
    for replicate in range(bootstrap):
        sampled = rng.integers(0, family_count, size=family_count)
        multiplicity = np.bincount(sampled, minlength=family_count)
        replicate_weights = weights * multiplicity[family_codes]
        replicate_denominator = np.bincount(
            cell_index,
            weights=replicate_weights,
            minlength=2 * len(POLICY_IDS),
        ).reshape(2, len(POLICY_IDS))
        replicate_numerator = np.bincount(
            cell_index,
            weights=replicate_weights * mismatch,
            minlength=2 * len(POLICY_IDS),
        ).reshape(2, len(POLICY_IDS))
        valid = replicate_denominator > 0
        boot_rates[replicate][valid] = (
            replicate_numerator[valid] / replicate_denominator[valid]
        )
    boot_differences = boot_rates[:, 1, :] - boot_rates[:, 0, :]
    complete = np.all(np.isfinite(boot_differences), axis=1)
    valid_count = int(complete.sum())
    valid_fraction = valid_count / bootstrap
    valid_boot = boot_differences[complete]

    contrast = helmert_matrix(len(POLICY_IDS))
    rank = 0
    wald = math.nan
    p_value = math.nan
    max_t_critical = math.nan
    simultaneous: dict[str, list[float]] = {
        policy: [math.nan, math.nan] for policy in POLICY_IDS
    }
    if valid_count >= 2:
        covariance = np.cov(valid_boot, rowvar=False, ddof=1)
        contrast_covariance = contrast @ covariance @ contrast.T
        rank = int(np.linalg.matrix_rank(contrast_covariance, tol=1e-12))
        inverse = np.linalg.pinv(contrast_covariance, rcond=1e-12)
        observed_contrast = contrast @ differences
        wald = float(observed_contrast @ inverse @ observed_contrast)
        centered_boot_contrast = (valid_boot - differences) @ contrast.T
        wald_boot = np.einsum(
            "bi,ij,bj->b", centered_boot_contrast, inverse, centered_boot_contrast
        )
        p_value = float((1 + np.sum(wald_boot >= wald)) / (valid_count + 1))

        centered = differences - differences.mean()
        centered_boot = valid_boot - valid_boot.mean(axis=1, keepdims=True)
        standard_error = centered_boot.std(axis=0, ddof=1)
        usable = standard_error > 0
        if np.all(usable):
            max_t = np.max(
                np.abs((centered_boot - centered) / standard_error), axis=1
            )
            max_t_critical = float(np.quantile(max_t, 0.95))
            simultaneous = {
                policy: [
                    float(centered[index] - max_t_critical * standard_error[index]),
                    float(centered[index] + max_t_critical * standard_error[index]),
                ]
                for index, policy in enumerate(POLICY_IDS)
            }

    raw_confusion = weighted_confusion(
        annotator_a, annotator_b, np.ones(n, dtype=np.float64)
    )
    weighted_ab_confusion = weighted_confusion(
        annotator_a, annotator_b, weights
    )
    reference_confusion = weighted_confusion(reference, human, weights)

    agreement_by_policy = {}
    for index, policy in enumerate(POLICY_IDS):
        selected = policies == index
        raw = weighted_confusion(
            annotator_a[selected],
            annotator_b[selected],
            np.ones(int(selected.sum()), dtype=np.float64),
        )
        weighted = weighted_confusion(
            annotator_a[selected], annotator_b[selected], weights[selected]
        )
        agreement_by_policy[policy] = {
            "raw_exact_agreement": float(np.trace(raw) / raw.sum()),
            "raw_kappa": cohen_kappa(raw),
            "weighted_exact_agreement": float(np.trace(weighted) / weighted.sum()),
            "weighted_kappa": cohen_kappa(weighted),
        }

    mismatch_by_domain_reference = {}
    mismatch_by_domain_policy_reference = {}
    for domain_index, domain in enumerate(DOMAINS):
        mismatch_by_domain_reference[domain] = {}
        mismatch_by_domain_policy_reference[domain] = {}
        for state_index, state in enumerate(LABELS):
            selected = (domains == domain_index) & (reference == state_index)
            mismatch_by_domain_reference[domain][state] = {
                "sample_cells": int(selected.sum()),
                "weighted_mismatch": weighted_rate(
                    mismatch[selected], weights[selected]
                ),
            }
        for policy_index, policy in enumerate(POLICY_IDS):
            mismatch_by_domain_policy_reference[domain][policy] = {}
            for state_index, state in enumerate(LABELS):
                selected = (
                    (domains == domain_index)
                    & (policies == policy_index)
                    & (reference == state_index)
                )
                mismatch_by_domain_policy_reference[domain][policy][state] = {
                    "sample_cells": int(selected.sum()),
                    "weighted_mismatch": weighted_rate(
                        mismatch[selected], weights[selected]
                    ),
                }

    differential = (
        percentile_ci(valid_boot.mean(axis=1))[0] > 0
        or percentile_ci(valid_boot.mean(axis=1))[1] < 0
        if valid_count
        else False
    )
    interaction = math.isfinite(p_value) and p_value < 0.05
    evaluable = valid_fraction >= 0.90 and rank == len(POLICY_IDS) - 1
    if not evaluable:
        verdict = "NON_EVALUABLE"
    elif differential or interaction:
        verdict = "DIFFERENTIAL_REFERENCE_ERROR"
    else:
        verdict = "NO_DIFFERENTIAL_ERROR_DETECTED"

    per_policy = {}
    for index, policy in enumerate(POLICY_IDS):
        per_policy[policy] = {
            "D0_weighted_mismatch": float(rates[0, index]),
            "new_D5_weighted_mismatch": float(rates[1, index]),
            "domain_difference": float(differences[index]),
            "domain_difference_95ci": percentile_ci(valid_boot[:, index]),
            "centered_domain_effect": float(
                differences[index] - mean_difference
            ),
            "simultaneous_centered_95ci": simultaneous[policy],
        }

    return {
        "protocol": {
            "bootstrap_unit": "lexical family_id",
            "bootstrap_replicates": bootstrap,
            "seed": seed,
            "criteria_fixed_not_resampled": True,
            "estimator": "inverse-probability-weighted Hajek mismatch rate",
            "interaction": "orthonormal Helmert Wald with recentered family bootstrap",
        },
        "sample": {
            "cells": n,
            "unique_families": len(families),
            "domain_policy_cells": 40,
        },
        "annotator_agreement": {
            "raw_exact_agreement": float(np.trace(raw_confusion) / raw_confusion.sum()),
            "raw_kappa": cohen_kappa(raw_confusion),
            "weighted_exact_agreement": float(
                np.trace(weighted_ab_confusion) / weighted_ab_confusion.sum()
            ),
            "weighted_kappa": cohen_kappa(weighted_ab_confusion),
            "raw_confusion_A_rows_B_columns": raw_confusion,
            "weighted_confusion_A_rows_B_columns": weighted_ab_confusion,
            "per_policy": agreement_by_policy,
        },
        "reference_validity": {
            "weighted_exact_agreement": float(
                np.trace(reference_confusion) / reference_confusion.sum()
            ),
            "weighted_mismatch": float(
                1.0 - np.trace(reference_confusion) / reference_confusion.sum()
            ),
            "weighted_confusion_reference_rows_human_columns": reference_confusion,
            "mismatch_by_domain_reference_state": mismatch_by_domain_reference,
            "mismatch_by_domain_policy_reference_state": (
                mismatch_by_domain_policy_reference
            ),
            "per_policy": per_policy,
            "mean_domain_difference": mean_difference,
            "mean_domain_difference_95ci": (
                percentile_ci(valid_boot.mean(axis=1))
                if valid_count
                else [math.nan, math.nan]
            ),
        },
        "domain_by_criterion_interaction": {
            "null": "the ten new-D5 minus D0 mismatch changes share one common shift",
            "rank": rank,
            "wald_statistic": wald,
            "recentered_bootstrap_p": p_value,
            "max_t_critical_95": max_t_critical,
            "valid_bootstrap_replicates": valid_count,
            "valid_bootstrap_fraction": valid_fraction,
        },
        "verdict": verdict,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=LOCKED_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    joined = validate_and_join(args.completed, args.private)
    result = analyze(joined, bootstrap=args.bootstrap, seed=args.seed)
    result["input_sha256"] = {
        "completed": sha256_file(args.completed),
        "private": sha256_file(args.private),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(to_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "verdict": result["verdict"],
                "cells": result["sample"]["cells"],
                "valid_bootstrap_replicates": result[
                    "domain_by_criterion_interaction"
                ]["valid_bootstrap_replicates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
