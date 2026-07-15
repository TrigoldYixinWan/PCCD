#!/usr/bin/env python3
"""Analyze Day-3 teacher targets and perturbation stability for G1.

This script is deliberately limited to teacher-produced labels.  It does not
load critic predictions and therefore does not estimate D0 critic F1, the
cross-policy F1 coefficient of variation, or an overall G1 verdict.

All uncertainty intervals use an item-cluster percentile bootstrap: an item is
resampled as one cluster, retaining its ten correlated policy labels.  The
paired homogeneity tests likewise treat the ten policy indicators as repeated
measurements on each item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2


POLICY_IDS = ("H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2")
LABEL_STATES = ("satisfied", "violated", "not_applicable")
LABEL_TO_INT = {label: i for i, label in enumerate(LABEL_STATES)}
REGISTERED_PERTURBATIONS = (
    "repeat_sampling",
    "policy_order_swap",
    "policy_paraphrase",
)
PERTURBATION_ALIASES = {
    "repeat_sampling": ("repeat_sampling", "repeat"),
    "policy_order_swap": ("policy_order_swap", "order_swap"),
    "policy_paraphrase": ("policy_paraphrase", "paraphrase"),
}


class ValidationError(ValueError):
    """Raised when an input cannot support a strict paired analysis."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValidationError(f"{path}:{line_no}: blank JSONL line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValidationError(f"{path}:{line_no}: record must be an object")
            item_id = record.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise ValidationError(f"{path}:{line_no}: id must be a non-empty string")
            if item_id in seen:
                raise ValidationError(
                    f"{path}:{line_no}: duplicate id {item_id!r} "
                    f"(first seen on line {seen[item_id]})"
                )
            seen[item_id] = line_no
            records.append(record)
    if not records:
        raise ValidationError(f"{path}: no records")
    return records


def _validate_label_object(value: Any, where: str, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, dict):
        raise ValidationError(f"{where}: labels must be an object or null")
    keys = set(value)
    expected = set(POLICY_IDS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValidationError(f"{where}: policy-key mismatch; missing={missing}, extra={extra}")
    for policy in POLICY_IDS:
        if value[policy] not in LABEL_TO_INT:
            raise ValidationError(
                f"{where}.{policy}: invalid label {value[policy]!r}; "
                f"expected one of {LABEL_STATES}"
            )


def _validate_expected_count(records: list[dict[str, Any]], expected: int, name: str) -> None:
    if len(records) != expected:
        raise ValidationError(f"{name}: expected {expected} unique items, found {len(records)}")


def _load_conflict(path: Path, expected: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    records = _read_jsonl(path)
    _validate_expected_count(records, expected, "conflict")
    matrix = np.full((len(records), len(POLICY_IDS)), -1, dtype=np.int8)
    for i, record in enumerate(records):
        source = record.get("source")
        if not isinstance(source, str) or not source:
            raise ValidationError(f"{path}: id={record['id']!r}: source must be a non-empty string")
        parse_ok = record.get("parse_ok")
        if not isinstance(parse_ok, bool):
            raise ValidationError(f"{path}: id={record['id']!r}: parse_ok must be boolean")
        labels = record.get("labels")
        _validate_label_object(labels, f"{path}: id={record['id']!r}.labels", allow_none=True)
        if parse_ok != (labels is not None):
            raise ValidationError(
                f"{path}: id={record['id']!r}: parse_ok disagrees with labels nullness"
            )
        if labels is not None:
            matrix[i] = [LABEL_TO_INT[labels[p]] for p in POLICY_IDS]
    return records, matrix


def _resolve_perturbation_fields(records: list[dict[str, Any]], path: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for registered in REGISTERED_PERTURBATIONS:
        aliases = PERTURBATION_ALIASES[registered]
        complete = [field for field in aliases if all(field in record for record in records)]
        if not complete:
            partial = {field: sum(field in record for record in records) for field in aliases}
            raise ValidationError(
                f"{path}: no uniform field for registered perturbation {registered!r}; "
                f"coverage={partial}"
            )
        chosen = complete[0]  # registered name is always listed before legacy aliases
        for alternative in complete[1:]:
            for record in records:
                if record[chosen] != record[alternative]:
                    raise ValidationError(
                        f"{path}: id={record['id']!r}: conflicting duplicate fields "
                        f"{chosen!r} and {alternative!r}"
                    )
        resolved[registered] = chosen
    return resolved


def _load_perturb(
    path: Path, expected: int
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, np.ndarray]]:
    records = _read_jsonl(path)
    _validate_expected_count(records, expected, "perturbation audit")
    if not all("canonical" in record for record in records):
        raise ValidationError(f"{path}: every record must contain canonical")
    fields = _resolve_perturbation_fields(records, path)
    order_aliases = ("policy_order_swap_order", "order_swap_order")
    order_fields = [field for field in order_aliases if all(field in r for r in records)]
    if not order_fields:
        coverage = {field: sum(field in r for r in records) for field in order_aliases}
        raise ValidationError(
            f"{path}: missing uniform policy-order permutation field; coverage={coverage}"
        )
    order_field = order_fields[0]
    canonical_order = list(POLICY_IDS)
    for record in records:
        order = record[order_field]
        valid_order = (
            isinstance(order, list)
            and len(order) == len(POLICY_IDS)
            and set(order) == set(POLICY_IDS)
        )
        if not valid_order:
            raise ValidationError(
                f"{path}: id={record['id']!r}.{order_field}: expected a permutation of {POLICY_IDS}"
            )
        if order == canonical_order:
            raise ValidationError(
                f"{path}: id={record['id']!r}.{order_field}: order-swap did not change order"
            )
    matrices: dict[str, np.ndarray] = {}
    logical_to_raw = {"canonical": "canonical", **fields}
    for logical, raw_field in logical_to_raw.items():
        matrix = np.full((len(records), len(POLICY_IDS)), -1, dtype=np.int8)
        for i, record in enumerate(records):
            labels = record[raw_field]
            _validate_label_object(
                labels,
                f"{path}: id={record['id']!r}.{raw_field}",
                allow_none=True,
            )
            if labels is not None:
                matrix[i] = [LABEL_TO_INT[labels[p]] for p in POLICY_IDS]
        matrices[logical] = matrix
    return records, fields, matrices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_weights(n: int, repetitions: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        raise ValueError("bootstrap requires at least one item")
    probabilities = np.full(n, 1.0 / n)
    # Multinomial weights are exactly equivalent to drawing n item indices with
    # replacement, but permit efficient matrix multiplication over all policies.
    return rng.multinomial(n, probabilities, size=repetitions)


def _ci(values: np.ndarray) -> tuple[list[float | None], int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [None, None], 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return [float(low), float(high)], int(finite.size)


def _ratio_metric(
    numerator: int,
    denominator: int,
    boot_numerator: np.ndarray,
    boot_denominator: np.ndarray,
) -> dict[str, Any]:
    estimate = float(numerator / denominator) if denominator else None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(
            np.asarray(boot_numerator, dtype=float),
            np.asarray(boot_denominator, dtype=float),
            out=np.full(np.asarray(boot_numerator).shape, np.nan, dtype=float),
            where=np.asarray(boot_denominator) != 0,
        )
    interval, valid = _ci(ratios)
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "estimate": estimate,
        "ci95": interval,
        "bootstrap_valid_replicates": valid,
    }


def _summarize_policy_labels(
    matrix: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    *,
    include_heterogeneity: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if matrix.ndim != 2 or matrix.shape[1] != len(POLICY_IDS):
        raise ValueError("label matrix has the wrong shape")
    if np.any(matrix < 0):
        raise ValueError("policy summary requires complete parsed rows")
    n = matrix.shape[0]
    weights = _bootstrap_weights(n, repetitions, rng)
    one_hot = matrix[:, :, None] == np.arange(len(LABEL_STATES))[None, None, :]
    counts = one_hot.sum(axis=0).astype(np.int64)
    boot_counts = (weights @ one_hot.reshape(n, -1)).reshape(
        repetitions, len(POLICY_IDS), len(LABEL_STATES)
    )

    policies: dict[str, Any] = {}
    for p_idx, policy in enumerate(POLICY_IDS):
        labels: dict[str, Any] = {}
        for s_idx, state in enumerate(LABEL_STATES):
            labels[state] = _ratio_metric(
                int(counts[p_idx, s_idx]),
                n,
                boot_counts[:, p_idx, s_idx],
                np.full(repetitions, n),
            )
        applicable = int(counts[p_idx, 0] + counts[p_idx, 1])
        boot_applicable = boot_counts[:, p_idx, 0] + boot_counts[:, p_idx, 1]
        policies[policy] = {
            "labels": labels,
            "applicability": _ratio_metric(
                applicable,
                n,
                boot_applicable,
                np.full(repetitions, n),
            ),
            "violation_among_applicable": _ratio_metric(
                int(counts[p_idx, 1]),
                applicable,
                boot_counts[:, p_idx, 1],
                boot_applicable,
            ),
        }

    heterogeneity = None
    if include_heterogeneity:
        heterogeneity = _heterogeneity(matrix, counts, boot_counts)
    return policies, heterogeneity


def _jsd_base2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(p > 0, p * np.log2(p / midpoint), 0.0)
        right = np.where(q > 0, q * np.log2(q / midpoint), 0.0)
    return 0.5 * (left.sum(axis=-1) + right.sum(axis=-1))


def _cochran_q(binary: np.ndarray) -> dict[str, Any]:
    """Asymptotic Cochran Q for k paired binary responses."""
    if binary.ndim != 2:
        raise ValueError("Cochran Q input must be two-dimensional")
    n, k = binary.shape
    column_sums = binary.sum(axis=0, dtype=float)
    row_sums = binary.sum(axis=1, dtype=float)
    total = float(column_sums.sum())
    denominator = k * total - float(np.square(row_sums).sum())
    numerator = (k - 1) * (k * float(np.square(column_sums).sum()) - total**2)
    if denominator <= 0:
        statistic = 0.0
        p_value = 1.0
        degenerate = True
    else:
        statistic = max(0.0, numerator / denominator)
        p_value = float(chi2.sf(statistic, k - 1))
        degenerate = False
    return {
        "n_paired_items": int(n),
        "policies": int(k),
        "statistic": float(statistic),
        "df": int(k - 1),
        "p_value_raw": p_value,
        "degenerate_no_informative_discordance": degenerate,
    }


def _stuart_maxwell(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Paired marginal-homogeneity test for two three-state label vectors."""
    states = len(LABEL_STATES)
    table = np.zeros((states, states), dtype=np.int64)
    np.add.at(table, (left, right), 1)
    row = table.sum(axis=1, dtype=float)
    column = table.sum(axis=0, dtype=float)
    difference = row[:-1] - column[:-1]
    covariance = np.empty((states - 1, states - 1), dtype=float)
    for i in range(states - 1):
        for j in range(states - 1):
            if i == j:
                covariance[i, i] = row[i] + column[i] - 2.0 * table[i, i]
            else:
                covariance[i, j] = -(table[i, j] + table[j, i])
    rank = int(np.linalg.matrix_rank(covariance))
    if rank == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = max(
            0.0,
            float(difference @ np.linalg.pinv(covariance) @ difference),
        )
        p_value = float(chi2.sf(statistic, rank))
    return {
        "method": "Stuart-Maxwell paired marginal-homogeneity test",
        "contingency_table_rows_left_columns_right": table.tolist(),
        "statistic": statistic,
        "df_covariance_rank": rank,
        "p_value_raw": p_value,
    }


def _holm_adjust(raw_p_values: list[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in the original order."""
    count = len(raw_p_values)
    order = np.argsort(np.asarray(raw_p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for position, original_index in enumerate(order):
        candidate = (count - position) * raw_p_values[int(original_index)]
        running = max(running, candidate)
        adjusted[int(original_index)] = min(1.0, running)
    return adjusted.tolist()


def _heterogeneity(
    matrix: np.ndarray, counts: np.ndarray, boot_counts: np.ndarray
) -> dict[str, Any]:
    n = matrix.shape[0]
    distributions = counts / n
    boot_distributions = boot_counts / n
    pair_rows: list[dict[str, Any]] = []
    boot_jsd_columns: list[np.ndarray] = []
    boot_tv_columns: list[np.ndarray] = []
    for left in range(len(POLICY_IDS)):
        for right in range(left + 1, len(POLICY_IDS)):
            p = distributions[left]
            q = distributions[right]
            bp = boot_distributions[:, left, :]
            bq = boot_distributions[:, right, :]
            point_jsd = float(_jsd_base2(p[None, :], q[None, :])[0])
            point_tv = float(0.5 * np.abs(p - q).sum())
            boot_jsd = _jsd_base2(bp, bq)
            boot_tv = 0.5 * np.abs(bp - bq).sum(axis=1)
            jsd_ci, jsd_valid = _ci(boot_jsd)
            tv_ci, tv_valid = _ci(boot_tv)
            pair_rows.append(
                {
                    "left": POLICY_IDS[left],
                    "right": POLICY_IDS[right],
                    "jensen_shannon_divergence_base2": {
                        "estimate": point_jsd,
                        "ci95": jsd_ci,
                        "bootstrap_valid_replicates": jsd_valid,
                    },
                    "total_variation_distance": {
                        "estimate": point_tv,
                        "ci95": tv_ci,
                        "bootstrap_valid_replicates": tv_valid,
                    },
                    "paired_marginal_homogeneity": _stuart_maxwell(
                        matrix[:, left], matrix[:, right]
                    ),
                }
            )
            boot_jsd_columns.append(boot_jsd)
            boot_tv_columns.append(boot_tv)

    point_jsd_mean = float(
        np.mean([row["jensen_shannon_divergence_base2"]["estimate"] for row in pair_rows])
    )
    point_tv_mean = float(
        np.mean([row["total_variation_distance"]["estimate"] for row in pair_rows])
    )
    boot_jsd_mean = np.column_stack(boot_jsd_columns).mean(axis=1)
    boot_tv_mean = np.column_stack(boot_tv_columns).mean(axis=1)
    mean_jsd_ci, mean_jsd_valid = _ci(boot_jsd_mean)
    mean_tv_ci, mean_tv_valid = _ci(boot_tv_mean)

    pair_adjusted = _holm_adjust(
        [row["paired_marginal_homogeneity"]["p_value_raw"] for row in pair_rows]
    )
    distinguishable_counts = {policy: 0 for policy in POLICY_IDS}
    for row, adjusted in zip(pair_rows, pair_adjusted):
        test = row["paired_marginal_homogeneity"]
        test["p_value_holm_45_pairs"] = float(adjusted)
        rejected = bool(adjusted < 0.05)
        test["reject_equal_marginals_at_0_05"] = rejected
        if rejected:
            distinguishable_counts[row["left"]] += 1
            distinguishable_counts[row["right"]] += 1

    tests: dict[str, Any] = {}
    raw_p_values: list[float] = []
    for s_idx, state in enumerate(LABEL_STATES):
        test = _cochran_q((matrix == s_idx).astype(np.int8))
        raw_p_values.append(test["p_value_raw"])
        tests[state] = test
    family_size = len(LABEL_STATES)
    for state, raw_p in zip(LABEL_STATES, raw_p_values):
        adjusted = min(1.0, family_size * raw_p)
        tests[state]["p_value_bonferroni"] = float(adjusted)
        tests[state]["reject_equal_policy_marginals_at_0_05"] = bool(adjusted < 0.05)
    global_bonferroni = min(1.0, family_size * min(raw_p_values))

    return {
        "estimand": (
            "Marginal three-state teacher target-label distributions across policies; "
            "this is not D0 critic behavior or per-policy critic F1."
        ),
        "n_paired_complete_items": int(n),
        "pair_count": len(pair_rows),
        "distance_definition": {
            "jensen_shannon": "Base-2 Jensen-Shannon divergence over satisfied/violated/not_applicable; range [0,1].",
            "total_variation": "One half of the L1 distance over the same three states; range [0,1].",
            "inference_caveat": (
                "Percentile bootstrap intervals quantify sampling uncertainty around the "
                "observed distances; they are not null-calibrated pairwise hypothesis tests."
            ),
        },
        "pairs": pair_rows,
        "pairwise_marginal_homogeneity": {
            "multiplicity_control": "Holm family-wise adjustment over all 45 policy pairs",
            "alpha": 0.05,
            "rejected_pairs": int(
                sum(
                    row["paired_marginal_homogeneity"]["reject_equal_marginals_at_0_05"]
                    for row in pair_rows
                )
            ),
            "all_45_pairs_rejected": bool(
                all(
                    row["paired_marginal_homogeneity"]["reject_equal_marginals_at_0_05"]
                    for row in pair_rows
                )
            ),
            "significant_partners_per_policy_out_of_9": distinguishable_counts,
        },
        "mean_over_45_pairs": {
            "jensen_shannon_divergence_base2": {
                "estimate": point_jsd_mean,
                "ci95": mean_jsd_ci,
                "bootstrap_valid_replicates": mean_jsd_valid,
            },
            "total_variation_distance": {
                "estimate": point_tv_mean,
                "ci95": mean_tv_ci,
                "bootstrap_valid_replicates": mean_tv_valid,
            },
        },
        "paired_homogeneity_test": {
            "method": (
                "Cochran Q applied separately to each label-state indicator over the ten "
                "paired policy measurements; three tests controlled by Bonferroni."
            ),
            "null": "All ten policies have the same marginal probability for each tested label state.",
            "tests": tests,
            "global_bonferroni_p_value": float(global_bonferroni),
            "reject_joint_equal_marginals_at_0_05": bool(global_bonferroni < 0.05),
            "caveat": (
                "The asymptotic test establishes marginal target-label heterogeneity, not "
                "that every one of the 45 policy pairs differs and not that a trained critic is heterogeneous."
            ),
        },
    }


def _source_stratification(
    records: list[dict[str, Any]],
    matrix: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sources = np.asarray([record["source"] for record in records], dtype=object)
    result: dict[str, Any] = {}
    total = len(records)
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        source_matrix = matrix[selected]
        complete = np.all(source_matrix >= 0, axis=1)
        n_source = int(selected.sum())
        composition_weights = _bootstrap_weights(total, repetitions, rng)
        composition_boot = composition_weights @ selected.astype(np.int8)
        composition = _ratio_metric(
            n_source,
            total,
            composition_boot,
            np.full(repetitions, total),
        )
        parse_weights = _bootstrap_weights(n_source, repetitions, rng)
        parse_boot = parse_weights @ complete.astype(np.int8)
        entry: dict[str, Any] = {
            "items": n_source,
            "composition": composition,
            "complete_label_items": int(complete.sum()),
            "complete_label_rate": _ratio_metric(
                int(complete.sum()),
                n_source,
                parse_boot,
                np.full(repetitions, n_source),
            ),
        }
        if complete.any():
            entry["per_policy"] = _summarize_policy_labels(
                source_matrix[complete],
                repetitions,
                rng,
                include_heterogeneity=False,
            )[0]
        else:
            entry["per_policy"] = None
        result[source] = entry
    return result


def _summarize_perturbation(
    matrices: dict[str, np.ndarray],
    raw_fields: dict[str, str],
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    canonical = matrices["canonical"]
    n = canonical.shape[0]
    weights = _bootstrap_weights(n, repetitions, rng)
    parsed = {name: np.all(matrix >= 0, axis=1) for name, matrix in matrices.items()}

    parse_rates: dict[str, Any] = {}
    for name in ("canonical", *REGISTERED_PERTURBATIONS):
        indicator = parsed[name].astype(np.int8)
        parse_rates[name] = _ratio_metric(
            int(indicator.sum()),
            n,
            weights @ indicator,
            np.full(repetitions, n),
        )

    variants: dict[str, Any] = {}
    for name in REGISTERED_PERTURBATIONS:
        other = matrices[name]
        paired = parsed["canonical"] & parsed[name]
        paired_i = paired.astype(np.int8)
        paired_n = int(paired.sum())
        boot_paired = weights @ paired_i
        equal_cells = (canonical == other) & paired[:, None]
        exact = paired & np.all(canonical == other, axis=1)
        exact_i = exact.astype(np.int8)
        cell_counts = equal_cells.sum(axis=1, dtype=np.int16)
        variant: dict[str, Any] = {
            "raw_input_field": raw_fields[name],
            "parse_rate": parse_rates[name],
            "paired_with_canonical": {
                "n": paired_n,
                "fraction_of_all_items": _ratio_metric(
                    paired_n,
                    n,
                    boot_paired,
                    np.full(repetitions, n),
                ),
            },
            "whole_record_exact_match": _ratio_metric(
                int(exact.sum()),
                paired_n,
                weights @ exact_i,
                boot_paired,
            ),
            "policy_cell_micro_agreement": _ratio_metric(
                int(cell_counts.sum()),
                paired_n * len(POLICY_IDS),
                weights @ cell_counts,
                boot_paired * len(POLICY_IDS),
            ),
            "per_policy_agreement": {},
        }
        for p_idx, policy in enumerate(POLICY_IDS):
            agree = equal_cells[:, p_idx].astype(np.int8)
            variant["per_policy_agreement"][policy] = _ratio_metric(
                int(agree.sum()),
                paired_n,
                weights @ agree,
                boot_paired,
            )
        variants[name] = variant

    return {
        "items": int(n),
        "canonical_parse_rate": parse_rates["canonical"],
        "registered_field_resolution": raw_fields,
        "variants": variants,
    }


def _path_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    conflict_path = Path(args.conflict)
    audit_pool_path = Path(args.audit_pool)
    perturb_path = Path(args.perturb)
    if args.bootstrap < 1:
        raise ValidationError("--bootstrap must be positive")
    if args.expected_items < 1:
        raise ValidationError("--expected-items must be positive")

    conflict_records, conflict_matrix = _load_conflict(conflict_path, args.expected_items)
    audit_pool_records = _read_jsonl(audit_pool_path)
    _validate_expected_count(audit_pool_records, args.expected_items, "audit pool")
    perturb_records, perturb_fields, perturb_matrices = _load_perturb(
        perturb_path, args.expected_items
    )
    conflict_ids = {record["id"] for record in conflict_records}
    audit_pool_ids = {record["id"] for record in audit_pool_records}
    perturb_ids = {record["id"] for record in perturb_records}
    missing_perturb = audit_pool_ids - perturb_ids
    extra_perturb = perturb_ids - audit_pool_ids
    if missing_perturb or extra_perturb:
        raise ValidationError(
            "perturbation rows do not exactly match audit-pool ids; "
            f"missing={len(missing_perturb)}, extra={len(extra_perturb)}"
        )
    audit_pool_order = [record["id"] for record in audit_pool_records]
    perturb_order = [record["id"] for record in perturb_records]
    if audit_pool_order != perturb_order:
        mismatch = next(
            i for i, (left, right) in enumerate(zip(audit_pool_order, perturb_order))
            if left != right
        )
        raise ValidationError(
            "perturbation row order does not match audit-pool order; "
            f"first mismatch at index {mismatch}: "
            f"audit={audit_pool_order[mismatch]!r}, perturb={perturb_order[mismatch]!r}"
        )
    overlap = conflict_ids & perturb_ids
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValidationError(
            "conflict and perturbation-audit inputs must be disjoint; "
            f"found {len(overlap)} overlapping ids (first: {preview})"
        )
    complete = np.all(conflict_matrix >= 0, axis=1)
    if not complete.any():
        raise ValidationError("conflict: no complete parsed label records")

    # One deterministic generator is used in a fixed analysis order.  Every
    # bootstrap independently resamples whole item clusters.
    rng = np.random.default_rng(args.seed)
    complete_i = complete.astype(np.int8)
    parse_weights = _bootstrap_weights(len(conflict_records), args.bootstrap, rng)
    parse_boot = parse_weights @ complete_i
    per_policy, heterogeneity = _summarize_policy_labels(
        conflict_matrix[complete],
        args.bootstrap,
        rng,
        include_heterogeneity=True,
    )
    assert heterogeneity is not None

    result = {
        "schema_version": "pccd.day3.g1_teacher_analysis.v1",
        "analysis_scope": {
            "estimates": [
                "teacher target-label distributions on conflict split",
                "teacher perturbation agreement on audit split",
            ],
            "does_not_estimate": [
                "D0 critic predictions or behavior",
                "per-policy D0 critic F1",
                "cross-policy D0 F1 coefficient of variation",
                "overall G1 pass/fail verdict",
            ],
        },
        "parameters": {
            "bootstrap_repetitions": int(args.bootstrap),
            "seed": int(args.seed),
            "confidence_level": 0.95,
            "bootstrap_unit": "item cluster retaining all ten policy labels",
            "bootstrap_interval": "percentile",
            "expected_items_per_input": int(args.expected_items),
            "policy_order": list(POLICY_IDS),
            "label_order": list(LABEL_STATES),
        },
        "inputs": {
            "conflict": _path_metadata(conflict_path),
            "audit_pool": _path_metadata(audit_pool_path),
            "perturb": _path_metadata(perturb_path),
            "audit_pool_to_perturb_id_match": True,
            "audit_pool_to_perturb_id_order_match": True,
            "cross_input_id_overlap": 0,
        },
        "conflict": {
            "items": len(conflict_records),
            "unique_ids": len({record["id"] for record in conflict_records}),
            "complete_label_items": int(complete.sum()),
            "complete_label_rate": _ratio_metric(
                int(complete.sum()),
                len(conflict_records),
                parse_boot,
                np.full(args.bootstrap, len(conflict_records)),
            ),
            "per_policy": per_policy,
            "source_stratification": _source_stratification(
                conflict_records,
                conflict_matrix,
                args.bootstrap,
                rng,
            ),
            "target_label_distribution_heterogeneity": heterogeneity,
        },
        "perturbation": {
            "unique_ids": len({record["id"] for record in perturb_records}),
            **_summarize_perturbation(
                perturb_matrices,
                perturb_fields,
                args.bootstrap,
                rng,
            ),
        },
    }
    return result


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Day-3 teacher target-label heterogeneity and perturbation stability."
    )
    parser.add_argument("--conflict", required=True, help="Merged conflict label JSONL")
    parser.add_argument("--audit-pool", required=True, help="Original audit pool JSONL")
    parser.add_argument("--perturb", required=True, help="Raw audit perturbation JSONL")
    parser.add_argument("--out", required=True, help="Machine-readable analysis JSON")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--expected-items",
        type=int,
        default=400,
        help="Strict expected count for each input (default: 400; useful override for tests)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        result = analyze(args)
    except (OSError, ValidationError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote G1 teacher-only analysis: {out_path}")
    print(f"  conflict complete: {result['conflict']['complete_label_items']}/{result['conflict']['items']}")
    print(
        "  mean pairwise JSD/TV: "
        f"{result['conflict']['target_label_distribution_heterogeneity']['mean_over_45_pairs']['jensen_shannon_divergence_base2']['estimate']:.6f}/"
        f"{result['conflict']['target_label_distribution_heterogeneity']['mean_over_45_pairs']['total_variation_distance']['estimate']:.6f}"
    )
    print("  scope: teacher targets/stability only; no D0 F1 or overall G1 verdict")


if __name__ == "__main__":
    main()
