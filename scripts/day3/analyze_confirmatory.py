#!/usr/bin/env python3
"""Analyze the single locked L1 confirmatory audit and issue its verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.audit_labels import (  # noqa: E402
    CONFIRMATORY_AUDIT_SHA256,
    validate_confirmatory_schedule,
)
from src.policy_defs import (  # noqa: E402
    LABEL_STATES,
    POLICY_IDS,
    parse_judgment_cells,
)


VARIANTS = ("canonical", "repeat_sampling", "policy_order_swap", "policy_paraphrase")
PERTURBATIONS = ("repeat_sampling", "policy_order_swap", "policy_paraphrase")
THRESHOLDS = {
    "repeat_sampling": 0.97,
    "policy_order_swap": 0.90,
    "policy_paraphrase": 0.90,
}
LABEL_TO_INT = {label: index for index, label in enumerate(LABEL_STATES)}


class ValidationError(ValueError):
    """Raised when confirmatory artifacts violate the locked contract."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValidationError(f"{path}:{line_number}: blank line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise ValidationError(f"{path}:{line_number}: invalid record/id")
            if record["id"] in seen:
                raise ValidationError(f"{path}:{line_number}: duplicate id={record['id']!r}")
            seen.add(record["id"])
            records.append(record)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_ratio(
    per_item_numerator: np.ndarray,
    per_item_denominator: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    numerator = int(per_item_numerator.sum())
    denominator = int(per_item_denominator.sum())
    if denominator <= 0:
        raise ValidationError("metric has zero parsed-cell denominator")
    boot_num = weights @ per_item_numerator
    boot_den = weights @ per_item_denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.divide(
            boot_num,
            boot_den,
            out=np.full(boot_num.shape, np.nan, dtype=float),
            where=boot_den > 0,
        )
    finite = boot[np.isfinite(boot)]
    if not len(finite):
        raise ValidationError("bootstrap has no valid replicate")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": float(numerator / denominator),
        "ci95": [float(lower), float(upper)],
        "bootstrap_valid_replicates": int(len(finite)),
    }


def validate_and_load(
    audit_path: Path,
    raw_path: Path,
    marker_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, list[list[str]]],
]:
    audit_records = read_jsonl(audit_path)
    raw_records = read_jsonl(raw_path)
    if len(audit_records) != 400 or len(raw_records) != 400:
        raise ValidationError(
            f"confirmatory inputs must each contain 400 rows: audit={len(audit_records)}, "
            f"raw={len(raw_records)}"
        )
    audit_ids = [record["id"] for record in audit_records]
    raw_ids = [record["id"] for record in raw_records]
    if audit_ids != raw_ids:
        raise ValidationError("raw confirmatory ids/order differ from the frozen audit pool")

    with marker_path.open("r", encoding="utf-8") as handle:
        marker = json.load(handle)
    if marker.get("status") != "complete" or marker.get("single_run_guard") is not True:
        raise ValidationError("confirmatory run marker is not sealed complete")
    if marker.get("audit_sha256") != sha256_file(audit_path):
        raise ValidationError("audit SHA-256 differs from run marker")
    if marker.get("audit_sha256") != CONFIRMATORY_AUDIT_SHA256:
        raise ValidationError("audit SHA-256 differs from the frozen Day-2/3 artifact")
    if marker.get("output_sha256") != sha256_file(raw_path):
        raise ValidationError("raw output SHA-256 differs from run marker")
    if marker.get("retries") != 0 or marker.get("temperature") != 0.0:
        raise ValidationError("run marker violates no-retry/temperature-zero rules")

    expected_orders, position_counts, schedule_sha256 = validate_confirmatory_schedule(audit_records)
    if marker.get("latin_position_counts") != position_counts:
        raise ValidationError("marker Latin position counts differ from recomputation")
    if marker.get("latin_schedule_sha256") != schedule_sha256:
        raise ValidationError("marker Latin schedule SHA-256 differs from recomputation")

    n_items, n_policies = len(raw_records), len(POLICY_IDS)
    matrices = {
        variant: np.full((n_items, n_policies), -1, dtype=np.int8)
        for variant in VARIANTS
    }
    strict = {
        variant: np.zeros(n_items, dtype=bool)
        for variant in VARIANTS
    }
    displayed_orders: dict[str, list[list[str]]] = {variant: [] for variant in VARIANTS}
    canonical_order = list(POLICY_IDS)

    for row, (record, expected_order) in enumerate(zip(raw_records, expected_orders)):
        if record.get("policy_order_swap_order") != expected_order:
            raise ValidationError(f"id={record['id']!r}: stored Latin order mismatch")
        variants = record.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
            raise ValidationError(f"id={record['id']!r}: variant keys mismatch")
        for variant in VARIANTS:
            stored = variants[variant]
            if not isinstance(stored, dict) or not isinstance(stored.get("raw_text"), str):
                raise ValidationError(f"id={record['id']!r}.{variant}: malformed result")
            reparsed = parse_judgment_cells(stored["raw_text"])
            for field in (
                "json_object_found",
                "strict_parse_ok",
                "cells",
                "missing_keys",
                "extra_keys",
                "duplicate_keys",
                "invalid_value_keys",
            ):
                if stored.get(field) != reparsed[field]:
                    raise ValidationError(
                        f"id={record['id']!r}.{variant}: stored parser field {field} mismatch"
                    )
            strict[variant][row] = reparsed["strict_parse_ok"]
            for column, pid in enumerate(POLICY_IDS):
                if pid in reparsed["cells"]:
                    matrices[variant][row, column] = LABEL_TO_INT[reparsed["cells"][pid]]
            displayed_orders[variant].append(
                expected_order if variant == "policy_order_swap" else canonical_order
            )
    return audit_records, raw_records, marker, matrices, strict, displayed_orders


def missing_diagnostics(
    matrix: np.ndarray,
    orders: list[list[str]],
) -> dict[str, Any]:
    missing = matrix < 0
    per_policy = {
        pid: {
            "missing": int(missing[:, column].sum()),
            "denominator": int(matrix.shape[0]),
            "rate": float(missing[:, column].mean()),
        }
        for column, pid in enumerate(POLICY_IDS)
    }
    by_position = []
    policy_position = {
        pid: [
            {"missing": 0, "denominator": 0, "rate": None}
            for _ in range(len(POLICY_IDS))
        ]
        for pid in POLICY_IDS
    }
    for position in range(len(POLICY_IDS)):
        count = 0
        for row, order in enumerate(orders):
            pid = order[position]
            column = POLICY_IDS.index(pid)
            is_missing = bool(missing[row, column])
            count += int(is_missing)
            cell = policy_position[pid][position]
            cell["missing"] += int(is_missing)
            cell["denominator"] += 1
        by_position.append(
            {
                "position": position + 1,
                "missing": count,
                "denominator": matrix.shape[0],
                "rate": float(count / matrix.shape[0]),
            }
        )
    for pid in POLICY_IDS:
        for cell in policy_position[pid]:
            if cell["denominator"]:
                cell["rate"] = float(cell["missing"] / cell["denominator"])
    return {
        "per_policy": per_policy,
        "per_position": by_position,
        "policy_by_position": policy_position,
    }


def schema_issue_diagnostics(
    raw_records: list[dict[str, Any]],
    variant: str,
    orders: list[list[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("missing_keys", "invalid_value_keys", "duplicate_keys"):
        issue_matrix = np.zeros((len(raw_records), len(POLICY_IDS)), dtype=bool)
        for row, record in enumerate(raw_records):
            for pid in record["variants"][variant][field]:
                if pid in POLICY_IDS:
                    issue_matrix[row, POLICY_IDS.index(pid)] = True
        # Reuse the position-aware counter: -1 denotes this specific issue,
        # while 0 denotes no issue (not a teacher label in this context).
        result[field] = missing_diagnostics(
            np.where(issue_matrix, -1, 0).astype(np.int8), orders
        )
    extra_keys: dict[str, int] = {}
    for record in raw_records:
        for key in record["variants"][variant]["extra_keys"]:
            extra_keys[key] = extra_keys.get(key, 0) + 1
    result["extra_keys"] = dict(sorted(extra_keys.items()))
    return result


def transition_tables(canonical: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for column, pid in enumerate(POLICY_IDS):
        table = np.zeros((len(LABEL_STATES), len(LABEL_STATES)), dtype=int)
        observed = (canonical[:, column] >= 0) & (other[:, column] >= 0)
        for left, right in zip(canonical[observed, column], other[observed, column]):
            table[int(left), int(right)] += 1
        tables[pid] = {
            "state_order": list(LABEL_STATES),
            "table": table.tolist(),
            "to_not_applicable": int(table[:2, 2].sum()),
            "from_not_applicable": int(table[2, :2].sum()),
            "both_parsed": int(observed.sum()),
        }
    return tables


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = Path(args.audit)
    raw_path = Path(args.raw)
    marker_path = Path(args.run_marker)
    frozen_path = Path(args.frozen_paraphrases)
    audit_records, raw_records, marker, matrices, strict, orders = validate_and_load(
        audit_path, raw_path, marker_path
    )
    if marker.get("frozen_paraphrases_sha256") != sha256_file(frozen_path):
        raise ValidationError("frozen paraphrase artifact SHA-256 differs from run marker")

    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(
        len(raw_records),
        np.full(len(raw_records), 1.0 / len(raw_records)),
        size=args.bootstrap,
    )
    canonical = matrices["canonical"]
    perturbation_results: dict[str, Any] = {}
    for variant in PERTURBATIONS:
        other = matrices[variant]
        observed = (canonical >= 0) & (other >= 0)
        agree = observed & (canonical == other)
        micro = bootstrap_ratio(agree.sum(axis=1), observed.sum(axis=1), weights)
        threshold = THRESHOLDS[variant]
        gate_pass = (
            micro["estimate"] >= threshold
            and micro["ci95"][0] >= threshold - 0.02
        )
        per_policy = {}
        for column, pid in enumerate(POLICY_IDS):
            per_policy[pid] = bootstrap_ratio(
                agree[:, column].astype(int),
                observed[:, column].astype(int),
                weights,
            )
        strict_pair = strict["canonical"] & strict[variant]
        exact = strict_pair & np.all(canonical == other, axis=1)
        perturbation_results[variant] = {
            "cell_micro_agreement": micro,
            "threshold": threshold,
            "required_ci_lower": threshold - 0.02,
            "passes_locked_rule": bool(gate_pass),
            "per_policy_agreement": per_policy,
            "strict_whole_record_exact_match": {
                "numerator": int(exact.sum()),
                "denominator": int(strict_pair.sum()),
                "estimate": float(exact.sum() / strict_pair.sum()) if strict_pair.sum() else None,
                "gate_role": "descriptive_only",
            },
            "na_transitions": transition_tables(canonical, other),
        }

    if not perturbation_results["repeat_sampling"]["passes_locked_rule"]:
        verdict = "FAIL"
    elif all(
        perturbation_results[variant]["passes_locked_rule"]
        for variant in PERTURBATIONS
    ):
        verdict = "PASS"
    else:
        verdict = "PARTIAL"

    parse_diagnostics = {}
    for variant in VARIANTS:
        parse_diagnostics[variant] = {
            "strict_10_key_success": {
                "numerator": int(strict[variant].sum()),
                "denominator": len(raw_records),
                "rate": float(strict[variant].mean()),
            },
            "valid_policy_cells": {
                "numerator": int((matrices[variant] >= 0).sum()),
                "denominator": len(raw_records) * len(POLICY_IDS),
                "rate": float((matrices[variant] >= 0).mean()),
            },
            "unparsed_policy_cells": missing_diagnostics(
                matrices[variant], orders[variant]
            ),
            "schema_issues": schema_issue_diagnostics(
                raw_records, variant, orders[variant]
            ),
        }

    return {
        "schema_version": "pccd.day3.l1_confirmatory_analysis.v1",
        "locked_protocol": "reports/PREREG_G1.md L1 confirmatory-run execution rules",
        "l1_verdict": verdict,
        "verdict_rule": (
            "PASS iff all three perturbations pass; PARTIAL iff repeat passes and another "
            "fails; FAIL iff repeat fails the locked point+CI rule"
        ),
        "gate_inputs_only": {
            variant: perturbation_results[variant]["cell_micro_agreement"]
            for variant in PERTURBATIONS
        },
        "perturbations": perturbation_results,
        "diagnostics_excluded_from_gate": {
            "parse_and_missing_keys": parse_diagnostics,
            "na_transitions_location": "perturbations.*.na_transitions",
            "whole_record_location": "perturbations.*.strict_whole_record_exact_match",
        },
        "bootstrap": {
            "unit": "item cluster retaining ten policy cells",
            "interval": "percentile",
            "replicates": args.bootstrap,
            "seed": args.seed,
            "confidence": 0.95,
        },
        "locked_thresholds": THRESHOLDS,
        "inputs": {
            "audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "raw": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
            "run_marker": {"path": str(marker_path), "sha256": sha256_file(marker_path)},
            "frozen_paraphrases": {
                "path": str(frozen_path),
                "sha256": sha256_file(frozen_path),
            },
            "ids_and_order_match": True,
            "items": len(audit_records),
        },
        "run_marker": marker,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--run-marker", required=True)
    parser.add_argument("--frozen-paraphrases", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the deterministic CPU metrics; never reruns teacher calls",
    )
    args = parser.parse_args()
    if args.bootstrap != 10_000 or args.seed != 20260715:
        parser.error("locked analysis requires bootstrap=10000 and seed=20260715")
    if args.out.exists() and not args.force:
        parser.error(f"refusing to overwrite existing confirmatory analysis: {args.out}")
    return args


def main() -> None:
    args = parse_args()
    try:
        result = analyze(args)
    except (OSError, ValidationError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.force else "x"
    with args.out.open(mode, encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("=== L1 CONFIRMATORY RESULT ===")
    for variant in PERTURBATIONS:
        entry = result["perturbations"][variant]
        metric = entry["cell_micro_agreement"]
        print(
            f"{variant:22s} {100*metric['estimate']:.3f}% "
            f"[{100*metric['ci95'][0]:.3f}, {100*metric['ci95'][1]:.3f}] "
            f"threshold={100*entry['threshold']:.1f}% "
            f"pass={entry['passes_locked_rule']}"
        )
    print(f"L1 VERDICT: {result['l1_verdict']}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
