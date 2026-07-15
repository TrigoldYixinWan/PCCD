#!/usr/bin/env python3
"""Analyze the pre-registered Day-3 diag100 structure ablation and D-3 check."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2_contingency


POLICY_IDS = ("H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2")
LABEL_STATES = ("satisfied", "violated", "not_applicable")
LABEL_TO_INT = {label: index for index, label in enumerate(LABEL_STATES)}
STRUCTURES = (
    "single_policy",
    "five_policy_block",
    "ten_policy_joint",
    "latin_square_order",
)
EXPECTED_CALLS_PER_ITEM = {
    "single_policy": 10,
    "five_policy_block": 2,
    "ten_policy_joint": 1,
    "latin_square_order": 1,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(record)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_labels(labels: Any, policies: Iterable[str], where: str) -> dict[str, str]:
    ids = tuple(policies)
    if not isinstance(labels, dict) or set(labels) != set(ids):
        raise ValueError(f"{where}: labels do not exactly match requested policies")
    if any(labels[pid] not in LABEL_TO_INT for pid in ids):
        raise ValueError(f"{where}: invalid label state")
    return labels


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, float | int]:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("bootstrap arrays must be equal-length vectors")
    point_denominator = float(denominator.sum())
    if point_denominator <= 0:
        return {
            "point": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "numerator": 0,
            "denominator": 0,
        }
    point = float(numerator.sum() / point_denominator)
    n_items = len(numerator)
    values: list[np.ndarray] = []
    remaining = replicates
    # Chunking bounds memory while preserving item-cluster resampling.
    while remaining:
        chunk = min(remaining, 1000)
        indices = rng.integers(0, n_items, size=(chunk, n_items))
        boot_num = numerator[indices].sum(axis=1)
        boot_den = denominator[indices].sum(axis=1)
        chunk_values = np.full(chunk, np.nan, dtype=float)
        np.divide(boot_num, boot_den, out=chunk_values, where=boot_den > 0)
        values.append(chunk_values)
        remaining -= chunk
    samples = np.concatenate(values)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        raise ValueError("all bootstrap replicates had a zero denominator")
    return {
        "point": point,
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "numerator": int(numerator.sum()),
        "denominator": int(denominator.sum()),
    }


def contingency(records: list[dict[str, Any]]) -> dict[str, Any]:
    table = np.zeros((3, 3), dtype=int)
    for record in records:
        labels = record["labels"]
        table[LABEL_TO_INT[labels["S2"]], LABEL_TO_INT[labels["S3"]]] += 1
    result: dict[str, Any] = {
        "n": len(records),
        "row_policy": "S2",
        "column_policy": "S3",
        "state_order": list(LABEL_STATES),
        "table": table.tolist(),
    }
    active_rows = table.sum(axis=1) > 0
    active_columns = table.sum(axis=0) > 0
    active = table[np.ix_(active_rows, active_columns)]
    if len(records) and min(active.shape) > 1:
        chi2, p_value, degrees, _ = chi2_contingency(active, correction=False)
        denom = len(records) * min(active.shape[0] - 1, active.shape[1] - 1)
        result.update(
            {
                "chi2": float(chi2),
                "degrees_of_freedom": int(degrees),
                "p_value": float(p_value),
                "cramers_v": float(np.sqrt(chi2 / denom)),
            }
        )
    return result


def chosen_soft_pole(meta: dict[str, Any]) -> str | None:
    target = meta.get("target_pole")
    matches = meta.get("intended_match")
    if target not in ("A", "B") or not isinstance(matches, bool):
        return None
    if matches:
        return target
    return "B" if target == "A" else "A"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    reference_records = read_jsonl(args.reference)
    if len(reference_records) != 100:
        raise ValueError(f"locked diag100 requires 100 references, got {len(reference_records)}")
    reference_by_id: dict[str, dict[str, Any]] = {}
    for record in reference_records:
        item_id = record.get("id")
        if not isinstance(item_id, str) or item_id in reference_by_id:
            raise ValueError("reference IDs must be non-empty and unique")
        validate_labels(record.get("labels"), POLICY_IDS, f"reference id={item_id}")
        reference_by_id[item_id] = record
    ids = sorted(reference_by_id)
    id_to_index = {item_id: index for index, item_id in enumerate(ids)}

    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_ids = sorted(item["id"] for item in manifest["selected_items"])
    if manifest_ids != ids:
        raise ValueError("manifest/reference ID mismatch")
    recorded_reference_sha = manifest["artifacts"]["reference_labels"]["sha256"]
    if recorded_reference_sha != sha256_file(args.reference):
        raise ValueError("reference SHA-256 differs from the diag100 manifest")

    calls = read_jsonl(args.ablation)
    expected_calls = len(ids) * sum(EXPECTED_CALLS_PER_ITEM.values())
    if len(calls) != expected_calls:
        raise ValueError(f"expected {expected_calls} ablation calls, found {len(calls)}")

    outputs = {
        structure: np.full((len(ids), len(POLICY_IDS)), -1, dtype=np.int8)
        for structure in STRUCTURES
    }
    cell_seen = {
        structure: np.zeros((len(ids), len(POLICY_IDS)), dtype=np.int8)
        for structure in STRUCTURES
    }
    call_counts: Counter[tuple[str, str]] = Counter()
    parse_counts: Counter[str] = Counter()
    latin_positions = np.full((len(ids), len(POLICY_IDS)), -1, dtype=np.int8)

    for call_number, call in enumerate(calls, 1):
        item_id = call.get("id")
        structure = call.get("structure")
        policies = call.get("policies")
        order = call.get("order")
        where = f"ablation call {call_number}"
        if item_id not in id_to_index or structure not in STRUCTURES:
            raise ValueError(f"{where}: unknown id or structure")
        if not isinstance(policies, list) or not isinstance(order, list):
            raise ValueError(f"{where}: policies/order must be lists")
        if len(set(policies)) != len(policies) or set(order) != set(policies):
            raise ValueError(f"{where}: invalid policy subset/order")
        if any(pid not in POLICY_IDS for pid in policies):
            raise ValueError(f"{where}: unknown policy")
        call_counts[(item_id, structure)] += 1
        labels = call.get("labels")
        if call.get("parse_ok") is True:
            labels = validate_labels(labels, policies, where)
            parse_counts[structure] += 1
        elif labels is not None:
            raise ValueError(f"{where}: parse_ok false but labels present")
        row = id_to_index[item_id]
        for pid in policies:
            column = POLICY_IDS.index(pid)
            if cell_seen[structure][row, column]:
                raise ValueError(f"{where}: duplicate output cell {item_id}/{structure}/{pid}")
            cell_seen[structure][row, column] = 1
            if labels is not None:
                outputs[structure][row, column] = LABEL_TO_INT[labels[pid]]
            if structure == "latin_square_order":
                latin_positions[row, column] = order.index(pid)

    for item_id in ids:
        for structure, expected in EXPECTED_CALLS_PER_ITEM.items():
            actual = call_counts[(item_id, structure)]
            if actual != expected:
                raise ValueError(
                    f"id={item_id}: {structure} has {actual} calls, expected {expected}"
                )
    for structure in STRUCTURES:
        if not np.all(cell_seen[structure] == 1):
            raise ValueError(f"{structure}: missing or duplicate policy cells")
    for position in range(len(POLICY_IDS)):
        if int((latin_positions == position).sum()) != len(ids):
            raise ValueError("Latin-square position allocation is not balanced")

    reference = np.asarray(
        [
            [LABEL_TO_INT[reference_by_id[item_id]["labels"][pid]] for pid in POLICY_IDS]
            for item_id in ids
        ],
        dtype=np.int8,
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    structure_results: dict[str, Any] = {}
    for structure in STRUCTURES:
        observed = outputs[structure] >= 0
        agree = (outputs[structure] == reference) & observed
        result: dict[str, Any] = {
            "parsed_calls": int(parse_counts[structure]),
            "total_calls": len(ids) * EXPECTED_CALLS_PER_ITEM[structure],
            "parsed_cells": int(observed.sum()),
            "total_cells": len(ids) * len(POLICY_IDS),
            "cell_micro_agreement": bootstrap_ratio(
                agree.sum(axis=1),
                observed.sum(axis=1),
                rng=rng,
                replicates=args.bootstrap_replicates,
            ),
            "per_policy": {},
        }
        for column, pid in enumerate(POLICY_IDS):
            result["per_policy"][pid] = {
                "agreement": bootstrap_ratio(
                    agree[:, column].astype(int),
                    observed[:, column].astype(int),
                    rng=rng,
                    replicates=args.bootstrap_replicates,
                ),
                "not_applicable_rate": bootstrap_ratio(
                    (outputs[structure][:, column] == 2).astype(int),
                    observed[:, column].astype(int),
                    rng=rng,
                    replicates=args.bootstrap_replicates,
                ),
                "reference_not_applicable_rate": float((reference[:, column] == 2).mean()),
            }
        structure_results[structure] = result

    position_results: dict[str, Any] = {}
    latin_observed = outputs["latin_square_order"] >= 0
    latin_agree = (outputs["latin_square_order"] == reference) & latin_observed
    for position in range(len(POLICY_IDS)):
        mask = latin_positions == position
        position_results[str(position + 1)] = bootstrap_ratio(
            (latin_agree & mask).sum(axis=1),
            (latin_observed & mask).sum(axis=1),
            rng=rng,
            replicates=args.bootstrap_replicates,
        )

    d3_records = [reference_by_id[item_id] for item_id in ids]
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_soft_axis: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in d3_records:
        by_source[record["source"]].append(record)
        if record["source"] == "soft_style":
            by_soft_axis[str(record.get("meta", {}).get("axis", "missing"))].append(record)
    soft_records = by_source.get("soft_style", [])
    explicit_dual_axis = 0
    potential_joint_template = 0
    chosen_poles = Counter()
    for record in soft_records:
        meta = record.get("meta", {})
        axis = meta.get("axis")
        # The generator chooses exactly one registered axis.  Its structure
        # templates also alter layout/length cues, so flag those separately as
        # a potential S2/S3 template confound rather than calling them dual-axis.
        if isinstance(axis, (list, tuple, set)) and {"verbosity", "structure"} <= set(axis):
            explicit_dual_axis += 1
        if axis == "structure":
            potential_joint_template += 1
        chosen_poles[str(chosen_soft_pole(meta))] += 1

    d3 = {
        "overall": contingency(d3_records),
        "by_source": {source: contingency(records) for source, records in sorted(by_source.items())},
        "soft_style_by_registered_axis": {
            axis: contingency(records) for axis, records in sorted(by_soft_axis.items())
        },
        "generator_audit": {
            "soft_style_items": len(soft_records),
            "explicitly_manipulated_axes_per_item": 1,
            "explicit_s2_and_s3_dual_axis_items": explicit_dual_axis,
            "explicit_s2_and_s3_dual_axis_fraction": (
                explicit_dual_axis / len(soft_records) if soft_records else None
            ),
            "potential_s2_s3_template_confound_items": potential_joint_template,
            "potential_s2_s3_template_confound_fraction": (
                potential_joint_template / len(soft_records) if soft_records else None
            ),
            "potential_confound_definition": (
                "registered axis=structure: structured/unstructured response templates also "
                "change layout and some length cues"
            ),
            "chosen_response_pole_counts": dict(sorted(chosen_poles.items())),
        },
    }

    return {
        "schema_version": "pccd.day3.diag_metrics.v1",
        "diagnostic_only": True,
        "gate_verdict": None,
        "bootstrap": {
            "method": "item-cluster percentile bootstrap",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "confidence": 0.95,
        },
        "artifacts": {
            "reference": {"path": str(args.reference), "sha256": sha256_file(args.reference)},
            "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
            "ablation": {"path": str(args.ablation), "sha256": sha256_file(args.ablation)},
        },
        "d2_structure_ablation": structure_results,
        "latin_square_position_agreement": position_results,
        "d3_s2_s3": d3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"refusing to overwrite diagnostic metrics: {args.out}")
    if args.bootstrap_replicates <= 0:
        parser.error("bootstrap-replicates must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = analyze(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("=== D-2 STRUCTURE ABLATION (diagnostic only; no gate verdict) ===")
    for structure in STRUCTURES:
        metric = result["d2_structure_ablation"][structure]["cell_micro_agreement"]
        print(
            f"{structure:20s} {100 * metric['point']:6.2f}% "
            f"[{100 * metric['ci_lower']:.2f}, {100 * metric['ci_upper']:.2f}] "
            f"cells={metric['denominator']}"
        )
    d3 = result["d3_s2_s3"]["overall"]
    print(f"D-3 S2xS3: n={d3['n']} Cramer's V={d3.get('cramers_v', float('nan')):.4f}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
