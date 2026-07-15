#!/usr/bin/env python3
"""Build the pre-registered balanced Day-3 ``diag100`` diagnostic view.

The source universe is the five frozen, mutually disjoint pool splits.  Frozen
teacher labels come from the merged train/calib/test/conflict label files and
the canonical column of the first, frozen audit perturbation run.  No teacher
is called and no label is changed by this script.

Selection is a deterministic mixed-integer program:

* exactly ``n`` items (default 100);
* source and frozen-split counts fixed to largest-remainder proportional quotas;
* at least ``min_per_state`` examples of every policy x label state;
* at least ``min_cross_policy`` items containing both a satisfied and a
  violated policy label;
* minimize total absolute deviation from a 1/3--1/3--1/3 state balance for
  each policy, with a stable ID-hash tie breaker.

This is a diagnostic-only balanced view.  It never replaces or modifies a
frozen split and is never used to score a gate or train a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


POLICY_IDS = ("H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2")
LABEL_STATES = ("satisfied", "violated", "not_applicable")
LABEL_TO_INT = {state: index for index, state in enumerate(LABEL_STATES)}
FROZEN_SPLITS = ("train", "calib", "test", "audit", "conflict")
LABELED_SPLITS = ("train", "calib", "test", "conflict")
EXPECTED_SPLIT_SIZES = {
    "train": 8000,
    "calib": 1000,
    "test": 1000,
    "audit": 400,
    "conflict": 400,
}


class InputError(ValueError):
    """Raised when a frozen input violates the expected contract."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise InputError(f"{path}:{line_number}: blank JSONL line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise InputError(f"{path}:{line_number}: record must be an object")
            item_id = record.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise InputError(f"{path}:{line_number}: missing/non-string id")
            if item_id in seen:
                raise InputError(f"{path}:{line_number}: duplicate id {item_id!r}")
            seen.add(item_id)
            records.append(record)
    return records


def validate_labels(labels: Any, where: str) -> dict[str, str]:
    if not isinstance(labels, dict) or set(labels) != set(POLICY_IDS):
        raise InputError(f"{where}: expected exactly the ten registered policy keys")
    for policy in POLICY_IDS:
        if labels[policy] not in LABEL_TO_INT:
            raise InputError(f"{where}.{policy}: invalid state {labels[policy]!r}")
    return labels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def largest_remainder(counts: Counter[str], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if population <= 0:
        raise ValueError("cannot allocate a quota over an empty population")
    exact = {key: total * value / population for key, value in counts.items()}
    quotas = {key: int(np.floor(value)) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def stable_tie_breaker(item_id: str, seed: int) -> float:
    payload = f"{seed}:{item_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_universe(pool_dir: Path, label_dir: Path) -> list[dict[str, Any]]:
    pool_by_id: dict[str, dict[str, Any]] = {}
    frozen_split_by_id: dict[str, str] = {}
    ordered_ids: list[str] = []
    for split in FROZEN_SPLITS:
        split_records = read_jsonl(pool_dir / f"{split}.jsonl")
        if len(split_records) != EXPECTED_SPLIT_SIZES[split]:
            raise InputError(
                f"frozen pool {split!r} has {len(split_records)} items; "
                f"expected {EXPECTED_SPLIT_SIZES[split]}"
            )
        for record in split_records:
            item_id = record["id"]
            if item_id in pool_by_id:
                raise InputError(f"frozen pool overlap/duplicate id: {item_id}")
            pool_by_id[item_id] = record
            frozen_split_by_id[item_id] = split
            ordered_ids.append(item_id)

    labels_by_id: dict[str, dict[str, str]] = {}
    label_record_by_id: dict[str, dict[str, Any]] = {}
    for split in LABELED_SPLITS:
        path = label_dir / f"{split}.jsonl"
        for record in read_jsonl(path):
            if record.get("parse_ok") is not True:
                raise InputError(f"{path}: id={record['id']!r} is not parse_ok")
            labels = validate_labels(record.get("labels"), f"{path}: id={record['id']}")
            if record["id"] in labels_by_id:
                raise InputError(f"duplicate frozen label id: {record['id']}")
            labels_by_id[record["id"]] = labels
            label_record_by_id[record["id"]] = record

    audit_path = label_dir / "audit_perturb.jsonl"
    for record in read_jsonl(audit_path):
        labels = validate_labels(
            record.get("canonical"), f"{audit_path}: id={record['id']}.canonical"
        )
        if record["id"] in labels_by_id:
            raise InputError(f"duplicate frozen audit label id: {record['id']}")
        labels_by_id[record["id"]] = labels

    missing = set(pool_by_id) - set(labels_by_id)
    extra = set(labels_by_id) - set(pool_by_id)
    if missing or extra:
        raise InputError(f"pool/label id mismatch: missing={len(missing)}, extra={len(extra)}")

    # The four ordinary merged label files duplicate the pool payload.  Verify
    # exact traceability before their labels can participate in selection.
    for item_id, label_record in label_record_by_id.items():
        pool_record = pool_by_id[item_id]
        for field in ("source", "prompt", "response"):
            if label_record.get(field) != pool_record.get(field):
                raise InputError(f"id={item_id!r}: pool/label mismatch in {field}")
        if label_record.get("meta", {}) != pool_record.get("meta", {}):
            raise InputError(f"id={item_id!r}: pool/label mismatch in meta")

    universe: list[dict[str, Any]] = []
    for item_id in ordered_ids:
        pool_record = pool_by_id[item_id]
        labels = labels_by_id[item_id]
        universe.append(
            {
                "id": item_id,
                "pool": pool_record,
                "labels": labels,
                "frozen_split": frozen_split_by_id[item_id],
                "label_record": label_record_by_id.get(item_id),
            }
        )
    return universe


def solve_selection(
    universe: list[dict[str, Any]],
    n_select: int,
    min_per_state: int,
    min_cross_policy: int,
    time_limit: float,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    n_items = len(universe)
    n_cells = len(POLICY_IDS) * len(LABEL_STATES)
    n_variables = n_items + 2 * n_cells

    label_matrix = np.asarray(
        [[LABEL_TO_INT[item["labels"][policy]] for policy in POLICY_IDS] for item in universe],
        dtype=np.int8,
    )
    cell_indicators = np.zeros((n_items, n_cells), dtype=np.int8)
    for policy_index in range(len(POLICY_IDS)):
        for state_index in range(len(LABEL_STATES)):
            cell = policy_index * len(LABEL_STATES) + state_index
            cell_indicators[:, cell] = label_matrix[:, policy_index] == state_index

    sources = [item["pool"]["source"] for item in universe]
    frozen_splits = [item["frozen_split"] for item in universe]
    source_quotas = largest_remainder(Counter(sources), n_select)
    split_quotas = largest_remainder(Counter(frozen_splits), n_select)
    cross_policy = np.asarray(
        [
            any(value == "satisfied" for value in item["labels"].values())
            and any(value == "violated" for value in item["labels"].values())
            for item in universe
        ],
        dtype=np.int8,
    )

    objective = np.zeros(n_variables, dtype=float)
    objective[:n_items] = np.asarray(
        [stable_tie_breaker(item["id"], seed) for item in universe], dtype=float
    ) * 1e-7
    objective[n_items:] = 1.0

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_row(coefficients: Iterable[tuple[int, float]], lower: float, upper: float) -> None:
        row = len(lower_bounds)
        for column, value in coefficients:
            if value:
                row_indices.append(row)
                column_indices.append(column)
                values.append(float(value))
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))

    add_row(((index, 1.0) for index in range(n_items)), n_select, n_select)

    for source, quota in sorted(source_quotas.items()):
        add_row(
            ((index, 1.0) for index, value in enumerate(sources) if value == source),
            quota,
            quota,
        )
    for split, quota in sorted(split_quotas.items()):
        add_row(
            ((index, 1.0) for index, value in enumerate(frozen_splits) if value == split),
            quota,
            quota,
        )

    target = n_select / len(LABEL_STATES)
    for cell in range(n_cells):
        members = np.flatnonzero(cell_indicators[:, cell])
        equality = [(int(index), 1.0) for index in members]
        equality.extend(
            [
                (n_items + cell, -1.0),
                (n_items + n_cells + cell, 1.0),
            ]
        )
        add_row(equality, target, target)
        add_row(((int(index), 1.0) for index in members), min_per_state, np.inf)

    add_row(
        ((int(index), 1.0) for index in np.flatnonzero(cross_policy)),
        min_cross_policy,
        np.inf,
    )

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower_bounds), n_variables),
        dtype=float,
    ).tocsr()
    constraints = LinearConstraint(
        matrix,
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
    )
    lower = np.zeros(n_variables, dtype=float)
    upper = np.full(n_variables, np.inf, dtype=float)
    upper[:n_items] = 1.0
    integrality = np.zeros(n_variables, dtype=np.int8)
    integrality[:n_items] = 1

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.x is None or result.status != 0:
        raise RuntimeError(
            f"diag100 MILP did not reach an optimal solution: status={result.status}, "
            f"message={result.message!r}"
        )
    selected = np.flatnonzero(result.x[:n_items] > 0.5).tolist()
    if len(selected) != n_select:
        raise RuntimeError(f"solver returned {len(selected)} selected items, expected {n_select}")
    solver_info = {
        "scipy_version": scipy.__version__,
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "mip_gap": (
            float(getattr(result, "mip_gap"))
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "source_quotas": source_quotas,
        "frozen_split_quotas": split_quotas,
        "state_target_per_policy": target,
    }
    return selected, solver_info


def build_outputs(
    universe: list[dict[str, Any]],
    selected_indices: list[int],
    out_pool: Path,
    out_reference: Path,
    manifest_path: Path,
    n_select: int,
    min_per_state: int,
    min_cross_policy: int,
    seed: int,
    solver_info: dict[str, Any],
) -> None:
    selected = sorted((universe[index] for index in selected_indices), key=lambda item: item["id"])
    pool_records = [item["pool"] for item in selected]
    reference_records = [
        {
            "id": item["id"],
            "source": item["pool"]["source"],
            "prompt": item["pool"]["prompt"],
            "response": item["pool"]["response"],
            "meta": item["pool"].get("meta", {}),
            "labels": item["labels"],
            "parse_ok": True,
            "frozen_split": item["frozen_split"],
        }
        for item in selected
    ]
    write_jsonl(out_pool, pool_records)
    write_jsonl(out_reference, reference_records)

    policy_counts = {
        policy: dict(
            Counter(item["labels"][policy] for item in selected)
        )
        for policy in POLICY_IDS
    }
    for counts in policy_counts.values():
        for state in LABEL_STATES:
            counts.setdefault(state, 0)
    source_counts = Counter(item["pool"]["source"] for item in selected)
    split_counts = Counter(item["frozen_split"] for item in selected)
    cross_item_ids = {
        item["id"]
        for item in selected
        if any(value == "satisfied" for value in item["labels"].values())
        and any(value == "violated" for value in item["labels"].values())
    }
    minimum_observed = min(
        policy_counts[policy][state] for policy in POLICY_IDS for state in LABEL_STATES
    )
    if minimum_observed < min_per_state:
        raise RuntimeError(
            f"selected set violates min_per_state: observed={minimum_observed}, "
            f"required={min_per_state}"
        )
    if len(cross_item_ids) < min_cross_policy:
        raise RuntimeError("selected set violates min_cross_policy")

    manifest = {
        "schema_version": "pccd.day3.diag100_manifest.v1",
        "git_revision": git_revision(),
        "purpose": (
            "Diagnostic-only balanced view; never replaces a frozen split and is never "
            "used for training or gate scoring."
        ),
        "selection_spec": {
            "n": n_select,
            "seed": seed,
            "universe": "five frozen deduplicated splits with frozen teacher labels",
            "minimum_per_policy_state": min_per_state,
            "minimum_cross_policy_items": min_cross_policy,
            "cross_policy_definition": (
                "at least one satisfied and at least one violated policy on the same item"
            ),
            "source_and_split_quota_rule": "largest-remainder proportional allocation",
            "objective": (
                "minimize total absolute deviation from 1/3 per state for each policy; "
                "stable SHA-256 ID tie breaker"
            ),
            "solver": solver_info,
        },
        "counts": {
            "items": len(selected),
            "sources": dict(sorted(source_counts.items())),
            "frozen_split_overlap": dict(sorted(split_counts.items())),
            "cross_policy_items": len(cross_item_ids),
            "minimum_observed_policy_state_count": minimum_observed,
            "per_policy": policy_counts,
        },
        "artifacts": {
            "pool": {"path": str(out_pool), "sha256": sha256_file(out_pool)},
            "reference_labels": {
                "path": str(out_reference),
                "sha256": sha256_file(out_reference),
            },
        },
        "selected_items": [
            {
                "id": item["id"],
                "source": item["pool"]["source"],
                "frozen_split": item["frozen_split"],
                "cross_policy_conflict": item["id"] in cross_item_ids,
            }
            for item in selected
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"diag100 selected: {len(selected)}")
    print(f"sources: {dict(sorted(source_counts.items()))}")
    print(f"frozen split overlap: {dict(sorted(split_counts.items()))}")
    print(f"cross-policy items: {len(cross_item_ids)}")
    print(f"minimum policy-state count: {minimum_observed}")
    for policy in POLICY_IDS:
        counts = policy_counts[policy]
        print(
            f"  {policy}: sat={counts['satisfied']:3d} vio={counts['violated']:3d} "
            f"na={counts['not_applicable']:3d}"
        )
    print(f"pool sha256: {sha256_file(out_pool)}")
    print(f"reference sha256: {sha256_file(out_reference)}")
    print(f"manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--out-pool", type=Path, required=True)
    parser.add_argument("--out-reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--min-per-state", type=int, default=20)
    parser.add_argument("--min-cross-policy", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing diagnostic outputs (off by default to preserve the first run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.min_per_state <= 0 or args.min_cross_policy < 0:
        raise SystemExit("n/minimum arguments must be positive")
    if args.min_per_state * len(LABEL_STATES) > args.n:
        raise SystemExit("min-per-state is infeasible for n")
    existing = [path for path in (args.out_pool, args.out_reference, args.manifest) if path.exists()]
    if existing and not args.force:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing diagnostic output(s): {paths}")
    universe = load_universe(args.pool_dir, args.label_dir)
    selected, solver_info = solve_selection(
        universe,
        args.n,
        args.min_per_state,
        args.min_cross_policy,
        args.time_limit,
        args.seed,
    )
    build_outputs(
        universe,
        selected,
        args.out_pool,
        args.out_reference,
        args.manifest,
        args.n,
        args.min_per_state,
        args.min_cross_policy,
        args.seed,
        solver_info,
    )


if __name__ == "__main__":
    main()
