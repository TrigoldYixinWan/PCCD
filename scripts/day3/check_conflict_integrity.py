#!/usr/bin/env python3
"""Strict integrity checks for the Day-3 conflict teacher labels.

The check deliberately reads every pool split, not just ``conflict.jsonl``:
the conflict split is only valid if it is disjoint from all Day-2 splits.  Any
malformed JSONL row or failed invariant is reported and causes a non-zero exit.

Example (after ``source scripts/setup/env.sh``)::

    python scripts/day3/check_conflict_integrity.py \
      --pool-dir "$PCCD_OUT/pool" --label-dir "$PCCD_OUT/labels" \
      --expected 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


POLICY_IDS = ("H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2")
LABEL_STATES = {"satisfied", "violated", "not_applicable"}
POOL_COUNTS = {
    "train": 8000,
    "calib": 1000,
    "test": 1000,
    "audit": 400,
    # ``conflict`` is set from --expected.
}
LABEL_FILENAMES = {
    "merged": "conflict.jsonl",
    "shardA": "conflict.shardA.jsonl",
    "shardB": "conflict.shardB.jsonl",
}
COPIED_FIELDS = ("id", "source", "prompt", "response", "meta")


class Checks:
    """Collect failures so one invocation exposes all actionable problems."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path, checks: Checks) -> list[dict[str, Any]]:
    """Read a JSONL file strictly, retaining valid object records for checks."""
    records: list[dict[str, Any]] = []
    if not path.is_file():
        checks.fail(f"missing file: {path}")
        return records

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    checks.fail(f"{path}:{line_no}: blank JSONL row")
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    checks.fail(f"{path}:{line_no}: invalid JSON: {exc}")
                    continue
                if not isinstance(value, dict):
                    checks.fail(f"{path}:{line_no}: JSON row is not an object")
                    continue
                records.append(value)
    except UnicodeDecodeError as exc:
        checks.fail(f"{path}: not valid UTF-8: {exc}")
    except OSError as exc:
        checks.fail(f"cannot read {path}: {exc}")
    return records


def record_ids(records: list[dict[str, Any]], path: Path, checks: Checks) -> list[str]:
    ids: list[str] = []
    for row_no, record in enumerate(records, 1):
        item_id = record.get("id")
        if not isinstance(item_id, str) or not item_id:
            checks.fail(f"{path}: record {row_no} has missing/non-string/empty id")
            continue
        ids.append(item_id)
    duplicates = sorted(item_id for item_id, n in Counter(ids).items() if n > 1)
    checks.require(
        not duplicates,
        f"{path}: duplicate ids ({len(duplicates)}): {preview(duplicates)}",
    )
    return ids


def preview(values: Any, limit: int = 8) -> str:
    ordered = list(values)
    shown = ", ".join(repr(v) for v in ordered[:limit])
    if len(ordered) > limit:
        shown += f", ... (+{len(ordered) - limit})"
    return shown or "<none>"


def validate_pool_record(record: dict[str, Any], path: Path, row_no: int,
                         checks: Checks) -> None:
    for field in COPIED_FIELDS:
        if field not in record:
            checks.fail(f"{path}: record {row_no} missing pool field {field!r}")
    for field in ("source", "prompt", "response"):
        if field in record and not isinstance(record[field], str):
            checks.fail(f"{path}: record {row_no} field {field!r} is not a string")
    if "meta" in record and not isinstance(record["meta"], dict):
        checks.fail(f"{path}: record {row_no} field 'meta' is not an object")


def validate_label_record(record: dict[str, Any], path: Path, row_no: int,
                          checks: Checks) -> None:
    if record.get("parse_ok") is not True:
        checks.fail(f"{path}: record {row_no} id={record.get('id')!r} parse_ok is not true")

    labels = record.get("labels")
    if not isinstance(labels, dict):
        checks.fail(f"{path}: record {row_no} id={record.get('id')!r} labels is not an object")
        return
    actual_keys = set(labels)
    expected_keys = set(POLICY_IDS)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        checks.fail(
            f"{path}: record {row_no} id={record.get('id')!r} label keys differ; "
            f"missing={missing}, extra={extra}"
        )
    for policy_id in POLICY_IDS:
        value = labels.get(policy_id)
        if policy_id in labels and (not isinstance(value, str) or value not in LABEL_STATES):
            checks.fail(
                f"{path}: record {row_no} id={record.get('id')!r} "
                f"{policy_id} has invalid state {value!r}"
            )


def require_id_set(actual_ids: list[str], expected_ids: list[str], description: str,
                   checks: Checks) -> None:
    actual, expected = set(actual_ids), set(expected_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    checks.require(
        not missing and not extra,
        f"{description}: id-set mismatch; missing={len(missing)} "
        f"[{preview(missing)}], extra={len(extra)} [{preview(extra)}]",
    )


def require_id_sequence(actual_ids: list[str], expected_ids: list[str], description: str,
                        checks: Checks) -> None:
    """Require order as well as membership and show the first divergent slot."""
    if actual_ids == expected_ids:
        return
    common = min(len(actual_ids), len(expected_ids))
    mismatch = next((i for i in range(common) if actual_ids[i] != expected_ids[i]), common)
    actual = actual_ids[mismatch] if mismatch < len(actual_ids) else "<end>"
    expected = expected_ids[mismatch] if mismatch < len(expected_ids) else "<end>"
    checks.fail(
        f"{description}: id order differs first at zero-based position {mismatch}; "
        f"actual={actual!r}, expected={expected!r}"
    )


def validate_copied_fields(label_records: list[dict[str, Any]],
                           pool_by_id: dict[str, dict[str, Any]], path: Path,
                           checks: Checks) -> None:
    for row_no, label in enumerate(label_records, 1):
        item_id = label.get("id")
        pool = pool_by_id.get(item_id) if isinstance(item_id, str) else None
        if pool is None:
            # The id-set check reports the missing/extra relationship.
            continue
        for field in COPIED_FIELDS:
            if field not in label:
                checks.fail(f"{path}: record {row_no} id={item_id!r} missing field {field!r}")
            elif label[field] != pool.get(field):
                checks.fail(
                    f"{path}: record {row_no} id={item_id!r} field {field!r} "
                    "does not exactly match conflict pool"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strict Day-3 conflict split and teacher-label integrity check."
    )
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=400,
                        help="expected conflict size (default: 400)")
    args = parser.parse_args(argv)
    if args.expected <= 0:
        parser.error("--expected must be positive")

    checks = Checks()
    pool_counts = dict(POOL_COUNTS, conflict=args.expected)
    pool_records: dict[str, list[dict[str, Any]]] = {}
    pool_ids: dict[str, list[str]] = {}
    related_paths: list[Path] = []

    print("== Pool splits ==")
    for split, expected_count in pool_counts.items():
        path = args.pool_dir / f"{split}.jsonl"
        related_paths.append(path)
        records = read_jsonl(path, checks)
        pool_records[split] = records
        checks.require(
            len(records) == expected_count,
            f"{path}: expected {expected_count} valid JSON object rows, got {len(records)}",
        )
        for row_no, record in enumerate(records, 1):
            validate_pool_record(record, path, row_no, checks)
        ids = record_ids(records, path, checks)
        pool_ids[split] = ids
        print(f"{split:8s} valid_rows={len(records):5d} unique_ids={len(set(ids)):5d} "
              f"expected={expected_count:5d}")

    split_names = list(pool_counts)
    print("\n== Cross-split ID isolation ==")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = sorted(set(pool_ids[left]) & set(pool_ids[right]))
            checks.require(
                not overlap,
                f"pool splits {left}/{right}: {len(overlap)} overlapping ids: {preview(overlap)}",
            )
            print(f"{left:8s} vs {right:8s} overlap={len(overlap)}")

    conflict_path = args.pool_dir / "conflict.jsonl"
    conflict_records = pool_records["conflict"]
    conflict_ids = pool_ids["conflict"]
    pool_by_id = {
        record["id"]: record
        for record in conflict_records
        if isinstance(record.get("id"), str) and record.get("id")
    }

    print("\n== Conflict composition ==")
    source_counts = Counter(
        record.get("source") if isinstance(record.get("source"), str) else "<invalid>"
        for record in conflict_records
    )
    print("sources=" + json.dumps(dict(sorted(source_counts.items())), sort_keys=True))
    pku_safe = Counter()
    for row_no, record in enumerate(conflict_records, 1):
        if record.get("source") != "pku_saferlhf":
            continue
        meta = record.get("meta")
        safety = meta.get("is_safe") if isinstance(meta, dict) else None
        if safety is True:
            pku_safe["safe"] += 1
        elif safety is False:
            pku_safe["unsafe"] += 1
        else:
            pku_safe["invalid_or_missing"] += 1
            checks.fail(
                f"{conflict_path}: PKU record {row_no} id={record.get('id')!r} "
                "meta.is_safe is not a boolean"
            )
    print("pku_safe_unsafe=" + json.dumps({
        "safe": pku_safe["safe"],
        "unsafe": pku_safe["unsafe"],
        "invalid_or_missing": pku_safe["invalid_or_missing"],
    }, sort_keys=True))

    print("\n== Conflict labels ==")
    label_records: dict[str, list[dict[str, Any]]] = {}
    label_ids: dict[str, list[str]] = {}
    expected_label_counts = {
        "merged": args.expected,
        "shardA": (args.expected + 1) // 2,
        "shardB": args.expected // 2,
    }
    for kind, filename in LABEL_FILENAMES.items():
        path = args.label_dir / filename
        related_paths.append(path)
        records = read_jsonl(path, checks)
        label_records[kind] = records
        checks.require(
            len(records) == expected_label_counts[kind],
            f"{path}: expected {expected_label_counts[kind]} valid JSON object rows, "
            f"got {len(records)}",
        )
        ids = record_ids(records, path, checks)
        label_ids[kind] = ids
        for row_no, record in enumerate(records, 1):
            validate_label_record(record, path, row_no, checks)
        validate_copied_fields(records, pool_by_id, path, checks)
        print(f"{kind:8s} valid_rows={len(records):4d} unique_ids={len(set(ids)):4d} "
              f"parse_ok={sum(r.get('parse_ok') is True for r in records):4d}")

    require_id_set(label_ids["merged"], conflict_ids, "merged labels vs conflict pool", checks)
    expected_a = conflict_ids[0::2]
    expected_b = conflict_ids[1::2]
    require_id_set(label_ids["shardA"], expected_a, "shardA vs even conflict-pool rows", checks)
    require_id_set(label_ids["shardB"], expected_b, "shardB vs odd conflict-pool rows", checks)
    require_id_sequence(
        label_ids["shardA"], expected_a, "shardA vs even conflict-pool rows", checks
    )
    require_id_sequence(
        label_ids["shardB"], expected_b, "shardB vs odd conflict-pool rows", checks
    )
    shard_overlap = sorted(set(label_ids["shardA"]) & set(label_ids["shardB"]))
    checks.require(
        not shard_overlap,
        f"conflict shards overlap on {len(shard_overlap)} ids: {preview(shard_overlap)}",
    )
    require_id_set(
        label_ids["shardA"] + label_ids["shardB"],
        conflict_ids,
        "union(shardA, shardB) vs conflict pool",
        checks,
    )
    expected_merged_records = label_records["shardA"] + label_records["shardB"]
    checks.require(
        label_records["merged"] == expected_merged_records,
        "merged conflict labels do not exactly equal shardA rows followed by shardB rows",
    )

    print("\n== SHA-256 ==")
    for path in related_paths:
        if path.is_file():
            try:
                print(f"{sha256_file(path)}  {path}")
            except OSError as exc:
                checks.fail(f"cannot hash {path}: {exc}")
        else:
            print(f"<missing>  {path}")

    if checks.errors:
        print(f"\nFAIL: {len(checks.errors)} integrity error(s)", file=sys.stderr)
        for index, error in enumerate(checks.errors, 1):
            print(f"  {index:03d}. {error}", file=sys.stderr)
        return 1

    print("\nPASS: conflict pool/labels are complete, isolated, schema-valid, and traceable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
