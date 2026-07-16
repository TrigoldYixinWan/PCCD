#!/usr/bin/env python3
"""Prepare, merge, and finalize the frozen blinded human-audit worksheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


VALID_STATES = ("satisfied", "violated", "not_applicable")
DISPLAY_FIELDS = ("audit_id", "policy_id", "policy", "prompt", "response", "rubric")


def refuse_overwrite(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {', '.join(existing)}")


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
        raise ValueError(f"no audit rows in {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    refuse_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def formula_safe(value: object) -> str:
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def deterministic_order(rows: list[dict], role: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{role}:{row['audit_id']}".encode("utf-8")
        ).digest(),
    )


def export_csv(path: Path, rows: list[dict], *, label_field: str = "label") -> None:
    refuse_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*DISPLAY_FIELDS, label_field]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {field: formula_safe(row.get(field)) for field in DISPLAY_FIELDS}
            output[label_field] = ""
            writer.writerow(output)


def export_worksheets(blind: Path, out_a: Path, out_b: Path) -> dict:
    rows = load_jsonl(blind)
    required = {*DISPLAY_FIELDS, "annotator_1", "annotator_2", "adjudicated"}
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{blind}:{index}: missing fields {sorted(missing)}")
        if any(row[field] is not None for field in ("annotator_1", "annotator_2", "adjudicated")):
            raise ValueError("frozen blind packet already contains annotations")
    refuse_overwrite(out_a, out_b)
    export_csv(out_a, deterministic_order(rows, "A"))
    export_csv(out_b, deterministic_order(rows, "B"))
    return {
        "rows": len(rows),
        "annotator_a": str(out_a.resolve()),
        "annotator_b": str(out_b.resolve()),
    }


def read_label_csv(path: Path, field: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "audit_id" not in (reader.fieldnames or ()) or field not in (reader.fieldnames or ()):
            raise ValueError(f"{path}: requires audit_id and {field} columns")
        for line_number, row in enumerate(reader, start=2):
            audit_id = (row.get("audit_id") or "").strip()
            label = (row.get(field) or "").strip().lower()
            if not audit_id or audit_id in labels:
                raise ValueError(f"{path}:{line_number}: missing or duplicate audit_id")
            if label not in VALID_STATES:
                raise ValueError(f"{path}:{line_number}: invalid {field}={label!r}")
            labels[audit_id] = label
    if not labels:
        raise ValueError(f"{path}: no labels")
    return labels


def merge_annotations(
    blind: Path,
    annotator_a: Path,
    annotator_b: Path,
    merged_out: Path,
    disagreements_out: Path,
) -> dict:
    rows = load_jsonl(blind)
    ids = {row["audit_id"] for row in rows}
    labels_a = read_label_csv(annotator_a, "label")
    labels_b = read_label_csv(annotator_b, "label")
    if set(labels_a) != ids or set(labels_b) != ids:
        raise ValueError("annotator worksheet ID set differs from frozen blind packet")
    merged: list[dict] = []
    disagreements: list[dict] = []
    for row in rows:
        audit_id = row["audit_id"]
        label_a, label_b = labels_a[audit_id], labels_b[audit_id]
        output = dict(row)
        output["annotator_1"] = label_a
        output["annotator_2"] = label_b
        output["adjudicated"] = label_a if label_a == label_b else None
        merged.append(output)
        if label_a != label_b:
            disagreement = {field: row.get(field) for field in DISPLAY_FIELDS}
            disagreement["annotator_1"] = label_a
            disagreement["annotator_2"] = label_b
            disagreement["adjudicated"] = None
            disagreements.append(disagreement)
    refuse_overwrite(merged_out, disagreements_out)
    write_jsonl(merged_out, merged)
    disagreements_out.parent.mkdir(parents=True, exist_ok=True)
    fields = [*DISPLAY_FIELDS, "annotator_1", "annotator_2", "adjudicated"]
    with disagreements_out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in deterministic_order(disagreements, "ADJUDICATOR"):
            writer.writerow(
                {
                    field: (
                        formula_safe(row.get(field))
                        if field in DISPLAY_FIELDS
                        else (row.get(field) or "")
                    )
                    for field in fields
                }
            )
    return {
        "rows": len(rows),
        "agreements": len(rows) - len(disagreements),
        "disagreements": len(disagreements),
        "merged": str(merged_out.resolve()),
        "adjudication": str(disagreements_out.resolve()),
    }


def finalize_annotations(merged: Path, adjudication: Path, out: Path) -> dict:
    rows = load_jsonl(merged)
    disagreements = {
        row["audit_id"]
        for row in rows
        if row.get("annotator_1") != row.get("annotator_2")
    }
    adjudicated = read_label_csv(adjudication, "adjudicated") if disagreements else {}
    if set(adjudicated) != disagreements:
        raise ValueError("adjudication ID set must equal the A/B disagreement set")
    finalized: list[dict] = []
    for row in rows:
        output = dict(row)
        label_a, label_b = output.get("annotator_1"), output.get("annotator_2")
        if label_a not in VALID_STATES or label_b not in VALID_STATES:
            raise ValueError(f"{output['audit_id']}: invalid annotator label")
        if label_a == label_b:
            output["adjudicated"] = label_a
        else:
            output["adjudicated"] = adjudicated[output["audit_id"]]
        finalized.append(output)
    write_jsonl(out, finalized)
    return {
        "rows": len(rows),
        "disagreements": len(disagreements),
        "completed": str(out.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--blind", type=Path, required=True)
    export.add_argument("--out_a", type=Path, required=True)
    export.add_argument("--out_b", type=Path, required=True)

    merge = commands.add_parser("merge")
    merge.add_argument("--blind", type=Path, required=True)
    merge.add_argument("--annotator_a", type=Path, required=True)
    merge.add_argument("--annotator_b", type=Path, required=True)
    merge.add_argument("--merged_out", type=Path, required=True)
    merge.add_argument("--disagreements_out", type=Path, required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--merged", type=Path, required=True)
    finalize.add_argument("--adjudication", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        result = export_worksheets(args.blind, args.out_a, args.out_b)
    elif args.command == "merge":
        result = merge_annotations(
            args.blind,
            args.annotator_a,
            args.annotator_b,
            args.merged_out,
            args.disagreements_out,
        )
    else:
        result = finalize_annotations(args.merged, args.adjudication, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
