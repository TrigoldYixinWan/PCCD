#!/usr/bin/env python3
"""Read-only provenance and support audit for AEGIS 2.0.

This script does not run a guard, score an outcome, or modify the dataset. It
audits the exact raw JSON files at the frozen Hugging Face revision below. The
six-way common taxonomy is diagnostic only: it tests whether even a favorable
aggregation could satisfy the pre-lock support gate. It is not a frozen map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_REPO = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
DATASET_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"

EXPECTED_FILES = {
    "train.json": {
        "sha256": "154fba82c71d9fa73abd2ca5588a198e693ddc816c83444df180a22f613e02f6",
        "kind": "base",
        "split": "train",
    },
    "validation.json": {
        "sha256": "a97200e226ad4f6ba6a639982f817f675909bc74513859df3e8fc9a92951dfcd",
        "kind": "base",
        "split": "validation",
    },
    "test.json": {
        "sha256": "b0a6d602260524866053cb34105194f074f2c2906e3691b68d43b9e6e9318f35",
        "kind": "base",
        "split": "test",
    },
    "refusals_train.json": {
        "sha256": "ff948d3696c9da94cf2523f5d4ae7f16cad7b3c0fd3cb46a331dad0ed717fbe2",
        "kind": "augmentation",
        "split": "train",
    },
    "refusals_validation.json": {
        "sha256": "aba81546da0108bc931ae7fb7662b687b42b0bb3e734cde34fbddfbe33cadfcf",
        "kind": "augmentation",
        "split": "validation",
    },
}

# Deliberately broad, diagnostic aggregation. A support failure therefore
# cannot be attributed to an overly narrow mapping.
DIAGNOSTIC_COMMON_MAP = {
    "violence_physical": {"Violence", "Threat"},
    "hate_identity": {"Hate/Identity Hate"},
    "sexual": {"Sexual", "Sexual (minor)"},
    "self_harm": {"Suicide and Self Harm"},
    "illegal_criminal": {
        "Criminal Planning/Confessions",
        "Guns and Illegal Weapons",
        "Controlled/Regulated Substances",
        "Illegal Activity",
        "Malware",
        "Fraud/Deception",
    },
    "privacy_pii": {"PII/Privacy"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is not a JSON array of objects")
    return rows


def parse_categories(value: Any) -> set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def label_source_table(rows: list[dict[str, Any]], prefix: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    source_key = f"{prefix}_label_source"
    label_key = f"{prefix}_label"
    for row in rows:
        source = row.get(source_key)
        label = row.get(label_key)
        counts[
            f"{source if source is not None else 'null'}:"
            f"{label if label is not None else 'null'}"
        ] += 1
    return dict(sorted(counts.items()))


def criterion_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_human_safe = sum(
        row.get("prompt_label_source") == "human"
        and row.get("prompt_label") == "safe"
        for row in rows
    )
    response_human_safe = sum(
        row.get("response_label_source") == "human"
        and row.get("response_label") == "safe"
        for row in rows
    )

    support: dict[str, Any] = {}
    for common_name, native_categories in DIAGNOSTIC_COMMON_MAP.items():
        prompt_positive = 0
        response_positive = 0
        for row in rows:
            categories = parse_categories(row.get("violated_categories"))
            mapped_positive = bool(categories & native_categories)
            if (
                mapped_positive
                and row.get("prompt_label_source") == "human"
                and row.get("prompt_label") == "unsafe"
            ):
                prompt_positive += 1
            if (
                mapped_positive
                and row.get("response_label_source") == "human"
                and row.get("response_label") == "unsafe"
            ):
                response_positive += 1
        support[common_name] = {
            "prompt_human_positive": prompt_positive,
            "prompt_human_conservative_negative": prompt_human_safe,
            "response_human_positive": response_positive,
            "response_human_conservative_negative": response_human_safe,
        }
    return support


def audit(data_dir: Path) -> dict[str, Any]:
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    file_records: dict[str, Any] = {}

    for filename, spec in EXPECTED_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {spec['sha256']}, "
                f"observed {actual_hash}"
            )
        rows = load_rows(path)
        rows_by_file[filename] = rows
        file_records[filename] = {
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
            "rows": len(rows),
            "kind": spec["kind"],
            "split": spec["split"],
        }

    base_by_split: dict[str, list[dict[str, Any]]] = {
        split: [
            row
            for filename, rows in rows_by_file.items()
            if EXPECTED_FILES[filename]["kind"] == "base"
            and EXPECTED_FILES[filename]["split"] == split
            for row in rows
        ]
        for split in ("train", "validation", "test")
    }
    base_rows = [row for rows in base_by_split.values() for row in rows]
    all_rows = [row for rows in rows_by_file.values() for row in rows]

    base_ids = [str(row.get("id")) for row in base_rows]
    all_ids = [str(row.get("id")) for row in all_rows]
    split_id_sets = {
        split: {str(row.get("id")) for row in rows}
        for split, rows in base_by_split.items()
    }
    cross_split_overlap = {
        "train_validation": len(split_id_sets["train"] & split_id_sets["validation"]),
        "train_test": len(split_id_sets["train"] & split_id_sets["test"]),
        "validation_test": len(split_id_sets["validation"] & split_id_sets["test"]),
    }

    prompt_human_safe_nonempty = 0
    prompt_human_safe_nonempty_excluding_caution = 0
    prompt_human_unsafe_empty = 0
    response_human_nonempty = 0
    response_human_unsafe = 0
    native_prompt_positive_test: Counter[str] = Counter()

    for row in base_rows:
        categories = parse_categories(row.get("violated_categories"))
        if (
            row.get("prompt_label_source") == "human"
            and row.get("prompt_label") == "safe"
            and categories
        ):
            prompt_human_safe_nonempty += 1
            if categories - {"Needs Caution"}:
                prompt_human_safe_nonempty_excluding_caution += 1
        if (
            row.get("prompt_label_source") == "human"
            and row.get("prompt_label") == "unsafe"
            and not categories
        ):
            prompt_human_unsafe_empty += 1
        if row.get("response_label_source") == "human":
            if categories:
                response_human_nonempty += 1
            if row.get("response_label") == "unsafe":
                response_human_unsafe += 1

    for row in base_by_split["test"]:
        if (
            row.get("prompt_label_source") == "human"
            and row.get("prompt_label") == "unsafe"
        ):
            native_prompt_positive_test.update(
                parse_categories(row.get("violated_categories"))
            )

    per_split = {}
    for split, rows in base_by_split.items():
        per_split[split] = {
            "base_rows": len(rows),
            "unique_ids": len({str(row.get("id")) for row in rows}),
            "prompt_label_source_x_label": label_source_table(rows, "prompt"),
            "response_label_source_x_label": label_source_table(rows, "response"),
            "diagnostic_common_support": criterion_support(rows),
        }

    response_human_safe = sum(
        row.get("response_label_source") == "human"
        and row.get("response_label") == "safe"
        for row in base_rows
    )

    return {
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "files": file_records,
        },
        "units": {
            "logical_rows_all_files": len(all_rows),
            "unique_ids_all_files": len(set(all_ids)),
            "base_rows": len(base_rows),
            "unique_base_ids": len(set(base_ids)),
            "duplicate_base_ids": len(base_ids) - len(set(base_ids)),
            "augmentation_rows": len(all_rows) - len(base_rows),
            "cross_base_split_id_overlap": cross_split_overlap,
        },
        "provenance_checks": {
            "base_prompt_label_source_all_human": all(
                row.get("prompt_label_source") == "human" for row in base_rows
            ),
            "human_response_safe": response_human_safe,
            "human_response_unsafe": response_human_unsafe,
            "human_response_with_nonempty_violated_categories": response_human_nonempty,
            "prompt_human_safe_with_nonempty_categories": prompt_human_safe_nonempty,
            "prompt_human_safe_with_non_caution_category": (
                prompt_human_safe_nonempty_excluding_caution
            ),
            "prompt_human_unsafe_with_empty_categories": prompt_human_unsafe_empty,
        },
        "per_split": per_split,
        "native_test_prompt_human_positive_support": dict(
            sorted(native_prompt_positive_test.items())
        ),
        "gate": {
            "required_response_human_positive_per_common_criterion": 100,
            "observed_response_human_positive_max": max(
                values["response_human_positive"]
                for split in per_split.values()
                for values in split["diagnostic_common_support"].values()
            ),
            "verdict": "FAIL",
            "reason": (
                "No base row has response_label_source=human and "
                "response_label=unsafe; response-side positive support is zero "
                "for every native or aggregated criterion."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the five exact raw AEGIS 2.0 JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON output path. The audit is also printed to stdout.",
    )
    args = parser.parse_args()

    result = audit(args.data_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
