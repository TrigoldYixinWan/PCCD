#!/usr/bin/env python3
"""Freeze the inference-free BeaverTails 330k_test support manifest.

The 330k files contain repeated crowdworker annotation rows for the same
prompt/response pair.  This script aggregates exact pairs by strict majority,
keeps category ties as missing for that category, and never loads a model.
Human references and blind item text are written to separate artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DATASET = "PKU-Alignment/BeaverTails"
REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
EXPECTED_INPUT_SHA256 = (
    "1c726c662b85bf4761913041fe9e098f80e5206344864597dbde86bc9a1bfc96"
)
SUPPORT_THRESHOLD = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pair_id(prompt: str, response: str) -> str:
    return sha256_text(f"{prompt}\0{response}")


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def strict_majority(positive_votes: int, total_votes: int) -> bool | None:
    doubled = 2 * positive_votes
    if doubled == total_votes:
        return None
    return doubled > total_votes


def read_aggregates(path: Path) -> tuple[dict[str, dict[str, Any]], list[str], int]:
    aggregates: dict[str, dict[str, Any]] = {}
    categories: list[str] | None = None
    annotation_rows = 0

    with lzma.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            annotation_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

            prompt = row.get("prompt")
            response = row.get("response")
            category = row.get("category")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(f"{path}:{line_number}: missing prompt/response text")
            if not isinstance(category, dict) or not category:
                raise ValueError(f"{path}:{line_number}: missing category flags")
            row_categories = sorted(category)
            if categories is None:
                categories = row_categories
            elif row_categories != categories:
                raise ValueError(f"{path}:{line_number}: category schema changed")

            item_id = pair_id(prompt, response)
            record = aggregates.setdefault(
                item_id,
                {
                    "prompt": prompt,
                    "response": response,
                    "annotation_count": 0,
                    "safe_votes": 0,
                    "category_votes": Counter(),
                },
            )
            if record["prompt"] != prompt or record["response"] != response:
                raise AssertionError("SHA-256 collision between distinct pairs")
            record["annotation_count"] += 1
            record["safe_votes"] += int(bool(row.get("is_safe")))
            for name in categories:
                record["category_votes"][name] += int(bool(category[name]))

    if categories is None:
        raise ValueError(f"{path}: empty file")
    return aggregates, categories, annotation_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_hash = sha256_file(args.input)
    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "330k_test hash mismatch: "
            f"expected {EXPECTED_INPUT_SHA256}, observed {input_hash}"
        )

    aggregates, categories, annotation_rows = read_aggregates(args.input)
    annotation_histogram = Counter(
        record["annotation_count"] for record in aggregates.values()
    )

    blind_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    safe_ties = 0
    category_ties = Counter()
    support = {
        category: {
            "human_positive": 0,
            "benign_negative": 0,
            "all_nonpositive": 0,
            "missing_tie": 0,
        }
        for category in categories
    }

    for item_id in sorted(aggregates):
        record = aggregates[item_id]
        n = record["annotation_count"]
        safe_label = strict_majority(record["safe_votes"], n)
        if safe_label is None:
            safe_ties += 1

        labels: dict[str, bool | None] = {}
        votes: dict[str, int] = {}
        for category in categories:
            positive_votes = int(record["category_votes"][category])
            label = strict_majority(positive_votes, n)
            labels[category] = label
            votes[category] = positive_votes
            if label is None:
                category_ties[category] += 1
                support[category]["missing_tie"] += 1
            elif label:
                support[category]["human_positive"] += 1
            else:
                support[category]["all_nonpositive"] += 1
                if safe_label is True:
                    support[category]["benign_negative"] += 1

        blind_rows.append(
            {
                "id": f"bt330test_{item_id[:24]}",
                "pair_sha256": item_id,
                "prompt": record["prompt"],
                "response": record["response"],
            }
        )
        reference_rows.append(
            {
                "id": f"bt330test_{item_id[:24]}",
                "pair_sha256": item_id,
                "annotation_count": n,
                "human_is_safe": safe_label,
                "human_safe_votes": record["safe_votes"],
                "human_category": labels,
                "human_category_positive_votes": votes,
            }
        )

    primary = [
        category
        for category in categories
        if support[category]["human_positive"] >= SUPPORT_THRESHOLD
        and support[category]["benign_negative"] >= SUPPORT_THRESHOLD
    ]
    rare = [category for category in categories if category not in primary]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "blind_items": args.out_dir / "beavertails_330k_test_blind_items.jsonl",
        "human_reference": args.out_dir
        / "beavertails_330k_test_human_reference.jsonl",
    }
    support_path = args.out_dir / "beavertails_330k_test_support.json"
    for path in [*paths.values(), support_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")

    paths["blind_items"].write_bytes(canonical_jsonl(blind_rows))
    paths["human_reference"].write_bytes(canonical_jsonl(reference_rows))

    summary = {
        "schema": "pccd.labelsource.beavertails_support.v1",
        "status": "FROZEN_PRELOCK_METADATA_ONLY",
        "dataset": DATASET,
        "revision": REVISION,
        "split": "330k_test",
        "input_sha256": input_hash,
        "unit": "unique exact prompt-response pair",
        "aggregation": {
            "rule": "strict majority independently for is_safe and each category",
            "category_tie": "missing for that category only",
            "is_safe_tie": "not eligible as a benign negative",
            "benign_negative": "majority is_safe=true and category majority=false",
        },
        "threshold": {
            "human_positive_min": SUPPORT_THRESHOLD,
            "benign_negative_min": SUPPORT_THRESHOLD,
            "primary_category_min": 6,
        },
        "counts": {
            "annotation_rows": annotation_rows,
            "unique_pairs": len(aggregates),
            "majority_safe": sum(
                row["human_is_safe"] is True for row in reference_rows
            ),
            "majority_unsafe": sum(
                row["human_is_safe"] is False for row in reference_rows
            ),
            "is_safe_ties": safe_ties,
            "annotation_count_histogram": {
                str(key): annotation_histogram[key]
                for key in sorted(annotation_histogram)
            },
            "category_ties": {
                category: category_ties[category] for category in categories
            },
        },
        "category_support": support,
        "primary_categories": primary,
        "rare_categories": rare,
        "gate": {
            "required_primary_categories": 6,
            "observed_primary_categories": len(primary),
            "pass": len(primary) >= 6,
        },
        "artifacts": {
            name: {
                "filename": path.name,
                "rows": len(blind_rows),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    support_path.write_bytes(
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not summary["gate"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
