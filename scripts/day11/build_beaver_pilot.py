#!/usr/bin/env python3
"""Build the frozen BeaverTails teacher-compatibility pilot.

The builder reads annotation metadata only. It never loads Qwen or any guard.
Human references are written separately from the blind teacher input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO


REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
SEED = "20260716"
POSITIVE_TARGET = 120
SAFE_TARGET = 400

EXPECTED_SHA256 = {
    "30k_train": "2e24b85dd599224eeb7379da297223bad50e11fea9fee6a7fdb4bd26aedec315",
    "30k_test": "0ce619bf4c421c12e578445b1fb8d1aedf4bf05ca3eb985f4ddf498a784769ff",
    "330k_train": "12f04c9200000929ebbc616c29cb53331f6df36e7341d73e4a12cffac0eb1c08",
    "330k_test": "1c726c662b85bf4761913041fe9e098f80e5206344864597dbde86bc9a1bfc96",
}

COMMON_MAP = {
    "violence_physical": {
        "violence,aiding_and_abetting,incitement",
        "animal_abuse",
        "child_abuse",
    },
    "hate_identity": {
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
    },
    "sexual": {"sexually_explicit,adult_content"},
    "self_harm": {"self_harm"},
    "illegal_criminal": {
        "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft",
        "terrorism,organized_crime",
    },
    "privacy_pii": {"privacy_violation"},
}
CRITERIA = tuple(COMMON_MAP)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pair_sha(row: dict[str, Any]) -> str:
    return sha256_text(f"{row['prompt']}\0{row['response']}")


def open_jsonl(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_rows(path: Path):
    with open_jsonl(path) as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row.get("category"), dict):
                raise ValueError(f"{path}:{line_number}: missing category dict")
            yield row


def verify_file(name: str, path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    expected = EXPECTED_SHA256[name]
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: expected {expected}, observed {actual}")
    return actual


def common_vote(row: dict[str, Any], criterion: str) -> bool:
    return any(bool(row["category"].get(native)) for native in COMMON_MAP[criterion])


def stable_order(stratum: str, pair: str) -> str:
    return sha256_text(f"{SEED}|{stratum}|{pair}")


def jsonl_hash(rows: list[dict[str, Any]]) -> str:
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return sha256_text(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--30k-train", type=Path, required=True)
    parser.add_argument("--30k-test", type=Path, required=True)
    parser.add_argument("--330k-train", type=Path, required=True)
    parser.add_argument("--330k-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "30k_train": args.__dict__["30k_train"],
        "30k_test": args.__dict__["30k_test"],
        "330k_train": args.__dict__["330k_train"],
        "330k_test": args.__dict__["330k_test"],
    }
    input_hashes = {name: verify_file(name, path) for name, path in paths.items()}

    excluded_pairs: set[str] = set()
    excluded_rows = {}
    for name in ("30k_train", "30k_test", "330k_test"):
        count = 0
        for row in iter_rows(paths[name]):
            excluded_pairs.add(pair_sha(row))
            count += 1
        excluded_rows[name] = count

    aggregates: dict[str, dict[str, Any]] = {}
    candidate_annotation_rows = 0
    excluded_annotation_rows = 0
    for row in iter_rows(paths["330k_train"]):
        candidate_annotation_rows += 1
        key = pair_sha(row)
        if key in excluded_pairs:
            excluded_annotation_rows += 1
            continue
        record = aggregates.setdefault(
            key,
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "n": 0,
                "safe_votes": 0,
                "criterion_votes": {criterion: 0 for criterion in CRITERIA},
            },
        )
        if record["prompt"] != row["prompt"] or record["response"] != row["response"]:
            raise AssertionError("SHA-256 pair collision")
        record["n"] += 1
        record["safe_votes"] += int(bool(row["is_safe"]))
        for criterion in CRITERIA:
            record["criterion_votes"][criterion] += int(common_vote(row, criterion))

    eligible: dict[str, dict[str, Any]] = {}
    tie_counts = {"overall_safe": 0, **{criterion: 0 for criterion in CRITERIA}}
    for key, record in aggregates.items():
        n = record["n"]
        if 2 * record["safe_votes"] == n:
            tie_counts["overall_safe"] += 1
            continue
        labels = {}
        tied = False
        for criterion in CRITERIA:
            votes = record["criterion_votes"][criterion]
            if 2 * votes == n:
                tie_counts[criterion] += 1
                tied = True
            labels[criterion] = votes * 2 > n
        if tied:
            continue
        record["safe"] = record["safe_votes"] * 2 > n
        record["labels"] = labels
        if record["safe"] and any(labels.values()):
            raise ValueError("majority-safe pair has a majority-positive common criterion")
        eligible[key] = record

    selected: set[str] = set()
    for criterion in CRITERIA:
        candidates = sorted(
            (key for key, row in eligible.items() if row["labels"][criterion]),
            key=lambda key: stable_order(criterion, key),
        )
        for key in candidates:
            if sum(eligible[item]["labels"][criterion] for item in selected) >= POSITIVE_TARGET:
                break
            selected.add(key)
        support = sum(eligible[item]["labels"][criterion] for item in selected)
        if support < POSITIVE_TARGET:
            raise ValueError(f"insufficient {criterion} support: {support}")

    safe_candidates = sorted(
        (key for key, row in eligible.items() if row["safe"]),
        key=lambda key: stable_order("safe", key),
    )
    if len(safe_candidates) < SAFE_TARGET:
        raise ValueError(f"insufficient safe support: {len(safe_candidates)}")
    selected.update(safe_candidates[:SAFE_TARGET])

    ordered = sorted(selected, key=lambda key: stable_order("final", key))
    items = []
    references = []
    for key in ordered:
        row = eligible[key]
        item_id = f"btp_{key[:24]}"
        items.append(
            {
                "id": item_id,
                "source": "beavertails_330k_train_exploratory_pilot",
                "prompt": row["prompt"],
                "response": row["response"],
                "meta": {"pair_sha256": key, "annotation_count": row["n"]},
            }
        )
        references.append(
            {
                "id": item_id,
                "pair_sha256": key,
                "human_safe": row["safe"],
                "human_safe_votes": row["safe_votes"],
                "annotation_count": row["n"],
                "human_labels": row["labels"],
                "human_positive_votes": row["criterion_votes"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    item_path = args.out_dir / "items.jsonl"
    reference_path = args.out_dir / "reference.jsonl"
    if item_path.exists() or reference_path.exists():
        raise FileExistsError("pilot output already exists; refusing to overwrite")

    item_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
    reference_text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in references
    )
    item_path.write_bytes(item_text.encode("utf-8"))
    reference_path.write_bytes(reference_text.encode("utf-8"))

    realized = {
        "items": len(items),
        "safe_negatives": sum(row["human_safe"] for row in references),
        "positive_support": {
            criterion: sum(row["human_labels"][criterion] for row in references)
            for criterion in CRITERIA
        },
        "non_unanimous_safe": sum(
            0 < row["human_safe_votes"] < row["annotation_count"] for row in references
        ),
        "non_unanimous_by_criterion": {
            criterion: sum(
                0 < row["human_positive_votes"][criterion] < row["annotation_count"]
                for row in references
            )
            for criterion in CRITERIA
        },
    }
    manifest = {
        "schema": "pccd.beaver_teacher_pilot.v1",
        "status": "FROZEN_BEFORE_QWEN_OUTPUT",
        "dataset": "PKU-Alignment/BeaverTails",
        "revision": REVISION,
        "input_sha256": input_hashes,
        "sampling": {
            "seed": SEED,
            "positive_target_per_criterion": POSITIVE_TARGET,
            "safe_target": SAFE_TARGET,
            "common_map": {key: sorted(value) for key, value in COMMON_MAP.items()},
            "excluded_pair_universe_size": len(excluded_pairs),
            "excluded_file_rows": excluded_rows,
            "330k_train_annotation_rows": candidate_annotation_rows,
            "330k_train_annotation_rows_excluded_by_pair": excluded_annotation_rows,
            "eligible_unique_pairs": len(eligible),
            "tie_counts": tie_counts,
        },
        "realized": realized,
        "artifacts": {
            "items": {"path": item_path.name, "sha256": sha256_file(item_path)},
            "reference": {
                "path": reference_path.name,
                "sha256": sha256_file(reference_path),
            },
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
