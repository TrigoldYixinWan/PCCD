#!/usr/bin/env python3
"""Join the frozen human/LLM labels and formal guard scores by exact item ID.

This is a lossless join. Malformed annotations, non-finite guard cells, native
scores, and no-mapping cells remain explicit; nothing is imputed or repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BLIND_SHA256 = "602de7160b536ae8567bda8a14ae6394a7413710c15b22157384c4d503a0c142"
HUMAN_SHA256 = "6c64b844c97ef123fe9395e3a5f150e1c7bb0b7eba04d342e6b3669d9b2a75e9"
TAXONOMY_SHA256 = "443f837bcc265ddafc35dff60edc3a49eadeae8e11e91e2422b7f484576bb8ed"
QWEN_SCHEMA_SHA256 = "3620d9c48a71ead307da7ec3a3c31d454051f2b4a83a8ef874589113a1c6e9aa"
GUARDS = {
    "llama_guard_3_8b": {
        "repo": "meta-llama/Llama-Guard-3-8B",
        "revision": "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
        "interfaces": ["full_s1_s14"],
        "family": "llama_guard_3",
    },
    "shieldgemma_2b": {
        "repo": "google/shieldgemma-2b",
        "revision": "d1dffc9c8c9237a90aab09c61383791e718ef9e8",
        "interfaces": [
            "dangerous_content",
            "harassment",
            "hate_speech",
            "sexually_explicit_information",
        ],
        "family": "shieldgemma",
    },
    "shieldgemma_9b": {
        "repo": "google/shieldgemma-9b",
        "revision": "b8b636016df4540721a098c7aab91c97ec6ee508",
        "interfaces": [
            "dangerous_content",
            "harassment",
            "hate_speech",
            "sexually_explicit_information",
        ],
        "family": "shieldgemma",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {observed} != {expected}")


def load_jsonl_by_id(path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            item_id = str(row["id"])
            if item_id in rows:
                raise ValueError(f"duplicate {label} id at line {line_number}: {item_id}")
            rows[item_id] = row
    return rows


def load_guard_scores(
    path: Path, guard_id: str
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    spec = GUARDS[guard_id]
    meta_path = Path(str(path) + ".meta.json")
    meta = json.loads(meta_path.read_text("utf-8"))
    if meta.get("status") != "FORMAL_COMPLETE" or meta.get("guard_id") != guard_id:
        raise ValueError(f"{guard_id} formal manifest is incomplete")
    if meta.get("repo") != spec["repo"] or meta.get("revision") != spec["revision"]:
        raise ValueError(f"{guard_id} identity differs from the locked registry")
    if meta.get("output_sha256") != sha256_file(path):
        raise ValueError(f"{guard_id} formal output hash mismatch")
    expected_rows = 11088 * len(spec["interfaces"])
    if meta.get("rows") != expected_rows:
        raise ValueError(f"{guard_id} manifest row count mismatch")

    scores: dict[str, dict[str, dict[str, Any]]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row_count += 1
            row = json.loads(line)
            if row.get("guard_id") != guard_id:
                raise ValueError(f"{guard_id} row identity mismatch at {line_number}")
            interface = str(row["interface"])
            if interface not in spec["interfaces"]:
                raise ValueError(f"{guard_id} unexpected interface: {interface}")
            item_scores = scores.setdefault(str(row["id"]), {})
            if interface in item_scores:
                raise ValueError(f"{guard_id} duplicate item/interface at {line_number}")
            probability = row.get("unsafe_probability")
            if probability is not None:
                probability = float(probability)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{guard_id} invalid retained probability")
            item_scores[interface] = {
                "unsafe_probability": probability,
                "score_ok": row.get("score_ok") is True,
                "error": row.get("error"),
                "input_tokens": row.get("input_tokens"),
            }
    if row_count != expected_rows or len(scores) != 11088:
        raise ValueError(f"{guard_id} score cardinality mismatch")
    return scores, meta


def mapped_probability(
    guard_id: str,
    native: dict[str, dict[str, Any]],
    mapping_cell: dict[str, Any],
) -> dict[str, Any]:
    if guard_id == "llama_guard_3_8b":
        targets = list(mapping_cell["target_ids"])
        cell = native["full_s1_s14"]
        return {
            "unsafe_probability": cell["unsafe_probability"],
            "score_ok": cell["score_ok"],
            "error": cell["error"],
            "target_ids": targets,
            "relation": mapping_cell["relation"],
            "aggregation": "omnibus_full_s1_s14_probability",
        }
    targets = list(mapping_cell["target_ids"])
    if not targets:
        return {
            "unsafe_probability": None,
            "score_ok": False,
            "error": "no_defensible_taxonomy_mapping",
            "target_ids": [],
            "relation": mapping_cell["relation"],
            "aggregation": "non_evaluable",
        }
    target_cells = [native[target] for target in targets]
    if any(cell["score_ok"] is not True for cell in target_cells):
        return {
            "unsafe_probability": None,
            "score_ok": False,
            "error": "mapped_native_score_non_evaluable",
            "target_ids": targets,
            "relation": mapping_cell["relation"],
            "aggregation": "max_union",
        }
    return {
        "unsafe_probability": max(float(cell["unsafe_probability"]) for cell in target_cells),
        "score_ok": True,
        "error": None,
        "target_ids": targets,
        "relation": mapping_cell["relation"],
        "aggregation": "max_union",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--qwen-schema", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--llama-guard", type=Path, required=True)
    parser.add_argument("--shieldgemma-2b", type=Path, required=True)
    parser.add_argument("--shieldgemma-9b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or Path(str(args.out) + ".meta.json").exists():
        raise FileExistsError("refusing to overwrite formal joined output")

    require_hash(args.blind, BLIND_SHA256, "blind manifest")
    require_hash(args.human, HUMAN_SHA256, "human reference")
    require_hash(args.qwen_schema, QWEN_SCHEMA_SHA256, "Qwen schema")
    require_hash(args.taxonomy, TAXONOMY_SHA256, "taxonomy map")
    qwen_meta_path = Path(str(args.qwen) + ".meta.json")
    qwen_meta = json.loads(qwen_meta_path.read_text("utf-8"))
    if (
        qwen_meta.get("status") != "FORMAL_COMPLETE"
        or qwen_meta.get("rows") != 11088
        or qwen_meta.get("num_shards") != 1
        or qwen_meta.get("output_sha256") != sha256_file(args.qwen)
        or qwen_meta.get("schema_sha256") != QWEN_SCHEMA_SHA256
        or qwen_meta.get("input_manifest_sha256") != BLIND_SHA256
    ):
        raise ValueError("Qwen formal manifest is incomplete or mismatched")
    blind = load_jsonl_by_id(args.blind, "blind")
    human = load_jsonl_by_id(args.human, "human")
    qwen = load_jsonl_by_id(args.qwen, "Qwen")
    if set(blind) != set(human) or set(blind) != set(qwen) or len(blind) != 11088:
        raise ValueError("blind/human/Qwen ID sets differ")

    qwen_schema = json.loads(args.qwen_schema.read_text("utf-8"))
    categories = list(qwen_schema["category_order"])
    for item_id, row in qwen.items():
        if row.get("input_manifest_sha256") != BLIND_SHA256:
            raise ValueError(f"Qwen input signature mismatch for {item_id}")
        if row.get("schema_sha256") != QWEN_SCHEMA_SHA256:
            raise ValueError(f"Qwen schema signature mismatch for {item_id}")
        labels = row.get("labels")
        if row.get("parse_ok") is True:
            if not isinstance(labels, dict) or set(labels) != set(categories + ["is_safe"]):
                raise ValueError(f"Qwen parsed schema mismatch for {item_id}")
        elif labels is not None:
            raise ValueError(f"Qwen malformed row was semantically repaired: {item_id}")

    taxonomy = json.loads(args.taxonomy.read_text("utf-8"))
    mappings = {row["source_id"]: row for row in taxonomy["mappings"]}
    if list(mappings) != categories:
        raise ValueError("taxonomy category order differs from Qwen schema")
    guard_paths = {
        "llama_guard_3_8b": args.llama_guard,
        "shieldgemma_2b": args.shieldgemma_2b,
        "shieldgemma_9b": args.shieldgemma_9b,
    }
    guard_scores: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    guard_meta: dict[str, Any] = {}
    for guard_id, path in guard_paths.items():
        guard_scores[guard_id], guard_meta[guard_id] = load_guard_scores(path, guard_id)
        if set(guard_scores[guard_id]) != set(blind):
            raise ValueError(f"{guard_id} item IDs differ from blind manifest")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    parse_failures = 0
    guard_failures = {guard_id: 0 for guard_id in GUARDS}
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        for item_id in blind:
            blind_row = blind[item_id]
            human_row = human[item_id]
            qwen_row = qwen[item_id]
            if human_row["pair_sha256"] != blind_row["pair_sha256"]:
                raise ValueError(f"human pair signature mismatch for {item_id}")
            if qwen_row.get("pair_sha256") != blind_row["pair_sha256"]:
                raise ValueError(f"Qwen pair signature mismatch for {item_id}")
            if qwen_row.get("parse_ok") is not True:
                parse_failures += 1
            guards: dict[str, Any] = {}
            for guard_id, spec in GUARDS.items():
                native = guard_scores[guard_id][item_id]
                mapped: dict[str, Any] = {}
                family = str(spec["family"])
                for category in categories:
                    cell = mapped_probability(
                        guard_id, native, mappings[category][family]
                    )
                    mapped[category] = cell
                    if cell["error"] not in (None, "no_defensible_taxonomy_mapping"):
                        guard_failures[guard_id] += 1
                guards[guard_id] = {"native": native, "mapped": mapped}
            record = {
                "schema": "pccd.labelsource.joined_item.v1",
                "id": item_id,
                "pair_sha256": blind_row["pair_sha256"],
                "human": {
                    "is_safe": human_row["human_is_safe"],
                    "category": human_row["human_category"],
                },
                "llm_proxy": {
                    "parse_ok": qwen_row["parse_ok"],
                    "labels": qwen_row["labels"],
                    "raw_output": qwen_row["raw_output"],
                },
                "guards": guards,
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    meta = {
        "schema": "pccd.labelsource.joined_manifest.v1",
        "status": "FORMAL_JOIN_COMPLETE",
        "rows": 11088,
        "categories": categories,
        "qwen_parse_failures": parse_failures,
        "guard_mapped_score_failures_excluding_no_mapping": guard_failures,
        "mapping_rules": {
            "llama_guard_3_8b": "omnibus full-S1-S14 P(unsafe) evaluated on each category-vs-benign stratum",
            "shieldgemma": "max native P(unsafe) across frozen mapped target policies; no-map is non-evaluable",
        },
        "input_sha256": {
            "blind": BLIND_SHA256,
            "human": HUMAN_SHA256,
            "qwen": sha256_file(args.qwen),
            "qwen_schema": QWEN_SCHEMA_SHA256,
            "taxonomy": TAXONOMY_SHA256,
            **{
                guard_id: guard_meta[guard_id]["output_sha256"] for guard_id in GUARDS
            },
        },
        "output_sha256": sha256_file(args.out),
    }
    meta_path = Path(str(args.out) + ".meta.json")
    meta_path.write_bytes((json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
