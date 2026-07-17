#!/usr/bin/env python3
"""Validate §11.1-§11.4 metadata and emit the human-lock package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CATEGORIES = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]
EXPECTED_QWEN_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
EXPECTED_BLIND_MANIFEST_SHA256 = (
    "602de7160b536ae8567bda8a14ae6394a7413710c15b22157384c4d503a0c142"
)
EXPECTED_GUARD_REVISIONS = {
    "meta-llama/Llama-Guard-3-8B": "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
    "google/shieldgemma-2b": "d1dffc9c8c9237a90aab09c61383791e718ef9e8",
    "google/shieldgemma-9b": "b8b636016df4540721a098c7aab91c97ec6ee508",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--guard-registry", type=Path, required=True)
    parser.add_argument("--guard-report", type=Path, required=True)
    parser.add_argument("--taxonomy-packet", type=Path, required=True)
    parser.add_argument("--taxonomy-map", type=Path, required=True)
    parser.add_argument("--qwen-schema", type=Path, required=True)
    parser.add_argument("--qwen-runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite lock package: {args.out}")

    prereg_text = args.prereg.read_text("utf-8")
    if "Status: **DRAFT" not in prereg_text or "NO GUARD SCORING" not in prereg_text:
        raise ValueError("pre-lock package must preserve DRAFT/no-scoring header")
    if "11,088 unique exact QA pairs" not in prereg_text:
        raise ValueError("prereg lacks corrected independent-unit count")

    support_text = args.support_report.read_text("utf-8")
    for required in (
        "Independent analysis units: **11,088 unique exact prompt-response pairs**",
        "Majority-safe / majority-unsafe pairs: **4,936 / 6,152**",
        "**10 PRIMARY categories** pass",
    ):
        if required not in support_text:
            raise ValueError(f"support audit lacks frozen fact: {required}")

    registry_text = args.guard_registry.read_text("utf-8")
    for repo, revision in EXPECTED_GUARD_REVISIONS.items():
        if repo not in registry_text or revision not in registry_text:
            raise ValueError(f"guard registry lacks pinned identity: {repo}@{revision}")
    for required in (
        "softmax([logit(token 3553), logit(token 1294)])[0]",
        "exact token `unsafe` (ID `39257`) at that",
        "semantic-label position. Safe/unsafe renormalization",
    ):
        if required not in registry_text:
            raise ValueError(f"guard registry lacks frozen interface: {required}")

    guard_text = args.guard_report.read_text("utf-8")
    if guard_text.count("| PASS |") != 3:
        raise ValueError("guard sanity report does not contain three passing interfaces")

    taxonomy = json.loads(args.taxonomy_map.read_text("utf-8"))
    if taxonomy.get("status") != "ADJUDICATED_BEFORE_OUTCOMES_PENDING_HUMAN_LOCK":
        raise ValueError("taxonomy map is not in adjudicated pre-lock state")
    if [row["source_id"] for row in taxonomy["mappings"]] != EXPECTED_CATEGORIES:
        raise ValueError("taxonomy map category order/set mismatch")
    reviews = taxonomy.get("blind_reviews", [])
    if len(reviews) != 2 or reviews[0]["reviewer_id"] == reviews[1]["reviewer_id"]:
        raise ValueError("taxonomy map lacks two distinct blind reviewers")
    for review in reviews:
        review_path = Path(review["path"])
        if not review_path.is_file() or sha256_file(review_path) != review["sha256"]:
            raise ValueError(f"review artifact signature mismatch: {review}")

    schema = json.loads(args.qwen_schema.read_text("utf-8"))
    if schema.get("status") != "PRELOCK_FROZEN_BEFORE_ANNOTATION":
        raise ValueError("Qwen schema is not pre-lock frozen")
    if schema.get("category_order") != EXPECTED_CATEGORIES:
        raise ValueError("Qwen category order differs from native BeaverTails order")
    output_keys = list(schema.get("output_schema", {}))
    if output_keys != EXPECTED_CATEGORIES + ["is_safe"]:
        raise ValueError("Qwen output key order/set mismatch")
    if any(value != "boolean" for value in schema["output_schema"].values()):
        raise ValueError("Qwen schema contains a non-boolean label")
    annotator = schema["annotator"]
    if (
        annotator.get("revision") != EXPECTED_QWEN_REVISION
        or annotator.get("temperature") != 0.0
        or annotator.get("calls_per_item") != 1
        or annotator.get("retries") != 0
    ):
        raise ValueError("Qwen model identity or one-call deterministic protocol mismatch")
    fingerprints = annotator.get("local_asset_fingerprints", {})
    expected_shards = {
        f"model-{index:05d}-of-00017.safetensors" for index in range(1, 18)
    }
    if not expected_shards.issubset(fingerprints):
        raise ValueError("Qwen fingerprint registry does not cover all 17 weight shards")
    if any(len(value) != 64 for value in fingerprints.values()):
        raise ValueError("Qwen fingerprint registry contains a non-SHA256 value")
    if schema["blindness"]["input_manifest_sha256"] != EXPECTED_BLIND_MANIFEST_SHA256:
        raise ValueError("Qwen schema blind-manifest signature mismatch")
    axis = schema["objective_subjective_axis"]
    objective = axis["objective"]
    subjective = axis["subjective"]
    if set(objective) & set(subjective):
        raise ValueError("objective/subjective groups overlap")
    if set(objective) | set(subjective) != set(EXPECTED_CATEGORIES):
        raise ValueError("objective/subjective groups do not partition 14 categories")

    payload = {
        "schema": "pccd.labelsource.prelock_package.v1",
        "status": "READY_FOR_HUMAN_LOCK_NO_OUTCOMES_AUTHORIZED",
        "formal_outcomes_authorized": False,
        "gates": {
            "11.1_support": "PASS_10_PRIMARY_OF_14",
            "11.2_guard_registry_and_sanity": "PASS_3_GUARDS_2_FAMILIES",
            "11.3_taxonomy_map": "PASS_TWO_BLIND_REVIEWS_ADJUDICATED",
            "11.4_qwen_annotation_protocol": "PASS_FROZEN_ONE_CALL_BLIND_SCHEMA",
        },
        "frozen_constants": {
            "independent_items": 11088,
            "majority_unsafe_items": 6152,
            "blind_manifest_sha256": EXPECTED_BLIND_MANIFEST_SHA256,
            "qwen_repo": annotator["model"],
            "qwen_revision": annotator["revision"],
            "objective_categories": objective,
            "subjective_categories": subjective,
        },
        "artifacts": {
            "prereg_draft": artifact(args.prereg),
            "support_report": artifact(args.support_report),
            "guard_registry": artifact(args.guard_registry),
            "guard_report": artifact(args.guard_report),
            "taxonomy_definition_packet": artifact(args.taxonomy_packet),
            "taxonomy_map": artifact(args.taxonomy_map),
            "qwen_schema": artifact(args.qwen_schema),
            "qwen_runner": artifact(args.qwen_runner),
        },
        "next_gate": (
            "Human project owner or PaperGuru reviews this package, records acceptance "
            "in DECISION_LOG, changes the prereg header to LOCKED in a commit, and only "
            "then authorizes the single formal run."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "sha256": sha256_file(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
