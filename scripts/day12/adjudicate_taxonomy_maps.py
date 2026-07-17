#!/usr/bin/env python3
"""Validate and adjudicate two outcome-blind taxonomy reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_PACKET_SHA256 = (
    "71ea53cb5e7a7f64b7b8e4fa5c5c1711b4b3db06bfae890d075bb61d07fe860d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("entries", payload.get("mappings"))
    if not isinstance(rows, list):
        raise ValueError("review lacks entries/mappings list")
    return rows


# Every decision is fixed from definitions before guard target outcomes.  The
# only target-set disagreement is handled conservatively: generic unethical
# behavior does not inherit S8 solely from an open-ended 'other unethical'
# clause when intellectual-property conduct is not named in the source rubric.
ADJUDICATIONS: dict[tuple[str, str], dict[str, Any]] = {
    ("animal_abuse", "llama_guard_3"): {
        "target_ids": ["S1"],
        "relation": "partial_overlap",
        "rationale": (
            "S1 explicitly includes animal abuse, but the source also includes "
            "neglect and distress outside S1's enable/encourage/endorse violence rule."
        ),
    },
    ("child_abuse", "llama_guard_3"): {
        "target_ids": ["S1", "S4"],
        "relation": "partial_overlap",
        "rationale": (
            "S1 and S4 cover physical and sexual child abuse; emotional abuse, "
            "neglect, exploitation, and general rights violations remain broader."
        ),
    },
    ("hate_speech,offensive_language", "llama_guard_3"): {
        "target_ids": ["S10"],
        "relation": "partial_overlap",
        "rationale": (
            "S10 covers protected-characteristic demeaning/dehumanization, while "
            "the source also includes generic offensive language and uses a partly "
            "different hostility boundary."
        ),
    },
    ("non_violent_unethical_behavior", "llama_guard_3"): {
        "target_ids": ["S2", "S5"],
        "relation": "partial_overlap",
        "rationale": (
            "S2 covers criminal non-violent conduct and S5 covers false "
            "reputation-harming rumors. S8 is excluded because intellectual-property "
            "conduct is not named in the source definition; mapping an open-ended "
            "'other unethical conduct' clause to every narrower policy would be "
            "non-conservative."
        ),
    },
    ("self_harm", "llama_guard_3"): {
        "target_ids": ["S11"],
        "relation": "partial_overlap",
        "rationale": (
            "Both cover suicide and self-injury, but the source includes emotional "
            "self-harm and general discussion while S11 separately names disordered eating."
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite adjudicated map: {args.out}")
    packet_hash = sha256_file(args.packet)
    if packet_hash != EXPECTED_PACKET_SHA256:
        raise ValueError(f"definition packet hash mismatch: {packet_hash}")

    packet = json.loads(args.packet.read_text("utf-8"))
    a = json.loads(args.review_a.read_text("utf-8"))
    b = json.loads(args.review_b.read_text("utf-8"))
    if a["packet_sha256"] != packet_hash or b["packet_sha256"] != packet_hash:
        raise ValueError("reviewer packet signature mismatch")
    if a["reviewer_id"] == b["reviewer_id"]:
        raise ValueError("reviewer identities are not distinct")

    expected_ids = [
        row["id"] for row in packet["sources"]["beavertails"]["categories"]
    ]
    by_a = {row["source_id"]: row for row in entries(a)}
    by_b = {row["source_id"]: row for row in entries(b)}
    if list(by_a) != expected_ids or list(by_b) != expected_ids:
        raise ValueError("review category order/set differs from frozen packet")

    disagreements = []
    final_rows = []
    for source_id in expected_ids:
        output: dict[str, Any] = {"source_id": source_id}
        for family in ("llama_guard_3", "shieldgemma"):
            cell_a = by_a[source_id][family]
            cell_b = by_b[source_id][family]
            same_targets = cell_a["target_ids"] == cell_b["target_ids"]
            same_relation = cell_a["relation"] == cell_b["relation"]
            if same_targets and same_relation:
                output[family] = {
                    **cell_a,
                    "reviewer_agreement": "full",
                }
                continue
            key = (source_id, family)
            decision = ADJUDICATIONS.get(key)
            if decision is None:
                raise ValueError(f"unadjudicated reviewer disagreement: {key}")
            output[family] = {
                **decision,
                "reviewer_agreement": (
                    "target_set_only" if same_targets else "adjudicated_target_set"
                ),
            }
            disagreements.append(
                {
                    "source_id": source_id,
                    "family": family,
                    "reviewer_a": {
                        "target_ids": cell_a["target_ids"],
                        "relation": cell_a["relation"],
                    },
                    "reviewer_b": {
                        "target_ids": cell_b["target_ids"],
                        "relation": cell_b["relation"],
                    },
                    "resolution": decision,
                }
            )
        final_rows.append(output)

    payload = {
        "schema": "pccd.labelsource.taxonomy_map.v1",
        "status": "ADJUDICATED_BEFORE_OUTCOMES_PENDING_HUMAN_LOCK",
        "definition_packet": {
            "path": str(args.packet),
            "sha256": packet_hash,
        },
        "blind_reviews": [
            {
                "reviewer_id": a["reviewer_id"],
                "path": str(args.review_a),
                "sha256": sha256_file(args.review_a),
            },
            {
                "reviewer_id": b["reviewer_id"],
                "path": str(args.review_b),
                "sha256": sha256_file(args.review_b),
            },
        ],
        "reviewer_provenance": (
            "Two isolated Codex reviewer instances read only the frozen definition "
            "packet and were blind to support and guard outcomes. Human project-owner "
            "acceptance remains required in the final lock commit."
        ),
        "disagreements": disagreements,
        "mappings": final_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": sha256_file(args.out),
                "disagreements": len(disagreements),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
