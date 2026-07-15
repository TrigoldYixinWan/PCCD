#!/usr/bin/env python3
"""Build and freeze the adequately-powered held-out prompt set for G2.

The candidate phase is label-free and draws new prompts from the local PKU,
UltraFeedback, and controlled-style sources while excluding every prompt in the
five frozen Day-2 splits.  The finalize phase may inspect *base-policy* teacher
labels, as explicitly allowed by PREREG_G2, and fixes a paired evaluation set
with ample base violated/satisfied support before any adapted generation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.policy_defs import POLICY_IDS
from src.sample_data import _soft_pairs


HARD_CATEGORIES = {
    "H1": {"Privacy Violation"},
    "H2": {"Cybercrime", "Economic Crime"},
    "H3": {"Violence", "Physical Harm", "Drugs", "Cybercrime", "Endangering Public Health"},
    "H4": {"Mental Manipulation"},
    "H5": {"Endangering National Security"},
}
LABEL_STATES = {"satisfied", "violated", "not_applicable"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def categories(record: dict) -> set[str]:
    result: set[str] = set()
    for index in (0, 1):
        raw = record.get(f"response_{index}_harm_category")
        if isinstance(raw, dict):
            result.update(key for key, value in raw.items() if value)
        elif isinstance(raw, list):
            result.update(raw)
    return result


def frozen_prompts(pool_dir: Path) -> tuple[set[str], dict[str, int]]:
    prompts: set[str] = set()
    counts = {}
    for split in ("train", "calib", "test", "audit", "conflict"):
        rows = read_jsonl(pool_dir / f"{split}.jsonl")
        counts[split] = len(rows)
        prompts.update(row["prompt"] for row in rows)
    return prompts, counts


def add_candidate(
    selected: dict[str, dict], prompt: str, source: str, stratum: str, metadata: dict
) -> None:
    if prompt not in selected:
        selected[prompt] = {
            "id": "g2eval_" + hashlib.sha256(prompt.encode()).hexdigest()[:20],
            "source": source,
            "prompt": prompt,
            "meta": {"g2_strata": [], "source_metadata": metadata},
        }
    if stratum not in selected[prompt]["meta"]["g2_strata"]:
        selected[prompt]["meta"]["g2_strata"].append(stratum)


def build_candidates(args: argparse.Namespace) -> None:
    output = Path(args.out)
    manifest_path = Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite frozen G2 candidate artifacts")
    excluded, frozen_counts = frozen_prompts(Path(args.pool_dir))
    selected: dict[str, dict] = {}
    stratum_counts: Counter = Counter()

    pku = load_dataset(str(Path(args.data_dir) / "pku-saferlhf"), split="train")
    hard_rows = []
    for index, row in enumerate(pku):
        prompt = row["prompt"]
        if prompt in excluded:
            continue
        cats = categories(row)
        if cats:
            hard_rows.append((digest(prompt, args.seed), index, prompt, cats))
    hard_rows.sort()
    for policy, mapped in HARD_CATEGORIES.items():
        for _, index, prompt, cats in hard_rows:
            if stratum_counts[policy] >= args.per_hard:
                break
            if not (cats & mapped):
                continue
            prior = policy in selected.get(prompt, {}).get("meta", {}).get("g2_strata", [])
            add_candidate(
                selected,
                prompt,
                "pku_saferlhf_prompt",
                policy,
                {"dataset_index": index, "pku_harm_categories": sorted(cats)},
            )
            if not prior:
                stratum_counts[policy] += 1
        if stratum_counts[policy] != args.per_hard:
            raise RuntimeError(f"insufficient new PKU prompts for {policy}: {stratum_counts[policy]}")

    rng = random.Random(args.seed + 1)
    soft_rows = _soft_pairs(max(20_000, args.per_soft * 20), rng)
    for axis in ("S1", "S2", "S3"):
        for row in sorted(soft_rows, key=lambda item: digest(item["prompt"], args.seed + 2)):
            if stratum_counts[axis] >= args.per_soft:
                break
            prompt = row["prompt"]
            if prompt in excluded:
                continue
            prior = axis in selected.get(prompt, {}).get("meta", {}).get("g2_strata", [])
            if row.get("meta", {}).get("axis") != {
                "S1": "formality", "S2": "verbosity", "S3": "structure"
            }[axis]:
                continue
            add_candidate(selected, prompt, "soft_style_prompt", axis, row.get("meta", {}))
            if not prior:
                stratum_counts[axis] += 1
        if stratum_counts[axis] != args.per_soft:
            raise RuntimeError(f"insufficient new controlled prompts for {axis}")

    uf = load_dataset(str(Path(args.data_dir) / "ultrafeedback"), split="train")
    uf_rows = sorted(
        (
            (digest(row["instruction"], args.seed + 3), index, row["instruction"])
            for index, row in enumerate(uf)
            if row["instruction"] not in excluded
        ),
        key=lambda item: item[0],
    )
    for _, index, prompt in uf_rows:
        if stratum_counts["TASK"] >= args.task:
            break
        if prompt in selected:
            continue
        add_candidate(
            selected,
            prompt,
            "ultrafeedback_prompt",
            "TASK",
            {"dataset_index": index},
        )
        stratum_counts["TASK"] += 1
    if stratum_counts["TASK"] != args.task:
        raise RuntimeError("insufficient new UltraFeedback prompts")

    rows = sorted(selected.values(), key=lambda row: digest(row["id"], args.seed + 4))
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("candidate ID collision")
    if any(row["prompt"] in excluded for row in rows):
        raise RuntimeError("candidate prompt leakage into a frozen split")
    write_jsonl(output, rows)
    manifest = {
        "phase": "candidate",
        "seed": args.seed,
        "candidate_count": len(rows),
        "stratum_counts": dict(stratum_counts),
        "excluded_frozen_prompt_count": len(excluded),
        "frozen_split_item_counts": frozen_counts,
        "prompt_overlap_with_any_frozen_split": 0,
        "candidate_sha256": sha256_file(output),
        "parameters": {
            "per_hard": args.per_hard,
            "per_soft": args.per_soft,
            "task": args.task,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def validate_alignment(*groups: list[dict]) -> None:
    ids = [[row["id"] for row in rows] for rows in groups]
    if not ids or any(group != ids[0] for group in ids[1:]):
        raise ValueError("candidate/base/teacher artifacts are not exactly ID-aligned")
    if len(ids[0]) != len(set(ids[0])):
        raise ValueError("duplicate candidate IDs")


def finalize(args: argparse.Namespace) -> None:
    outputs = [Path(args.out), Path(args.base_out), Path(args.labels_out), Path(args.manifest)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite frozen G2 evaluation artifacts")
    candidates = read_jsonl(Path(args.candidates))
    base = read_jsonl(Path(args.base))
    labels = read_jsonl(Path(args.labels))
    validate_alignment(candidates, base, labels)
    label_by_id = {row["id"]: row for row in labels}
    candidate_by_id = {row["id"]: row for row in candidates}
    for row in labels:
        if set(row.get("labels", {})) != set(POLICY_IDS):
            raise ValueError(f"non-strict teacher JSON for {row['id']}")
        if not set(row["labels"].values()) <= LABEL_STATES:
            raise ValueError(f"invalid teacher state for {row['id']}")

    support = {
        policy: Counter(row["labels"][policy] for row in labels) for policy in POLICY_IDS
    }
    under = [policy for policy in POLICY_IDS if support[policy]["violated"] < args.minimum]
    if under:
        raise RuntimeError(
            "candidate base support below locked minimum; augment and regenerate base: "
            + ", ".join(f"{p}={support[p]['violated']}" for p in under)
        )

    selected: set[str] = set()
    for policy in POLICY_IDS:
        for state in ("violated", "satisfied"):
            eligible = [
                row["id"] for row in labels if row["labels"][policy] == state
            ]
            eligible.sort(key=lambda item: digest(f"{policy}:{state}:{item}", args.seed))
            selected.update(eligible[: args.target_per_state])
    remaining = [row["id"] for row in candidates if row["id"] not in selected]
    remaining.sort(key=lambda item: digest(item, args.seed + 1))
    selected.update(remaining[: max(0, args.target_total - len(selected))])
    ordered_ids = sorted(selected, key=lambda item: digest(item, args.seed + 2))
    if len(ordered_ids) < args.target_total:
        raise RuntimeError("candidate set is too small for target_total")

    base_by_id = {row["id"]: row for row in base}
    final_prompts = [candidate_by_id[item] for item in ordered_ids]
    final_base = [base_by_id[item] for item in ordered_ids]
    final_labels = [label_by_id[item] for item in ordered_ids]
    final_support = {
        policy: Counter(row["labels"][policy] for row in final_labels) for policy in POLICY_IDS
    }
    final_under = [
        policy for policy in POLICY_IDS if final_support[policy]["violated"] < args.minimum
    ]
    if final_under:
        raise RuntimeError(f"selection unexpectedly underpowered: {final_under}")
    write_jsonl(Path(args.out), final_prompts)
    write_jsonl(Path(args.base_out), final_base)
    write_jsonl(Path(args.labels_out), final_labels)
    manifest = {
        "phase": "final",
        "seed": args.seed,
        "candidate_count": len(candidates),
        "final_count": len(final_prompts),
        "minimum_locked_violated_support": args.minimum,
        "target_per_state": args.target_per_state,
        "target_total": args.target_total,
        "candidate_base_support": {p: dict(support[p]) for p in POLICY_IDS},
        "final_base_support": {p: dict(final_support[p]) for p in POLICY_IDS},
        "underpowered_base_policies": [],
        "artifacts": {
            "prompts": {"path": str(Path(args.out).resolve()), "sha256": sha256_file(Path(args.out))},
            "base_responses": {"path": str(Path(args.base_out).resolve()), "sha256": sha256_file(Path(args.base_out))},
            "base_teacher": {"path": str(Path(args.labels_out).resolve()), "sha256": sha256_file(Path(args.labels_out))},
        },
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    outputs = Path(os.environ.get("PCCD_OUT", "outputs"))
    data = Path(os.environ.get("DATA_DIR", "data"))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidate")
    candidate.add_argument("--data_dir", default=str(data))
    candidate.add_argument("--pool_dir", default=str(outputs / "pool"))
    candidate.add_argument("--out", required=True)
    candidate.add_argument("--manifest", required=True)
    candidate.add_argument("--per_hard", type=int, default=800)
    candidate.add_argument("--per_soft", type=int, default=600)
    candidate.add_argument("--task", type=int, default=1200)
    candidate.add_argument("--seed", type=int, default=20260716)
    final = sub.add_parser("finalize")
    final.add_argument("--candidates", required=True)
    final.add_argument("--base", required=True)
    final.add_argument("--labels", required=True)
    final.add_argument("--out", required=True)
    final.add_argument("--base_out", required=True)
    final.add_argument("--labels_out", required=True)
    final.add_argument("--manifest", required=True)
    final.add_argument("--minimum", type=int, default=30)
    final.add_argument("--target_per_state", type=int, default=80)
    final.add_argument("--target_total", type=int, default=3000)
    final.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "candidate":
        build_candidates(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
