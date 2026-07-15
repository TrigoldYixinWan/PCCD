#!/usr/bin/env python3
"""Freeze critic-blind, teacher-violated PKU preference pairs for G2.

Chosen responses are human-written PKU responses that the label-only teacher
marks as violating at least one hard policy while the already-frozen D0 critic
does not predict violated for that policy.  Rejected responses use the paired
PKU-safe answer when one exists and an invariant generic safe refusal otherwise.
This directly operationalizes
"violated-but-fluent and hidden from the critic" without consulting any G2
adapted outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic_model import POLICY_IDS

HARD = ("H1", "H2", "H3", "H4", "H5")
GENERIC_SAFE_REJECTION = (
    "I can't assist with that request. I can instead provide high-level, lawful, "
    "and safety-conscious information that does not enable harm, unauthorized "
    "access, confidential disclosure, deception, or other policy violations."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def softmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array -= array.max()
    result = np.exp(array)
    return result / result.sum()


def raw_pair_index(dataset) -> dict[tuple[str, str], dict]:
    result = {}
    for index, row in enumerate(dataset):
        prompt = row["prompt"]
        safe = [bool(row[f"is_response_{i}_safe"]) for i in (0, 1)]
        if safe[0] == safe[1]:
            continue
        for chosen_index in (0, 1):
            if safe[chosen_index]:
                continue
            rejected_index = 1 - chosen_index
            result[(prompt, row[f"response_{chosen_index}"])] = {
                "rejected": row[f"response_{rejected_index}"],
                "dataset_index": index,
                "chosen_index": chosen_index,
                "rejected_index": rejected_index,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_labels", required=True)
    parser.add_argument("--train_logits", required=True)
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--eval_prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--per_policy_floor", type=int, default=20)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    output = Path(args.out)
    manifest_path = Path(args.manifest)
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite frozen hidden-pair artifacts")
    labels = read_jsonl(Path(args.train_labels))
    logits = read_jsonl(Path(args.train_logits))
    if [row["id"] for row in labels] != [row["id"] for row in logits]:
        raise ValueError("train labels/logits are not exactly ID-aligned")
    eval_prompt_text = {row["prompt"] for row in read_jsonl(Path(args.eval_prompts))}

    pku = load_dataset(str(Path(args.data_dir) / "pku-saferlhf"), split="train")
    paired = raw_pair_index(pku)
    candidates = []
    for label_row, logit_row in zip(labels, logits):
        if label_row.get("source") != "pku_saferlhf":
            continue
        if label_row.get("meta", {}).get("is_safe") is not False:
            continue
        response = label_row["response"]
        if not args.min_chars <= len(response) <= args.max_chars:
            continue
        raw = paired.get((label_row["prompt"], response))
        if label_row["prompt"] in eval_prompt_text:
            continue
        teacher_violated = [p for p in HARD if label_row["labels"][p] == "violated"]
        if not teacher_violated:
            continue
        probabilities = {
            policy: softmax(logit_row["logits"][policy]).tolist() for policy in POLICY_IDS
        }
        hidden = [
            policy
            for policy in teacher_violated
            if int(np.argmax(probabilities[policy])) != 1
        ]
        if not hidden:
            continue
        hidden_probability = min(probabilities[policy][1] for policy in hidden)
        order = hashlib.sha256(f"{args.seed}:{label_row['id']}".encode()).hexdigest()
        candidates.append(
            (
                hidden_probability,
                order,
                {
                    "id": "g2pair_" + hashlib.sha256(label_row["id"].encode()).hexdigest()[:20],
                    "source_id": label_row["id"],
                    "prompt": label_row["prompt"],
                    "chosen": response,
                    "rejected": raw["rejected"] if raw else GENERIC_SAFE_REJECTION,
                    "meta": {
                        "teacher_labels": label_row["labels"],
                        "teacher_violated_hard": teacher_violated,
                        "critic_hidden_hard": hidden,
                        "critic_violated_probabilities": {
                            policy: probabilities[policy][1] for policy in teacher_violated
                        },
                        "raw_dataset_index": raw["dataset_index"] if raw else None,
                        "chosen_response_index": raw["chosen_index"] if raw else label_row.get("meta", {}).get("resp_idx"),
                        "rejected_response_index": raw["rejected_index"] if raw else None,
                        "rejected_source": "paired_pku_safe" if raw else "generic_safe_refusal",
                        "chosen_pku_safe": False,
                        "rejected_pku_safe": True,
                    },
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < args.count:
        raise RuntimeError(
            f"only {len(candidates)} critic-blind teacher-violated pairs; need {args.count}"
        )
    selected_by_id = {}
    for policy in HARD:
        eligible = [item for item in candidates if policy in item[2]["meta"]["critic_hidden_hard"]]
        if len(eligible) < args.per_policy_floor:
            raise RuntimeError(
                f"hidden pair support for {policy}={len(eligible)} below floor={args.per_policy_floor}"
            )
        for item in eligible[: args.per_policy_floor]:
            selected_by_id[item[2]["id"]] = item[2]
    for _, _, row in candidates:
        if len(selected_by_id) >= args.count:
            break
        selected_by_id[row["id"]] = row
    selected = list(selected_by_id.values())
    selected.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['id']}".encode()).hexdigest())
    if len({row["id"] for row in selected}) != args.count:
        raise RuntimeError("duplicate hidden-pair IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    hidden_counts = Counter(
        policy for row in selected for policy in row["meta"]["critic_hidden_hard"]
    )
    teacher_counts = Counter(
        policy for row in selected for policy in row["meta"]["teacher_violated_hard"]
    )
    selected_probabilities = [
        min(
            row["meta"]["critic_violated_probabilities"][policy]
            for policy in row["meta"]["critic_hidden_hard"]
        )
        for row in selected
    ]
    manifest = {
        "objective": "teacher-violated fluent PKU chosen response hidden from frozen D0 critic; paired PKU-safe or invariant generic-safe response rejected",
        "selection_precedes_all_g2_adapted_generation": True,
        "seed": args.seed,
        "available_candidates": len(candidates),
        "selected_count": len(selected),
        "eval_prompt_overlap": 0,
        "teacher_violated_hard_counts": dict(teacher_counts),
        "critic_hidden_hard_counts": dict(hidden_counts),
        "rejected_source_counts": dict(Counter(row["meta"]["rejected_source"] for row in selected)),
        "per_policy_floor": args.per_policy_floor,
        "selected_hidden_violated_probability": {
            "mean": float(np.mean(selected_probabilities)),
            "min": float(np.min(selected_probabilities)),
            "max": float(np.max(selected_probabilities)),
        },
        "length_filter_chars": [args.min_chars, args.max_chars],
        "out": str(output.resolve()),
        "sha256": sha256_file(output),
    }
    if not all(math.isfinite(value) for value in manifest["selected_hidden_violated_probability"].values()):
        raise RuntimeError("non-finite selection probability")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
