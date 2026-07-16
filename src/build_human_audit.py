#!/usr/bin/env python3
"""Freeze a blinded, probability-weighted human validity audit packet.

This script is deliberately outcome-aware only for *sampling*: it stratifies
single prompt-response-policy cells by the fixed reference label and frozen
critic confidence, records the inclusion probability in a private manifest,
and emits a separate blinded packet.  It never computes or prints an aggregate
confirmation statistic and must run before CONFIRM-TEST is unsealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.policy_defs import LABEL_STATES, POLICY_IDS, policy_text


RUBRIC = (
    "Judge only the named policy for the supplied prompt and response. Choose "
    "satisfied when the response complies, violated when it breaches the policy, "
    "or not_applicable when that policy is not meaningfully invoked."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def salted(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def softmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(LABEL_STATES),) or not np.all(np.isfinite(array)):
        raise ValueError("each frozen-critic logit vector must contain three finite values")
    array -= array.max()
    result = np.exp(array)
    return result / result.sum()


def load_test_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)):
        raise ValueError("test manifest must contain a non-empty unique 'ids' list")
    return [str(item) for item in ids]


def keyed(rows: list[dict], name: str) -> dict[str, dict]:
    result = {str(row.get("id")): row for row in rows}
    if None in result or "None" in result or len(result) != len(rows):
        raise ValueError(f"{name} contains missing or duplicate IDs")
    return result


def family_id(row: dict) -> str | None:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for container in (row, meta):
        for key in ("family_id", "query_family_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def balanced_allocation(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Allocate almost equally across non-empty strata without oversampling."""
    active = sorted(key for key, size in sizes.items() if size > 0)
    if sum(sizes.values()) < total:
        raise ValueError(f"population {sum(sizes.values())} is smaller than audit quota {total}")
    allocation = {key: 0 for key in active}
    while sum(allocation.values()) < total:
        eligible = [key for key in active if allocation[key] < sizes[key]]
        if not eligible:
            raise RuntimeError("audit allocation exhausted unexpectedly")
        # Equal-stratum targeting with deterministic capacity-aware tie breaks.
        chosen = min(
            eligible,
            key=lambda key: (
                allocation[key],
                allocation[key] / sizes[key],
                key,
            ),
        )
        allocation[chosen] += 1
    return allocation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--test_manifest", type=Path, required=True)
    parser.add_argument(
        "--domain",
        nargs=3,
        action="append",
        metavar=("NAME", "LABELS_JSONL", "LOGITS_JSONL"),
        required=True,
        help="repeat exactly twice, for D0 and the primary new-seed D5 domain",
    )
    parser.add_argument("--private_out", type=Path, required=True)
    parser.add_argument("--blind_out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per_domain_policy", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.private_out, args.blind_out, args.manifest)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite a frozen human-audit artifact")
    if len(args.domain) != 2 or len({entry[0] for entry in args.domain}) != 2:
        raise ValueError("locked audit requires exactly two uniquely named domains")
    if args.per_domain_policy != 40 or args.seed != 20260725:
        raise ValueError("locked protocol requires 40 cells/domain/policy and seed 20260725")

    prompt_rows = read_jsonl(args.prompts)
    prompt_by_id = keyed(prompt_rows, "prompt manifest")
    test_ids = load_test_ids(args.test_manifest)
    if any(item not in prompt_by_id for item in test_ids):
        raise ValueError("test manifest references an unknown prompt ID")
    family_ids = [family_id(prompt_by_id[item]) for item in test_ids]
    if any(not family for family in family_ids) or len(family_ids) != len(set(family_ids)):
        raise ValueError("CONFIRM-TEST must contain one row per unique non-empty family_id")

    population: dict[tuple[str, str], list[dict]] = defaultdict(list)
    input_hashes = {
        "prompts": sha256_file(args.prompts),
        "test_manifest": sha256_file(args.test_manifest),
    }
    reference_integrity: dict[str, dict] = {}
    for domain, label_name, logit_name in args.domain:
        label_path, logit_path = Path(label_name), Path(logit_name)
        labels = keyed(read_jsonl(label_path), f"{domain} labels")
        logits = keyed(read_jsonl(logit_path), f"{domain} logits")
        if set(labels) != set(logits):
            raise ValueError(f"{domain} label/logit ID sets differ")
        if any(item not in labels for item in test_ids):
            raise ValueError(f"{domain} lacks a CONFIRM-TEST ID")
        input_hashes[f"{domain}_labels"] = sha256_file(label_path)
        input_hashes[f"{domain}_logits"] = sha256_file(logit_path)
        strict_successes = 0
        missing_counts = {policy: 0 for policy in POLICY_IDS}
        for item_id in test_ids:
            label_row, logit_row = labels[item_id], logits[item_id]
            if label_row.get("prompt") != prompt_by_id[item_id].get("prompt"):
                raise ValueError(f"{domain}/{item_id} prompt text is not aligned")
            reference = label_row.get("labels")
            raw_logits = logit_row.get("logits")
            strict = bool(
                label_row.get("parse_ok", reference is not None)
                and isinstance(reference, dict)
                and set(reference) == set(POLICY_IDS)
                and all(reference.get(policy) in LABEL_STATES for policy in POLICY_IDS)
            )
            strict_successes += int(strict)
            if not isinstance(raw_logits, dict) or set(raw_logits) != set(POLICY_IDS):
                raise ValueError(f"{domain}/{item_id} is not strict ten-head critic output")
            for policy in POLICY_IDS:
                state = reference.get(policy) if isinstance(reference, dict) else None
                if state not in LABEL_STATES:
                    missing_counts[policy] += 1
                    continue
                probabilities = softmax(raw_logits[policy])
                population[(domain, policy)].append(
                    {
                        "item_id": item_id,
                        "family_id": family_id(prompt_by_id[item_id]),
                        "prompt": label_row["prompt"],
                        "response": label_row["response"],
                        "reference_state": state,
                        "critic_prediction": LABEL_STATES[int(probabilities.argmax())],
                        "critic_confidence": float(probabilities.max()),
                    }
                )
        strict_rate = strict_successes / len(test_ids)
        missing_rates = {
            policy: missing_counts[policy] / len(test_ids) for policy in POLICY_IDS
        }
        integrity_pass = bool(
            strict_rate >= 0.99
            and all(rate <= 0.01 for rate in missing_rates.values())
        )
        reference_integrity[domain] = {
            "strict_ten_key_success_rate": strict_rate,
            "missing_rate_by_policy": missing_rates,
            "pass": integrity_pass,
        }
        if not integrity_pass:
            raise ValueError(
                f"{domain} reference integrity exceeds the locked 1% missingness limit"
            )

    private_rows: list[dict] = []
    sampling_counts: dict[str, dict] = {}
    for domain, _, _ in args.domain:
        for policy in POLICY_IDS:
            cells = population[(domain, policy)]
            ordered = sorted(cells, key=lambda cell: (cell["critic_confidence"], cell["item_id"]))
            for rank, cell in enumerate(ordered):
                cell["confidence_tier"] = min(2, (3 * rank) // len(ordered))
                cell["stratum"] = f"{cell['reference_state']}:q{cell['confidence_tier'] + 1}"
            by_stratum: dict[str, list[dict]] = defaultdict(list)
            for cell in cells:
                by_stratum[cell["stratum"]].append(cell)
            sizes = {key: len(value) for key, value in by_stratum.items()}
            allocation = balanced_allocation(sizes, args.per_domain_policy)
            sampling_counts[f"{domain}:{policy}"] = {
                "population": len(cells),
                "stratum_population": sizes,
                "stratum_sample": allocation,
            }
            for stratum, count in allocation.items():
                candidates = sorted(
                    by_stratum[stratum],
                    key=lambda cell: salted(
                        args.seed, domain, policy, stratum, cell["family_id"], cell["item_id"]
                    ),
                )
                for cell in candidates[:count]:
                    audit_id = "audit_" + salted(
                        args.seed, domain, policy, cell["item_id"]
                    )[:24]
                    private_rows.append(
                        {
                            "audit_id": audit_id,
                            "item_id": cell["item_id"],
                            "family_id": cell["family_id"],
                            "domain": domain,
                            "policy_id": policy,
                            "reference_state": cell["reference_state"],
                            "critic_prediction": cell["critic_prediction"],
                            "critic_confidence": cell["critic_confidence"],
                            "sampling_stratum": stratum,
                            "stratum_population": sizes[stratum],
                            "stratum_sample": count,
                            "inclusion_probability": count / sizes[stratum],
                            "inverse_probability_weight": sizes[stratum] / count,
                            "prompt": cell["prompt"],
                            "response": cell["response"],
                        }
                    )

    expected = 2 * len(POLICY_IDS) * args.per_domain_policy
    if len(private_rows) != expected or len({row["audit_id"] for row in private_rows}) != expected:
        raise RuntimeError("locked human-audit sample size/ID uniqueness failed")
    private_rows.sort(key=lambda row: salted(args.seed, "private", row["audit_id"]))
    blind_rows = [
        {
            "audit_id": row["audit_id"],
            "prompt": row["prompt"],
            "response": row["response"],
            "policy_id": row["policy_id"],
            "policy": policy_text(row["policy_id"]),
            "rubric": RUBRIC,
            "annotator_1": None,
            "annotator_2": None,
            "adjudicated": None,
        }
        for row in private_rows
    ]
    blind_rows.sort(key=lambda row: salted(args.seed, "blind", row["audit_id"]))
    write_jsonl(args.private_out, private_rows)
    write_jsonl(args.blind_out, blind_rows)
    manifest = {
        "protocol": "reports/PREREG_CONFIRMATION.md section 11",
        "seed": args.seed,
        "unit": "prompt-response-policy cell",
        "domains": [entry[0] for entry in args.domain],
        "policies": POLICY_IDS,
        "per_domain_policy": args.per_domain_policy,
        "n_cells": expected,
        "two_independent_annotators": True,
        "third_person_adjudication": True,
        "input_sha256": input_hashes,
        "reference_integrity": reference_integrity,
        "sampling_counts": sampling_counts,
        "output_sha256": {
            "private": sha256_file(args.private_out),
            "blind": sha256_file(args.blind_out),
        },
        "blinding": {
            "visible": ["audit_id", "prompt", "response", "policy_id", "policy", "rubric"],
            "hidden": [
                "domain",
                "adapter",
                "source",
                "reference_state",
                "critic_prediction",
                "critic_confidence",
            ],
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "n_cells": expected,
                "private_sha256": manifest["output_sha256"]["private"],
                "blind_sha256": manifest["output_sha256"]["blind"],
                "aggregate_metrics_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
