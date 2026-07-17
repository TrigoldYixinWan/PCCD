#!/usr/bin/env python3
"""Freeze the complete machine-readable guard registry before sanity output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prelock-dir", type=Path, required=True)
    parser.add_argument("--guard-score", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    specs = [
        (
            "llama_guard_3_8b",
            "meta-llama/Llama-Guard-3-8B",
            "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
            "llama3.1",
            (
                "full-vocabulary softmax for unsafe at the first semantic "
                "safety-label token after the fixed template prefix '\\n\\n'"
            ),
            args.prelock_dir / "llama_guard_tokenizer_registry.json",
            args.prelock_dir / "llama_guard_weight_verification.json",
        ),
        (
            "shieldgemma_2b",
            "google/shieldgemma-2b",
            "d1dffc9c8c9237a90aab09c61383791e718ef9e8",
            "gemma",
            "softmax([Yes_logit, No_logit])[Yes]",
            args.prelock_dir / "shieldgemma_2b_tokenizer_registry.json",
            args.prelock_dir / "shieldgemma_2b_weight_verification.json",
        ),
        (
            "shieldgemma_9b",
            "google/shieldgemma-9b",
            "b8b636016df4540721a098c7aab91c97ec6ee508",
            "gemma",
            "softmax([Yes_logit, No_logit])[Yes]",
            args.prelock_dir / "shieldgemma_9b_tokenizer_registry.json",
            args.prelock_dir / "shieldgemma_9b_weight_verification.json",
        ),
    ]

    guards = {}
    for key, repo, revision, license_name, rule, tokenizer_path, weights_path in specs:
        tokenizer = json.loads(tokenizer_path.read_text("utf-8"))
        weights = json.loads(weights_path.read_text("utf-8"))
        if tokenizer["repo"] != repo or tokenizer["revision"] != revision:
            raise ValueError(f"tokenizer registry mismatch for {key}")
        if weights["repo"] != repo or weights["revision"] != revision or not weights["pass"]:
            raise ValueError(f"weight verification mismatch/failure for {key}")
        guards[key] = {
            "repo": repo,
            "revision": revision,
            "license": license_name,
            "probability_rule": rule,
            "tokenizer_registry": artifact(tokenizer_path),
            "weight_verification": artifact(weights_path),
            "chat_template_sha256": tokenizer["chat_template_sha256"],
            "verbalizers": tokenizer["verbalizers"],
        }

    payload = {
        "schema": "pccd.labelsource.guard_registry.v1",
        "status": "FROZEN_BEFORE_DISTRIBUTION_SANITY",
        "guards": guards,
        "sanity_cases": artifact(args.cases),
        "guard_score_implementation": artifact(args.guard_score),
        "formal_outcomes_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite frozen registry: {args.out}")
    args.out.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps({"out": str(args.out), "sha256": sha256_file(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
