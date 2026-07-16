#!/usr/bin/env python3
"""Verify that the new D5 adapter differs from frozen D5 only by seed/outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LOCKED_NEW_SEED = 20260723
SETTING_PATHS = (
    ("point",),
    ("objective",),
    ("method",),
    ("base_model",),
    ("independent_from_frozen_critic",),
    ("pairs", "sha256"),
    ("pairs", "count"),
    ("lora", "r"),
    ("lora", "alpha"),
    ("lora", "dropout"),
    ("lora", "targets"),
    ("dpo_beta",),
    ("training", "epochs"),
    ("training", "global_step"),
    ("training", "effective_batch"),
    ("training", "per_device_batch"),
    ("training", "world_size"),
    ("training", "gradient_accumulation"),
    ("training", "learning_rate"),
    ("training", "max_length"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(payload: dict, path: tuple[str, ...]):
    value = payload
    for key in path:
        value = value[key]
    return value


def compare_metadata(old: dict, new: dict) -> dict:
    differences = []
    for path in SETTING_PATHS:
        old_value = nested(old, path)
        new_value = nested(new, path)
        if old_value != new_value:
            differences.append(
                {
                    "path": ".".join(path),
                    "old": old_value,
                    "new": new_value,
                }
            )
    checks = {
        "new_seed_is_locked": new.get("seed") == LOCKED_NEW_SEED,
        "seed_is_independent": new.get("seed") != old.get("seed"),
        "all_registered_settings_match": not differences,
        "locked_point_is_D5": new.get("point") == "D5",
        "locked_method_is_SFT": str(new.get("method", "")).upper() == "SFT",
        "locked_rank_is_32": new.get("lora", {}).get("r") == 32,
    }
    return {
        "checks": checks,
        "differences": differences,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite adapter validation: {args.out}")
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    result = compare_metadata(old, new)
    pairs_sha = sha256_file(args.pairs)
    result["checks"]["old_pair_hash_matches_file"] = (
        old.get("pairs", {}).get("sha256") == pairs_sha
    )
    result["checks"]["new_pair_hash_matches_file"] = (
        new.get("pairs", {}).get("sha256") == pairs_sha
    )
    result["pass"] = bool(result["pass"] and all(result["checks"].values()))
    result["input_sha256"] = {
        "old_metadata": sha256_file(args.old),
        "new_metadata": sha256_file(args.new),
        "pairs": pairs_sha,
    }
    result["locked_new_seed"] = LOCKED_NEW_SEED
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
