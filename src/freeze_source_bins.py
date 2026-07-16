#!/usr/bin/env python3
"""Freeze source-base-calib confidence-bin edges before confirmation outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.policy_defs import POLICY_IDS


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()
    if args.bins != 15:
        raise ValueError("locked confirmation sensitivity requires 15 bins")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite frozen bin edges: {args.out}")
    rows = [
        json.loads(line)
        for line in args.logits.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("source logits must contain non-empty unique IDs")
    quantiles = np.linspace(0.0, 1.0, args.bins + 1)
    edges = {}
    for policy in POLICY_IDS:
        raw = []
        for row in rows:
            if set(row.get("logits", {})) != set(POLICY_IDS):
                raise ValueError("source logits do not contain exactly ten policy heads")
            values = np.asarray(row["logits"][policy], dtype=np.float64)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid logits for {policy}")
            raw.append(values)
        confidence = softmax(np.stack(raw)).max(axis=1)
        policy_edges = np.quantile(confidence, quantiles, method="linear")
        policy_edges[0], policy_edges[-1] = 0.0, 1.0
        if np.any(np.diff(policy_edges) < 0):
            raise RuntimeError("quantile edges are not monotone")
        edges[policy] = policy_edges.tolist()
    payload = {
        "role": "SOURCE-BASE-CALIB",
        "frozen_before_confirmation_outcomes": True,
        "input": str(args.logits.resolve()),
        "input_sha256": sha256_file(args.logits),
        "n_items": len(rows),
        "bins": args.bins,
        "method": "per-policy top-confidence linear quantiles; endpoints fixed to 0 and 1",
        "edges": edges,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"out": str(args.out), "sha256": sha256_file(args.out)}, indent=2))


if __name__ == "__main__":
    main()
