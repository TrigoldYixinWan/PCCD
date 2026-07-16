#!/usr/bin/env python3
"""Mechanically reorder/subset JSONL rows to a frozen ID manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_manifest_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    if isinstance(payload, dict):
        ids = payload.get("ids")
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        ids = [row.get("id") for row in payload]
    else:
        ids = payload
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("manifest must contain a non-empty unique ordered ID list")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require_exhaustive",
        action="store_true",
        help="also reject input IDs that are absent from the manifest",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite subset: {args.out}")
    ids = load_manifest_ids(args.manifest)
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row.get("id"): row for row in rows}
    if None in by_id or len(by_id) != len(rows):
        raise ValueError("input contains missing or duplicate IDs")
    missing = [item for item in ids if item not in by_id]
    if missing:
        raise ValueError(f"input lacks {len(missing)} frozen manifest IDs")
    if args.require_exhaustive and set(by_id) != set(ids):
        raise ValueError("input IDs are not an exhaustive match to the frozen manifest")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        for item in ids:
            handle.write(json.dumps(by_id[item], ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "rows": len(ids),
                "input_sha256": sha256_file(args.input),
                "manifest_sha256": sha256_file(args.manifest),
                "output_sha256": sha256_file(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
