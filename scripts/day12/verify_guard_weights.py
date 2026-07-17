#!/usr/bin/env python3
"""Verify downloaded guard shards against official pinned LFS metadata."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text("utf-8"))
    model_dir = Path(manifest["model_dir"])
    results = []
    for expected in manifest["files"]:
        path = model_dir / expected["filename"]
        if not path.is_file():
            results.append({**expected, "exists": False, "pass": False})
            continue
        size = path.stat().st_size
        observed_hash = sha256_file(path) if size == expected["size"] else None
        results.append(
            {
                **expected,
                "exists": True,
                "observed_size": size,
                "observed_sha256": observed_hash,
                "pass": size == expected["size"]
                and observed_hash == expected["sha256"],
            }
        )
    payload = {
        "schema": "pccd.labelsource.guard_weight_verification.v1",
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "source_manifest": str(args.manifest),
        "files": results,
        "pass": all(row["pass"] for row in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
