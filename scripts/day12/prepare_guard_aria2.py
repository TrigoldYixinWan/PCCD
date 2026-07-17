#!/usr/bin/env python3
"""Prepare a private aria2 input and official hash manifest for guard weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from huggingface_hub import HfApi, get_token, hf_hub_url


OFFICIAL_HOSTS = {"huggingface.co", "www.huggingface.co"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_url(repo: str, revision: str, filename: str, token: str) -> str:
    current = hf_hub_url(repo, filename, revision=revision)
    with httpx.Client(follow_redirects=False, timeout=60.0, trust_env=True) as client:
        for _ in range(8):
            parsed = urlparse(current)
            headers = {}
            if parsed.hostname in OFFICIAL_HOSTS:
                headers["Authorization"] = f"Bearer {token}"
            response = client.head(current, headers=headers)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise RuntimeError("redirect without Location")
                next_url = urljoin(current, location)
                if urlparse(next_url).hostname not in OFFICIAL_HOSTS:
                    return next_url
                current = next_url
                continue
            response.raise_for_status()
            raise RuntimeError("official endpoint did not issue an external signed URL")
    raise RuntimeError("too many redirects")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--aria-input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("no saved Hugging Face token")
    info = HfApi(token=token).model_info(
        args.repo, revision=args.revision, files_metadata=True
    )
    if info.sha != args.revision:
        raise ValueError(f"revision mismatch: expected {args.revision}, got {info.sha}")

    selected = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if "/" in name or not name.endswith(".safetensors"):
            continue
        if not name.startswith("model-"):
            continue
        lfs = sibling.lfs
        if lfs is None or not lfs.sha256 or not lfs.size:
            raise ValueError(f"missing official LFS metadata for {name}")
        selected.append(
            {
                "filename": name,
                "size": int(lfs.size),
                "sha256": str(lfs.sha256),
            }
        )
    selected.sort(key=lambda row: row["filename"])
    if not selected:
        raise ValueError("no root model safetensor shards found")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in selected:
        destination = args.model_dir / row["filename"]
        aria_control = Path(f"{destination}.aria2")
        if destination.exists() and destination.stat().st_size == row["size"]:
            observed_sha256 = sha256_file(destination)
            row["observed_sha256"] = observed_sha256
            if observed_sha256 == row["sha256"]:
                row["already_size_complete"] = True
                # A verified complete payload makes any leftover aria2 range
                # state stale.  Removing only that control file prevents a
                # later invocation from reopening a known-good shard.
                aria_control.unlink(missing_ok=True)
                continue
            if not aria_control.exists():
                raise ValueError(
                    f"size-complete hash mismatch without aria2 resume state: "
                    f"{destination}"
                )
        row["already_size_complete"] = False
        url = signed_url(args.repo, args.revision, row["filename"], token)
        lines.extend(
            [
                url,
                f"  dir={args.model_dir}",
                f"  out={row['filename']}",
            ]
        )

    args.aria_input.parent.mkdir(parents=True, exist_ok=True)
    args.aria_input.write_text("\n".join(lines) + ("\n" if lines else ""), "utf-8")
    os.chmod(args.aria_input, 0o600)
    manifest = {
        "schema": "pccd.labelsource.guard_weight_download.v1",
        "status": "PRELOCK_OFFICIAL_PINNED_TRANSFER",
        "repo": args.repo,
        "revision": args.revision,
        "resolved_revision": info.sha,
        "model_dir": str(args.model_dir),
        "files": selected,
        "total_bytes": sum(row["size"] for row in selected),
        "token_in_aria_input": False,
        "signed_url_hosts": sorted(
            {
                urlparse(lines[index]).hostname
                for index in range(0, len(lines), 3)
            }
        ),
    }
    args.manifest.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        json.dumps(
            {
                "repo": args.repo,
                "revision": info.sha,
                "files": len(selected),
                "queued": len(lines) // 3,
                "total_bytes": manifest["total_bytes"],
                "hosts": manifest["signed_url_hosts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
