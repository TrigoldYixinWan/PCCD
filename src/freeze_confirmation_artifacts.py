#!/usr/bin/env python3
"""Freeze pre-unseal confirmation artifacts and the execution environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


LOCKED_PACKAGES = (
    "accelerate",
    "datasets",
    "numpy",
    "peft",
    "probmetrics",
    "scipy",
    "torch",
    "transformers",
    "trl",
    "vllm",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
    ).strip()


def collect_files(paths: list[Path]) -> list[Path]:
    files: dict[str, Path] = {}
    for original in paths:
        path = original.resolve()
        if not path.exists():
            raise FileNotFoundError(f"required freeze input is missing: {path}")
        candidates = [path] if path.is_file() else sorted(
            child for child in path.rglob("*") if child.is_file()
        )
        if not candidates:
            raise ValueError(f"freeze input contains no files: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            files[str(resolved)] = resolved
    return [files[key] for key in sorted(files)]


def relative_manifest_path(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path, manifest.parent)).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata_out", type=Path, required=True)
    parser.add_argument("--path", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    metadata_out = args.metadata_out.resolve()
    if out == metadata_out:
        raise ValueError("hash manifest and metadata output must differ")
    if out.exists() or metadata_out.exists():
        raise FileExistsError("refusing to overwrite a pre-unseal freeze artifact")
    if git_output(repo, "status", "--porcelain"):
        raise RuntimeError("repository must be clean before the pre-unseal freeze")

    packages = {}
    for package in LOCKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata = {
        "schema": "pccd.confirmation.preunseal_environment.v1",
        "frozen_before_aggregate_metrics": True,
        "git": {
            "commit": git_output(repo, "rev-parse", "HEAD"),
            "branch": git_output(repo, "branch", "--show-current"),
            "status_porcelain": "",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": packages,
        },
        "protocol": "reports/PREREG_CONFIRMATION.md",
        "requested_paths": [str(path.resolve()) for path in args.path],
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    files = collect_files([*args.path, metadata_out])
    if out in files:
        raise ValueError("hash manifest cannot include itself")
    lines = [
        f"{sha256_file(path)}  {relative_manifest_path(path, out)}\n"
        for path in files
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "manifest": str(out),
                "manifest_sha256": sha256_file(out),
                "metadata": str(metadata_out),
                "files": len(files),
                "git_commit": metadata["git"]["commit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
