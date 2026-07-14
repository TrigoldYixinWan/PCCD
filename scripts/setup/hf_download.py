#!/usr/bin/env python
"""Resumable Hub snapshot download with fresh Xet read tokens.

AutoDL's academic proxy caches the public Xet token route beyond the token's
expiration. Add ``Cache-Control: no-cache`` only to token refresh requests;
all snapshot selection, hashing, resumption, and file transfer remain in the
official ``huggingface_hub`` implementation.
"""
from __future__ import annotations

import argparse

import huggingface_hub.utils._xet as xet_utils
from huggingface_hub import snapshot_download


def _install_xet_token_refresh() -> None:
    original = xet_utils._fetch_xet_connection_info_with_url

    def fetch_no_cache(url, headers, params=None):
        fresh_headers = dict(headers)
        fresh_headers["Cache-Control"] = "no-cache"
        return original(url, fresh_headers, params)

    xet_utils._fetch_xet_connection_info_with_url = fetch_no_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"),
                        default="model")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    _install_xet_token_refresh()
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
