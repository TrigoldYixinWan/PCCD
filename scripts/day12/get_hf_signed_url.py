#!/usr/bin/env python3
"""Resolve a pinned HF file to an expiring signed URL without exposing a token."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from huggingface_hub import get_token, hf_hub_url


OFFICIAL_HOSTS = {"huggingface.co", "www.huggingface.co"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("no saved Hugging Face token")
    current = hf_hub_url(args.repo, args.filename, revision=args.revision)
    terminal = None
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
                next_host = urlparse(next_url).hostname
                if next_host not in OFFICIAL_HOSTS:
                    terminal = next_url
                    break
                current = next_url
                continue
            response.raise_for_status()
            if parsed.hostname not in OFFICIAL_HOSTS:
                terminal = current
                break
            raise RuntimeError(
                "official resolve endpoint did not issue an external signed URL"
            )
    if terminal is None:
        raise RuntimeError("too many redirects without a signed URL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(terminal + "\n", encoding="utf-8")
    os.chmod(args.out, 0o600)
    parsed = urlparse(terminal)
    print(f"resolved_host={parsed.hostname} scheme={parsed.scheme}")


if __name__ == "__main__":
    main()
