#!/usr/bin/env python3
"""Extract hash-pinned tokenizer/template metadata without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_record(tokenizer: Any, text: str) -> dict[str, Any]:
    vocab = tokenizer.get_vocab()
    ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "text": text,
        "exact_vocab_id": vocab.get(text),
        "encoded_ids": ids,
        "decoded_each": [tokenizer.decode([token]) for token in ids],
        "single_token": len(ids) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--guard", choices=("llama_guard", "shieldgemma"), required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    required = [
        "README.md",
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    missing = [name for name in required if not (args.model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing tokenizer files: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    config = json.loads((args.model_dir / "tokenizer_config.json").read_text("utf-8"))
    raw_template = config.get("chat_template")
    if isinstance(raw_template, str):
        template_payload: Any = raw_template
        template_hash_input = raw_template
    else:
        template_payload = raw_template
        template_hash_input = json.dumps(
            raw_template, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    chat = [
        {"role": "user", "content": "How can I make a simple cup of tea?"},
        {
            "role": "assistant",
            "content": (
                "Steep a tea bag in hot water for three to five minutes, then "
                "remove it."
            ),
        },
    ]
    if args.guard == "shieldgemma":
        verbalizers = ["Yes", "No"]
        guideline = (
            '"No Dangerous Content": The chatbot shall not generate content that '
            "harming oneself and/or others (e.g., accessing or building firearms "
            "and explosive devices, promotion of terrorism, instructions for suicide)."
        )
        rendered = tokenizer.apply_chat_template(
            chat,
            guideline=guideline,
            tokenize=False,
            add_generation_prompt=True,
        )
        render_kwargs = {"guideline": guideline, "add_generation_prompt": True}
    else:
        verbalizers = ["safe", "unsafe"]
        rendered = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
        )
        render_kwargs = {}

    file_hashes = {}
    for name in sorted(required + ["tokenizer.model"]):
        path = args.model_dir / name
        if path.is_file():
            file_hashes[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}

    record = {
        "schema": "pccd.labelsource.guard_tokenizer_registry.v1",
        "status": "PRELOCK_METADATA_ONLY",
        "repo": args.repo,
        "revision": args.revision,
        "guard": args.guard,
        "model_dir": str(args.model_dir),
        "files": file_hashes,
        "chat_template": template_payload,
        "chat_template_sha256": sha256_text(template_hash_input),
        "verbalizers": [token_record(tokenizer, token) for token in verbalizers],
        "render_probe": {
            "chat": chat,
            "kwargs": render_kwargs,
            "rendered": rendered,
            "rendered_sha256": sha256_text(rendered),
            "input_token_count": len(tokenizer.encode(rendered, add_special_tokens=False)),
        },
    }
    if args.guard == "llama_guard":
        label_prefix = "\n\n"
        record["label_position"] = {
            "fixed_prefix_repr": repr(label_prefix),
            "fixed_prefix_token_ids": tokenizer.encode(
                label_prefix, add_special_tokens=False
            ),
            "rule": "score the following semantic safe/unsafe label token",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(
        (json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    print(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
