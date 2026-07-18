#!/usr/bin/env python3
"""Outcome-free token-length audit for the locked formal input serialization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.guard_score import FORMAL_GUARDS, render_formal_task
from src.label_beavertails_qwen import load_blind_items, load_schema


def batched(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def token_lengths(tokenizer: Any, rendered: list[str]) -> list[int]:
    lengths: list[int] = []
    for batch in batched(rendered, 256):
        ids = tokenizer(
            batch, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]
        lengths.extend(map(len, ids))
    return lengths


def summary(lengths: list[int], context: int) -> dict[str, Any]:
    values = np.asarray(lengths)
    return {
        "count": len(lengths),
        "min": int(values.min()),
        "median": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
        "context": context,
        "over_context": int(np.sum(values > context)),
        "pass": bool(np.all(values <= context)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--blind-sha256", required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--qwen-schema", type=Path, required=True)
    parser.add_argument("--qwen-schema-sha256", required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--shield-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {args.out}")

    items = load_blind_items(args.blind, args.blind_sha256, 0, 1)
    schema = load_schema(args.qwen_schema, args.qwen_schema_sha256)
    result: dict[str, Any] = {
        "schema": "pccd.labelsource.input_length_preflight.v1",
        "status": "OUTCOME_FREE_PRE_FORMAL",
        "blind_sha256": args.blind_sha256,
        "interfaces": {},
    }

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_model, local_files_only=True, trust_remote_code=False
    )
    qwen_rendered = [
        qwen_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": schema["system_prompt"]},
                {
                    "role": "user",
                    "content": schema["user_template"].format(
                        prompt=item["prompt"], response=item["response"]
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in items
    ]
    result["interfaces"]["qwen32b"] = summary(
        token_lengths(qwen_tokenizer, qwen_rendered), 4096
    )
    del qwen_rendered, qwen_tokenizer

    for guard_id, model_path in (
        ("llama_guard_3_8b", args.llama_model),
        ("shieldgemma_2b", args.shield_model),
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        context = int(getattr(config, "max_position_embeddings", tokenizer.model_max_length))
        family = str(FORMAL_GUARDS[guard_id]["family"])
        for interface in FORMAL_GUARDS[guard_id]["interfaces"]:
            rendered = [
                render_formal_task(tokenizer, family, item, str(interface))
                for item in items
            ]
            result["interfaces"][f"{guard_id}:{interface}"] = summary(
                token_lengths(tokenizer, rendered), context
            )
        del tokenizer

    result["pass"] = all(cell["pass"] for cell in result["interfaces"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
