#!/usr/bin/env python3
"""Blind, one-call Qwen-32B annotation for the locked BeaverTails manifest.

This runner intentionally cannot read the private human-reference file.  It
performs no retry or semantic repair: malformed model output is retained as a
null label record for later complete-cell reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schema(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256.lower():
        raise ValueError(
            f"schema hash mismatch: expected {expected_sha256}, observed {observed}"
        )
    schema = json.loads(path.read_text("utf-8"))
    if schema.get("status") != "PRELOCK_FROZEN_BEFORE_ANNOTATION":
        raise ValueError("annotation schema is not marked pre-lock frozen")
    return schema


def validate_model_asset(model_path: Path, schema: dict[str, Any]) -> None:
    """Fail closed if the local model metadata differs from the frozen asset."""
    expected = schema["annotator"]["local_asset_fingerprints"]
    for relative_path, expected_sha256 in expected.items():
        candidate = model_path / relative_path
        if not candidate.is_file():
            raise FileNotFoundError(f"frozen model metadata is missing: {candidate}")
        observed = sha256_file(candidate)
        if observed != expected_sha256:
            raise ValueError(
                f"model asset hash mismatch for {relative_path}: "
                f"expected {expected_sha256}, observed {observed}"
            )


def load_blind_items(
    path: Path, expected_sha256: str, shard: int, num_shards: int
) -> list[dict[str, Any]]:
    observed = sha256_file(path)
    if observed != expected_sha256.lower():
        raise ValueError(
            f"blind manifest hash mismatch: expected {expected_sha256}, observed {observed}"
        )
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % num_shards != shard:
                continue
            row = json.loads(line)
            expected_fields = {"id", "pair_sha256", "prompt", "response"}
            if set(row) != expected_fields:
                raise ValueError(
                    f"blind-manifest field mismatch for {row.get('id')}: "
                    f"expected {sorted(expected_fields)}, observed {sorted(row)}"
                )
            rows.append(row)
    return rows


def strict_parse(text: str, ordered_keys: list[str]) -> dict[str, bool] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != set(ordered_keys):
        return None
    if any(type(payload[key]) is not bool for key in ordered_keys):
        return None
    return {key: payload[key] for key in ordered_keys}


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            item_id = str(row["id"])
            if item_id in completed:
                raise ValueError(f"duplicate completed id in output: {item_id}")
            completed.add(item_id)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--schema-sha256", required=True)
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted shard without regenerating completed IDs.",
    )
    args = parser.parse_args()
    if not (0 <= args.shard < args.num_shards):
        raise ValueError("invalid shard index")
    if args.out.exists() and not args.resume:
        raise FileExistsError("output exists; use --resume only for interruption recovery")

    schema = load_schema(args.schema, args.schema_sha256)
    validate_model_asset(args.model, schema)
    frozen_input_sha = schema["blindness"]["input_manifest_sha256"]
    if args.input_sha256.lower() != frozen_input_sha:
        raise ValueError("CLI input hash differs from the schema-frozen blind manifest hash")
    rows = load_blind_items(
        args.input_path, args.input_sha256, args.shard, args.num_shards
    )
    completed = load_completed(args.out) if args.resume else set()
    todo = [row for row in rows if str(row["id"]) not in completed]

    category_order = list(schema["category_order"])
    ordered_keys = category_order + ["is_safe"]
    if list(schema["output_schema"]) != ordered_keys:
        raise ValueError("output schema key order differs from category_order + is_safe")
    messages = [
        [
            {"role": "system", "content": schema["system_prompt"]},
            {
                "role": "user",
                "content": schema["user_template"].format(
                    prompt=row["prompt"], response=row["response"]
                ),
            },
        ]
        for row in todo
    ]

    from vllm import LLM, SamplingParams

    model = LLM(
        model=str(args.model),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        temperature=float(schema["annotator"]["temperature"]),
        max_tokens=int(schema["annotator"]["max_tokens"]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "x"
    with args.out.open(mode, encoding="utf-8") as output:
        for start in range(0, len(todo), args.batch_size):
            batch_rows = todo[start : start + args.batch_size]
            batch_messages = messages[start : start + args.batch_size]
            generated = model.chat(batch_messages, sampling)
            for row, result in zip(batch_rows, generated, strict=True):
                raw = result.outputs[0].text
                labels = strict_parse(raw, ordered_keys)
                record = {
                    "id": row["id"],
                    "pair_sha256": row.get("pair_sha256"),
                    "labels": labels,
                    "parse_ok": labels is not None,
                    "raw_output": raw,
                    "schema_sha256": args.schema_sha256.lower(),
                    "input_manifest_sha256": args.input_sha256.lower(),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
