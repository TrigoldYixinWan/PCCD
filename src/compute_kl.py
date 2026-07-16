#!/usr/bin/env python3
"""Estimate KL(adapted||base) on a G2 variant's held-out continuations.

Primary estimator (fixed for later D points): ancestral Monte Carlo under the
adapted policy, teacher-forced under both policies, aggregated as the token-weighted
mean log p_adapt(y_t|x,y_<t) - log p_base(y_t|x,y_<t).  No negative item is clipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    models = Path(os.environ.get("MODELS_DIR", "models"))
    outputs = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(models / "qwen7b"))
    parser.add_argument("--adapter")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--items_out")
    parser.add_argument(
        "--tokens_out",
        help="new JSONL containing the lossless per-token log-ratio vector for each item",
    )
    parser.add_argument(
        "--reference_items",
        help="frozen legacy *_kl_items.jsonl to reproduce before tokens_out is finalized",
    )
    parser.add_argument("--reproduction_tolerance", type=float, default=1e-6)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def continuation_log_probs(model, input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    continuation = input_ids[:, prompt_length:]
    predictive_logits = outputs.logits[:, prompt_length - 1 : -1, :].float()
    return torch.log_softmax(predictive_logits, dim=-1).gather(
        -1, continuation.unsqueeze(-1)
    ).squeeze(-1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("KL computation requires CUDA")
    rows = [json.loads(line) for line in Path(args.generations).open(encoding="utf-8")]
    variants = {row.get("policy_variant") for row in rows}
    if not rows or len(variants) != 1 or None in variants:
        raise ValueError("KL requires one non-empty, single-variant generation file")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate generation IDs")
    if any("base_prompt_token_ids" not in row for row in rows):
        raise ValueError("G2 KL requires stored base_prompt_token_ids")

    # Early G2 response artifacts were produced under Transformers 5, whose
    # apply_chat_template(tokenize=True) returns a BatchEncoding.  The
    # serializer stored its field names instead of input_ids.  Reconstruct the
    # deterministic user-only base context from the frozen prompt; responses
    # and adapted prompt token IDs are left untouched.
    invalid_base_ids = any(
        not isinstance(row["base_prompt_token_ids"], list)
        or not row["base_prompt_token_ids"]
        or not all(isinstance(token, int) for token in row["base_prompt_token_ids"])
        for row in rows
    )
    if invalid_base_ids:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        for row in rows:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            row["base_prompt_token_ids"] = list(encoded)

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).cuda().eval()
    else:
        model = base.cuda().eval()
    item_results = []
    token_results = []
    with torch.inference_mode():
        for row in tqdm(rows, desc="KL teacher-forcing"):
            adapted_prompt_ids = list(row["prompt_token_ids"])
            base_prompt_ids = list(row["base_prompt_token_ids"])
            generated_ids = list(row["generated_token_ids"])
            if not generated_ids:
                raise ValueError(f"empty adapted continuation for {row['id']}")
            available_prompt = args.max_length - len(generated_ids)
            if available_prompt < 1:
                raise ValueError(f"continuation exceeds max length for {row['id']}")
            adapted_prompt_ids = adapted_prompt_ids[-available_prompt:]
            base_prompt_ids = base_prompt_ids[-available_prompt:]
            adapted_ids = torch.tensor(
                [adapted_prompt_ids + generated_ids], dtype=torch.long, device="cuda"
            )
            base_ids = torch.tensor(
                [base_prompt_ids + generated_ids], dtype=torch.long, device="cuda"
            )
            adapted = continuation_log_probs(model, adapted_ids, len(adapted_prompt_ids))
            if args.adapter:
                with model.disable_adapter():
                    base_logp = continuation_log_probs(model, base_ids, len(base_prompt_ids))
            else:
                base_logp = continuation_log_probs(model, base_ids, len(base_prompt_ids))
            differences = (adapted - base_logp).double().cpu().numpy()
            item_results.append(
                {
                    "id": row["id"],
                    "tokens": len(generated_ids),
                    "log_ratio_sum": float(differences.sum()),
                    "log_ratio_mean": float(differences.mean()),
                    "adapted_logp_sum": float(adapted.double().sum().cpu()),
                    "base_logp_sum": float(base_logp.double().sum().cpu()),
                }
            )
            if args.tokens_out:
                token_results.append(
                    {
                        "id": row["id"],
                        "tokens": len(generated_ids),
                        "log_ratios": differences.tolist(),
                    }
                )

    reproduction = None
    if args.tokens_out:
        if not args.reference_items:
            raise ValueError("--tokens_out requires --reference_items for the frozen check")
        reference_rows = [
            json.loads(line)
            for line in Path(args.reference_items).open(encoding="utf-8")
            if line.strip()
        ]
        if [row["id"] for row in reference_rows] != [row["id"] for row in item_results]:
            raise RuntimeError("new token rows are not ID-aligned with frozen reference items")
        sum_errors = np.asarray(
            [
                abs(new["log_ratio_sum"] - old["log_ratio_sum"])
                for new, old in zip(item_results, reference_rows)
            ],
            dtype=np.float64,
        )
        mean_errors = np.asarray(
            [
                abs(new["log_ratio_mean"] - old["log_ratio_mean"])
                for new, old in zip(item_results, reference_rows)
            ],
            dtype=np.float64,
        )
        reproduction = {
            "reference_items": str(Path(args.reference_items).resolve()),
            "tolerance": args.reproduction_tolerance,
            "max_abs_sum_error": float(sum_errors.max(initial=0.0)),
            "max_abs_mean_error": float(mean_errors.max(initial=0.0)),
            "items_over_tolerance": int(
                np.sum(
                    (sum_errors > args.reproduction_tolerance)
                    | (mean_errors > args.reproduction_tolerance)
                )
            ),
        }
        if reproduction["items_over_tolerance"]:
            raise RuntimeError(
                "per-token recomputation failed frozen reproduction check: "
                + json.dumps(reproduction, sort_keys=True)
            )
        tokens_output = Path(args.tokens_out)
        if tokens_output.exists():
            raise FileExistsError(f"refusing to overwrite frozen token artifact: {tokens_output}")
        tokens_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = tokens_output.with_suffix(tokens_output.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in token_results:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            temporary.replace(tokens_output)
        finally:
            if temporary.exists():
                temporary.unlink()

    sums = np.asarray([row["log_ratio_sum"] for row in item_results], dtype=np.float64)
    counts = np.asarray([row["tokens"] for row in item_results], dtype=np.int64)
    primary = float(sums.sum() / counts.sum())
    prompt_mean = float(np.mean(sums / counts))
    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.bootstrap, dtype=np.float64)
    for replicate in range(args.bootstrap):
        indices = rng.integers(0, len(rows), size=len(rows))
        boot[replicate] = sums[indices].sum() / counts[indices].sum()
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    summary = {
        "estimator": "adapted-sample teacher-forced token-weighted mean log(p_adapt/p_base)",
        "sampling": "one ancestral continuation per fixed prompt; temperature=1, top_p=1",
        "units": "nats/token",
        "n_prompts": len(rows),
        "n_generated_tokens": int(counts.sum()),
        "variant": next(iter(variants)),
        "kl_adapted_base": primary,
        "kl_95ci_prompt_bootstrap": ci,
        "mean_item_log_ratio": prompt_mean,
        "negative_item_fraction": float(np.mean(sums / counts < 0)),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "model": str(Path(args.model).resolve()),
        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        "generations": str(Path(args.generations).resolve()),
        "per_token_artifact": str(Path(args.tokens_out).resolve()) if args.tokens_out else None,
        "frozen_reproduction": reproduction,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    items_output = (
        Path(args.items_out)
        if args.items_out
        else (None if args.tokens_out else output.with_suffix(".items.jsonl"))
    )
    if items_output is not None:
        with items_output.open("w", encoding="utf-8") as handle:
            for row in item_results:
                handle.write(json.dumps(row) + "\n")
    if not math.isfinite(primary):
        raise RuntimeError("non-finite KL estimate")
    print(json.dumps(summary, indent=2))
    if items_output is not None:
        print(f"items={items_output}")
    if args.tokens_out:
        print(f"tokens={args.tokens_out} reproduction={json.dumps(reproduction, sort_keys=True)}")


if __name__ == "__main__":
    main()
