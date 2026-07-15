#!/usr/bin/env python3
"""Estimate KL(adapted||base) on D3-generated held-out continuations.

Primary estimator (fixed for later D points): ancestral Monte Carlo under the
adapted policy, teacher-forced under both policies, aggregated as the token-weighted
mean log p_adapt(y_t|x,y_<t) - log p_base(y_t|x,y_<t).  No negative item is clipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    models = Path(os.environ.get("MODELS_DIR", "models"))
    outputs = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(models / "qwen7b"))
    parser.add_argument("--adapter", default=str(outputs / "policy" / "d3_lora_r8"))
    parser.add_argument("--generations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--items_out")
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
    if len(rows) != 1000 or any(row.get("policy_variant") != "d3" for row in rows):
        raise ValueError("KL requires exactly the 1000 adapted-policy generations")

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).cuda().eval()
    item_results = []
    with torch.inference_mode():
        for row in tqdm(rows, desc="KL teacher-forcing"):
            prompt_ids = list(row["prompt_token_ids"])
            generated_ids = list(row["generated_token_ids"])
            if not generated_ids:
                raise ValueError(f"empty adapted continuation for {row['id']}")
            available_prompt = args.max_length - len(generated_ids)
            if available_prompt < 1:
                raise ValueError(f"continuation exceeds max length for {row['id']}")
            prompt_ids = prompt_ids[-available_prompt:]
            ids = torch.tensor([prompt_ids + generated_ids], dtype=torch.long, device="cuda")
            adapted = continuation_log_probs(model, ids, len(prompt_ids))
            with model.disable_adapter():
                base_logp = continuation_log_probs(model, ids, len(prompt_ids))
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
        "kl_adapted_base": primary,
        "kl_95ci_prompt_bootstrap": ci,
        "mean_item_log_ratio": prompt_mean,
        "negative_item_fraction": float(np.mean(sums / counts < 0)),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "model": str(Path(args.model).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "generations": str(Path(args.generations).resolve()),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    items_output = Path(args.items_out) if args.items_out else output.with_suffix(".items.jsonl")
    with items_output.open("w", encoding="utf-8") as handle:
        for row in item_results:
            handle.write(json.dumps(row) + "\n")
    if not math.isfinite(primary):
        raise RuntimeError("non-finite KL estimate")
    print(json.dumps(summary, indent=2))
    print(f"items={items_output}")


if __name__ == "__main__":
    main()
