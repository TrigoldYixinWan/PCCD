#!/usr/bin/env python3
"""Generate one fixed-seed response per held-out test prompt from base or D3 policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    models = Path(os.environ.get("MODELS_DIR", "models"))
    outputs = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(models / "qwen7b"))
    parser.add_argument("--prompts", default=str(outputs / "labels" / "test.jsonl"))
    parser.add_argument("--adapter")
    parser.add_argument("--variant", required=True, choices=("base", "d3"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.variant == "d3") != bool(args.adapter):
        raise ValueError("D3 requires --adapter; base must not receive one")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite generation: {output}")
    records = [json.loads(line) for line in Path(args.prompts).open(encoding="utf-8")]
    if len(records) != 1000 or len({record["id"] for record in records}) != 1000:
        raise ValueError("fixed held-out prompt set must contain exactly 1000 unique test IDs")

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    if args.adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=8)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    messages = [[{"role": "user", "content": record["prompt"]}] for record in records]
    if args.adapter:
        from vllm.lora.request import LoRARequest

        generated = llm.chat(
            messages,
            sampling_params=sampling,
            lora_request=LoRARequest("d3_r8", 1, args.adapter),
        )
    else:
        generated = llm.chat(messages, sampling_params=sampling)
    if len(generated) != len(records):
        raise RuntimeError("vLLM returned a different number of generations")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record, request_output in zip(records, generated):
            candidate = request_output.outputs[0]
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "source": record["source"],
                        "prompt": record["prompt"],
                        "response": candidate.text,
                        "prompt_token_ids": list(request_output.prompt_token_ids),
                        "generated_token_ids": list(candidate.token_ids),
                        "finish_reason": candidate.finish_reason,
                        "policy_variant": args.variant,
                        "meta": {
                            "original_test_id": record["id"],
                            "generation": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": args.max_tokens,
                                "seed": args.seed,
                                "adapter": args.adapter,
                            },
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    total_tokens = sum(len(row.outputs[0].token_ids) for row in generated)
    print(
        f"variant={args.variant} prompts={len(records)} generated_tokens={total_tokens} "
        f"mean_tokens={total_tokens/len(records):.3f} out={output}"
    )


if __name__ == "__main__":
    main()
