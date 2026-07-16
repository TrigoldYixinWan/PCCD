#!/usr/bin/env python3
"""Generate paired fixed-seed responses for any G2 policy variant."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path


def parse_args() -> argparse.Namespace:
    models = Path(os.environ.get("MODELS_DIR", "models"))
    outputs = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(models / "qwen7b"))
    parser.add_argument("--prompts", default=str(outputs / "labels" / "test.jsonl"))
    parser.add_argument("--adapter")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--system_prompt")
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--expected_count", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--gpu_mem_util", type=float, default=0.85)
    return parser.parse_args()


def build_response_record(
    record: dict,
    *,
    response: str,
    prompt_token_ids: list[int],
    base_prompt_token_ids: list[int],
    generated_token_ids: list[int],
    finish_reason: str | None,
    variant: str,
    adapter: str | None,
    system_prompt: str | None,
    max_tokens: int,
    seed: int,
) -> dict:
    inherited_meta = (
        dict(record["meta"]) if isinstance(record.get("meta"), dict) else {}
    )
    family_id = record.get(
        "family_id",
        inherited_meta.get("family_id", inherited_meta.get("query_family_id")),
    )
    return {
        "id": record["id"],
        "source": record["source"],
        "prompt": record["prompt"],
        "response": response,
        "family_id": family_id,
        "prompt_token_ids": prompt_token_ids,
        "base_prompt_token_ids": base_prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "finish_reason": finish_reason,
        "policy_variant": variant,
        "meta": {
            **inherited_meta,
            "original_prompt_id": record["id"],
            "generation": {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "seed": seed,
                "adapter": adapter,
                "system_prompt": system_prompt,
            },
        },
    }


def main() -> None:
    args = parse_args()
    if args.adapter and args.system_prompt:
        raise ValueError("an adapter and a system-prompt intervention cannot be combined")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite generation: {output}")
    records = [json.loads(line) for line in Path(args.prompts).open(encoding="utf-8")]
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("fixed held-out prompt set must contain unique IDs")
    if args.expected_count is not None and len(records) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} fixed prompts, observed {len(records)}"
        )

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    if args.adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    messages = []
    for record in records:
        conversation = []
        if args.system_prompt:
            conversation.append({"role": "system", "content": args.system_prompt})
        conversation.append({"role": "user", "content": record["prompt"]})
        messages.append(conversation)
    if args.adapter:
        from vllm.lora.request import LoRARequest

        generated = llm.chat(
            messages,
            sampling_params=sampling,
            lora_request=LoRARequest(args.variant, 1, args.adapter),
        )
    else:
        generated = llm.chat(messages, sampling_params=sampling)
    if len(generated) != len(records):
        raise RuntimeError("vLLM returned a different number of generations")

    tokenizer = llm.get_tokenizer()
    base_prompt_ids = []
    for record in records:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        # Transformers 5 returns a BatchEncoding by default.  Iterating it
        # yields field names rather than token IDs, so unwrap input_ids first.
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        base_prompt_ids.append(list(encoded))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record, request_output, base_ids in zip(records, generated, base_prompt_ids):
            candidate = request_output.outputs[0]
            handle.write(
                json.dumps(
                    build_response_record(
                        record,
                        response=candidate.text,
                        prompt_token_ids=list(request_output.prompt_token_ids),
                        base_prompt_token_ids=list(base_ids),
                        generated_token_ids=list(candidate.token_ids),
                        finish_reason=candidate.finish_reason,
                        variant=args.variant,
                        adapter=args.adapter,
                        system_prompt=args.system_prompt,
                        max_tokens=args.max_tokens,
                        seed=args.seed,
                    ),
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
