#!/usr/bin/env python3
"""Inspect Llama Guard's fixed formatting prefix before the safety label.

This is a pre-lock interface diagnostic.  It uses only the already frozen
distribution-sanity cases and does not compute benchmark or reliability
metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chat_for(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(case["prompt"])},
        {"role": "assistant", "content": str(case["response"])},
    ]


def token_record(tokenizer: Any, token_id: int) -> dict[str, Any]:
    return {
        "id": int(token_id),
        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
        "decoded_repr": repr(tokenizer.decode([int(token_id)])),
    }


@torch.inference_mode()
def probe(
    model: Any,
    tokenizer: Any,
    text: str,
    safe_id: int,
    unsafe_id: int,
) -> dict[str, Any]:
    inputs = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    logits = model(**inputs).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=8)
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
    )[0, inputs["input_ids"].shape[-1] :].cpu()
    pair = torch.softmax(torch.stack([logits[safe_id], logits[unsafe_id]]), dim=0)
    return {
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "text_tail_repr": repr(text[-120:]),
        "p_safe_full_vocab": probs[safe_id].item(),
        "p_unsafe_full_vocab": probs[unsafe_id].item(),
        "p_unsafe_two_token_sensitivity": pair[1].item(),
        "top_next_tokens": [
            {**token_record(tokenizer, token_id), "probability": probability.item()}
            for probability, token_id in zip(top_probs.cpu(), top_ids.cpu())
        ],
        "generated_ids": [int(value) for value in generated],
        "generated_tokens": [
            token_record(tokenizer, int(value)) for value in generated
        ],
        "generated_text_repr": repr(tokenizer.decode(generated)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {args.out}")

    cases_payload = json.loads(args.cases.read_text("utf-8"))
    cases = cases_payload["cases"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    safe_ids = tokenizer.encode("safe", add_special_tokens=False)
    unsafe_ids = tokenizer.encode("unsafe", add_special_tokens=False)
    if len(safe_ids) != 1 or len(unsafe_ids) != 1:
        raise ValueError(f"non-single-token labels: safe={safe_ids}, unsafe={unsafe_ids}")
    safe_id, unsafe_id = safe_ids[0], unsafe_ids[0]

    rows = []
    for case in cases:
        chat = chat_for(case)
        official_text = tokenizer.apply_chat_template(chat, tokenize=False)
        current_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        variants = {
            "official_card_default": official_text,
            "current_add_generation_prompt": current_text,
            "official_plus_two_newlines": official_text + "\n\n",
        }
        rows.append(
            {
                "case_id": case["id"],
                "stratum": case["stratum"],
                "official_equals_current": official_text == current_text,
                "variants": {
                    name: probe(model, tokenizer, text, safe_id, unsafe_id)
                    for name, text in variants.items()
                },
            }
        )

    output = {
        "schema": "pccd.labelsource.llama_guard_token_diagnostic.v1",
        "status": "PRELOCK_INTERFACE_DIAGNOSTIC_ONLY",
        "explicitly_not_computed": [
            "accuracy",
            "ECE",
            "AUROC",
            "F1",
            "reliability",
            "threshold selection",
            "guard ranking",
        ],
        "cases_sha256": sha256_file(args.cases),
        "model_path": str(args.model),
        "label_tokens": {
            "safe": token_record(tokenizer, safe_id),
            "unsafe": token_record(tokenizer, unsafe_id),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
