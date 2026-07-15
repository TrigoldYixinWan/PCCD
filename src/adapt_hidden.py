#!/usr/bin/env python3
"""Train a locked G2 hidden-violation policy adapter.

D2/D4/D5 use chosen-only SFT at ranks 4/16/32. D6 uses DPO beta=0.1
at rank 16 on the same 512 frozen critic-blind PKU preference pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]


def parse_args() -> argparse.Namespace:
    models = Path(os.environ.get("MODELS_DIR", "models"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(models / "qwen7b"))
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--point", required=True, choices=("D2", "D4", "D5", "D6"))
    parser.add_argument("--method", required=True, choices=("sft", "dpo"))
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    args = parse_args()
    expected = {"D2": ("sft", 4), "D4": ("sft", 16), "D5": ("sft", 32), "D6": ("dpo", 16)}
    if (args.method, args.rank) != expected[args.point]:
        raise ValueError(f"locked {args.point} configuration is {expected[args.point]}")
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-4 if args.method == "sft" else 5e-5
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() > 1 and world_size == 1:
        raise RuntimeError("multiple visible GPUs require explicit one-process-per-GPU launch")
    global_micro = args.per_device_batch * world_size
    if args.effective_batch % global_micro:
        raise ValueError("effective batch must divide global micro-batch")
    gradient_accumulation = args.effective_batch // global_micro
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite adapter: {output}")
    pair_path = Path(args.pairs)
    pairs = [json.loads(line) for line in pair_path.open(encoding="utf-8") if line.strip()]
    if len(pairs) != 512 or len({row["id"] for row in pairs}) != 512:
        raise ValueError("locked hidden adaptation corpus must contain 512 unique pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset_rows = []
    for row in pairs:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if args.method == "sft":
            dataset_rows.append({"prompt": prompt, "completion": row["chosen"]})
        else:
            dataset_rows.append(
                {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}
            )
    dataset = Dataset.from_list(dataset_rows)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    common = dict(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    if args.method == "sft":
        training_args = SFTConfig(
            completion_only_loss=True,
            packing=False,
            shuffle_dataset=True,
            **common,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.05,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    else:
        training_args = DPOConfig(beta=args.beta, **common)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.05,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    result = trainer.train()
    trainer.save_model(str(output))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "tokenizer")
        metadata = {
            "point": args.point,
            "objective": "hidden_violation",
            "method": args.method.upper(),
            "base_model": str(Path(args.model).resolve()),
            "independent_from_frozen_critic": True,
            "pairs": {"path": str(pair_path.resolve()), "sha256": sha256_file(pair_path), "count": len(pairs)},
            "seed": args.seed,
            "lora": {
                "r": args.rank,
                "alpha": 2 * args.rank,
                "dropout": 0.05,
                "targets": LORA_TARGET_MODULES,
            },
            "dpo_beta": args.beta if args.method == "dpo" else None,
            "training": {
                "epochs": args.epochs,
                "global_step": trainer.state.global_step,
                "effective_batch": args.effective_batch,
                "per_device_batch": args.per_device_batch,
                "world_size": world_size,
                "gradient_accumulation": gradient_accumulation,
                "learning_rate": learning_rate,
                "max_length": args.max_len,
                "training_loss": result.training_loss,
                "log_history": trainer.state.log_history,
            },
        }
        (output / "adaptation_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(output), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
