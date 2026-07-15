#!/usr/bin/env python3
"""Fit the single pre-registered D3 policy adapter (LoRA r=8) with SFT.

This policy model is an independent Qwen2.5-7B instance.  It never loads or
updates the frozen D0 critic.  The deterministic adaptation corpus is drawn only
from the Day-2 train split: 512 independently safe PKU responses and 512
high-scoring UltraFeedback responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from src.critic_model import load_labeled_jsonl

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]


def parse_args() -> argparse.Namespace:
    models = Path(os.environ.get("MODELS_DIR", "models"))
    outputs = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(models / "qwen7b"))
    parser.add_argument("--train", default=str(outputs / "labels" / "train.jsonl"))
    parser.add_argument("--heldout", default=str(outputs / "labels" / "test.jsonl"))
    parser.add_argument("--out", default=str(outputs / "policy" / "d3_lora_r8"))
    parser.add_argument("--per_source", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def deterministic_order(record: dict, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest()


def ultra_scores(record: dict) -> list[int] | None:
    keys = ("instruction_following", "truthfulness", "honesty", "helpfulness")
    try:
        return [int(record["meta"][key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return None


def select_adaptation_records(
    train_records: list[dict], heldout_records: list[dict], per_source: int, seed: int
) -> list[dict]:
    heldout_ids = {record["id"] for record in heldout_records}
    if any(record["id"] in heldout_ids for record in train_records):
        raise ValueError("adaptation train records overlap the fixed held-out prompt set")

    pku_safe = [
        record for record in train_records
        if record["source"] == "pku_saferlhf" and record.get("meta", {}).get("is_safe") is True
    ]
    ultra_high = []
    for record in train_records:
        if record["source"] != "ultrafeedback":
            continue
        scores = ultra_scores(record)
        if scores is not None and min(scores) >= 4:
            ultra_high.append(record)
    if len(pku_safe) < per_source or len(ultra_high) < per_source:
        raise ValueError(
            f"insufficient adaptation candidates: pku_safe={len(pku_safe)}, "
            f"ultra_all_ge4={len(ultra_high)}, requested={per_source} each"
        )
    selected = sorted(pku_safe, key=lambda row: deterministic_order(row, seed))[:per_source]
    selected += sorted(ultra_high, key=lambda row: deterministic_order(row, seed))[:per_source]
    if len({record["id"] for record in selected}) != 2 * per_source:
        raise ValueError("duplicate IDs in adaptation selection")
    return sorted(selected, key=lambda row: deterministic_order(row, seed + 1))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.device_count() > 1 and world_size == 1:
        raise RuntimeError("multiple visible GPUs require one explicit process per GPU")
    global_micro = args.per_device_batch * world_size
    if args.effective_batch % global_micro:
        raise ValueError("effective batch must divide per-device batch times WORLD_SIZE")
    gradient_accumulation = args.effective_batch // global_micro

    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing D3 adapter: {output}")
    train_records = load_labeled_jsonl(args.train)
    heldout_records = load_labeled_jsonl(args.heldout)
    selected = select_adaptation_records(
        train_records, heldout_records, args.per_source, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset_rows = []
    for record in selected:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        dataset_rows.append({"prompt": prompt_text, "completion": record["response"]})
    dataset = Dataset.from_list(dataset_rows)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    config = SFTConfig(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        completion_only_loss=True,
        packing=False,
        shuffle_dataset=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
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
        source_counts = {
            source: sum(record["source"] == source for record in selected)
            for source in ("pku_saferlhf", "ultrafeedback")
        }
        metadata = {
            "diagnostic_point": "D3",
            "method": "SFT",
            "base_model": str(Path(args.model).resolve()),
            "independent_from_critic": True,
            "seed": args.seed,
            "selection": {
                "split": str(Path(args.train).resolve()),
                "heldout": str(Path(args.heldout).resolve()),
                "heldout_overlap": 0,
                "source_counts": source_counts,
                "pku_rule": "meta.is_safe == true",
                "ultrafeedback_rule": "all four source ratings >= 4",
                "selected_ids": [record["id"] for record in selected],
            },
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "targets": LORA_TARGET_MODULES,
            },
            "training": {
                "epochs": args.epochs,
                "global_step": trainer.state.global_step,
                "effective_batch": args.effective_batch,
                "per_device_batch": args.per_device_batch,
                "world_size": world_size,
                "gradient_accumulation": gradient_accumulation,
                "learning_rate": args.learning_rate,
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
