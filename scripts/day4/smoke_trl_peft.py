#!/usr/bin/env python3
"""One-step TRL/PEFT compatibility smoke test for the Day-4 training gate.

This intentionally uses a randomly initialized tiny Qwen2 model and synthetic data.
It verifies library/API execution only; it does not define the D0 critic architecture
or any scientific training protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path

import peft
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    models_dir = os.environ.get("MODELS_DIR")
    out_root = os.environ.get("PCCD_OUT", "outputs")
    default_tokenizer = str(Path(models_dir) / "qwen32b") if models_dir else None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        default=default_tokenizer,
        required=default_tokenizer is None,
        help="Local tokenizer path (model weights are not loaded).",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(out_root) / "smoke" / "day4_trl_peft"),
    )
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def tiny_model(tokenizer: AutoTokenizer) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        use_cache=False,
    )
    return Qwen2ForCausalLM(config)


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def common_args(out_dir: Path) -> dict:
    return {
        "output_dir": str(out_dir),
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
        "gradient_checkpointing": False,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": False,
        "seed": 20260715,
        "data_seed": 20260715,
    }


def run_sft(tokenizer: AutoTokenizer, out_dir: Path) -> dict:
    dataset = Dataset.from_dict(
        {
            "text": [
                "User: Say hello in one word.\nAssistant: Hello!",
                "User: Name a primary color.\nAssistant: Blue.",
            ]
        }
    )
    args = SFTConfig(max_length=64, dataset_text_field="text", **common_args(out_dir))
    trainer = SFTTrainer(
        model=tiny_model(tokenizer),
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    result = trainer.train()
    if trainer.state.global_step != 1 or not math.isfinite(result.training_loss):
        raise RuntimeError(
            f"SFT smoke failed: step={trainer.state.global_step}, loss={result.training_loss}"
        )
    return {
        "global_step": trainer.state.global_step,
        "training_loss": result.training_loss,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }


def run_dpo(tokenizer: AutoTokenizer, out_dir: Path) -> dict:
    dataset = Dataset.from_dict(
        {
            "prompt": ["User: Say hello.\nAssistant:", "User: Give a safe greeting.\nAssistant:"],
            "chosen": [" Hello!", " Welcome!"],
            "rejected": [" Go away.", " I refuse to greet you."],
        }
    )
    args = DPOConfig(max_length=64, beta=0.1, **common_args(out_dir))
    trainer = DPOTrainer(
        model=tiny_model(tokenizer),
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(),
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    result = trainer.train()
    if trainer.state.global_step != 1 or not math.isfinite(result.training_loss):
        raise RuntimeError(
            f"DPO smoke failed: step={trainer.state.global_step}, loss={result.training_loss}"
        )
    return {
        "global_step": trainer.state.global_step,
        "training_loss": result.training_loss,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pre-Day-4 smoke test")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "purpose": "API compatibility only; not a D0 architecture or training protocol",
        "seed": args.seed,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "sft_lora": run_sft(tokenizer, out_dir / "sft"),
        "dpo_lora": run_dpo(tokenizer, out_dir / "dpo"),
        "verdict": "PASS",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
