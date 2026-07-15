#!/usr/bin/env python3
"""Train the pre-registered frozen D0 multi-policy critic.

Launch with one process per GPU, for example:
  accelerate launch --num_processes 2 src/train_d0.py ...
or expose exactly one GPU.  Multiple visible GPUs with one process are rejected
to prevent Transformers' implicit DataParallel path.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from src.critic_model import (
    CriticBatchCollator,
    LabeledCriticDataset,
    MultiPolicyCritic,
    load_labeled_jsonl,
    save_critic_checkpoint,
)
from src.eval_critic import l3_per_policy_f1

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    models_dir = os.environ.get("MODELS_DIR")
    outputs = os.environ.get("PCCD_OUT", "outputs")
    default_model = str(Path(models_dir) / "qwen7b") if models_dir else None
    labels_dir = Path(outputs) / "labels"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model, required=default_model is None)
    parser.add_argument("--train", default=str(labels_dir / "train.jsonl"))
    parser.add_argument("--calib", default=str(labels_dir / "calib.jsonl"))
    parser.add_argument("--output", default=str(Path(outputs) / "critic" / "d0"))
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--effective_batch", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--head_hidden_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def assert_explicit_gpu_strategy(accelerator: Accelerator) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("D0 training requires CUDA/bf16")
    if torch.cuda.device_count() > 1 and accelerator.num_processes == 1:
        raise RuntimeError(
            "multiple GPUs are visible to one process; use `accelerate launch`/`torchrun` "
            "with one process per GPU, or set CUDA_VISIBLE_DEVICES to one GPU"
        )


def build_optimizer(
    model: MultiPolicyCritic,
    lora_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    lora_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = list(model.heads.parameters())
    if not lora_parameters:
        raise RuntimeError("no trainable LoRA parameters found")
    if not head_parameters or not all(parameter.requires_grad for parameter in head_parameters):
        raise RuntimeError("all ten classification heads must be fully trainable")
    return torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": lora_lr, "weight_decay": weight_decay},
            {"params": head_parameters, "lr": head_lr, "weight_decay": weight_decay},
        ]
    )


@torch.no_grad()
def calibration_macro_f1(
    model: MultiPolicyCritic,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> tuple[float, dict[str, float]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in dataloader:
        batch.pop("row_indices")
        labels = batch["labels"]
        logits = model(**batch).logits
        logits, labels = accelerator.gather_for_metrics((logits, labels))
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().cpu())
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    predictions = logits_np.argmax(axis=-1)
    per_policy = l3_per_policy_f1(labels_np, predictions)
    score = float(np.mean(list(per_policy.values())))
    model.train()
    return score, per_policy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    declared_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_micro_batch = args.per_device_batch * declared_world_size
    if args.effective_batch % global_micro_batch:
        raise ValueError("effective_batch must be divisible by per-device batch * process count")
    gradient_accumulation_steps = args.effective_batch // global_micro_batch
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    assert_explicit_gpu_strategy(accelerator)
    if accelerator.num_processes != declared_world_size:
        raise RuntimeError(
            f"launcher WORLD_SIZE={declared_world_size} but Accelerate initialized "
            f"{accelerator.num_processes} processes"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("locked D0 protocol requires bf16-capable GPUs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModel.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    base.config.use_cache = False
    peft_backbone = get_peft_model(
        base,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ),
    )
    model = MultiPolicyCritic(
        peft_backbone,
        head_hidden_size=args.head_hidden_size,
    )

    train_records = load_labeled_jsonl(args.train)
    calib_records = load_labeled_jsonl(args.calib)
    collator = CriticBatchCollator(tokenizer, max_length=args.max_len)
    train_loader = DataLoader(
        LabeledCriticDataset(train_records),
        batch_size=args.per_device_batch,
        shuffle=True,
        collate_fn=collator,
    )
    calib_loader = DataLoader(
        LabeledCriticDataset(calib_records),
        batch_size=args.per_device_batch,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = build_optimizer(model, args.lora_lr, args.head_lr, args.weight_decay)
    updates_per_epoch = math.ceil(len(train_records) / args.effective_batch)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer, train_loader, calib_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, calib_loader, scheduler
    )

    if accelerator.is_main_process:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"train={len(train_records)} calib={len(calib_records)} processes={accelerator.num_processes} "
            f"micro={args.per_device_batch} grad_accum={gradient_accumulation_steps} "
            f"effective_batch={args.effective_batch} steps={total_steps} trainable={trainable}/{total}"
        )

    best_score = -math.inf
    epochs_without_improvement = 0
    output_dir = Path(args.output)
    for epoch in range(1, args.epochs + 1):
        model.train()
        policy_loss_sum = torch.zeros(10, device=accelerator.device)
        batches = 0
        for batch in train_loader:
            batch.pop("row_indices")
            with accelerator.accumulate(model):
                outputs = model(**batch)
                assert outputs.loss is not None and outputs.per_policy_loss is not None
                accelerator.backward(outputs.loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            policy_loss_sum += outputs.per_policy_loss.detach()
            batches += 1

        loss_stats = accelerator.reduce(policy_loss_sum, reduction="sum")
        batch_stats = accelerator.reduce(torch.tensor(batches, device=accelerator.device), reduction="sum")
        calib_score, per_policy_f1 = calibration_macro_f1(model, calib_loader, accelerator)
        improved = calib_score > best_score
        if improved:
            best_score = calib_score
            epochs_without_improvement = 0
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                save_critic_checkpoint(
                    unwrapped,
                    tokenizer,
                    output_dir,
                    base_model=args.model,
                    metadata={
                        "seed": args.seed,
                        "epoch": epoch,
                        "calib_macro_f1": calib_score,
                        "calib_per_policy_f1": per_policy_f1,
                        "effective_batch": args.effective_batch,
                        "max_length": args.max_len,
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "lora_lr": args.lora_lr,
                        "head_lr": args.head_lr,
                    },
                )
            accelerator.wait_for_everyone()
        else:
            epochs_without_improvement += 1

        if accelerator.is_main_process:
            mean_losses = (loss_stats / batch_stats.clamp_min(1)).float().cpu().tolist()
            print(
                f"epoch={epoch} per_policy_train_loss={mean_losses} "
                f"calib_macro_f1={calib_score:.6f} per_policy_f1={per_policy_f1} "
                f"best={best_score:.6f} improved={improved}"
            )
        if epochs_without_improvement >= args.patience:
            if accelerator.is_main_process:
                print(f"early_stop epoch={epoch} patience={args.patience}")
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"D0 training complete; best checkpoint={output_dir} calib_macro_f1={best_score:.6f}")


if __name__ == "__main__":
    main()
