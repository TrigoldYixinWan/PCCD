#!/usr/bin/env python3
"""CPU-only deterministic self-test for the pre-registered D0 implementation."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

from src.critic_model import (
    CriticBatchCollator,
    LabeledCriticDataset,
    MultiPolicyCritic,
    load_labeled_jsonl,
    multihead_cross_entropy,
)
from src.eval_critic import (
    adaptive_multiclass_ece,
    coefficient_of_variation,
    compute_l3_metrics,
    compute_p1_calibration,
    l3_per_policy_f1,
    multiclass_ece,
)
from src.train_d0 import LORA_TARGET_MODULES, build_optimizer


class TinyRandomBackbone(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, return_dict=True, **kwargs):
        hidden = self.projection(self.embedding(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


def main() -> None:
    torch.manual_seed(20260715)
    model = MultiPolicyCritic(TinyRandomBackbone(), head_hidden_size=8)
    input_ids = torch.randint(0, 32, (4, 7))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0],
        ]
    )
    labels = torch.randint(0, 3, (4, 10), dtype=torch.long)
    output = model(input_ids, attention_mask, labels=labels)
    assert output.logits.shape == (4, 10, 3)
    assert output.pooled_hidden.shape == (4, 16)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.per_policy_loss is not None and output.per_policy_loss.shape == (10,)
    direct_loss, direct_per_policy = multihead_cross_entropy(output.logits, labels)
    assert torch.allclose(output.loss, direct_loss)
    assert torch.allclose(output.per_policy_loss, direct_per_policy)
    output.loss.backward()
    assert all(parameter.grad is not None for parameter in model.heads.parameters())

    qwen_config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    tiny_qwen = get_peft_model(
        Qwen2Model(qwen_config),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ),
    )
    qwen_critic = MultiPolicyCritic(tiny_qwen, head_hidden_size=8)
    qwen_output = qwen_critic(input_ids, attention_mask, labels=labels)
    assert qwen_output.logits.shape == (4, 10, 3) and torch.isfinite(qwen_output.loss)
    assert all(
        "lora_" in name for name, parameter in qwen_critic.backbone.named_parameters()
        if parameter.requires_grad
    )
    optimizer = build_optimizer(qwen_critic, lora_lr=1e-4, head_lr=1e-3, weight_decay=0.01)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-3]

    truth = np.zeros((12, 10), dtype=np.int64)
    truth[1::3, :] = 1
    truth[2::3, :] = 2
    prediction = truth.copy()
    prediction[:6, 0] = 1
    prediction[:3, 1] = 0
    f1 = l3_per_policy_f1(truth, prediction)
    assert list(f1) == ["H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2"]
    assert all(np.isfinite(list(f1.values())))
    assert np.isfinite(coefficient_of_variation(np.asarray(list(f1.values()))))
    l3 = compute_l3_metrics(truth, prediction, replicates=50, seed=20260715)
    assert l3["bootstrap_replicates"] == 50 and len(l3["cv_95ci"]) == 2

    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    calibration_labels = np.array([0, 1])
    assert np.isclose(multiclass_ece(probabilities, calibration_labels, n_bins=2), 0.2)
    assert np.isclose(adaptive_multiclass_ece(probabilities, calibration_labels, n_bins=2), 0.2)
    probability_cube = np.tile(probabilities[:, None, :], (1, 10, 1))
    label_matrix = np.tile(calibration_labels[:, None], (1, 10))
    p1 = compute_p1_calibration(
        label_matrix, probability_cube, replicates=20, seed=20260715, n_bins=2
    )
    assert all(np.isclose(metrics["ece"], 0.2) for metrics in p1.values())

    canonical_batch = "not_available"
    models_dir = os.environ.get("MODELS_DIR")
    output_root = os.environ.get("PCCD_OUT")
    if models_dir and output_root:
        tokenizer_path = Path(models_dir) / "qwen7b"
        label_path = Path(output_root) / "labels" / "test.jsonl"
        if tokenizer_path.exists() and label_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            records = load_labeled_jsonl(label_path)[:2]
            batch = CriticBatchCollator(tokenizer, max_length=4096)(
                [LabeledCriticDataset(records)[index] for index in range(2)]
            )
            assert batch["labels"].shape == (2, 10)
            assert batch["input_ids"].shape[0] == 2 and batch["input_ids"].shape[1] <= 4096
            canonical_batch = str(tuple(batch["input_ids"].shape))
    print(
        "PASS cpu-self-test: dummy+tiny-Qwen2 head_shape=(4,10,3), seven-target LoRA, "
        "dual-lr optimizer, summed_3way_ce, backward, L3 F1/CV/bootstrap, "
        f"P1 ECE/adaptive-ECE/bootstrap, canonical_batch={canonical_batch}"
    )


if __name__ == "__main__":
    main()
