"""Shared Qwen2.5 multi-policy critic components for the frozen D0 critic.

The locked architecture is one shared causal-LM backbone plus ten independent
Linear -> GELU -> Linear heads over the last non-padding token.  Each head emits
three logits in the registered label order: satisfied, violated, not_applicable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from src.policy_defs import LABEL_STATES, POLICY_IDS, build_messages

NUM_POLICIES = len(POLICY_IDS)
NUM_LABELS = len(LABEL_STATES)
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_STATES)}
ID_TO_LABEL = dict(enumerate(LABEL_STATES))


@dataclass
class CriticOutput:
    logits: torch.Tensor
    pooled_hidden: torch.Tensor
    loss: torch.Tensor | None = None
    per_policy_loss: torch.Tensor | None = None


class PolicyClassificationHead(nn.Module):
    """The locked lightweight per-policy 3-way MLP head."""

    def __init__(self, backbone_hidden_size: int, mlp_hidden_size: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(backbone_hidden_size, mlp_hidden_size)
        self.activation = nn.GELU()
        self.out = nn.Linear(mlp_hidden_size, NUM_LABELS)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation(self.proj(hidden)))


def last_non_pad_hidden(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool the final mask-positive token, robust to either padding side."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, sequence, hidden]")
    if attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("attention_mask must match [batch, sequence]")
    mask = attention_mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every example must contain at least one non-padding token")
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    last_positions = positions.masked_fill(~mask, -1).max(dim=1).values
    batch_positions = torch.arange(mask.shape[0], device=mask.device)
    return last_hidden_state[batch_positions, last_positions]


def multihead_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-weight sum of ten 3-way CEs; N/A (class 2) is never masked."""
    if logits.ndim != 3 or logits.shape[1:] != (NUM_POLICIES, NUM_LABELS):
        raise ValueError(
            f"logits must have shape [batch,{NUM_POLICIES},{NUM_LABELS}], got {tuple(logits.shape)}"
        )
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels must have shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}"
        )
    if labels.dtype != torch.long:
        labels = labels.long()
    if torch.any((labels < 0) | (labels >= NUM_LABELS)):
        raise ValueError("labels must contain only registered 3-way class indices")
    per_policy = torch.stack(
        [F.cross_entropy(logits[:, index, :], labels[:, index]) for index in range(NUM_POLICIES)]
    )
    return per_policy.sum(), per_policy


class MultiPolicyCritic(nn.Module):
    """Shared backbone with independent H1-H5/S1-S3/T1-T2 classification heads."""

    def __init__(
        self,
        backbone: nn.Module,
        backbone_hidden_size: int | None = None,
        head_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone_hidden_size or getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("backbone hidden size is unavailable; pass backbone_hidden_size")
        self.backbone_hidden_size = int(hidden_size)
        self.head_hidden_size = int(head_hidden_size)
        self.heads = nn.ModuleDict(
            {
                policy_id: PolicyClassificationHead(self.backbone_hidden_size, self.head_hidden_size)
                for policy_id in POLICY_IDS
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> CriticOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **backbone_kwargs,
        )
        pooled = last_non_pad_hidden(outputs.last_hidden_state, attention_mask)
        logits = torch.stack([self.heads[policy_id](pooled) for policy_id in POLICY_IDS], dim=1)
        if labels is None:
            return CriticOutput(logits=logits, pooled_hidden=pooled)
        loss, per_policy = multihead_cross_entropy(logits, labels)
        return CriticOutput(
            logits=logits,
            pooled_hidden=pooled,
            loss=loss,
            per_policy_loss=per_policy,
        )


def load_labeled_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            labels = record.get("labels")
            if not record.get("parse_ok", labels is not None) or not isinstance(labels, dict):
                raise ValueError(f"{path}:{line_number}: record has no valid teacher labels")
            missing = [policy_id for policy_id in POLICY_IDS if labels.get(policy_id) not in LABEL_TO_ID]
            if missing:
                raise ValueError(f"{path}:{line_number}: invalid/missing labels for {missing}")
            if not isinstance(record.get("prompt"), str) or not isinstance(record.get("response"), str):
                raise ValueError(f"{path}:{line_number}: prompt/response must be strings")
            records.append(record)
    if not records:
        raise ValueError(f"no labeled records found in {path}")
    return records


class LabeledCriticDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"row_index": index, "record": self.records[index]}


class CriticBatchCollator:
    """Render the frozen canonical teacher prompt without a teacher answer."""

    def __init__(self, tokenizer: Any, max_length: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def _render(self, prompt: str, response: str) -> str:
        messages = build_messages(prompt, response)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        records = [example["record"] for example in examples]
        texts = [self._render(record["prompt"], record["response"]) for record in records]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        labels = torch.tensor(
            [
                [LABEL_TO_ID[record["labels"][policy_id]] for policy_id in POLICY_IDS]
                for record in records
            ],
            dtype=torch.long,
        )
        encoded["labels"] = labels
        encoded["row_indices"] = torch.tensor(
            [example["row_index"] for example in examples], dtype=torch.long
        )
        return encoded


def save_critic_checkpoint(
    critic: MultiPolicyCritic,
    tokenizer: Any,
    output_dir: str | Path,
    base_model: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save only LoRA adapter + ten heads, never a duplicate 7B base checkpoint."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not hasattr(critic.backbone, "save_pretrained"):
        raise TypeError("critic backbone must be a PEFT model with save_pretrained")
    critic.backbone.save_pretrained(output / "adapter", safe_serialization=True)
    torch.save(critic.heads.state_dict(), output / "heads.pt")
    tokenizer.save_pretrained(output / "tokenizer")
    config = {
        "checkpoint_version": 1,
        "base_model": str(base_model),
        "backbone_hidden_size": critic.backbone_hidden_size,
        "head_hidden_size": critic.head_hidden_size,
        "policy_ids": POLICY_IDS,
        "label_states": LABEL_STATES,
        "pooling": "last_non_pad_token",
        "head_structure": "Linear-GELU-Linear",
        "metadata": metadata or {},
    }
    (output / "critic_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
