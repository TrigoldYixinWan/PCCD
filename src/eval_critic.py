#!/usr/bin/env python3
"""Evaluate the frozen D0 critic and emit locked L3 plus P1 calibration metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.critic_model import (
    ID_TO_LABEL,
    LABEL_STATES,
    POLICY_IDS,
    CriticBatchCollator,
    LabeledCriticDataset,
    MultiPolicyCritic,
    load_labeled_jsonl,
)


def binary_violated_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positive = 1  # LABEL_STATES[1] == "violated"
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    fp = int(np.sum((y_true != positive) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred != positive)))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def l3_per_policy_f1(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Locked L3 F1: violated-positive, teacher-N/A rows excluded per policy."""
    if labels.shape != predictions.shape or labels.ndim != 2 or labels.shape[1] != len(POLICY_IDS):
        raise ValueError("labels/predictions must both have shape [items, 10]")
    result: dict[str, float] = {}
    na_class = LABEL_STATES.index("not_applicable")
    for index, policy_id in enumerate(POLICY_IDS):
        applicable = labels[:, index] != na_class
        if not np.any(applicable):
            raise ValueError(f"policy {policy_id} has no applicable items")
        result[policy_id] = binary_violated_f1(
            labels[applicable, index], predictions[applicable, index]
        )
    return result


def coefficient_of_variation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if mean <= 0.0:
        return math.nan
    return float(values.std(ddof=0) / mean)


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def multiclass_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    total = len(labels)
    result = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(selected):
            result += (selected.sum() / total) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(result)


def adaptive_multiclass_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ordered = np.argsort(confidence)
    result = 0.0
    for selected in np.array_split(ordered, min(n_bins, len(ordered))):
        if len(selected):
            result += (len(selected) / len(labels)) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(result)


def bootstrap_item_metric(
    n_items: int,
    metric: Callable[[np.ndarray], float],
    replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, n_items, size=n_items)
        values[replicate] = metric(indices)
    return values


def compute_l3_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    replicates: int = 10_000,
    seed: int = 20260715,
) -> dict:
    per_policy = l3_per_policy_f1(labels, predictions)
    f1_vector = np.asarray([per_policy[policy_id] for policy_id in POLICY_IDS])
    cv = coefficient_of_variation(f1_vector)

    cv_bootstrap = bootstrap_item_metric(
        len(labels),
        lambda indices: coefficient_of_variation(
            np.asarray(
                list(l3_per_policy_f1(labels[indices], predictions[indices]).values())
            )
        ),
        replicates,
        seed,
    )
    cv_ci = percentile_ci(cv_bootstrap)
    per_policy_ci = {}
    for policy_index, policy_id in enumerate(POLICY_IDS):
        def sampled_f1(indices: np.ndarray, index: int = policy_index) -> float:
            sampled_labels = labels[indices, index]
            sampled_predictions = predictions[indices, index]
            applicable = sampled_labels != LABEL_STATES.index("not_applicable")
            if not np.any(applicable):
                return math.nan
            return binary_violated_f1(
                sampled_labels[applicable], sampled_predictions[applicable]
            )

        per_policy_ci[policy_id] = list(
            percentile_ci(
                bootstrap_item_metric(len(labels), sampled_f1, replicates, seed + policy_index + 1)
            )
        )
    return {
        "definition": "violated-positive F1 on teacher-applicable items; N/A excluded",
        "per_policy_f1": per_policy,
        "per_policy_f1_95ci": per_policy_ci,
        "macro_mean_f1": float(f1_vector.mean()),
        "cv": cv,
        "cv_95ci": list(cv_ci),
        "threshold": 0.15,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "verdict": "PASS" if cv > 0.15 and cv_ci[0] > 0.15 else "FAIL",
    }


def compute_p1_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    replicates: int = 10_000,
    seed: int = 20260715,
    n_bins: int = 15,
) -> dict[str, dict]:
    result = {}
    for index, policy_id in enumerate(POLICY_IDS):
        policy_probs = probabilities[:, index, :]
        policy_labels = labels[:, index]
        ece = multiclass_ece(policy_probs, policy_labels, n_bins)
        adaptive = adaptive_multiclass_ece(policy_probs, policy_labels, n_bins)
        ece_boot = bootstrap_item_metric(
            len(labels),
            lambda rows: multiclass_ece(policy_probs[rows], policy_labels[rows], n_bins),
            replicates,
            seed + 100 + index,
        )
        adaptive_boot = bootstrap_item_metric(
            len(labels),
            lambda rows: adaptive_multiclass_ece(policy_probs[rows], policy_labels[rows], n_bins),
            replicates,
            seed + 200 + index,
        )
        result[policy_id] = {
            "ece": ece,
            "ece_95ci": list(percentile_ci(ece_boot)),
            "adaptive_ece": adaptive,
            "adaptive_ece_95ci": list(percentile_ci(adaptive_boot)),
            "n": int(len(labels)),
            "bins": n_bins,
        }
    return result


def save_reliability_diagrams(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    n_bins: int = 15,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index, (axis, policy_id) in enumerate(zip(axes.flat, POLICY_IDS)):
        policy_probs = probabilities[:, index, :]
        confidence = policy_probs.max(axis=1)
        correct = policy_probs.argmax(axis=1) == labels[:, index]
        xs, ys = [], []
        for bin_index in range(n_bins):
            if bin_index == n_bins - 1:
                selected = (confidence >= edges[bin_index]) & (confidence <= edges[bin_index + 1])
            else:
                selected = (confidence >= edges[bin_index]) & (confidence < edges[bin_index + 1])
            if np.any(selected):
                xs.append(float(confidence[selected].mean()))
                ys.append(float(correct[selected].mean()))
        axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        axis.plot(xs, ys, marker="o")
        axis.set_title(policy_id)
        axis.grid(alpha=0.2)
    figure.supxlabel("Mean confidence")
    figure.supylabel("Empirical accuracy")
    figure.suptitle("D0 base-test per-policy reliability (3-way, including N/A)")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    outputs = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(outputs / "critic" / "d0"))
    parser.add_argument("--data", default=str(outputs / "labels" / "test.jsonl"))
    parser.add_argument("--out", default=str(outputs / "critic" / "d0_test_logits.jsonl"))
    parser.add_argument("--metrics", default=str(outputs / "critic" / "d0_test_metrics.json"))
    parser.add_argument("--plot", default=str(outputs / "critic" / "d0_reliability.png"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def load_checkpoint(checkpoint: Path) -> tuple[MultiPolicyCritic, AutoTokenizer, dict]:
    config = json.loads((checkpoint / "critic_config.json").read_text(encoding="utf-8"))
    if config["policy_ids"] != POLICY_IDS or config["label_states"] != LABEL_STATES:
        raise ValueError("checkpoint policy/label schema differs from the locked registry")
    tokenizer_path = checkpoint / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    base = AutoModel.from_pretrained(
        config["base_model"],
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    base.config.use_cache = False
    backbone = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False)
    model = MultiPolicyCritic(
        backbone,
        backbone_hidden_size=config["backbone_hidden_size"],
        head_hidden_size=config["head_hidden_size"],
    )
    model.heads.load_state_dict(torch.load(checkpoint / "heads.pt", map_location="cpu", weights_only=True))
    return model, tokenizer, config


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    if torch.cuda.device_count() > 1 and accelerator.num_processes == 1:
        raise RuntimeError("multiple visible GPUs require one explicit process per GPU")
    checkpoint = Path(args.checkpoint)
    model, tokenizer, checkpoint_config = load_checkpoint(checkpoint)
    records = load_labeled_jsonl(args.data)
    loader = DataLoader(
        LabeledCriticDataset(records),
        batch_size=args.batch,
        shuffle=False,
        collate_fn=CriticBatchCollator(tokenizer, max_length=args.max_len),
    )
    model, loader = accelerator.prepare(model, loader)
    model.eval()
    gathered_rows, gathered_logits, gathered_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            row_indices = batch.pop("row_indices")
            labels = batch["labels"]
            logits = model(**batch).logits
            row_indices, logits, labels = accelerator.gather_for_metrics(
                (row_indices, logits, labels)
            )
            gathered_rows.append(row_indices.cpu())
            gathered_logits.append(logits.float().cpu())
            gathered_labels.append(labels.cpu())

    if accelerator.is_main_process:
        rows = torch.cat(gathered_rows).numpy()
        logits = torch.cat(gathered_logits).numpy()
        labels = torch.cat(gathered_labels).numpy()
        order = np.argsort(rows)
        rows, logits, labels = rows[order], logits[order], labels[order]
        if not np.array_equal(rows, np.arange(len(records))):
            raise RuntimeError("distributed evaluation did not return each test row exactly once")
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        predictions = logits.argmax(axis=-1)

        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row_index, record in enumerate(records):
                handle.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "source": record.get("source"),
                            "labels": record["labels"],
                            "logits": {
                                policy_id: logits[row_index, policy_index].tolist()
                                for policy_index, policy_id in enumerate(POLICY_IDS)
                            },
                            "predictions": {
                                policy_id: ID_TO_LABEL[int(predictions[row_index, policy_index])]
                                for policy_index, policy_id in enumerate(POLICY_IDS)
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        metrics = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_config": checkpoint_config,
            "data": str(Path(args.data).resolve()),
            "n_items": len(records),
            "l3": compute_l3_metrics(labels, predictions, args.bootstrap, args.seed),
            "p1_calibration": compute_p1_calibration(
                labels, probabilities, args.bootstrap, args.seed
            ),
        }
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        save_reliability_diagrams(probabilities, labels, args.plot)
        print(json.dumps(metrics["l3"], indent=2))
        print(f"logits={output_path} metrics={metrics_path} reliability={args.plot}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
