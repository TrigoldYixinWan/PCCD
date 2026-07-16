#!/usr/bin/env python3
"""Pre-specified non-gating Day-8 alternative-divergence analysis.

This consumes newly materialized per-token log-ratios on the exact frozen G2
responses.  It does not load any model.  The original KL/G3 verdict remains
primary and frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic_model import POLICY_IDS
from src.eval_critic import multiclass_ece
from src.fit_g3 import (
    BOOTSTRAP_SEED,
    LOCKED_G2_SHA256,
    POINTS,
    exact_permutation,
    load_labels_logits,
    point_fit,
    softmax,
)


PREDICTORS = ("kl_adapted_base", "chi2_adapted_base", "reverse_kl", "total_variation")
_BOOT_DATA: dict | None = None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        filename = filename.lstrip("* ")
        entries[str(Path(filename).resolve())] = digest
    return entries


def validate_manifest(path: Path, required: list[Path]) -> dict[str, str]:
    entries = parse_sha256_manifest(path)
    observed = {}
    for item in required:
        resolved = str(item.resolve())
        if resolved not in entries:
            raise ValueError(f"token artifact is absent from frozen hash manifest: {item}")
        digest = sha256(item)
        if digest != entries[resolved]:
            raise ValueError(f"token artifact hash changed: {item}")
        observed[str(item)] = digest
    return observed


def load_token_contributions(
    token_path: Path,
    reference_path: Path,
    expected_ids: list[str],
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    token_rows = read_jsonl(token_path)
    reference_rows = read_jsonl(reference_path)
    token_ids = [row["id"] for row in token_rows]
    if token_ids != expected_ids or token_ids != [row["id"] for row in reference_rows]:
        raise ValueError(f"token/reference/response IDs differ for {token_path}")
    counts = np.empty(len(token_rows), dtype=np.float64)
    contributions = np.empty((len(token_rows), len(PREDICTORS)), dtype=np.float64)
    max_sum_error = 0.0
    max_mean_error = 0.0
    for index, (row, reference) in enumerate(zip(token_rows, reference_rows)):
        ell = np.asarray(row["log_ratios"], dtype=np.float64)
        # compute_kl retains the model's singleton batch dimension in the
        # frozen token artifact.  Accept only that exact [1, tokens] shape
        # (or a future lossless [tokens] representation), then validate the
        # continuation length against both frozen records.
        if ell.ndim == 2 and ell.shape[0] == 1:
            ell = ell[0]
        if ell.ndim != 1 or len(ell) != int(row["tokens"]) or len(ell) != int(reference["tokens"]):
            raise ValueError(f"token count mismatch for {row['id']}")
        sum_error = abs(float(ell.sum()) - float(reference["log_ratio_sum"]))
        mean_error = abs(float(ell.mean()) - float(reference["log_ratio_mean"]))
        max_sum_error = max(max_sum_error, sum_error)
        max_mean_error = max(max_mean_error, mean_error)
        if sum_error > tolerance or mean_error > tolerance:
            raise RuntimeError(
                f"frozen reproduction failed for {row['id']}: sum={sum_error} mean={mean_error}"
            )
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                exp_pos = np.exp(ell)
                exp_neg = np.exp(-ell)
            except FloatingPointError as error:
                raise RuntimeError(f"non-finite likelihood ratio for {row['id']}") from error
        # Histories and observed tokens are sampled under P_adapt.  Importance
        # identities give chi2(P||Q)=E_P[exp(ell)]-1,
        # KL(Q||P)=E_P[-ell*exp(-ell)], and
        # TV(P,Q)=0.5*E_P[abs(1-exp(-ell))].
        contributions[index] = (
            float(ell.sum()),
            float(np.expm1(ell).sum()),
            float((-ell * exp_neg).sum()),
            float((0.5 * np.abs(1.0 - exp_neg)).sum()),
        )
        if not np.all(np.isfinite(contributions[index])):
            raise RuntimeError(f"non-finite divergence contribution for {row['id']}")
        counts[index] = len(ell)
    return counts, contributions, {
        "max_abs_sum_error": max_sum_error,
        "max_abs_mean_error": max_mean_error,
        "tolerance": tolerance,
    }


def _bootstrap_worker(index_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _BOOT_DATA is None:
        raise RuntimeError("bootstrap worker not initialized")
    labels = _BOOT_DATA["labels"]
    probabilities = _BOOT_DATA["probabilities"]
    counts = _BOOT_DATA["counts"]
    contributions = _BOOT_DATA["contributions"]
    ece = np.empty((len(index_chunk), len(POINTS) + 1, len(POLICY_IDS)), dtype=np.float64)
    predictors = np.empty((len(index_chunk), len(POINTS), len(PREDICTORS)), dtype=np.float64)
    for replicate, indices in enumerate(index_chunk):
        for d in range(len(POINTS) + 1):
            for p in range(len(POLICY_IDS)):
                ece[replicate, d, p] = multiclass_ece(
                    probabilities[d][indices, p], labels[d][indices, p], 15
                )
        for d in range(len(POINTS)):
            denominator = counts[d, indices].sum()
            predictors[replicate, d] = contributions[d, indices].sum(axis=0) / denominator
    return ece, predictors


def ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [math.nan, math.nan]


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    root = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2_dir", type=Path, default=root / "g2")
    parser.add_argument("--g2_analysis", type=Path, default=root / "results" / "g2_analysis.json")
    parser.add_argument("--g3_analysis", type=Path, default=root / "results" / "g3_scaling.json")
    parser.add_argument("--token_manifest", type=Path, default=root / "g2" / "kl_tokens.sha256")
    parser.add_argument("--out", type=Path, default=root / "results" / "day8_divergence.json")
    parser.add_argument("--plot", type=Path, default=root / "results" / "day8_divergence.png")
    parser.add_argument("--summary_plot", type=Path, default=Path("reports/figures/day8_divergence.png"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap != 10_000 or args.seed != BOOTSTRAP_SEED:
        raise ValueError("locked G3 protocol requires 10,000 replicates and seed 20260720")
    if sha256(args.g2_analysis) != LOCKED_G2_SHA256:
        raise ValueError("authoritative G2 artifact hash changed")

    token_paths = [args.g2_dir / f"{point}_kl_tokens.jsonl" for point in POINTS]
    token_hashes = validate_manifest(args.token_manifest, token_paths)
    labels, probabilities, ids = [], [], None
    input_hashes = {
        "g2_analysis": sha256(args.g2_analysis),
        "g3_analysis": sha256(args.g3_analysis),
        "token_manifest": sha256(args.token_manifest),
    }
    for point in ("D0",) + POINTS:
        label_path = args.g2_dir / f"{point}_teacher.jsonl"
        logit_path = args.g2_dir / f"{point}_logits.jsonl"
        point_ids, point_labels, point_logits = load_labels_logits(label_path, logit_path)
        if ids is None:
            ids = point_ids
        elif point_ids != ids:
            raise ValueError(f"{point} IDs differ from D0")
        labels.append(point_labels)
        probabilities.append(softmax(point_logits))
        input_hashes[f"{point}_labels"] = sha256(label_path)
        input_hashes[f"{point}_logits"] = sha256(logit_path)
    assert ids is not None

    count_rows, contribution_rows, reproduction = [], [], {}
    for point, token_path in zip(POINTS, token_paths):
        reference = args.g2_dir / f"{point}_kl_items.jsonl"
        counts, contributions, check = load_token_contributions(token_path, reference, ids)
        count_rows.append(counts)
        contribution_rows.append(contributions)
        reproduction[point] = check
        input_hashes[f"{point}_reference_items"] = sha256(reference)
    counts_array = np.asarray(count_rows)
    contributions_array = np.asarray(contribution_rows)
    predictor_values = contributions_array.sum(axis=1) / counts_array.sum(axis=1)[:, None]

    frozen_g3 = json.loads(args.g3_analysis.read_text(encoding="utf-8"))
    mean_delta = np.asarray(
        [frozen_g3["observed"]["mean_delta_ece"][point] for point in POINTS],
        dtype=np.float64,
    )
    frozen_kl = np.asarray(
        [frozen_g3["observed"]["kl_nats_per_token"][point] for point in POINTS],
        dtype=np.float64,
    )
    if not np.allclose(predictor_values[:, 0], frozen_kl, rtol=0.0, atol=1e-12):
        raise RuntimeError("per-token KL does not reproduce frozen G3 predictor")

    observed_results = {}
    for j, predictor in enumerate(PREDICTORS):
        values = predictor_values[:, j]
        fit = point_fit(values, mean_delta)
        permutation = exact_permutation(values, mean_delta)
        observed_results[predictor] = {
            "point_values": dict(zip(POINTS, values)),
            "fit": fit,
            "exact_permutation": permutation,
            "negative_point_count": int(np.sum(values < 0)),
        }

    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, len(ids), size=(args.bootstrap, len(ids)), dtype=np.int32)
    global _BOOT_DATA
    _BOOT_DATA = {
        "labels": labels_array,
        "probabilities": probabilities_array,
        "counts": counts_array,
        "contributions": contributions_array,
    }
    chunks = [chunk for chunk in np.array_split(bootstrap_indices, args.jobs) if len(chunk)]
    if args.jobs == 1:
        worker_results = [_bootstrap_worker(chunks[0])]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=min(args.jobs, len(chunks))) as pool:
            worker_results = pool.map(_bootstrap_worker, chunks)
    ece_boot = np.concatenate([result[0] for result in worker_results])
    predictor_boot = np.concatenate([result[1] for result in worker_results])
    delta_boot = ece_boot[:, 1:] - ece_boot[:, :1]
    mean_delta_boot = delta_boot.mean(axis=2)

    fit_boot = np.empty((args.bootstrap, len(PREDICTORS), 3), dtype=np.float64)
    for replicate in range(args.bootstrap):
        for j in range(len(PREDICTORS)):
            fitted = point_fit(predictor_boot[replicate, :, j], mean_delta_boot[replicate])
            fit_boot[replicate, j] = fitted["slope"], fitted["lodo_r2"], fitted["lodo_rmse"]
    kl_lodo_boot = fit_boot[:, 0, 1]
    for j, predictor in enumerate(PREDICTORS):
        result = observed_results[predictor]
        result["fit"]["slope_95ci"] = ci(fit_boot[:, j, 0])
        result["fit"]["lodo_r2_95ci"] = ci(fit_boot[:, j, 1])
        result["fit"]["lodo_rmse_95ci"] = ci(fit_boot[:, j, 2])
        result["lodo_r2_difference_vs_kl"] = result["fit"]["lodo_r2"] - observed_results[
            "kl_adapted_base"
        ]["fit"]["lodo_r2"]
        result["lodo_r2_difference_vs_kl_95ci"] = ci(fit_boot[:, j, 1] - kl_lodo_boot)
        result["negative_bootstrap_point_fraction"] = float(
            np.mean(predictor_boot[:, :, j] < 0)
        )

    result = {
        "status": "pre-specified non-gating mechanistic analysis; frozen KL/G3 verdict unchanged",
        "protocol": {
            "n_prompts": len(ids),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "model": "sqrt(predictor) OLS with free intercept",
            "holdout": "leave one D point out",
            "permutation_assignments": 720,
            "estimators_under_adapted_token_sampling": {
                "kl_adapted_base": "mean(ell)",
                "chi2_adapted_base": "mean(exp(ell)-1) = E_base[(p_adapt/p_base-1)^2]",
                "reverse_kl": "mean(-ell*exp(-ell))",
                "total_variation": "0.5*mean(abs(1-exp(-ell)))",
            },
            "input_sha256": input_hashes,
            "token_artifact_sha256": token_hashes,
            "reproduction": reproduction,
        },
        "mean_delta_ece": dict(zip(POINTS, mean_delta)),
        "predictors": observed_results,
        "frozen_g3_reference": {
            "verdict": frozen_g3["gate"]["verdict"],
            "kl_lodo_r2": frozen_g3["primary_all_six"]["lodo_r2"],
            "unchanged": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(result), indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    values = [observed_results[name]["fit"]["lodo_r2"] for name in PREDICTORS]
    lows = [
        values[j] - observed_results[name]["fit"]["lodo_r2_95ci"][0]
        for j, name in enumerate(PREDICTORS)
    ]
    highs = [
        observed_results[name]["fit"]["lodo_r2_95ci"][1] - values[j]
        for j, name in enumerate(PREDICTORS)
    ]
    axis.bar(np.arange(len(PREDICTORS)), values, yerr=np.asarray([lows, highs]), capsize=4)
    axis.axhline(frozen_g3["primary_all_six"]["lodo_r2"], color="tab:red", linestyle="--", label="Frozen KL G3")
    axis.set_xticks(np.arange(len(PREDICTORS)), ("KL", "chi-squared", "reverse KL", "TV"))
    axis.set_ylabel("LODO R² (95% prompt-bootstrap CI)")
    axis.set_title("Alternative divergence predictors (non-gating)")
    axis.legend()
    figure.tight_layout()
    for path in (args.plot, args.summary_plot):
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
    plt.close(figure)

    print(json.dumps({
        name: {
            "lodo_r2": observed_results[name]["fit"]["lodo_r2"],
            "lodo_r2_95ci": observed_results[name]["fit"]["lodo_r2_95ci"],
            "delta_vs_kl": observed_results[name]["lodo_r2_difference_vs_kl"],
            "delta_vs_kl_95ci": observed_results[name]["lodo_r2_difference_vs_kl_95ci"],
            "permutation_p": observed_results[name]["exact_permutation"]["p_one_sided"],
        }
        for name in PREDICTORS
    }, indent=2))
    print(f"result={args.out} plot={args.plot}")


if __name__ == "__main__":
    main()
