#!/usr/bin/env python3
"""Run the locked G3 KL-to-calibration scaling analysis.

The implementation follows reports/PREREG_G3.md.  It never calls the teacher or
loads the critic checkpoint: all inputs are frozen labels, logits, and per-prompt
KL records produced by G2.
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
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.critic_model import LABEL_TO_ID, POLICY_IDS
from src.eval_critic import multiclass_ece


POINTS = ("D1", "D2", "D3_control", "D4", "D5", "D6")
HIDDEN_POINTS = ("D2", "D4", "D5")
BOOTSTRAP_SEED = 20260720
LOCKED_G2_SHA256 = "b7520277cb83d4c44e8808f977578ae8a90660841cd6f06998582d51c012d200"
_BOOT_DATA: dict | None = None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def load_labels_logits(labels_path: Path, logits_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    label_rows = read_jsonl(labels_path)
    logit_rows = read_jsonl(logits_path)
    label_ids = [row["id"] for row in label_rows]
    if label_ids != [row["id"] for row in logit_rows]:
        raise ValueError(f"ID mismatch: {labels_path} vs {logits_path}")
    labels = np.asarray(
        [[LABEL_TO_ID[row["labels"][policy]] for policy in POLICY_IDS] for row in label_rows],
        dtype=np.int8,
    )
    logits = np.asarray(
        [[row["logits"][policy] for policy in POLICY_IDS] for row in logit_rows],
        dtype=np.float64,
    )
    return label_ids, labels, logits


def load_kl(path: Path, expected_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = read_jsonl(path)
    if [row["id"] for row in rows] != expected_ids:
        raise ValueError(f"KL item IDs are not aligned: {path}")
    tokens = np.asarray([row["tokens"] for row in rows], dtype=np.float64)
    sums = np.asarray([row["log_ratio_sum"] for row in rows], dtype=np.float64)
    if np.any(tokens <= 0) or not np.all(np.isfinite(sums)):
        raise ValueError(f"invalid KL records: {path}")
    return tokens, sums


def transform_x(kl: np.ndarray, form: str) -> np.ndarray:
    kl = np.asarray(kl, dtype=np.float64)
    if form == "sqrt":
        return np.sqrt(np.maximum(kl, 0.0))
    if form == "linear":
        return kl
    if form == "log1p":
        return np.log1p(kl)
    raise ValueError(form)


def ols(x: np.ndarray, y: np.ndarray, *, intercept: bool = True) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x]) if intercept else x[:, None]
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual = y - predicted
    sse = float(residual @ residual)
    centered = y - y.mean()
    sst = float(centered @ centered)
    r2 = math.nan if sst == 0 else 1.0 - sse / sst
    return {
        "intercept": float(coefficients[0]) if intercept else 0.0,
        "slope": float(coefficients[-1]),
        "sse": sse,
        "r2": r2,
    }


def lodo(x: np.ndarray, y: np.ndarray, *, intercept: bool = True) -> dict:
    predictions = np.empty(len(y), dtype=np.float64)
    for held_out in range(len(y)):
        keep = np.arange(len(y)) != held_out
        fit = ols(x[keep], y[keep], intercept=intercept)
        predictions[held_out] = fit["intercept"] + fit["slope"] * x[held_out]
    errors = y - predictions
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "predictions": predictions,
        "r2": math.nan if denominator == 0 else 1.0 - float(errors @ errors) / denominator,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def aicc(n: int, k: int, sse: float) -> float | None:
    if n <= k + 1:
        return None
    safe_sse = max(sse, np.finfo(float).tiny)
    aic = n * math.log(safe_sse / n) + 2 * k
    return float(aic + 2 * k * (k + 1) / (n - k - 1))


def point_fit(kl: np.ndarray, y: np.ndarray, *, form: str = "sqrt", intercept: bool = True) -> dict:
    x = transform_x(kl, form)
    fit = ols(x, y, intercept=intercept)
    held = lodo(x, y, intercept=intercept)
    return {
        "form": form,
        "intercept_included": intercept,
        **fit,
        "lodo_r2": held["r2"],
        "lodo_mae": held["mae"],
        "lodo_rmse": held["rmse"],
        "lodo_predictions": held["predictions"],
        "aicc": aicc(len(y), 2 if intercept else 1, fit["sse"]),
    }


def pooled_fit(kl: np.ndarray, cells: np.ndarray) -> dict:
    """Policy fixed effects plus common sqrt(KL) slope, grouped LODO by D."""
    n_points, n_policies = cells.shape
    x = transform_x(kl, "sqrt")

    def design(point_indices: Iterable[int], include_slope: bool) -> tuple[np.ndarray, np.ndarray]:
        rows, outcomes = [], []
        for d in point_indices:
            for p in range(n_policies):
                row = np.zeros(n_policies + int(include_slope), dtype=np.float64)
                row[p] = 1.0
                if include_slope:
                    row[-1] = x[d]
                rows.append(row)
                outcomes.append(cells[d, p])
        return np.asarray(rows), np.asarray(outcomes)

    full_x, full_y = design(range(n_points), True)
    coefficients = np.linalg.lstsq(full_x, full_y, rcond=None)[0]
    predictions = np.empty_like(cells)
    baseline_predictions = np.empty_like(cells)
    for held_out in range(n_points):
        train = [d for d in range(n_points) if d != held_out]
        train_x, train_y = design(train, True)
        coef = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
        baseline_x, baseline_y = design(train, False)
        baseline_coef = np.linalg.lstsq(baseline_x, baseline_y, rcond=None)[0]
        predictions[held_out] = coef[:n_policies] + coef[-1] * x[held_out]
        baseline_predictions[held_out] = baseline_coef
    residual = cells - predictions
    baseline_residual = cells - baseline_predictions
    sse = float(np.sum(residual**2))
    baseline_sse = float(np.sum(baseline_residual**2))
    denominator = float(np.sum((cells - cells.mean()) ** 2))
    return {
        "slope": float(coefficients[-1]),
        "policy_intercepts": coefficients[:-1],
        "lodo_predictions": predictions,
        "lodo_r2": math.nan if denominator == 0 else 1.0 - sse / denominator,
        "lodo_rmse": float(np.sqrt(np.mean(residual**2))),
        "incremental_lodo_r2": math.nan if baseline_sse == 0 else 1.0 - sse / baseline_sse,
        "policy_only_sse": baseline_sse,
        "full_sse": sse,
    }


def exact_permutation(kl: np.ndarray, y: np.ndarray) -> dict:
    observed = point_fit(kl, y)
    observed_stat = observed["lodo_r2"] if observed["slope"] > 0 else -math.inf
    statistics = []
    for permutation in itertools.permutations(kl.tolist()):
        candidate = point_fit(np.asarray(permutation), y)
        statistics.append(candidate["lodo_r2"] if candidate["slope"] > 0 else -math.inf)
    values = np.asarray(statistics)
    return {
        "assignments": math.factorial(len(kl)),
        "observed_statistic": observed_stat,
        "p_one_sided": float(np.mean(values >= observed_stat)),
        "exceedances_including_ties": int(np.sum(values >= observed_stat)),
    }


def exact_positive_slope_test(kl: np.ndarray, y: np.ndarray) -> dict:
    """Exact one-sided slope test used only for the ten descriptive policy fits."""
    observed = point_fit(kl, y)["slope"]
    slopes = np.asarray([
        point_fit(np.asarray(permutation), y)["slope"]
        for permutation in itertools.permutations(kl.tolist())
    ])
    return {
        "assignments": math.factorial(len(kl)),
        "observed_slope": observed,
        "p_one_sided": float(np.mean(slopes >= observed)),
        "exceedances_including_ties": int(np.sum(slopes >= observed)),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (m - rank) * p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def _bootstrap_worker(index_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if _BOOT_DATA is None:
        raise RuntimeError("bootstrap worker was not initialized")
    labels = _BOOT_DATA["labels"]
    probabilities = _BOOT_DATA["probabilities"]
    kl_tokens = _BOOT_DATA["kl_tokens"]
    kl_sums = _BOOT_DATA["kl_sums"]
    n_rep = len(index_chunk)
    ece = np.empty((n_rep, len(POINTS) + 1, len(POLICY_IDS)), dtype=np.float64)
    kl = np.empty((n_rep, len(POINTS)), dtype=np.float64)
    negative = 0
    for replicate, indices in enumerate(index_chunk):
        for d in range(len(POINTS) + 1):
            for p in range(len(POLICY_IDS)):
                ece[replicate, d, p] = multiclass_ece(
                    probabilities[d][indices, p], labels[d][indices, p], n_bins=15
                )
        for d in range(len(POINTS)):
            kl[replicate, d] = float(
                kl_sums[d][indices].sum() / kl_tokens[d][indices].sum()
            )
            negative += int(kl[replicate, d] < 0)
    return ece, kl, negative


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


def ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else [math.nan, math.nan]


def parse_args() -> argparse.Namespace:
    root = Path(os.environ.get("PCCD_OUT", "outputs"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2_dir", type=Path, default=root / "g2")
    parser.add_argument("--g2_analysis", type=Path, default=root / "results" / "g2_analysis.json")
    parser.add_argument("--out", type=Path, default=root / "results" / "g3_scaling.json")
    parser.add_argument("--predictions", type=Path, default=root / "results" / "g3_predictions.jsonl")
    parser.add_argument("--plot", type=Path, default=root / "results" / "g3_scaling.png")
    parser.add_argument("--summary_plot", type=Path, default=Path("reports/figures/day6_g3_scaling.png"))
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != BOOTSTRAP_SEED:
        raise ValueError(f"locked seed is {BOOTSTRAP_SEED}")
    if args.bootstrap != 10_000:
        raise ValueError("locked analysis requires 10,000 bootstrap replicates")
    observed_g2_hash = sha256(args.g2_analysis)
    if observed_g2_hash != LOCKED_G2_SHA256:
        raise ValueError(
            f"authoritative G2 hash changed: {observed_g2_hash} != {LOCKED_G2_SHA256}"
        )

    paths: dict[str, Path] = {"authoritative_g2_analysis": args.g2_analysis}
    labels, probabilities, ids = [], [], None
    for point in ("D0",) + POINTS:
        label_path = args.g2_dir / f"{point}_teacher.jsonl"
        logit_path = args.g2_dir / f"{point}_logits.jsonl"
        paths[f"{point}_labels"] = label_path
        paths[f"{point}_logits"] = logit_path
        point_ids, point_labels, point_logits = load_labels_logits(label_path, logit_path)
        if ids is None:
            ids = point_ids
        elif point_ids != ids:
            raise ValueError(f"{point} prompt IDs differ from D0")
        labels.append(point_labels)
        probabilities.append(softmax(point_logits))
    assert ids is not None

    kl_tokens, kl_sums = [], []
    for point in POINTS:
        path = args.g2_dir / f"{point}_kl_items.jsonl"
        paths[f"{point}_kl_items"] = path
        tokens, sums = load_kl(path, ids)
        kl_tokens.append(tokens)
        kl_sums.append(sums)

    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    kl_tokens_array = np.asarray(kl_tokens)
    kl_sums_array = np.asarray(kl_sums)
    kl_observed = kl_sums_array.sum(axis=1) / kl_tokens_array.sum(axis=1)
    ece_observed = np.empty((len(POINTS) + 1, len(POLICY_IDS)), dtype=np.float64)
    all_indices = np.arange(len(ids))
    for d in range(len(POINTS) + 1):
        for p in range(len(POLICY_IDS)):
            ece_observed[d, p] = multiclass_ece(
                probabilities_array[d, all_indices, p], labels_array[d, all_indices, p], 15
            )
    delta_observed = ece_observed[1:] - ece_observed[0]
    mean_observed = delta_observed.mean(axis=1)
    frozen_g2 = json.loads(args.g2_analysis.read_text(encoding="utf-8"))
    for d, point in enumerate(POINTS):
        recorded_kl = frozen_g2["variants"][point]["kl"]["kl_adapted_base"]
        recorded_delta = frozen_g2["variants"][point]["aggregate"]["mean_delta_ece"]
        if not math.isclose(kl_observed[d], recorded_kl, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{point} recomputed KL differs from frozen G2")
        if not math.isclose(mean_observed[d], recorded_delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{point} recomputed mean Delta-ECE differs from frozen G2")

    primary = point_fit(kl_observed, mean_observed)
    permutation = exact_permutation(kl_observed, mean_observed)
    pooled = pooled_fit(kl_observed, delta_observed)
    hidden_indices = np.asarray([POINTS.index(point) for point in HIDDEN_POINTS])
    hidden = point_fit(kl_observed[hidden_indices], mean_observed[hidden_indices])

    sensitivities = {}
    for name, form, intercept in (
        ("linear", "linear", True),
        ("log1p", "log1p", True),
        ("sqrt_through_origin", "sqrt", False),
    ):
        sensitivities[name] = point_fit(kl_observed, mean_observed, form=form, intercept=intercept)

    per_policy = {}
    policy_p_values = {}
    for p, policy in enumerate(POLICY_IDS):
        fitted = point_fit(kl_observed, delta_observed[:, p])
        permuted = exact_positive_slope_test(kl_observed, delta_observed[:, p])
        policy_p_values[policy] = permuted["p_one_sided"]
        per_policy[policy] = {**fitted, "p_one_sided_exact": permuted["p_one_sided"]}
    adjusted = holm(policy_p_values)
    for policy in POLICY_IDS:
        per_policy[policy]["p_holm"] = adjusted[policy]

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, len(ids), size=(args.bootstrap, len(ids)), dtype=np.int32
    )
    global _BOOT_DATA
    _BOOT_DATA = {
        "labels": labels_array,
        "probabilities": probabilities_array,
        "kl_tokens": kl_tokens_array,
        "kl_sums": kl_sums_array,
    }
    chunks = [chunk for chunk in np.array_split(bootstrap_indices, args.jobs) if len(chunk)]
    if args.jobs == 1:
        worker_results = [_bootstrap_worker(chunks[0])]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=min(args.jobs, len(chunks))) as pool:
            worker_results = pool.map(_bootstrap_worker, chunks)
    ece_boot = np.concatenate([item[0] for item in worker_results], axis=0)
    kl_boot = np.concatenate([item[1] for item in worker_results], axis=0)
    negative_kl = sum(item[2] for item in worker_results)
    delta_boot = ece_boot[:, 1:] - ece_boot[:, :1]

    primary_boot = np.empty((args.bootstrap, 3), dtype=np.float64)
    hidden_boot = np.empty((args.bootstrap, 2), dtype=np.float64)
    pooled_boot = np.empty((args.bootstrap, 3), dtype=np.float64)
    per_policy_slope_boot = np.empty((args.bootstrap, len(POLICY_IDS)), dtype=np.float64)
    for replicate in range(args.bootstrap):
        means = delta_boot[replicate].mean(axis=1)
        fitted = point_fit(kl_boot[replicate], means)
        primary_boot[replicate] = fitted["slope"], fitted["lodo_r2"], fitted["lodo_rmse"]
        hfit = point_fit(kl_boot[replicate, hidden_indices], means[hidden_indices])
        hidden_boot[replicate] = hfit["slope"], hfit["lodo_r2"]
        pfit = pooled_fit(kl_boot[replicate], delta_boot[replicate])
        pooled_boot[replicate] = (
            pfit["slope"], pfit["lodo_r2"], pfit["incremental_lodo_r2"]
        )
        for p in range(len(POLICY_IDS)):
            per_policy_slope_boot[replicate, p] = point_fit(
                kl_boot[replicate], delta_boot[replicate, :, p]
            )["slope"]

    primary["slope_95ci"] = ci(primary_boot[:, 0])
    primary["lodo_r2_95ci"] = ci(primary_boot[:, 1])
    primary["lodo_rmse_95ci"] = ci(primary_boot[:, 2])
    hidden["slope_95ci"] = ci(hidden_boot[:, 0])
    hidden["lodo_r2_95ci"] = ci(hidden_boot[:, 1])
    pooled["slope_95ci"] = ci(pooled_boot[:, 0])
    pooled["lodo_r2_95ci"] = ci(pooled_boot[:, 1])
    pooled["incremental_lodo_r2_95ci"] = ci(pooled_boot[:, 2])
    for p, policy in enumerate(POLICY_IDS):
        per_policy[policy]["slope_95ci"] = ci(per_policy_slope_boot[:, p])

    primary_conditions = {
        "positive_slope_ci": primary["slope"] > 0 and primary["slope_95ci"][0] > 0,
        "lodo_r2_at_least_0_70": primary["lodo_r2"] >= 0.70,
        "permutation_p_at_most_0_05": permutation["p_one_sided"] <= 0.05,
    }
    primary_pass = all(primary_conditions.values())
    pooled_conditions = {
        "positive_slope": pooled["slope"] > 0,
        "incremental_lodo_r2_at_least_0_30": pooled["incremental_lodo_r2"] >= 0.30,
    }
    verdict = "FAIL"
    if primary_pass:
        verdict = "PASS" if all(pooled_conditions.values()) else "PARTIAL"

    result = {
        "protocol": {
            "preregistration": "reports/PREREG_G3.md (LOCKED 2026-07-16)",
            "points": list(POINTS),
            "hidden_sft_family": list(HIDDEN_POINTS),
            "n_prompts": len(ids),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
            "bootstrap_unit": "paired prompt; identical indices for D0 and D1-D6",
            "negative_kl_bootstrap_cells": negative_kl,
            "negative_kl_bootstrap_fraction": negative_kl / (args.bootstrap * len(POINTS)),
            "input_sha256": {key: sha256(path) for key, path in paths.items()},
        },
        "observed": {
            "kl_nats_per_token": dict(zip(POINTS, kl_observed)),
            "mean_delta_ece": dict(zip(POINTS, mean_observed)),
            "per_policy_delta_ece": {
                point: dict(zip(POLICY_IDS, delta_observed[d])) for d, point in enumerate(POINTS)
            },
        },
        "primary_all_six": primary,
        "exact_permutation": permutation,
        "hidden_sft_parallel": hidden,
        "cross_objective_heterogeneity": {
            "known_non_monotonic": bool(
                mean_observed[POINTS.index("D3_control")] > mean_observed[POINTS.index("D5")]
            ),
            "interpretation_rule": "hidden-family result cannot change the all-six primary verdict",
        },
        "pooled_policy_fixed_effects": pooled,
        "per_policy": per_policy,
        "sensitivities_non_gating": sensitivities,
        "gate": {
            "primary_conditions": primary_conditions,
            "pooled_conditions": pooled_conditions,
            "verdict": verdict,
        },
    }
    safe_result = to_jsonable(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(safe_result, indent=2) + "\n", encoding="utf-8")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as handle:
        for d, point in enumerate(POINTS):
            handle.write(json.dumps({
                "analysis": "primary_all_six_lodo",
                "point": point,
                "kl": kl_observed[d],
                "observed_mean_delta_ece": mean_observed[d],
                "predicted_mean_delta_ece": primary["lodo_predictions"][d],
                "residual": mean_observed[d] - primary["lodo_predictions"][d],
            }) + "\n")
        for j, d in enumerate(hidden_indices):
            handle.write(json.dumps({
                "analysis": "hidden_sft_parallel_lodo",
                "point": POINTS[d],
                "kl": kl_observed[d],
                "observed_mean_delta_ece": mean_observed[d],
                "predicted_mean_delta_ece": hidden["lodo_predictions"][j],
                "residual": mean_observed[d] - hidden["lodo_predictions"][j],
            }) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x_grid = np.linspace(0, max(kl_observed) * 1.05, 200)
    axes[0].scatter(kl_observed, mean_observed, color="tab:blue")
    axes[0].plot(x_grid, primary["intercept"] + primary["slope"] * np.sqrt(x_grid), color="tab:blue")
    for d, point in enumerate(POINTS):
        axes[0].annotate(point, (kl_observed[d], mean_observed[d]), xytext=(4, 4), textcoords="offset points")
    axes[0].set_title(f"All six (G3 {verdict}; LODO R²={primary['lodo_r2']:.2f})")
    axes[0].set_xlabel("KL(adapted || base), nats/token")
    axes[0].set_ylabel("Mean ΔECE")
    hidden_kl = kl_observed[hidden_indices]
    hidden_y = mean_observed[hidden_indices]
    hidden_grid = np.linspace(hidden_kl.min() * 0.98, hidden_kl.max() * 1.02, 200)
    axes[1].scatter(hidden_kl, hidden_y, color="tab:orange")
    axes[1].plot(hidden_grid, hidden["intercept"] + hidden["slope"] * np.sqrt(hidden_grid), color="tab:orange")
    for j, point in enumerate(HIDDEN_POINTS):
        axes[1].annotate(point, (hidden_kl[j], hidden_y[j]), xytext=(4, 4), textcoords="offset points")
    axes[1].set_title(f"Hidden-SFT family (descriptive; LODO R²={hidden['lodo_r2']:.2f})")
    axes[1].set_xlabel("KL(adapted || base), nats/token")
    axes[1].set_ylabel("Mean ΔECE")
    figure.tight_layout()
    for path in (args.plot, args.summary_plot):
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
    plt.close(figure)

    print(json.dumps(to_jsonable(result["gate"]), indent=2))
    print(f"primary slope={primary['slope']:.9f} CI={primary['slope_95ci']} LODO_R2={primary['lodo_r2']:.9f} p={permutation['p_one_sided']:.9f}")
    print(f"hidden slope={hidden['slope']:.9f} CI={hidden['slope_95ci']} LODO_R2={hidden['lodo_r2']:.9f}")
    print(f"outputs={args.out} predictions={args.predictions} plot={args.plot}")


if __name__ == "__main__":
    main()
