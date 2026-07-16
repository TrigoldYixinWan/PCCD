#!/usr/bin/env python3
"""Pure-CPU smoke tests for the P8 structured recalibration implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.fit_g4 import to_jsonable
import src.fit_g6 as fit_g6_module
from src.fit_g6 import (
    METRIC_INDEX,
    POLICY_IDS,
    analyze_arrays,
    classify_verdict,
    metric_vector,
    load_labels_logits_with_missing,
    probmetrics_status,
    simultaneous_auroc_interval,
)


class FakeStructuredCalibrator:
    """Small deterministic sklearn-style stand-in used only for bootstrap tests."""

    def __init__(self, exponent: float):
        self.exponent = exponent

    def fit(self, probabilities: np.ndarray, labels: np.ndarray):
        counts = np.bincount(labels, minlength=3).astype(np.float64) + 1.0
        self.prior_ = counts / counts.sum()
        self.classes_ = [0, 1, 2]
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        adjusted = np.power(np.clip(probabilities, 1e-12, 1.0), self.exponent)
        adjusted *= np.power(self.prior_[None, :], 0.05)
        return adjusted / adjusted.sum(axis=1, keepdims=True)


def fake_factory(method: str) -> FakeStructuredCalibrator:
    return FakeStructuredCalibrator(0.72 if method == "SMS" else 0.82)


def synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.empty((n, len(POLICY_IDS)), dtype=np.int8)
    logits = rng.normal(0.0, 0.7, size=(n, len(POLICY_IDS), 3))
    for policy in range(len(POLICY_IDS)):
        labels[:, policy] = (np.arange(n) + policy) % 3
        logits[np.arange(n), policy, labels[:, policy]] += 2.2
        # Deliberately make confidence too sharp so calibration methods have work.
        logits[:, policy] *= 1.8 + 0.03 * policy
    return labels, logits


def test_metrics() -> None:
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = np.full((len(labels), 3), 0.005)
    probabilities[np.arange(len(labels)), labels] = 0.99
    values = metric_vector(labels, probabilities)
    assert values.shape == (5,)
    assert values[METRIC_INDEX["ece"]] < 0.02
    assert values[METRIC_INDEX["f1"]] == 1.0
    assert values[METRIC_INDEX["auroc"]] == 1.0
    assert values[METRIC_INDEX["nll"]] < 0.02
    assert values[METRIC_INDEX["brier"]] < 0.001
    with_missing = metric_vector(
        np.asarray([0, -1, 2]),
        np.asarray([[0.99, 0.005, 0.005], [0.2, 0.7, 0.1], [0.005, 0.005, 0.99]]),
    )
    without_missing = metric_vector(
        np.asarray([0, 2]),
        np.asarray([[0.99, 0.005, 0.005], [0.005, 0.005, 0.99]]),
    )
    assert np.allclose(with_missing, without_missing, equal_nan=True)


def test_reference_missingness_loader() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        labels_path = root / "labels.jsonl"
        logits_path = root / "logits.jsonl"
        label_rows = []
        logit_rows = []
        for index in range(100):
            labels = {
                policy: ("satisfied", "violated", "not_applicable")[
                    (index + policy_index) % 3
                ]
                for policy_index, policy in enumerate(POLICY_IDS)
            }
            label_rows.append(
                {
                    "id": f"item_{index}",
                    "labels": None if index == 0 else labels,
                    "parse_ok": index != 0,
                }
            )
            logit_rows.append(
                {
                    "id": f"item_{index}",
                    "logits": {policy: [1.0, 0.0, -1.0] for policy in POLICY_IDS},
                }
            )
        labels_path.write_text(
            "".join(json.dumps(row) + "\n" for row in label_rows),
            encoding="utf-8",
        )
        logits_path.write_text(
            "".join(json.dumps(row) + "\n" for row in logit_rows),
            encoding="utf-8",
        )
        _, labels, _, integrity = load_labels_logits_with_missing(
            labels_path, logits_path
        )
        assert integrity["pass"] is True
        assert np.all(labels[0] == -1)
        label_rows[1]["labels"] = None
        label_rows[1]["parse_ok"] = False
        labels_path.write_text(
            "".join(json.dumps(row) + "\n" for row in label_rows),
            encoding="utf-8",
        )
        _, _, _, integrity = load_labels_logits_with_missing(labels_path, logits_path)
        assert integrity["pass"] is False


def test_simultaneous_bounds() -> None:
    observed = np.asarray([0.01, -0.005, 0.002])
    bootstrap = np.tile(observed, (20, 1))
    lower, upper, critical = simultaneous_auroc_interval(observed, bootstrap)
    assert abs(critical) < 1e-15
    assert np.allclose(lower, observed)
    assert np.allclose(upper, observed)
    bad = bootstrap.copy()
    bad[0, 0] = np.nan
    lower, upper, critical = simultaneous_auroc_interval(observed, bad)
    assert np.allclose(lower, observed)
    assert np.allclose(upper, observed)
    assert abs(critical) < 1e-15


def test_verdicts() -> None:
    all_true = {
        "sms_reduction_vs_raw_ci_lower_gt_0": True,
        "sms_mean_ece_ci_upper_le_0_05": True,
        "sms_improvement_vs_P7_T_ci_lower_gt_0": True,
        "mean_f1_ci_lower_ge_minus_0_005": True,
        "mean_auroc_ci_lower_ge_minus_0_01": True,
        "all_policy_simultaneous_auroc_lower_ge_minus_0_02": True,
    }
    assert classify_verdict(
        mode="confirmation", p2_status="PASS", evaluable=True,
        conditions=all_true, harm=False,
    )[0] == "SUCCESS"
    partial = dict(all_true)
    partial["sms_mean_ece_ci_upper_le_0_05"] = False
    assert classify_verdict(
        mode="confirmation", p2_status="PASS", evaluable=True,
        conditions=partial, harm=False,
    )[0] == "PARTIAL_SUPPORT"
    not_established = dict(partial)
    not_established["sms_reduction_vs_raw_ci_lower_gt_0"] = False
    assert classify_verdict(
        mode="confirmation", p2_status="PASS", evaluable=True,
        conditions=not_established, harm=False,
    )[0] == "NOT_ESTABLISHED"
    assert classify_verdict(
        mode="confirmation", p2_status="PASS", evaluable=True,
        conditions=all_true, harm=True,
    )[0] == "CONTRADICTED_OR_HARM"
    assert classify_verdict(
        mode="confirmation", p2_status="FAIL", evaluable=True,
        conditions=all_true, harm=False,
    )[0] == "NOT_REACHED"
    assert classify_verdict(
        mode="development", p2_status="PASS", evaluable=True,
        conditions=all_true, harm=False,
    )[0] == "DEVELOPMENT_ONLY"


def test_small_paired_bootstrap() -> None:
    calib_labels, calib_logits = synthetic(500, 101)
    test_labels, test_logits = synthetic(90, 202)
    # fit_temperature itself is already tested by Day 7.  Replacing only its
    # optimizer here keeps this bootstrap orchestration test sub-second while
    # preserving the per-policy refit call on every calibration resample.
    original_fit_temperature = fit_g6_module.fit_temperature

    def fake_temperature(logits, labels, *, diagnostics=False):
        del logits, labels, diagnostics
        return {"success": True, "temperature": 1.25, "message": "test"}

    fit_g6_module.fit_temperature = fake_temperature
    try:
        first = analyze_arrays(
            calib_labels,
            calib_logits,
            test_labels,
            test_logits,
            budgets=(500,),
            bootstrap_replicates=4,
            seed=303,
            jobs=1,
            mode="development",
            p2_status="UNKNOWN",
            factory=fake_factory,
        )
        second = analyze_arrays(
            calib_labels,
            calib_logits,
            test_labels,
            test_logits,
            budgets=(500,),
            bootstrap_replicates=4,
            seed=303,
            jobs=1,
            mode="development",
            p2_status="UNKNOWN",
            factory=fake_factory,
        )
    finally:
        fit_g6_module.fit_temperature = original_fit_temperature
    # Identical seed means identical shared calibration/test resamples and output.
    assert json.dumps(to_jsonable(first), sort_keys=True) == json.dumps(
        to_jsonable(second), sort_keys=True
    )
    assert first["protocol"]["paired_two_stage"] is True
    assert first["protocol"]["shared_resamples_across_methods"] is True
    assert first["protocol"]["refit_all_methods_each_calibration_resample"] is True
    assert first["budgets"]["500"]["role"] == "PRIMARY"
    assert set(first["budgets"]["500"]["methods"]) == {
        "P7_per_policy_T", "SMS", "SVS"
    }
    assert first["verdict"]["P8"] == "DEVELOPMENT_ONLY"
    assert first["fit_health"]["bootstrap_failure_counts"] == {}


def test_published_api(require: bool) -> None:
    status = probmetrics_status()
    if not status["available"]:
        if require:
            raise AssertionError(status)
        print("probmetrics integration: SKIP", status.get("error"))
        return
    from probmetrics.calibrators import SMSCalibrator, SVSCalibrator

    labels, logits = synthetic(60, 404)
    probabilities = np.exp(logits[:, 0] - logits[:, 0].max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    for calibrator_type in (SMSCalibrator, SVSCalibrator):
        calibrator = calibrator_type()
        calibrator.fit(probabilities, labels[:, 0])
        result = calibrator.predict_proba(probabilities)
        assert result.shape == probabilities.shape
        assert np.all(np.isfinite(result))
        assert np.allclose(result.sum(axis=1), 1.0)
    print("probmetrics integration: PASS", status["version"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require_probmetrics", action="store_true")
    args = parser.parse_args()
    test_metrics()
    test_reference_missingness_loader()
    test_simultaneous_bounds()
    test_verdicts()
    test_small_paired_bootstrap()
    test_published_api(args.require_probmetrics)
    print("PASS: fit_g6 CPU metrics, verdict, paired-bootstrap, and dependency tests")


if __name__ == "__main__":
    main()
