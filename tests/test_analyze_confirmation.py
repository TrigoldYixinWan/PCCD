from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analyze_confirmation import (
    BOOTSTRAP_SEED,
    ID_TO_LABEL,
    mutually_exclusive_verdict,
    run_analysis,
)
from src.policy_defs import POLICY_IDS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ConfirmationSyntheticTest(unittest.TestCase):
    @staticmethod
    def refresh_hash_manifest(args: argparse.Namespace) -> None:
        payload = json.loads(args.hash_manifest.read_text(encoding="utf-8"))
        for path in (
            args.d0_labels,
            args.d0_logits,
            args.new_d5_labels,
            args.new_d5_logits,
            args.old_d5_labels,
            args.old_d5_logits,
        ):
            payload["files"][path.name] = digest(path)
        args.hash_manifest.write_text(json.dumps(payload), encoding="utf-8")

    def build_fixture(self, root: Path, n: int = 360) -> argparse.Namespace:
        ids = [f"confirm_{index:04d}" for index in range(n)]
        truth = np.asarray(
            [[(row + policy) % 3 for policy in range(len(POLICY_IDS))] for row in range(n)],
            dtype=np.int8,
        )

        def make_probabilities(fractions: list[float], offsets: list[int]) -> np.ndarray:
            result = np.full((n, len(POLICY_IDS), 3), 0.01, dtype=np.float64)
            for policy, fraction in enumerate(fractions):
                wrong_count = int(round(fraction * n))
                order = (np.arange(n) * (2 * policy + 3) + offsets[policy]) % n
                wrong = order < wrong_count
                predicted = truth[:, policy].copy()
                predicted[wrong] = (predicted[wrong] + 1) % 3
                result[np.arange(n), policy, predicted] = 0.98
            return result

        base_prob = make_probabilities([0.0] * 10, [0] * 10)
        new_prob = make_probabilities(
            [0.05 + 0.03 * policy for policy in range(10)],
            [17 * policy for policy in range(10)],
        )
        old_prob = make_probabilities([0.08] * 10, [23 * policy for policy in range(10)])

        label_rows = [
            {
                "id": ids[row],
                "labels": {
                    policy: ID_TO_LABEL[int(truth[row, index])]
                    for index, policy in enumerate(POLICY_IDS)
                },
            }
            for row in range(n)
        ]

        def logit_rows(probabilities: np.ndarray) -> list[dict]:
            return [
                {
                    "id": ids[row],
                    "logits": {
                        policy: np.log(probabilities[row, index]).tolist()
                        for index, policy in enumerate(POLICY_IDS)
                    },
                }
                for row in range(n)
            ]

        d0_labels = root / "d0_labels.jsonl"
        d0_logits = root / "d0_logits.jsonl"
        new_labels = root / "new_labels.jsonl"
        new_logits = root / "new_logits.jsonl"
        old_labels = root / "old_labels.jsonl"
        old_logits = root / "old_logits.jsonl"
        write_jsonl(d0_labels, label_rows)
        write_jsonl(new_labels, label_rows)
        write_jsonl(old_labels, label_rows)
        write_jsonl(d0_logits, logit_rows(base_prob))
        write_jsonl(new_logits, logit_rows(new_prob))
        write_jsonl(old_logits, logit_rows(old_prob))

        test_manifest = root / "test_manifest.json"
        test_manifest.write_text(
            json.dumps(
                {
                    "role": "CONFIRM-TEST",
                    "frozen_before_outcomes": True,
                    "overlap_with_target_calib": 0,
                    "items": [
                        {"id": item, "family_id": f"family_{index:04d}"}
                        for index, item in enumerate(ids)
                    ],
                }
            ),
            encoding="utf-8",
        )
        source_edges = root / "source_edges.json"
        source_edges.write_text(
            json.dumps(
                {
                    "role": "SOURCE-BASE-CALIB",
                    "edges": {
                        policy: np.linspace(0.0, 1.0, 16).tolist() for policy in POLICY_IDS
                    },
                }
            ),
            encoding="utf-8",
        )
        inputs = [
            test_manifest,
            source_edges,
            d0_labels,
            d0_logits,
            new_labels,
            new_logits,
            old_labels,
            old_logits,
        ]
        hash_manifest = root / "hashes.json"
        hash_manifest.write_text(
            json.dumps({"files": {path.name: digest(path) for path in inputs}}),
            encoding="utf-8",
        )
        return argparse.Namespace(
            test_manifest=test_manifest,
            hash_manifest=hash_manifest,
            source_bin_edges=source_edges,
            d0_labels=d0_labels,
            d0_logits=d0_logits,
            new_d5_labels=new_labels,
            new_d5_logits=new_logits,
            old_d5_labels=old_labels,
            old_d5_logits=old_logits,
            out=root / "result.json",
            expected_test_count=n,
            bootstrap=250,
            seed=BOOTSTRAP_SEED,
        )

    def test_end_to_end_positive_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            result = run_analysis(args)
        self.assertEqual(result["reported_verdict"], "TEST_MODE_ONLY")
        primary = result["primary_new_D5"]
        self.assertTrue(primary["P2"]["lower_ci_gt_zero"])
        self.assertTrue(primary["P2"]["D0_anchor_upper_le_0_05"])
        self.assertTrue(primary["P3"]["omnibus_significant"])
        self.assertTrue(primary["P3"]["material_lower_ci_gt_0_01"])
        self.assertEqual(primary["statistical_verdict"], "P2_P3_CONFIRMED")
        self.assertEqual(result["secondary"]["old_D5"]["statistical_verdict"], "SECONDARY_NO_VERDICT")

    def test_hash_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory), n=90)
            with args.new_d5_logits.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run_analysis(args)

    def test_reference_missingness_at_or_below_one_percent_is_not_imputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            rows = [
                json.loads(line)
                for line in args.new_d5_labels.read_text(encoding="utf-8").splitlines()
            ]
            for row in rows[:3]:
                row["labels"] = None
                row["parse_ok"] = False
            write_jsonl(args.new_d5_labels, rows)
            self.refresh_hash_manifest(args)
            result = run_analysis(args)
        integrity = result["protocol"]["reference_integrity"]["new_D5"]
        self.assertTrue(integrity["pass"])
        self.assertAlmostEqual(integrity["strict_ten_key_success_rate"], 357 / 360)
        self.assertNotEqual(
            result["primary_new_D5"]["statistical_verdict"], "NON_EVALUABLE"
        )

    def test_reference_missingness_above_one_percent_is_non_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            rows = [
                json.loads(line)
                for line in args.new_d5_labels.read_text(encoding="utf-8").splitlines()
            ]
            for row in rows[:4]:
                row["labels"] = None
                row["parse_ok"] = False
            write_jsonl(args.new_d5_labels, rows)
            self.refresh_hash_manifest(args)
            result = run_analysis(args)
        self.assertFalse(result["protocol"]["reference_integrity"]["new_D5"]["pass"])
        self.assertEqual(
            result["primary_new_D5"]["statistical_verdict"], "NON_EVALUABLE"
        )

    def test_verdict_partition_is_mutually_exclusive(self) -> None:
        p3_yes = {"evaluable": True, "omnibus_significant": True}
        p3_no = {"evaluable": True, "omnibus_significant": False}
        self.assertEqual(
            mutually_exclusive_verdict(True, True, [0.01, 0.03], p3_yes),
            "P2_P3_CONFIRMED",
        )
        self.assertEqual(
            mutually_exclusive_verdict(True, True, [0.01, 0.03], p3_no),
            "P2_ONLY",
        )
        self.assertEqual(
            mutually_exclusive_verdict(True, True, [-0.01, 0.02], p3_yes),
            "CORE_NOT_ESTABLISHED",
        )
        self.assertEqual(
            mutually_exclusive_verdict(True, True, [-0.03, 0.0], p3_yes),
            "P2_CONTRADICTED",
        )
        self.assertEqual(
            mutually_exclusive_verdict(True, False, [0.01, 0.03], p3_yes),
            "BASE_ANCHOR_NOT_REPLICATED",
        )
        self.assertEqual(
            mutually_exclusive_verdict(False, True, [0.01, 0.03], p3_yes),
            "NON_EVALUABLE",
        )


if __name__ == "__main__":
    unittest.main()
