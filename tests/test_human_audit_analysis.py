import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from src.analyze_human_audit import (
    DOMAINS,
    LABELS,
    POLICY_IDS,
    analyze,
    validate_and_join,
)
from src.human_audit_workflow import (
    export_worksheets,
    finalize_annotations,
    load_jsonl,
    merge_annotations,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def fill_csv(path: Path, field: str, labels: dict[str, str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        row[field] = labels[row["audit_id"]]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class HumanAuditWorkflowTests(unittest.TestCase):
    def test_export_merge_and_finalize_are_blind_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blind = root / "blind.jsonl"
            rows = []
            for index in range(3):
                rows.append(
                    {
                        "audit_id": f"a{index}",
                        "policy_id": "H1",
                        "policy": "Policy",
                        "prompt": "=formula" if index == 0 else f"Prompt {index}",
                        "response": f"Response {index}",
                        "rubric": "Rubric",
                        "annotator_1": None,
                        "annotator_2": None,
                        "adjudicated": None,
                    }
                )
            write_jsonl(blind, rows)
            out_a, out_b = root / "a.csv", root / "b.csv"
            result = export_worksheets(blind, out_a, out_b)
            self.assertEqual(result["rows"], 3)
            self.assertIn("'=formula", out_a.read_text(encoding="utf-8-sig"))

            labels_a = {"a0": "satisfied", "a1": "violated", "a2": "not_applicable"}
            labels_b = {"a0": "satisfied", "a1": "satisfied", "a2": "not_applicable"}
            fill_csv(out_a, "label", labels_a)
            fill_csv(out_b, "label", labels_b)
            merged, adjudication = root / "merged.jsonl", root / "adjudication.csv"
            merged_result = merge_annotations(
                blind, out_a, out_b, merged, adjudication
            )
            self.assertEqual(merged_result["disagreements"], 1)
            fill_csv(adjudication, "adjudicated", {"a1": "violated"})
            completed = root / "completed.jsonl"
            final = finalize_annotations(merged, adjudication, completed)
            self.assertEqual(final["rows"], 3)
            finalized = {row["audit_id"]: row for row in load_jsonl(completed)}
            self.assertEqual(finalized["a0"]["adjudicated"], "satisfied")
            self.assertEqual(finalized["a1"]["adjudicated"], "violated")

    def test_locked_analysis_runs_with_family_cluster_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed_rows, private_rows = [], []
            rng = random.Random(9917)
            for domain_index, domain in enumerate(DOMAINS):
                for policy_index, policy in enumerate(POLICY_IDS):
                    for item in range(40):
                        audit_id = f"{domain}-{policy}-{item}"
                        reference = LABELS[(item + policy_index) % len(LABELS)]
                        probability = 0.08 + 0.01 * (policy_index % 4)
                        if domain_index == 1:
                            probability += 0.012 * (policy_index - 4.5)
                        probability = min(0.35, max(0.02, probability))
                        mismatch = rng.random() < probability
                        human_index = (
                            (LABELS.index(reference) + 1) % len(LABELS)
                            if mismatch
                            else LABELS.index(reference)
                        )
                        human = LABELS[human_index]
                        label_b = (
                            LABELS[(human_index + 1) % len(LABELS)]
                            if (item + policy_index + domain_index) % 19 == 0
                            else human
                        )
                        completed_rows.append(
                            {
                                "audit_id": audit_id,
                                "annotator_1": human,
                                "annotator_2": label_b,
                                "adjudicated": human,
                            }
                        )
                        private_rows.append(
                            {
                                "audit_id": audit_id,
                                "domain": domain,
                                "policy_id": policy,
                                "reference_state": reference,
                                "family_id": f"family-{domain}-{policy}-{item}",
                                "inverse_probability_weight": 1.0
                                + ((item + policy_index) % 4) / 10.0,
                            }
                        )
            completed, private = root / "completed.jsonl", root / "private.jsonl"
            write_jsonl(completed, completed_rows)
            write_jsonl(private, private_rows)
            joined = validate_and_join(completed, private)
            result = analyze(joined, bootstrap=500, seed=123)
            self.assertEqual(result["sample"]["cells"], 800)
            self.assertEqual(
                result["domain_by_criterion_interaction"]["rank"], 9
            )
            self.assertEqual(
                result["domain_by_criterion_interaction"][
                    "valid_bootstrap_replicates"
                ],
                500,
            )
            self.assertIn(
                result["verdict"],
                {
                    "DIFFERENTIAL_REFERENCE_ERROR",
                    "NO_DIFFERENTIAL_ERROR_DETECTED",
                },
            )


if __name__ == "__main__":
    unittest.main()
