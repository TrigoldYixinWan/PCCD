from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.eval_critic import load_scoring_jsonl
from src.gen_policy_responses import build_response_record
from src.subset_jsonl_by_manifest import load_manifest_ids
from src.validate_confirmation_adapter import compare_metadata


ROOT = Path(__file__).resolve().parents[1]


class ConfirmationHelperTest(unittest.TestCase):
    def test_response_record_preserves_family_metadata(self) -> None:
        record = {
            "id": "confirm_1",
            "source": "pku_saferlhf_prompt",
            "prompt": "Example prompt",
            "family_id": "qf_123",
            "meta": {
                "family_id": "qf_123",
                "query_family_id": "qf_123",
                "confirmation_stratum": "pku_h1_proxy",
            },
        }
        result = build_response_record(
            record,
            response="Example response",
            prompt_token_ids=[1, 2],
            base_prompt_token_ids=[1, 2],
            generated_token_ids=[3],
            finish_reason="stop",
            variant="new_D5",
            adapter="/adapter",
            system_prompt=None,
            max_tokens=256,
            seed=20260723,
        )
        self.assertEqual(result["family_id"], "qf_123")
        self.assertEqual(result["meta"]["query_family_id"], "qf_123")
        self.assertEqual(result["meta"]["confirmation_stratum"], "pku_h1_proxy")
        self.assertEqual(result["meta"]["generation"]["seed"], 20260723)

    def test_manifest_loader_accepts_json_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "ids.json"
            jsonl_path = root / "prompts.jsonl"
            json_path.write_text(json.dumps({"ids": ["a", "b"]}), encoding="utf-8")
            jsonl_path.write_text(
                json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_manifest_ids(json_path), ["a", "b"])
            self.assertEqual(load_manifest_ids(jsonl_path), ["a", "b"])

    def test_adapter_validator_allows_seed_and_training_outcomes_only(self) -> None:
        old = {
            "point": "D5",
            "objective": "hidden_violation",
            "method": "SFT",
            "base_model": "/models/qwen7b",
            "independent_from_frozen_critic": True,
            "pairs": {"sha256": "a" * 64, "count": 512},
            "seed": 20260716,
            "lora": {
                "r": 32,
                "alpha": 64,
                "dropout": 0.05,
                "targets": ["q_proj"],
            },
            "dpo_beta": None,
            "training": {
                "epochs": 4,
                "global_step": 64,
                "effective_batch": 32,
                "per_device_batch": 1,
                "world_size": 1,
                "gradient_accumulation": 32,
                "learning_rate": 2e-4,
                "max_length": 1024,
                "training_loss": 1.0,
                "log_history": [],
            },
        }
        new = json.loads(json.dumps(old))
        new["seed"] = 20260723
        new["training"]["training_loss"] = 2.0
        new["training"]["log_history"] = [{"loss": 2.0}]
        self.assertTrue(compare_metadata(old, new)["pass"])
        new["training"]["world_size"] = 2
        self.assertFalse(compare_metadata(old, new)["pass"])

    def test_logits_only_loader_preserves_missing_reference_without_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "item",
                        "prompt": "Prompt",
                        "response": "Response",
                        "labels": None,
                        "parse_ok": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = load_scoring_jsonl(path)[0]
            self.assertIsNone(row["_original_labels"])
            self.assertEqual(set(row["labels"]), {
                "H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2"
            })

    def test_logits_only_loader_accepts_complete_reference_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            labels = {
                policy: "satisfied"
                for policy in ("H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2")
            }
            path.write_text(
                json.dumps(
                    {
                        "id": "item",
                        "prompt": "Prompt",
                        "response": "Response",
                        "labels": labels,
                        "parse_ok": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = load_scoring_jsonl(path)[0]
            self.assertEqual(row["_original_labels"], labels)
            self.assertEqual(row["labels"], labels)

    def test_preunseal_freeze_records_clean_commit_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("frozen\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=PCCD Test",
                    "-c",
                    "user.email=pccd@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )
            artifact = root.parent / f"{root.name}_artifact.json"
            manifest = root.parent / f"{root.name}_freeze.sha256"
            metadata = root.parent / f"{root.name}_environment.json"
            artifact.write_text('{"frozen": true}\n', encoding="utf-8")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "src" / "freeze_confirmation_artifacts.py"),
                        "--repo",
                        str(root),
                        "--out",
                        str(manifest),
                        "--metadata_out",
                        str(metadata),
                        "--path",
                        str(artifact),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                self.assertTrue(payload["frozen_before_aggregate_metrics"])
                self.assertEqual(len(payload["git"]["commit"]), 40)
                text = manifest.read_text(encoding="utf-8")
                self.assertIn(artifact.name, text)
                self.assertIn(metadata.name, text)
            finally:
                for path in (artifact, manifest, metadata):
                    path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
