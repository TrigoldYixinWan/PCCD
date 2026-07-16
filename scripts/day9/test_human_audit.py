#!/usr/bin/env python3
"""CPU regression test for the blinded human-audit packet builder."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ["H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2"]
STATES = ["satisfied", "violated", "not_applicable"]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        prompts = [
            {
                "id": f"item_{index:03d}",
                "source": "synthetic",
                "prompt": f"Prompt {index}",
                "meta": {"family_id": f"family_{index:03d}"},
            }
            for index in range(60)
        ]
        prompt_path = tmp / "prompts.jsonl"
        test_path = tmp / "test.json"
        write_jsonl(prompt_path, prompts)
        test_path.write_text(
            json.dumps({"ids": [row["id"] for row in prompts]}) + "\n",
            encoding="utf-8",
        )
        domain_args: list[str] = []
        for domain_index, domain in enumerate(("D0", "D5_seed2")):
            labels, logits = [], []
            for index, prompt in enumerate(prompts):
                state_map = {
                    policy: STATES[(index + policy_index + domain_index) % 3]
                    for policy_index, policy in enumerate(POLICIES)
                }
                logit_map = {
                    policy: [
                        0.1 * ((index + policy_index) % 7),
                        0.1 * ((2 * index + policy_index) % 9),
                        0.1 * ((3 * index + policy_index) % 11),
                    ]
                    for policy_index, policy in enumerate(POLICIES)
                }
                labels.append(
                    {
                        "id": prompt["id"],
                        "source": "synthetic",
                        "prompt": prompt["prompt"],
                        "response": f"{domain} response {index}",
                        "labels": state_map,
                        "parse_ok": True,
                    }
                )
                logits.append({"id": prompt["id"], "labels": state_map, "logits": logit_map})
            label_path, logit_path = tmp / f"{domain}_labels.jsonl", tmp / f"{domain}_logits.jsonl"
            write_jsonl(label_path, labels)
            write_jsonl(logit_path, logits)
            domain_args.extend(["--domain", domain, str(label_path), str(logit_path)])

        private, blind, manifest = tmp / "private.jsonl", tmp / "blind.jsonl", tmp / "manifest.json"
        command = [
            sys.executable,
            str(ROOT / "src" / "build_human_audit.py"),
            "--prompts",
            str(prompt_path),
            "--test_manifest",
            str(test_path),
            *domain_args,
            "--private_out",
            str(private),
            "--blind_out",
            str(blind),
            "--manifest",
            str(manifest),
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        private_rows = [json.loads(line) for line in private.read_text(encoding="utf-8").splitlines()]
        blind_rows = [json.loads(line) for line in blind.read_text(encoding="utf-8").splitlines()]
        assert len(private_rows) == len(blind_rows) == 800
        assert len({row["audit_id"] for row in private_rows}) == 800
        assert all(row["inclusion_probability"] > 0 for row in private_rows)
        forbidden = {"domain", "reference_state", "critic_prediction", "critic_confidence", "source"}
        assert all(not (forbidden & set(row)) for row in blind_rows)
        assert {row["audit_id"] for row in private_rows} == {row["audit_id"] for row in blind_rows}
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["n_cells"] == 800
        print("PASS human-audit packet CPU regression")


if __name__ == "__main__":
    main()
