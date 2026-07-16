#!/usr/bin/env python3
"""CPU-only tests for the outcome-blind confirmation lockbox builder."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.build_confirmation_lockbox import (  # noqa: E402
    JACCARD_DENOMINATOR,
    JACCARD_NUMERATOR,
    LOCKED_PROTOCOL,
    OUTPUT_FILENAMES,
    REQUIRED_EXCLUSION_RELATIVE_PATHS,
    Document,
    build_similarity_components,
    construct_lockbox,
    family_text,
    jaccard_at_least,
    load_historical_documents,
    new_soft_candidates,
    normalize_text,
    sha256_bytes,
    word_five_shingles,
    write_payloads,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_document(index: int, shingles: set[str]) -> Document:
    fingerprint = hashlib.sha256("|".join(sorted(shingles)).encode()).hexdigest()
    return Document(
        node_id=index,
        kind="historical" if index == 0 else "candidate",
        source="fixture",
        prompt=f"fixture {index}",
        family_text=f"fixture {index}",
        shingles=frozenset(shingles),
        fingerprint=fingerprint,
        candidate_key=None if index == 0 else f"fixture:{index}",
        metadata={},
    )


class NormalizationTests(unittest.TestCase):
    def test_nfkc_lower_and_whitespace(self) -> None:
        self.assertEqual(normalize_text("  ＡBC\n  DeF\t"), "abc def")

    def test_punctuation_and_whitespace_runs_become_ascii_space(self) -> None:
        self.assertEqual(normalize_text("Alpha—beta,\n gamma's test."), "alpha beta gamma s test")

    def test_old_soft_style_suffix_is_removed(self) -> None:
        base = "Explain a revised archival process for the regional office."
        formal = base + " Please respond in a formal tone."
        casual = base + " Please respond casually."
        self.assertEqual(family_text(formal, soft=True), family_text(casual, soft=True))
        self.assertEqual(
            family_text(formal, soft=True),
            normalize_text(base).rstrip(" ."),
        )

    def test_new_soft_style_suffix_is_removed(self) -> None:
        base = "Draft a handoff for the archive team."
        structured = base + " Organize the answer with labeled headings and bullet points."
        paragraph = base + " Write the answer as one continuous paragraph without bullets."
        self.assertEqual(
            family_text(structured, soft=True),
            family_text(paragraph, soft=True),
        )

    def test_new_soft_registry_is_480_distinct_families(self) -> None:
        rows = new_soft_candidates()
        families = {
            family_text(row["prompt"], soft=True)
            for row in rows
        }
        self.assertEqual(len(rows), 480)
        self.assertEqual(len(families), 480)


class ExactSimilarityTests(unittest.TestCase):
    def test_connected_component_transitively_excludes(self) -> None:
        base = {f"s{i}" for i in range(20)}
        middle = (base - {"s0"}) | {"x0"}
        tail = (middle - {"s1"}) | {"x1"}
        self.assertTrue(jaccard_at_least(frozenset(base), frozenset(middle)))
        self.assertTrue(jaccard_at_least(frozenset(middle), frozenset(tail)))
        self.assertFalse(jaccard_at_least(frozenset(base), frozenset(tail)))
        documents = [
            make_document(0, base),
            make_document(1, middle),
            make_document(2, tail),
        ]
        components, _, edges = build_similarity_components(documents)
        self.assertEqual(edges, 2)
        self.assertEqual(components.find(0), components.find(2))

    def test_prefix_filter_has_no_false_negatives_against_bruteforce(self) -> None:
        rng = random.Random(20260723)
        sets: list[set[str]] = []
        universe = [f"token_{index}" for index in range(160)]
        for _ in range(45):
            sets.append(set(rng.sample(universe, rng.randint(12, 35))))
        # Insert exact and high-similarity pairs in addition to arbitrary sets.
        for index in range(8):
            original = set(rng.sample(universe, 30))
            changed = set(original)
            changed.remove(sorted(changed)[index])
            changed.add(f"replacement_{index}")
            sets.extend([original, changed])
        documents = [make_document(index, values) for index, values in enumerate(sets)]
        _, _, observed_edges = build_similarity_components(documents)
        expected_edges = sum(
            jaccard_at_least(frozenset(sets[left]), frozenset(sets[right]))
            for left in range(len(sets))
            for right in range(left)
        )
        self.assertEqual(observed_edges, expected_edges)

    def test_threshold_is_exact_rational(self) -> None:
        left = frozenset(f"w{i}" for i in range(20))
        # 17 intersection / 23 union is below .85.
        right = frozenset([*(f"w{i}" for i in range(17)), "a", "b", "c"])
        self.assertFalse(jaccard_at_least(left, right))
        self.assertEqual(JACCARD_NUMERATOR, 17)
        self.assertEqual(JACCARD_DENOMINATOR, 20)

    def test_short_prompts_have_exact_nonempty_sentinel(self) -> None:
        first = word_five_shingles("one two three")
        second = word_five_shingles("one two four")
        self.assertEqual(len(first), 1)
        self.assertNotEqual(first, second)


class BuilderIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"
        for index, relative in enumerate(REQUIRED_EXCLUSION_RELATIVE_PATHS):
            write_jsonl(
                self.outputs / relative,
                [
                    {
                        "id": f"history_{index}",
                        "source": "fixture_history",
                        "prompt": (
                            f"Historical family {index} contains archive marker hist{index}a "
                            f"hist{index}b hist{index}c hist{index}d hist{index}e hist{index}f."
                        ),
                        "meta": {},
                    }
                ],
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def pku_fixture(count: int = 2400) -> list[dict]:
        categories = (
            "Privacy Violation",
            "Cybercrime",
            "Violence",
            "Mental Manipulation",
            "Endangering National Security",
        )
        rows = []
        for index in range(count):
            category_index = index // 260
            active = {}
            if category_index < len(categories):
                active[categories[category_index]] = True
            # Exercise fixed multi-category assignment without reducing any
            # stratum below its locked quota.
            if index < 40:
                active["Cybercrime"] = True
            rows.append({
                "prompt": (
                    f"PKU confirmation scenario {index} uses exclusive markers "
                    f"p{index}a p{index}b p{index}c p{index}d p{index}e p{index}f "
                    f"p{index}g p{index}h p{index}i p{index}j for a distinct request."
                ),
                "response_0_harm_category": active,
                "response_1_harm_category": {},
            })
        return rows

    @staticmethod
    def uf_fixture(count: int = 1200) -> list[dict]:
        return [
            {
                "instruction": (
                    f"UltraFeedback confirmation task {index} includes exclusive markers "
                    f"u{index}a u{index}b u{index}c u{index}d u{index}e u{index}f "
                    f"u{index}g u{index}h u{index}i u{index}j and asks for an original note."
                )
            }
            for index in range(count)
        ]

    def test_missing_historical_file_fails_closed(self) -> None:
        missing = self.outputs / REQUIRED_EXCLUSION_RELATIVE_PATHS[-1]
        missing.unlink()
        with self.assertRaises(FileNotFoundError) as context:
            load_historical_documents(self.outputs)
        self.assertIn(str(missing), str(context.exception))

    def test_full_locked_quotas_hashes_and_byte_reproducibility(self) -> None:
        historical, exclusions = load_historical_documents(self.outputs)
        arguments = dict(
            historical_documents=historical,
            exclusion_artifacts=exclusions,
            pku_rows=self.pku_fixture(),
            uf_rows=self.uf_fixture(),
            protocol=LOCKED_PROTOCOL,
            dataset_provenance={"pku": {"fixture": True}, "uf": {"fixture": True}},
        )
        first = construct_lockbox(**arguments)
        second = construct_lockbox(**arguments)
        self.assertEqual(first, second)

        out_one = self.root / "run_one"
        out_two = self.root / "run_two"
        write_payloads(out_one, first)
        write_payloads(out_two, second)
        for filename in OUTPUT_FILENAMES.values():
            self.assertEqual(
                (out_one / filename).read_bytes(),
                (out_two / filename).read_bytes(),
                filename,
            )

        prompts = [
            json.loads(line)
            for line in (out_one / OUTPUT_FILENAMES["prompts"]).read_text(encoding="utf-8").splitlines()
        ]
        calib = json.loads((out_one / OUTPUT_FILENAMES["calib"]).read_text(encoding="utf-8"))
        test = json.loads((out_one / OUTPUT_FILENAMES["test"]).read_text(encoding="utf-8"))
        family = json.loads((out_one / OUTPUT_FILENAMES["family"]).read_text(encoding="utf-8"))
        self.assertEqual(len(prompts), 4000)
        self.assertTrue(
            all(
                "response" not in row and "labels" not in row and "logits" not in row
                for row in prompts
            )
        )
        self.assertEqual(calib["n_ids"], 500)
        self.assertEqual(test["n_ids"], 3500)
        self.assertEqual(calib["source_counts"], {"pku": 293, "soft": 60, "uf": 147})
        self.assertEqual(test["source_counts"], {"pku": 2047, "soft": 420, "uf": 1033})
        expected_calib_strata = {
            "pku_h1_proxy": 30,
            "pku_h2_proxy": 30,
            "pku_h3_proxy": 30,
            "pku_h4_proxy": 30,
            "pku_h5_proxy": 30,
            "pku_general": 143,
            "ultrafeedback": 147,
            "soft_s1": 20,
            "soft_s2": 20,
            "soft_s3": 20,
        }
        expected_test_strata = {
            "pku_h1_proxy": 210,
            "pku_h2_proxy": 210,
            "pku_h3_proxy": 210,
            "pku_h4_proxy": 210,
            "pku_h5_proxy": 210,
            "pku_general": 997,
            "ultrafeedback": 1033,
            "soft_s1": 140,
            "soft_s2": 140,
            "soft_s3": 140,
        }
        self.assertEqual(calib["stratum_counts"], expected_calib_strata)
        self.assertEqual(test["stratum_counts"], expected_test_strata)
        self.assertEqual(calib["nested_prefix_stratum_counts"]["500"], expected_calib_strata)
        self.assertFalse(set(calib["ids"]) & set(test["ids"]))
        self.assertTrue(calib["frozen_before_outcomes"])
        self.assertTrue(test["frozen_before_outcomes"])
        self.assertEqual(test["overlap_with_target_calib"], 0)
        self.assertEqual(calib["overlap_with_confirm_test"], 0)
        self.assertEqual(calib["family_ids"], calib["query_family_ids"])
        self.assertEqual(test["family_ids"], test["query_family_ids"])
        self.assertEqual(
            calib["items"],
            [{"id": item_id, "family_id": family_id} for item_id, family_id in zip(calib["ids"], calib["family_ids"])],
        )
        self.assertEqual(len({row["meta"]["family_id"] for row in prompts}), 4000)
        self.assertTrue(
            all(
                row["family_id"] == row["meta"]["family_id"] == row["meta"]["query_family_id"]
                for row in prompts
            )
        )
        self.assertEqual(family["integrity"]["historical_component_overlap_selected"], 0)
        self.assertEqual(family["integrity"]["selected_query_family_duplicates"], 0)
        self.assertEqual(family["selected"]["stratum_counts"], {
            key: expected_calib_strata[key] + expected_test_strata[key]
            for key in expected_calib_strata
        })

        prompt_bytes = (out_one / OUTPUT_FILENAMES["prompts"]).read_bytes()
        self.assertEqual(calib["prompt_artifact_sha256"], sha256_bytes(prompt_bytes))
        self.assertEqual(test["prompt_artifact_sha256"], sha256_bytes(prompt_bytes))
        sha_entries = {
            line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
            for line in (out_one / OUTPUT_FILENAMES["sha256"]).read_text(encoding="utf-8").splitlines()
        }
        for key in ("prompts", "calib", "test", "family"):
            filename = OUTPUT_FILENAMES[key]
            self.assertEqual(sha_entries[filename], sha256_bytes((out_one / filename).read_bytes()))

        with self.assertRaises(FileExistsError):
            write_payloads(out_one, first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
