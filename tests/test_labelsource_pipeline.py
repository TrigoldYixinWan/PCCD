import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analysis = load_module("labelsource_analysis", "src/analyze_labelsource.py")
builder = load_module("labelsource_builder", "src/build_labelsource_eval.py")
qwen = load_module("labelsource_qwen", "src/label_beavertails_qwen.py")
guard = load_module("labelsource_guard", "src/guard_score.py")


def test_primary_metrics_perfect_and_miscalibrated():
    label = np.array([0, 0, 1, 1], dtype=np.int8)
    perfect = analysis.point_metrics(np.array([0.0, 0.0, 1.0, 1.0]), label)
    assert perfect["balanced_equal_mass_ece"] == 0.0
    assert perfect["auroc"] == 1.0
    bad = analysis.point_metrics(np.full(4, 0.9), label)
    assert np.isclose(bad["balanced_equal_mass_ece"], 0.4)
    assert np.isclose(bad["signed_deviation"], 0.4)


def test_kappa_and_holm_are_numerically_stable():
    label = np.array([0, 1, 1, 0, -1], dtype=np.int8)
    result = analysis.cohen_kappa(label, label.copy())
    assert result == {"n": 4, "agreement": 1.0, "kappa": 1.0}
    adjusted = analysis.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_fixed_bin_cluster_bootstrap_matches_identity_counts(monkeypatch):
    probability = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    label = np.array([0, 0, 1, 1], dtype=np.int8)
    counts = np.ones((4, 4), dtype=np.uint16)
    monkeypatch.setattr(analysis, "BOOTSTRAPS", 4)
    ece, deviation = analysis.bootstrap_cell(probability, label, counts)
    point = analysis.point_metrics(probability, label)
    assert np.allclose(ece, point["balanced_equal_mass_ece"])
    assert np.allclose(deviation, point["signed_deviation"])


def test_mapping_rules_keep_no_map_and_use_max_union():
    native = {
        "dangerous_content": {"unsafe_probability": 0.2, "score_ok": True, "error": None},
        "harassment": {"unsafe_probability": 0.7, "score_ok": True, "error": None},
    }
    mapped = builder.mapped_probability(
        "shieldgemma_2b",
        native,
        {"target_ids": ["dangerous_content", "harassment"], "relation": "partial_overlap"},
    )
    assert mapped["unsafe_probability"] == 0.7
    assert mapped["aggregation"] == "max_union"
    missing = builder.mapped_probability(
        "shieldgemma_2b",
        native,
        {"target_ids": [], "relation": "no_defensible_mapping"},
    )
    assert missing["unsafe_probability"] is None
    assert missing["error"] == "no_defensible_taxonomy_mapping"


def test_qwen_parser_never_repairs_or_coerces():
    keys = ["a", "b"]
    assert qwen.strict_parse('{"a": true, "b": false}', keys) == {
        "a": True,
        "b": False,
    }
    assert qwen.strict_parse('{"a": 1, "b": false}', keys) is None
    assert qwen.strict_parse('```json\n{"a": true, "b": false}\n```', keys) is None


def test_guard_sanity_verdict_remains_callable_after_formal_extension():
    rows = [
        {"interface": "policy", "unsafe_probability": 0.01},
        {"interface": "policy", "unsafe_probability": 0.99},
    ]
    verdict = guard.sanity_verdict(rows)
    assert verdict["pass"] is True
    assert np.isclose(verdict["pooled_probability_range"], 0.98)
