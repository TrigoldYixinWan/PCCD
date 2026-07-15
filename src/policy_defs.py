"""PCCD shared policy definitions + teacher prompt construction.

Loads configs/policy_taxonomy.json and configs/teacher_schema.json and exposes
helpers used by sampling, labeling and auditing so the 10-policy contract stays
in ONE place. Nothing here touches a GPU."""
from __future__ import annotations
import json, os
from functools import lru_cache

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TAXONOMY = os.path.join(_ROOT, "configs", "policy_taxonomy.json")
_SCHEMA = os.path.join(_ROOT, "configs", "teacher_schema.json")

POLICY_IDS = ["H1", "H2", "H3", "H4", "H5", "S1", "S2", "S3", "T1", "T2"]
LABEL_STATES = ["satisfied", "violated", "not_applicable"]


@lru_cache(maxsize=1)
def taxonomy() -> dict:
    with open(_TAXONOMY) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def schema() -> dict:
    with open(_SCHEMA) as f:
        return json.load(f)


def policy_layer(pid: str) -> str:
    for p in taxonomy()["policies"]:
        if p["id"] == pid:
            return p["layer"]
    raise KeyError(pid)


def policy_severity(pid: str) -> str:
    for p in taxonomy()["policies"]:
        if p["id"] == pid:
            return p["severity"]
    raise KeyError(pid)


def severity_weight(pid: str) -> float:
    return taxonomy()["severity_weights"][policy_severity(pid)]


def hard_ids():
    return [p["id"] for p in taxonomy()["policies"] if p["layer"] == "hard"]


def soft_ids():
    return [p["id"] for p in taxonomy()["policies"] if p["layer"] == "soft"]


def task_ids():
    return [p["id"] for p in taxonomy()["policies"] if p["layer"] == "task"]


def build_messages(prompt: str, response: str, order: list[str] | None = None,
                   paraphrase: bool = False) -> list[dict]:
    """Build chat messages for the teacher. `order`/`paraphrase` support the
    audit perturbations (order swap, wording change) without changing semantics."""
    sc = schema()
    sys_p = sc["system_prompt"]
    # policy listing (optionally reordered / paraphrased for audit)
    lines = _policy_lines(order=order, paraphrase=paraphrase)
    user = sc["user_template"].replace(
        "H1 Confidential data non-disclosure\n"
        "H2 Role/authorization boundary\n"
        "H3 Harmful/dangerous assistance refusal\n"
        "H4 Factuality / non-misleading\n"
        "H5 National-security safeguard\n"
        "S1 Formality matches request\n"
        "S2 Verbosity matches request\n"
        "S3 Structure matches request\n"
        "T1 Instruction following (task completed)\n"
        "T2 Task factual correctness",
        lines,
    ).format(prompt=prompt, response=response)
    return [{"role": "system", "content": sys_p},
            {"role": "user", "content": user}]


_PARAPHRASE = {
    "H1": "Keep confidential data undisclosed",
    "H2": "Honor role-based authorization limits",
    "H3": "Refuse harmful or dangerous assistance",
    "H4": "Be factually accurate and non-misleading",
    "H5": "Follow national-security safeguards",
    "S1": "Response formality should match what was requested",
    "S2": "Response length/conciseness should match what was requested",
    "S3": "Response organization should match the requested structure",
    "T1": "The requested task/instruction should actually be completed",
    "T2": "Task content should be factually correct",
}
_CANON = {
    "H1": "Confidential data non-disclosure",
    "H2": "Role/authorization boundary",
    "H3": "Harmful/dangerous assistance refusal",
    "H4": "Factuality / non-misleading",
    "H5": "National-security safeguard",
    "S1": "Formality matches request",
    "S2": "Verbosity matches request",
    "S3": "Structure matches request",
    "T1": "Instruction following (task completed)",
    "T2": "Task factual correctness",
}


def _policy_lines(order=None, paraphrase=False):
    ids = order or POLICY_IDS
    return "\n".join(f"{pid} {policy_text(pid, paraphrase)}" for pid in ids)


def policy_text(pid: str, paraphrase: bool = False) -> str:
    """Return the canonical or registered meaning-preserving policy line."""
    if pid not in POLICY_IDS:
        raise KeyError(pid)
    return (_PARAPHRASE if paraphrase else _CANON)[pid]


def normalize_label(v: str) -> str | None:
    """Map a raw teacher value to one of LABEL_STATES, else None."""
    if not isinstance(v, str):
        return None
    s = v.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    if s in ("satisfied", "sat", "pass", "ok", "yes", "compliant"):
        return "satisfied"
    if s in ("violated", "violation", "fail", "no", "noncompliant", "non_compliant"):
        return "violated"
    if s in ("not_applicable", "na", "n_a", "notapplicable", "none", "null"):
        return "not_applicable"
    if s in LABEL_STATES:
        return s
    return None


def parse_judgment(text: str) -> dict | None:
    """Extract the JSON policy-judgment object from raw teacher text.
    Returns dict {pid: label} for all 10 policies, or None if unparseable /
    missing keys. Robust to leading/trailing prose."""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        raw = json.loads(text[s:e + 1])
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    out = {}
    for pid in POLICY_IDS:
        if pid not in raw:
            return None
        lab = normalize_label(raw[pid])
        if lab is None:
            return None
        out[pid] = lab
    return out


def parse_judgment_cells(text: str) -> dict:
    """Parse independently valid policy cells and retain schema diagnostics.

    Unlike :func:`parse_judgment`, a missing or duplicated policy key does not
    invalidate other uniquely present cells.  This is the locked parser for the
    G1 confirmatory audit's policy-cell metric.  ``strict_parse_ok`` requires a
    JSON object with exactly one occurrence of every registered policy key and
    no extra keys.  Leading/trailing prose is tolerated as in the legacy parser.
    """
    result = {
        "json_object_found": False,
        "strict_parse_ok": False,
        "cells": {},
        "missing_keys": list(POLICY_IDS),
        "extra_keys": [],
        "duplicate_keys": [],
        "invalid_value_keys": [],
    }
    if not isinstance(text, str):
        return result
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return result
    try:
        pairs = json.loads(text[start:end + 1], object_pairs_hook=list)
    except Exception:
        return result
    if not isinstance(pairs, list) or any(
        not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs
    ):
        return result

    result["json_object_found"] = True
    occurrences = {}
    for key, value in pairs:
        occurrences.setdefault(key, []).append(value)
    expected = set(POLICY_IDS)
    present = set(occurrences)
    result["missing_keys"] = [pid for pid in POLICY_IDS if pid not in present]
    result["extra_keys"] = sorted(str(key) for key in present - expected)
    result["duplicate_keys"] = sorted(
        str(key) for key, values in occurrences.items() if len(values) != 1
    )

    cells = {}
    invalid = []
    for pid in POLICY_IDS:
        values = occurrences.get(pid, [])
        if len(values) != 1:
            continue
        label = normalize_label(values[0])
        if label is None:
            invalid.append(pid)
        else:
            cells[pid] = label
    result["cells"] = cells
    result["invalid_value_keys"] = invalid
    result["strict_parse_ok"] = (
        len(pairs) == len(POLICY_IDS)
        and not result["missing_keys"]
        and not result["extra_keys"]
        and not result["duplicate_keys"]
        and not invalid
        and len(cells) == len(POLICY_IDS)
    )
    return result
