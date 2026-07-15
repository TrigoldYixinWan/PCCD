#!/usr/bin/env python3
"""CPU-only preflight for the single locked L1 confirmatory audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.audit_labels import (  # noqa: E402
    CONFIRMATORY_AUDIT_SHA256,
    _read_labels,
    _sha256_file,
    validate_confirmatory_schedule,
    validate_frozen_paraphrases,
)
from src.policy_defs import POLICY_IDS, build_messages, parse_judgment_cells  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--frozen-paraphrases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-marker", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists() or args.run_marker.exists():
        raise SystemExit(
            "FAIL: confirmatory output/marker already exists; do not start a second run "
            f"(output={args.out.exists()}, marker={args.run_marker.exists()})"
        )
    items = _read_labels(args.audit)
    audit_sha256 = _sha256_file(args.audit)
    if audit_sha256 != CONFIRMATORY_AUDIT_SHA256:
        raise SystemExit(
            "FAIL: audit pool SHA-256 differs from the frozen Day-2/3 artifact: "
            f"observed={audit_sha256}, expected={CONFIRMATORY_AUDIT_SHA256}"
        )
    orders, position_counts, schedule_sha256 = validate_confirmatory_schedule(items)
    paraphrases, paraphrase_sha256 = validate_frozen_paraphrases(args.frozen_paraphrases)

    # Confirm that canonical and paraphrase prompts differ only in registered
    # policy lines; input content and system prompt remain identical.
    probe = items[0]
    canonical = build_messages(probe["prompt"], probe["response"])
    paraphrased = build_messages(probe["prompt"], probe["response"], paraphrase=True)
    if canonical[0] != paraphrased[0]:
        raise SystemExit("FAIL: paraphrase changed the system prompt")
    if probe["prompt"] not in paraphrased[1]["content"] or probe["response"] not in paraphrased[1]["content"]:
        raise SystemExit("FAIL: paraphrase prompt lost the audit prompt/response")

    # Parser contract: one missing key preserves nine cells; duplicates make
    # only the duplicated cell invalid; strict success requires all ten once.
    complete = json.dumps({pid: "satisfied" for pid in POLICY_IDS})
    parsed_complete = parse_judgment_cells(complete)
    if not parsed_complete["strict_parse_ok"] or len(parsed_complete["cells"]) != 10:
        raise SystemExit("FAIL: strict parser contract")
    missing = {pid: "satisfied" for pid in POLICY_IDS if pid != "S3"}
    parsed_missing = parse_judgment_cells(json.dumps(missing))
    if parsed_missing["strict_parse_ok"] or len(parsed_missing["cells"]) != 9:
        raise SystemExit("FAIL: partial-cell parser contract")
    duplicate = complete[:-1] + ', "H1": "violated"}'
    parsed_duplicate = parse_judgment_cells(duplicate)
    if "H1" in parsed_duplicate["cells"] or len(parsed_duplicate["cells"]) != 9:
        raise SystemExit("FAIL: duplicate-key parser contract")

    print("=== L1 CONFIRMATORY PREFLIGHT: PASS ===")
    print(f"audit items: {len(items)} unique")
    print(f"audit sha256: {audit_sha256}")
    print(f"frozen paraphrases sha256: {paraphrase_sha256}")
    for pid in POLICY_IDS:
        print(f"  {pid}: {paraphrases[pid]}")
    print(f"Latin schedule sha256: {schedule_sha256}")
    print("Latin balance: every policy x position = 40")
    print(f"unique Latin rows: {len({tuple(order) for order in orders})}")
    print(f"unchanged canonical rows: {sum(order == list(POLICY_IDS) for order in orders)}")
    print(f"position counts: {position_counts}")
    print("parser checks: strict=10/10; missing-one=9/10 retained; duplicate-one=9/10 retained")
    print(f"single-run guard paths absent: {args.out}; {args.run_marker}")


if __name__ == "__main__":
    main()
