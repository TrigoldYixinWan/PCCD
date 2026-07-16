#!/usr/bin/env python3
"""Timing-only P8 benchmark on the already-consumed P7 development split."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fit_g6 import analyze_arrays, load_frozen_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="+", default=[500])
    args = parser.parse_args()
    root = Path(os.environ.get("PCCD_OUT", "outputs"))
    frozen = argparse.Namespace(
        mode="development",
        labels=root / "g2" / "D5_teacher.jsonl",
        logits=root / "g2" / "D5_logits.jsonl",
        calib_manifest=root / "results" / "g5_target_calib_ids.json",
        test_manifest=root / "results" / "g5_target_test_ids.json",
        split_hash_manifest=root / "results" / "g5_split_manifests.sha256",
    )
    calib_y, calib_z, test_y, test_z, _, integrity = load_frozen_inputs(frozen)
    started = time.perf_counter()
    result = analyze_arrays(
        calib_y,
        calib_z,
        test_y,
        test_z,
        budgets=tuple(args.budgets),
        bootstrap_replicates=args.replicates,
        seed=20260724,
        jobs=args.jobs,
        mode="development",
        p2_status="UNKNOWN",
        reference_integrity=bool(integrity["pass"]),
        reference_integrity_details=integrity,
    )
    elapsed = time.perf_counter() - started
    print(
        {
            "development_only": True,
            "replicates": args.replicates,
            "budgets": args.budgets,
            "jobs": args.jobs,
            "seconds": elapsed,
            "seconds_per_replicate_wall": elapsed / args.replicates,
            "start_method": result["protocol"]["multiprocessing_start_method"],
            "fit_failures": result["fit_health"]["bootstrap_failure_counts"],
        }
    )


if __name__ == "__main__":
    main()
