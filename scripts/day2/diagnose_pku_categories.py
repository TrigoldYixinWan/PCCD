#!/usr/bin/env python
"""PCCD — diagnose PKU-SafeRLHF harm-category coverage to redefine H5.

H5 "Approval-workflow adherence" mapped to NO PKU category, so the teacher (correctly)
labeled ~99.9% not_applicable and H5 has no satisfied/violated coverage for G2. We are
replacing H5 with a data-backed hard-safety dimension. This script reports the ACTUAL
category distribution in the local PKU snapshot so the replacement is chosen from real,
well-populated, currently-UNUSED categories — never guessed.

Run on AutoDL:
  source scripts/setup/env.sh
  python scripts/day2/diagnose_pku_categories.py --data_dir "$DATA_DIR"
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter

# categories already consumed by H1-H4 (must NOT reuse for the new H5)
USED = {
    "H1": ["Privacy Violation"],
    "H2": ["Cybercrime", "Economic Crime"],
    "H3": ["Violence", "Physical Harm", "Drugs", "Cybercrime", "Endangering Public Health"],
    "H4": ["Mental Manipulation"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "/root/autodl-tmp/pccd/data"))
    ap.add_argument("--max_rows", type=int, default=0, help="0 = scan all")
    args = ap.parse_args()

    from datasets import load_dataset
    path = os.path.join(args.data_dir, "pku-saferlhf")
    ds = load_dataset(path, split="train")
    n = len(ds) if args.max_rows == 0 else min(args.max_rows, len(ds))
    print(f"scanning {n} rows of {path}")

    # PKU stores per-response category dicts: category_{k}: bool (harm present).
    # Count, per category, how many responses are flagged harmful, split by safe flag.
    cat_harm = Counter()          # category -> # responses where that harm is present
    cat_in_unsafe = Counter()     # category present AND response marked unsafe
    total_resp = 0
    example_keys = None
    for i in range(n):
        r = ds[i]
        if example_keys is None:
            example_keys = sorted(r.keys())
        for which in (0, 1):
            total_resp += 1
            safe = r.get(f"is_response_{which}_safe", None)
            cats = r.get(f"response_{which}_harm_category", None)
            # harm_category may be a dict{name:bool} or a list of names depending on config
            names = []
            if isinstance(cats, dict):
                names = [k for k, v in cats.items() if v]
            elif isinstance(cats, list):
                names = list(cats)
            for name in names:
                cat_harm[name] += 1
                if safe is False:
                    cat_in_unsafe[name] += 1

    print(f"\nresponse-level fields present (sample): {example_keys}")
    print(f"total responses scanned: {total_resp}\n")
    print(f"{'category':32s} {'harm_present':>12s} {'in_unsafe':>10s}   used_by")
    used_lookup = {c: pid for pid, cs in USED.items() for c in cs}
    for name, cnt in cat_harm.most_common():
        owner = used_lookup.get(name, "-")
        print(f"{name:32s} {cnt:12d} {cat_in_unsafe[name]:10d}   {owner}")

    print("\n>>> Candidate H5 replacements = high-count categories with used_by == '-'.")
    print(">>> A good replacement should have several thousand harm-present responses so")
    print(">>> both satisfied and violated coverage is guaranteed after sampling.")


if __name__ == "__main__":
    main()
