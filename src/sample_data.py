"""PCCD Day-2 data sampler.

Builds the 10.8k pool of (prompt, response) items to be labeled by the teacher,
drawing from:
  * PKU-SafeRLHF   -> hard-safety coverage (H1-H5), keep both safe & unsafe responses
  * UltraFeedback  -> task quality (T1/T2) with varied response quality
  * self-built     -> controlled soft-style pairs (S1-S3: formality/verbosity/structure)

Each output item is a dict:
  {id, source, prompt, response, meta:{...}}
written to outputs/pool/<split>.jsonl

Splits: train(8000) / calib(1000) / test(1000) / audit(400) / conflict(400).
The audit split is a copy of items reused with perturbations at labeling time.
CPU-only; run once before teacher labeling.

Usage:
  python src/sample_data.py --data_dir /root/data --out outputs/pool --seed 0
"""
from __future__ import annotations
import argparse, json, os, random, hashlib

# ------------------------- self-built soft-style pairs -------------------------
# Each base task is rendered in two contrasting styles so the critic sees clear
# S1/S2/S3 signal. Responses are templated (no external data), fully local.
_SOFT_TASKS = [
    "Summarize the quarterly sales trend for the team.",
    "Explain the new expense-reimbursement rule to a new hire.",
    "Draft a reply declining a vendor meeting request.",
    "Describe how to reset an internal dashboard password.",
    "Write an update on the migration project status.",
    "Give feedback on a colleague's draft proposal.",
    "Announce the office closure for a public holiday.",
    "Outline the steps to onboard a new contractor.",
    "Explain why a ticket was deprioritized.",
    "Recommend a time-management approach for the sprint.",
]
_SOFT_TEAMS = [
    "analytics", "customer success", "design", "engineering", "finance",
    "legal", "marketing", "operations", "product", "research",
]
_SOFT_INITIATIVES = [
    "access review", "budget forecast", "client migration", "compliance refresh",
    "dashboard redesign", "documentation audit", "incident follow-up",
    "onboarding update", "policy rollout", "quality review", "release planning",
    "security assessment", "vendor transition",
]
_SOFT_PERIODS = [
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October",
]
_FORMAL = ("Dear team, please find below a concise account. {body} "
           "Kindly reach out should any clarification be required. Regards.")
_CASUAL = ("hey! quick note — {body} lmk if anything's unclear, cheers")
_TERSE = "{body}"
_VERBOSE = ("To provide full context, {body} In addition, there are several "
            "nuances worth noting, and I am happy to elaborate on each point in "
            "as much detail as you would find useful, at any time.")
_STRUCTURED = ("Summary:\n- Point 1: {body}\n- Point 2: follow-up needed\n- Next step: TBD")
_UNSTRUCTURED = ("{body} and there are a few other things to keep in mind as well")


def _soft_pairs(n: int, rng: random.Random):
    items = []
    axes = [("S1", _FORMAL, _CASUAL, "formality"),
            ("S2", _VERBOSE, _TERSE, "verbosity"),
            ("S3", _STRUCTURED, _UNSTRUCTURED, "structure")]
    i = 0
    while len(items) < n:
        task = rng.choice(_SOFT_TASKS)
        # 10 teams x 13 initiatives x 10 periods = 1,300 natural, unique
        # scenarios. Uniqueness is in the text, so hash-based cross-split dedup
        # cannot collapse the controlled soft-policy coverage.
        team = _SOFT_TEAMS[i % len(_SOFT_TEAMS)]
        initiative = _SOFT_INITIATIVES[(i // len(_SOFT_TEAMS)) % len(_SOFT_INITIATIVES)]
        period_stride = len(_SOFT_TEAMS) * len(_SOFT_INITIATIVES)
        period = _SOFT_PERIODS[(i // period_stride) % len(_SOFT_PERIODS)]
        context = (f"Context: this concerns the {initiative} initiative for the "
                   f"{team} team during {period}.")
        pid, tmpl_a, tmpl_b, axis = rng.choice(axes)
        body = f"{context} The key facts concern: {task.lower()}"
        # prompt explicitly requests one pole; response may match or mismatch
        want = rng.choice(["A", "B"])
        req = {"S1": "in a formal tone", "S2": "in a very detailed way",
               "S3": "as a structured bullet list"}[pid] if want == "A" else \
              {"S1": "casually", "S2": "very briefly",
               "S3": "as a single flowing paragraph"}[pid]
        prompt = f"{task} {context} Please respond {req}."
        # produce one response (matching or mismatching) — critic judges S-policy
        matches = rng.random() < 0.6
        chosen = tmpl_a if (want == "A") == matches else tmpl_b
        resp = chosen.format(body=body)
        items.append({"source": "soft_style", "prompt": prompt, "response": resp,
                      "meta": {"axis": axis, "target_pole": want, "intended_match": matches}})
        i += 1
    return items


# ------------------------------ HF loaders ------------------------------------
def _resolve_local(data_dir, subdir, hub_id, local_only):
    """Prefer a local snapshot; fall back to Hub id only when local_only=False.
    On a disconnected AutoDL box, local_only=True raises a clear error instead of
    hanging on network retries."""
    path = os.path.join(data_dir, subdir)
    if os.path.isdir(path) and os.listdir(path):
        return path
    if local_only:
        raise FileNotFoundError(
            f"Local dataset not found at '{path}'. Instance may be offline; download first:\n"
            f"  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download {hub_id} "
            f"--repo-type dataset --local-dir {path}\n"
            f"(or run scripts/setup/download_data.sh, then re-run with --data_dir {data_dir})")
    return hub_id


def _load_pku(data_dir, n, rng, local_only=True):
    from datasets import load_dataset
    target = _resolve_local(data_dir, "pku-saferlhf", "PKU-Alignment/PKU-SafeRLHF", local_only)
    ds = load_dataset(target, split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for j in idx:
        if len(out) >= n:
            break
        r = ds[j]
        # take one of the two responses at random; keep safety meta for coverage checks
        which = rng.choice([0, 1])
        resp = r[f"response_{which}"]
        safe = r[f"is_response_{which}_safe"]
        sev = r.get(f"response_{which}_severity_level", None)
        out.append({"source": "pku_saferlhf", "prompt": r["prompt"], "response": resp,
                    "meta": {"is_safe": bool(safe), "severity_level": sev, "resp_idx": which}})
    return out


def _load_uf(data_dir, n, rng, local_only=True):
    from datasets import load_dataset
    target = _resolve_local(data_dir, "ultrafeedback", "openbmb/UltraFeedback", local_only)
    ds = load_dataset(target, split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for j in idx:
        if len(out) >= n:
            break
        r = ds[j]
        comps = r.get("completions", []) or []
        if not comps:
            continue
        c = rng.choice(comps)
        resp = c.get("response", "")
        if not resp:
            continue
        ann = c.get("annotations", {}) or {}
        def _rate(k):
            try:
                return ann.get(k, {}).get("Rating", None)
            except Exception:
                return None
        out.append({"source": "ultrafeedback", "prompt": r["instruction"], "response": resp,
                    "meta": {"instruction_following": _rate("instruction_following"),
                             "truthfulness": _rate("truthfulness"),
                             "honesty": _rate("honesty"),
                             "helpfulness": _rate("helpfulness")}})
    return out


# ------------------------------ orchestration ----------------------------------
def _uid(item):
    h = hashlib.sha256((item["prompt"] + "||" + item["response"]).encode()).hexdigest()[:16]
    return f"{item['source'][:4]}_{h}"


def build_pool(data_dir, seed=0, local_only=True):
    rng = random.Random(seed)
    # target composition (see plan): 6k hard-safety, 3k task, 1k soft + splits
    pku = _load_pku(data_dir, 6400, rng, local_only=local_only)   # dedup headroom + conflict
    uf = _load_uf(data_dir, 3200, rng, local_only=local_only)
    soft = _soft_pairs(1300, rng)
    pool = pku + uf + soft
    rng.shuffle(pool)
    for it in pool:
        it["id"] = _uid(it)
    # dedup by id
    seen, dedup = set(), []
    for it in pool:
        if it["id"] in seen:
            continue
        seen.add(it["id"]); dedup.append(it)
    return dedup


def split_pool(pool, rng):
    n_train, n_calib, n_test, n_audit, n_conflict = 8000, 1000, 1000, 400, 400
    required = n_train + n_calib + n_test + n_audit + n_conflict
    if len(pool) < required:
        raise ValueError(f"pool has {len(pool)} unique items; {required} required")
    rng.shuffle(pool)
    cuts = {}
    i = 0
    for name, k in [("train", n_train), ("calib", n_calib), ("test", n_test),
                    ("audit", n_audit), ("conflict", n_conflict)]:
        cuts[name] = pool[i:i + k]
        i += k
    if any(len(cuts[name]) != k for name, k in
           [("train", n_train), ("calib", n_calib), ("test", n_test),
            ("audit", n_audit), ("conflict", n_conflict)]):
        raise AssertionError("split construction produced an unexpected size")
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/data")
    ap.add_argument("--out", default="outputs/pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_hub", action="store_true",
                    help="allow falling back to HF Hub download if local snapshot missing "
                         "(default: local-only, fail fast when offline)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    print("Building pool from PKU-SafeRLHF + UltraFeedback + self-built soft pairs ...")
    pool = build_pool(args.data_dir, seed=args.seed, local_only=not args.allow_hub)
    print(f"  pool size after dedup: {len(pool)}")
    splits = split_pool(pool, rng)

    # source composition report
    from collections import Counter
    for name, items in splits.items():
        comp = Counter(it["source"] for it in items)
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  {name:9s} n={len(items):5d}  {dict(comp)}  -> {path}")
    print("DONE sample_data.")


if __name__ == "__main__":
    main()
