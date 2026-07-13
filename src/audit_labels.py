"""PCCD Day-2 teacher-label quality gates.

Two modes:

(1) STATIC audit over a produced label file (fast, CPU-only):
      python src/audit_labels.py static --labels outputs/labels/train.jsonl
    Reports: JSON parse-success rate, per-policy label distribution,
    not_applicable share, positive/negative coverage per hard policy.

(2) PERTURBATION audit (needs the teacher on GPU; run on the 400-item audit split):
      CUDA_VISIBLE_DEVICES=0 python src/audit_labels.py perturb \
        --model /root/models/qwen32b --audit outputs/pool/audit.jsonl \
        --out outputs/labels/audit_perturb.jsonl
    Re-labels each item 3 ways — canonical, policy-order-swapped, policy-paraphrased —
    and reports agreement, i.e. how stable teacher judgments are to superficial
    prompt changes. This is the Day-3 Go/No-Go for label reliability.

Gates (from configs/teacher_schema.json):
    json_parse_success >= 0.99
    order_swap_consistency >= 0.90
    paraphrase_consistency >= 0.90
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.policy_defs import (POLICY_IDS, hard_ids, build_messages,  # noqa: E402
                             parse_judgment, schema)


def _read_labels(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def static_audit(path):
    recs = _read_labels(path)
    n = len(recs)
    ok = sum(1 for r in recs if r.get("parse_ok"))
    print(f"=== STATIC AUDIT {path} ===")
    print(f"items                 : {n}")
    print(f"json parse-success    : {ok}/{n} = {100*ok/max(1,n):.2f}%  "
          f"(gate >=99%: {'PASS' if ok/max(1,n) >= 0.99 else 'FAIL'})")
    attempts = Counter(r.get("attempts", 1) for r in recs if r.get("parse_ok"))
    print(f"attempts distribution : {dict(sorted(attempts.items()))}")

    # per-policy label distribution + N/A share + pos/neg coverage
    dist = {p: Counter() for p in POLICY_IDS}
    for r in recs:
        lab = r.get("labels") or {}
        for p in POLICY_IDS:
            if p in lab:
                dist[p][lab[p]] += 1
    print("\nper-policy label distribution (sat / vio / n_a):")
    hard = set(hard_ids())
    for p in POLICY_IDS:
        c = dist[p]
        tot = sum(c.values()) or 1
        sat, vio, na = c["satisfied"], c["violated"], c["not_applicable"]
        flag = ""
        if p in hard and (vio == 0 or sat == 0):
            flag = "  <-- WARN: no pos/neg coverage on a HARD policy"
        na_share = na / tot
        na_flag = "  <-- WARN: N/A>60%" if na_share > 0.60 else ""
        print(f"  {p}: {sat:5d} / {vio:5d} / {na:5d}  (N/A {100*na_share:4.1f}%)"
              f"{flag}{na_flag}")
    print("\nStatic audit done.")


def _labels_equal(a, b, ids=None):
    ids = ids or POLICY_IDS
    if a is None or b is None:
        return False
    return all(a.get(p) == b.get(p) for p in ids)


def perturb_audit(model, audit_path, out_path, max_tokens=256,
                  gpu_mem_util=0.90, max_model_len=4096, seed=0):
    from vllm import LLM, SamplingParams
    rng = random.Random(seed)
    items = _read_labels(audit_path)
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util,
              max_model_len=max_model_len, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # build 3 variants per item
    canon, swapped, para = [], [], []
    for it in items:
        p, r = it["prompt"], it["response"]
        canon.append(build_messages(p, r))
        order = POLICY_IDS[:]
        rng.shuffle(order)
        swapped.append(build_messages(p, r, order=order))
        para.append(build_messages(p, r, paraphrase=True))

    def _label(msgs):
        outs = llm.chat(msgs, sp)
        return [parse_judgment(o.outputs[0].text) for o in outs]

    lc, ls, lp = _label(canon), _label(swapped), _label(para)

    n = len(items)
    ok_c = sum(1 for x in lc if x is not None)
    # consistency: canonical vs perturbed, over items where both parsed
    def _consistency(base, other, ids=None):
        num = den = 0
        per_policy = defaultdict(lambda: [0, 0])  # pid -> [agree, total]
        for a, b in zip(base, other):
            if a is None or b is None:
                continue
            den += 1
            if _labels_equal(a, b, ids):
                num += 1
            for pid in (ids or POLICY_IDS):
                per_policy[pid][1] += 1
                if a.get(pid) == b.get(pid):
                    per_policy[pid][0] += 1
        return (num / den if den else 0.0), per_policy, den

    swap_cons, swap_pp, swap_den = _consistency(lc, ls)
    para_cons, para_pp, para_den = _consistency(lc, lp)

    gates = schema()["gates"]
    print(f"=== PERTURBATION AUDIT (n={n}) ===")
    print(f"json parse-success (canonical): {ok_c}/{n} = {100*ok_c/n:.2f}%  "
          f"(gate >={100*gates['json_parse_success']:.0f}%: "
          f"{'PASS' if ok_c/n >= gates['json_parse_success'] else 'FAIL'})")
    print(f"order-swap consistency  : {100*swap_cons:.1f}%  (n={swap_den})  "
          f"(gate >={100*gates['order_swap_consistency']:.0f}%: "
          f"{'PASS' if swap_cons >= gates['order_swap_consistency'] else 'FAIL'})")
    print(f"paraphrase consistency  : {100*para_cons:.1f}%  (n={para_den})  "
          f"(gate >={100*gates['paraphrase_consistency']:.0f}%: "
          f"{'PASS' if para_cons >= gates['paraphrase_consistency'] else 'FAIL'})")
    print("\nper-policy paraphrase agreement (lowest first):")
    rows = sorted(((pid, a / t if t else 0) for pid, (a, t) in para_pp.items()),
                  key=lambda x: x[1])
    for pid, ag in rows:
        print(f"  {pid}: {100*ag:.1f}%")

    # persist raw for later inspection
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for it, a, b, c in zip(items, lc, ls, lp):
            f.write(json.dumps({"id": it["id"], "canonical": a,
                                "order_swap": b, "paraphrase": c},
                               ensure_ascii=False) + "\n")
    print(f"\nWrote per-item perturbation labels -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("static")
    s.add_argument("--labels", required=True)
    p = sub.add_parser("perturb")
    p.add_argument("--model", default="/root/models/qwen32b")
    p.add_argument("--audit", required=True)
    p.add_argument("--out", default="outputs/labels/audit_perturb.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "static":
        static_audit(args.labels)
    else:
        perturb_audit(args.model, args.audit, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
