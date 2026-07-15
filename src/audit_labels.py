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
    Labels each item once canonically and under 3 registered perturbations — an
    independent repeat of the canonical call, policy-order swap, and policy
    paraphrase — and reports agreement with the canonical labels. This is the
    Day-3 Go/No-Go for label reliability.

Gates (from configs/teacher_schema.json):
    json_parse_success >= 0.99
    order_swap_consistency >= 0.90
    paraphrase_consistency >= 0.90
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.policy_defs import (POLICY_IDS, hard_ids, build_messages,  # noqa: E402
                             parse_judgment, parse_judgment_cells,
                             policy_text, schema)

CONFIRMATORY_AUDIT_SHA256 = "f7f2a84a5a30c0411ed251ef1dc7a30fd2a6475d797ca25fc651206bc92df6a1"


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

    # Build canonical plus the two prompt-changing perturbations. The repeat-
    # sampling perturbation deliberately reuses `canon` in a separate llm.chat
    # call below; canonical itself must not be counted as the repeat.
    canon, swapped, para, swap_orders = [], [], [], []
    for it in items:
        p, r = it["prompt"], it["response"]
        canon.append(build_messages(p, r))
        order = POLICY_IDS[:]
        rng.shuffle(order)
        if order == POLICY_IDS:
            # A swap perturbation must actually change the presentation order.
            order = order[1:] + order[:1]
        swap_orders.append(order)
        swapped.append(build_messages(p, r, order=order))
        para.append(build_messages(p, r, paraphrase=True))

    def _label(msgs):
        outs = llm.chat(msgs, sp)
        return [parse_judgment(o.outputs[0].text) for o in outs]

    lc = _label(canon)
    lr = _label(canon)  # independent repeat_sampling call at registered temperature=0
    ls = _label(swapped)
    lp = _label(para)

    n = len(items)
    parse_ok = {
        "canonical": sum(1 for x in lc if x is not None),
        "repeat_sampling": sum(1 for x in lr if x is not None),
        "policy_order_swap": sum(1 for x in ls if x is not None),
        "policy_paraphrase": sum(1 for x in lp if x is not None),
    }
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

    repeat_cons, repeat_pp, repeat_den = _consistency(lc, lr)
    swap_cons, swap_pp, swap_den = _consistency(lc, ls)
    para_cons, para_pp, para_den = _consistency(lc, lp)

    gates = schema()["gates"]
    print(f"=== PERTURBATION AUDIT (n={n}) ===")
    ok_c = parse_ok["canonical"]
    print(f"json parse-success (canonical      ): {ok_c}/{n} = {100*ok_c/n:.2f}%  "
          f"(gate >={100*gates['json_parse_success']:.0f}%: "
          f"{'PASS' if ok_c/n >= gates['json_parse_success'] else 'FAIL'})")
    for variant in ("repeat_sampling", "policy_order_swap", "policy_paraphrase"):
        ok = parse_ok[variant]
        print(f"json parse-success ({variant:15s}): {ok}/{n} = {100*ok/n:.2f}%  "
              "(descriptive; no separate registered threshold)")

    def _micro_agreement(per_policy):
        agree = sum(a for a, _ in per_policy.values())
        total = sum(t for _, t in per_policy.values())
        return agree / total if total else 0.0, total

    repeat_micro, repeat_cells = _micro_agreement(repeat_pp)
    swap_micro, swap_cells = _micro_agreement(swap_pp)
    para_micro, para_cells = _micro_agreement(para_pp)
    print(f"repeat-sampling whole-record exact-match: {100*repeat_cons:.1f}%  "
          f"(n={repeat_den}); policy-cell agreement: {100*repeat_micro:.1f}% "
          f"(n={repeat_cells})")
    print(f"order-swap whole-record exact-match  : {100*swap_cons:.1f}%  (n={swap_den})  "
          f"(gate >={100*gates['order_swap_consistency']:.0f}%: "
          f"{'PASS' if swap_cons >= gates['order_swap_consistency'] else 'FAIL'}); "
          f"policy-cell agreement: {100*swap_micro:.1f}% (n={swap_cells})")
    print(f"paraphrase whole-record exact-match : {100*para_cons:.1f}%  (n={para_den})  "
          f"(gate >={100*gates['paraphrase_consistency']:.0f}%: "
          f"{'PASS' if para_cons >= gates['paraphrase_consistency'] else 'FAIL'}); "
          f"policy-cell agreement: {100*para_micro:.1f}% (n={para_cells})")
    print("\nper-policy agreement with canonical:")
    print("  policy  repeat_sampling  order_swap  paraphrase")
    for pid in POLICY_IDS:
        ra, rt = repeat_pp[pid]
        sa, st = swap_pp[pid]
        pa, pt = para_pp[pid]
        rag = ra / rt if rt else 0.0
        sag = sa / st if st else 0.0
        pag = pa / pt if pt else 0.0
        print(f"  {pid:>4s}       {100*rag:6.1f}%       {100*sag:6.1f}%      {100*pag:6.1f}%")

    # persist raw for later inspection
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        for it, order, a, r, b, c in zip(items, swap_orders, lc, lr, ls, lp):
            f.write(json.dumps({"id": it["id"], "canonical": a,
                                "repeat_sampling": r,
                                "policy_order_swap_order": order,
                                "policy_order_swap": b,
                                "policy_paraphrase": c},
                               ensure_ascii=False) + "\n")
    print(f"\nWrote per-item perturbation labels -> {out_path}")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value):
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def confirmatory_latin_order(row_index):
    """Return a non-canonical cyclic Latin-square row for one audit item.

    Swapping the first two policies in the base permutation makes every one of
    the ten rotations a real order perturbation while preserving exact balance:
    over every ten rows, every policy occupies every position once.
    """
    base = POLICY_IDS[:]
    base[0], base[1] = base[1], base[0]
    shift = row_index % len(base)
    return base[shift:] + base[:shift]


def validate_confirmatory_schedule(items):
    if len(items) != 400:
        raise ValueError(f"confirmatory audit requires exactly 400 items, found {len(items)}")
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("every audit item must have a non-empty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("confirmatory audit item ids must be unique")
    for item in items:
        if not isinstance(item.get("prompt"), str) or not isinstance(item.get("response"), str):
            raise ValueError(f"audit id={item.get('id')!r}: prompt/response must be strings")

    orders = [confirmatory_latin_order(index) for index in range(len(items))]
    canonical = list(POLICY_IDS)
    if any(order == canonical for order in orders):
        raise ValueError("confirmatory order perturbation contains an unchanged canonical row")
    position_counts = {
        pid: [sum(order[position] == pid for order in orders) for position in range(10)]
        for pid in POLICY_IDS
    }
    if any(count != 40 for counts in position_counts.values() for count in counts):
        raise ValueError(f"Latin-square position balance failed: {position_counts}")
    schedule = [
        {"id": item["id"], "order": order}
        for item, order in zip(items, orders)
    ]
    return orders, position_counts, _sha256_json(schedule)


def validate_frozen_paraphrases(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    expected = {pid: policy_text(pid, paraphrase=True) for pid in POLICY_IDS}
    if not isinstance(frozen, dict) or list(frozen) != POLICY_IDS:
        raise ValueError("frozen paraphrase artifact must contain POLICY_IDS in registered order")
    if frozen != expected:
        differences = {
            pid: {"frozen": frozen.get(pid), "code": expected[pid]}
            for pid in POLICY_IDS
            if frozen.get(pid) != expected[pid]
        }
        raise ValueError(f"frozen paraphrases differ from src.policy_defs: {differences}")
    return frozen, _sha256_file(path)


def confirmatory_audit(
    model,
    audit_path,
    out_path,
    frozen_paraphrases,
    marker_path=None,
    max_tokens=256,
    gpu_mem_util=0.90,
    max_model_len=4096,
    seed=20260715,
):
    """Run the single locked L1 confirmatory teacher audit without retries."""
    from vllm import LLM, SamplingParams

    audit_path = Path(audit_path)
    out_path = Path(out_path)
    marker_path = Path(marker_path) if marker_path else Path(str(out_path) + ".run.json")
    if out_path.name == "audit_perturb.jsonl":
        raise ValueError("confirmatory mode refuses to overwrite the frozen first-run artifact")
    if out_path.exists() or marker_path.exists():
        raise FileExistsError(
            "confirmatory output or run marker already exists; the locked protocol forbids "
            f"a second run (output={out_path.exists()}, marker={marker_path.exists()})"
        )

    items = _read_labels(audit_path)
    orders, position_counts, schedule_sha256 = validate_confirmatory_schedule(items)
    paraphrases, paraphrase_sha256 = validate_frozen_paraphrases(frozen_paraphrases)
    audit_sha256 = _sha256_file(audit_path)
    if audit_sha256 != CONFIRMATORY_AUDIT_SHA256:
        raise ValueError(
            "confirmatory audit pool differs from the frozen Day-2/3 artifact: "
            f"observed={audit_sha256}, expected={CONFIRMATORY_AUDIT_SHA256}"
        )
    started_at = datetime.now(timezone.utc).isoformat()
    marker = {
        "schema_version": "pccd.day3.l1_confirmatory_run.v1",
        "status": "started",
        "single_run_guard": True,
        "started_at_utc": started_at,
        "git_revision": _git_revision(),
        "model": str(model),
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha256,
        "output_path": str(out_path),
        "frozen_paraphrases_path": str(frozen_paraphrases),
        "frozen_paraphrases_sha256": paraphrase_sha256,
        "frozen_paraphrases": paraphrases,
        "latin_schedule_sha256": schedule_sha256,
        "latin_base_order": confirmatory_latin_order(0),
        "latin_unique_rows": len({tuple(order) for order in orders}),
        "latin_position_counts": position_counts,
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_mem_util,
        "retries": 0,
    }
    _write_json_atomic(marker_path, marker)

    print("=== L1 CONFIRMATORY AUDIT: LOCKED SINGLE RUN ===", flush=True)
    print(f"started: {started_at}", flush=True)
    print(f"git revision: {marker['git_revision']}", flush=True)
    print(f"audit: {audit_path} sha256={audit_sha256}", flush=True)
    print(
        f"frozen paraphrases: {frozen_paraphrases} sha256={paraphrase_sha256}",
        flush=True,
    )
    print(f"Latin schedule sha256={schedule_sha256}", flush=True)
    print("Latin position balance: every policy x position = 40", flush=True)
    print(f"run marker: {marker_path}", flush=True)
    print("calls: canonical + independent repeat + Latin order + paraphrase; no retries", flush=True)

    canonical_messages = []
    swapped_messages = []
    paraphrase_messages = []
    for item, order in zip(items, orders):
        prompt, response = item["prompt"], item["response"]
        canonical_messages.append(build_messages(prompt, response))
        swapped_messages.append(build_messages(prompt, response, order=order))
        paraphrase_messages.append(build_messages(prompt, response, paraphrase=True))

    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=seed,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def label_once(name, messages):
        print(f"starting independent call batch: {name} ({len(messages)} items)", flush=True)
        outputs = llm.chat(messages, sampling)
        if len(outputs) != len(messages):
            raise RuntimeError(f"{name}: vLLM returned {len(outputs)} outputs")
        parsed = []
        for output in outputs:
            raw_text = output.outputs[0].text
            cell_parse = parse_judgment_cells(raw_text)
            parsed.append({"raw_text": raw_text, **cell_parse})
        strict = sum(record["strict_parse_ok"] for record in parsed)
        cells = sum(len(record["cells"]) for record in parsed)
        print(
            f"finished {name}: strict={strict}/{len(parsed)} valid_cells={cells}/{10*len(parsed)}",
            flush=True,
        )
        return parsed

    canonical = label_once("canonical", canonical_messages)
    repeat = label_once("repeat_sampling", canonical_messages)
    swapped = label_once("policy_order_swap", swapped_messages)
    paraphrased = label_once("policy_paraphrase", paraphrase_messages)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = out_path.with_name(out_path.name + ".tmp")
    with temporary_output.open("x", encoding="utf-8") as handle:
        for item, order, canonical_result, repeat_result, swap_result, para_result in zip(
            items, orders, canonical, repeat, swapped, paraphrased
        ):
            record = {
                "id": item["id"],
                "policy_order_swap_order": order,
                "variants": {
                    "canonical": canonical_result,
                    "repeat_sampling": repeat_result,
                    "policy_order_swap": swap_result,
                    "policy_paraphrase": para_result,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_output, out_path)

    marker.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": _sha256_file(out_path),
            "strict_parse_counts": {
                "canonical": sum(record["strict_parse_ok"] for record in canonical),
                "repeat_sampling": sum(record["strict_parse_ok"] for record in repeat),
                "policy_order_swap": sum(record["strict_parse_ok"] for record in swapped),
                "policy_paraphrase": sum(record["strict_parse_ok"] for record in paraphrased),
            },
            "valid_cell_counts": {
                "canonical": sum(len(record["cells"]) for record in canonical),
                "repeat_sampling": sum(len(record["cells"]) for record in repeat),
                "policy_order_swap": sum(len(record["cells"]) for record in swapped),
                "policy_paraphrase": sum(len(record["cells"]) for record in paraphrased),
            },
        }
    )
    _write_json_atomic(marker_path, marker)
    print(f"CONFIRMATORY RUN COMPLETE: {out_path}", flush=True)
    print(f"output sha256={marker['output_sha256']}", flush=True)
    print("single-run marker sealed with status=complete", flush=True)


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
    c = sub.add_parser("confirmatory")
    c.add_argument("--model", required=True)
    c.add_argument("--audit", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--frozen-paraphrases", required=True)
    c.add_argument("--run-marker")
    c.add_argument("--seed", type=int, default=20260715)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--gpu-mem-util", type=float, default=0.90)
    c.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()
    if args.mode == "static":
        static_audit(args.labels)
    elif args.mode == "perturb":
        perturb_audit(args.model, args.audit, args.out, seed=args.seed)
    else:
        confirmatory_audit(
            args.model,
            args.audit,
            args.out,
            args.frozen_paraphrases,
            marker_path=args.run_marker,
            max_tokens=args.max_tokens,
            gpu_mem_util=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
