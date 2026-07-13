"""PCCD Day-2 teacher labeling (label-only).

Loads the 32B teacher on ONE card via vLLM and produces per-policy JSON judgments
for a pool split. Designed to run as two independent processes (one per GPU) on
two shards of the data — no cross-GPU communication, no memory pooling.

Key properties:
  * label-only: teacher outputs ONLY the 10-policy JSON, never a rewritten response
    (avoids the More-is-Less shortcut where teacher-written text leaks provenance).
  * greedy decoding (temperature=0) for reproducible labels.
  * automatic JSON repair via up to `--retries` re-generations with a stricter
    reminder; unparseable items are written with label=None and flagged.
  * resumable: skips ids already present in the output file.

Output: JSONL, one line per item:
  {id, source, prompt, response, labels:{H1..T2}, parse_ok, attempts, meta}

Usage (per GPU, run two in parallel — see scripts/day2_label.sh):
  CUDA_VISIBLE_DEVICES=0 python src/gen_labels.py \
     --model /root/models/qwen32b --in outputs/pool/train.jsonl \
     --out outputs/labels/train.shardA.jsonl --shard 0 --num_shards 2
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.policy_defs import build_messages, parse_judgment, POLICY_IDS  # noqa: E402


def load_pool(path, shard, num_shards):
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if num_shards > 1 and (i % num_shards) != shard:
                continue
            items.append(json.loads(line))
    return items


def load_done_ids(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/qwen32b")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--retries", type=int, default=2, help="re-gen attempts for bad JSON")
    ap.add_argument("--batch", type=int, default=256, help="chat batch size")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from vllm import LLM, SamplingParams

    items = load_pool(args.inp, args.shard, args.num_shards)
    done = load_done_ids(args.out)
    todo = [it for it in items if it["id"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] pool={len(items)} done={len(done)} todo={len(todo)}")
    if not todo:
        print("nothing to do"); return

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, trust_remote_code=True)
    print(f"  teacher loaded in {time.time()-t0:.1f}s")

    sp_greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    # slightly higher temp on retries to escape a bad greedy formatting loop
    sp_retry = SamplingParams(temperature=0.3, max_tokens=args.max_tokens)

    fout = open(args.out, "a")
    n_ok = 0
    t_label0 = time.time()
    for b0 in range(0, len(todo), args.batch):
        batch = todo[b0:b0 + args.batch]
        msgs = [build_messages(it["prompt"], it["response"]) for it in batch]
        outs = llm.chat(msgs, sp_greedy)
        results: list = [None] * len(batch)   # each slot becomes a dict or stays None
        attempts = [1] * len(batch)
        pending = []
        for i, o in enumerate(outs):
            j = parse_judgment(o.outputs[0].text)
            if j is not None:
                results[i] = j
            else:
                pending.append(i)
        # retry loop for unparseable
        for _ in range(args.retries):
            if not pending:
                break
            rmsgs = [build_messages(batch[i]["prompt"], batch[i]["response"]) for i in pending]
            # append a stricter reminder
            for m in rmsgs:
                m[-1]["content"] += ("\n\nIMPORTANT: Output ONLY a valid JSON object with "
                                     "exactly keys " + ",".join(POLICY_IDS) +
                                     " and values satisfied/violated/not_applicable.")
            routs = llm.chat(rmsgs, sp_retry)
            still = []
            for k, i in enumerate(pending):
                attempts[i] += 1
                j = parse_judgment(routs[k].outputs[0].text)
                if j is not None:
                    results[i] = j
                else:
                    still.append(i)
            pending = still
        # write batch
        for i, it in enumerate(batch):
            rec = {"id": it["id"], "source": it["source"],
                   "prompt": it["prompt"], "response": it["response"],
                   "labels": results[i], "parse_ok": results[i] is not None,
                   "attempts": attempts[i], "meta": it.get("meta", {})}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if results[i] is not None:
                n_ok += 1
        fout.flush()
        rate = (b0 + len(batch)) / max(1e-6, time.time() - t_label0) * 3600
        print(f"  [{b0+len(batch)}/{len(todo)}] parse_ok_so_far={n_ok} "
              f"~{rate:.0f}/hour/GPU")
    fout.close()
    dt = time.time() - t_label0
    print(f"DONE shard {args.shard}: labeled {len(todo)} in {dt/60:.1f} min "
          f"({len(todo)/dt*3600:.0f}/hour), parse_ok={n_ok}/{len(todo)} "
          f"= {100*n_ok/len(todo):.1f}%")


if __name__ == "__main__":
    main()
