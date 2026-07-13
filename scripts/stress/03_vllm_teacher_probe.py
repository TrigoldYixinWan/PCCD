#!/usr/bin/env python
"""PCCD Day-1 stress — Part 3: load 32B teacher on ONE card via vLLM and measure
labeling throughput + peak memory. This is the single most important Day-1 gate:
it confirms the teacher fits on one 96GB card with KV cache and hits the throughput
budget (>=500 labeled prompts/hour/GPU) that the 9-day plan assumes.

Run (GPU0): CUDA_VISIBLE_DEVICES=0 python scripts/stress/03_vllm_teacher_probe.py \
              --model /root/models/qwen32b --n 64 2>&1 | tee logs/stress_03.log
"""
import argparse, time, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/qwen32b")
    ap.add_argument("--n", type=int, default=64, help="number of probe prompts")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    # A representative policy-judgment prompt (short output, JSON) approximating real load.
    sys_p = ("You are a compliance critic. For the response, output a JSON object with keys "
             "H1..H5,S1..S3,T1,T2, each one of 'satisfied','violated','not_applicable'.")
    user_tmpl = ("PROMPT: {q}\nRESPONSE: {r}\nReturn ONLY the JSON policy-judgment object.")
    probe = [user_tmpl.format(q=f"Example user question number {i} about internal data.",
                              r=f"Example assistant response number {i} with some detail.")
             for i in range(args.n)]

    print(f"Loading teacher: {args.model}")
    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=4096, trust_remote_code=True)
    print(f"  load time: {time.time()-t0:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    msgs = [[{"role": "system", "content": sys_p},
             {"role": "user", "content": p}] for p in probe]

    t0 = time.time()
    outs = llm.chat(msgs, sp)
    dt = time.time() - t0
    thru_hr = args.n / dt * 3600
    print(f"\n[THROUGHPUT] {args.n} prompts in {dt:.1f}s "
          f"=> {args.n/dt:.2f}/s = {thru_hr:.0f}/hour/GPU")
    print(f"[GATE] target >=500/hour/GPU : {'PASS' if thru_hr >= 500 else 'CHECK'}")

    # quick JSON-parse success rate on outputs
    ok = 0
    for o in outs:
        txt = o.outputs[0].text
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            try:
                json.loads(txt[s:e+1]); ok += 1
            except Exception:
                pass
    print(f"[JSON parse] {ok}/{args.n} = {100*ok/args.n:.0f}%  (gate >=99% on real prompts)")
    print("DONE stress 03. Also check `nvidia-smi` peak mem < 96GB during this run.")

if __name__ == "__main__":
    main()
