#!/usr/bin/env python
"""PCCD Day-1 stress — auto-verdict. Parses logs/stress_*.log and prints PASS/CHECK/FAIL
against the 4 gates so you (and the assistant) get an immediate summary.
Run AFTER run_all.sh:  python scripts/stress/verdict.py
"""
import os, re, glob

LOGDIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")

def read(name):
    p = os.path.join(LOGDIR, name)
    return open(p, errors="ignore").read() if os.path.exists(p) else ""

def gate_a(t):  # 2x 96GB visible
    n = len(re.findall(r"RTX PRO 6000", t)) or t.count("cc")  # cc lines from torch props
    mem96 = len(re.findall(r"9[0-9]\.\d\s*GB|9[0-9]\d{3}\s*MiB|97871MiB|9[0-9]\.\dGB", t))
    gpus = re.search(r"gpu count\s*:\s*(\d+)", t)
    ng = int(gpus.group(1)) if gpus else 0
    ok = ng >= 2
    return ("PASS" if ok else "CHECK"), f"torch gpu count={ng}; RTXPRO6000 mentions={n}"

def gate_b(t):  # thermal / throttle
    temps = [int(x) for x in re.findall(r"^\s*\d+\s+\d+\s+(\d{2,3})\s", t, re.M)]
    maxt = max(temps) if temps else -1
    throttles = re.findall(r"0x[0-9a-fA-F]{4,}", t)
    persist = [x for x in throttles if x not in ("0x0", "0x0000000000000000")]
    ok = (0 < maxt < 85) and (len(persist) == 0)
    return ("PASS" if ok else "CHECK"), f"max_temp={maxt}C; nonzero_throttle_samples={len(persist)}"

def gate_c(t):  # NVMe
    speeds = re.findall(r"([\d.]+)\s*(GB|MB)/s", t)
    vals = []
    for v, u in speeds:
        gb = float(v) * (1 if u == "GB" else 1/1024)
        vals.append(gb)
    best = max(vals) if vals else -1
    ok = best >= 1.0
    return ("PASS" if ok else "CHECK"), f"best_io={best:.2f} GB/s"

def gate_d(t):  # 32B teacher: throughput + JSON
    if "SKIP 03" in t or not t.strip():
        return ("PENDING", "teacher probe not run yet (model not downloaded)")
    thr = re.search(r"=\s*(\d+)/hour/GPU", t)
    js = re.search(r"\[JSON parse\].*?=\s*(\d+)%", t)
    thr_v = int(thr.group(1)) if thr else -1
    js_v = int(js.group(1)) if js else -1
    ok = thr_v >= 500 and js_v >= 99
    return ("PASS" if ok else "CHECK"), f"throughput={thr_v}/hr; json_ok={js_v}%"

def main():
    logs = {p.split("/")[-1]: read(p.split("/")[-1])
            for p in glob.glob(os.path.join(LOGDIR, "stress_*.log"))}
    allt = "\n".join(logs.values())
    print("=" * 60)
    print("PCCD Day-1 STRESS VERDICT")
    print("=" * 60)
    for gate, fn in [("A 2x96GB visible", gate_a),
                     ("B thermal/throttle", gate_b),
                     ("C NVMe >1GB/s", gate_c),
                     ("D 32B teacher", gate_d)]:
        status, detail = fn(allt)
        print(f"[{status:7s}] Gate {gate:22s} — {detail}")
    print("-" * 60)
    print("PASS=go, CHECK=inspect log, PENDING=run after model download, FAIL=blocker")

if __name__ == "__main__":
    main()
