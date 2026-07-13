#!/usr/bin/env python
"""PCCD Day-1 stress — Part 2: sustained dual-GPU matmul burn + thermal/power/throttle log.
Run: python scripts/stress/02_burn_thermal.py --minutes 20 2>&1 | tee logs/stress_02.log

Purpose: confirm both RTX PRO 6000 hold sustained load without thermal throttling or
power anomalies before committing to multi-hour teacher labeling / training.
Samples nvidia-smi every 15s in the background while both GPUs run large bf16 matmuls."""
import argparse, subprocess, threading, time, torch

def smi_logger(stop_evt, period=15):
    q = ("index,name,temperature.gpu,utilization.gpu,power.draw,power.limit,"
         "clocks.sm,clocks_throttle_reasons.active,memory.used,memory.total")
    print(f"{'t(s)':>6} {'gpu':>3} {'temp':>4} {'util':>4} {'pW':>6} {'limW':>6} "
          f"{'smMHz':>6} {'throttle':>10} {'memGB':>6}")
    t0 = time.time()
    while not stop_evt.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                text=True)
            for line in out.strip().splitlines():
                f = [x.strip() for x in line.split(",")]
                idx, name, temp, util, pw, lim, sm, thr, mu, mt = f
                thr_flag = "0x0" if thr in ("0x0000000000000000", "Not Active", "0x0") else thr
                print(f"{time.time()-t0:6.0f} {idx:>3} {temp:>4} {util:>4} {pw:>6} {lim:>6} "
                      f"{sm:>6} {thr_flag:>10} {float(mu)/1024:>6.1f}")
        except Exception as e:
            print("smi err:", e)
        stop_evt.wait(period)

def burn(gpu, stop_evt, dim=8192):
    torch.cuda.set_device(gpu)
    a = torch.randn(dim, dim, dtype=torch.bfloat16, device=f"cuda:{gpu}")
    b = torch.randn(dim, dim, dtype=torch.bfloat16, device=f"cuda:{gpu}")
    while not stop_evt.is_set():
        c = a @ b
        a = (c * 1e-4).to(torch.bfloat16)
        torch.cuda.synchronize(gpu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--dim", type=int, default=8192)
    args = ap.parse_args()
    ng = torch.cuda.device_count()
    print(f"Burning {ng} GPU(s) for {args.minutes} min, matmul dim={args.dim}")
    stop = threading.Event()
    logger = threading.Thread(target=smi_logger, args=(stop,), daemon=True)
    logger.start()
    burners = [threading.Thread(target=burn, args=(g, stop, args.dim), daemon=True)
               for g in range(ng)]
    for t in burners: t.start()
    try:
        time.sleep(args.minutes * 60)
    finally:
        stop.set()
        time.sleep(2)
    print("DONE stress 02. Check: temp<85C, no persistent throttle flag, power near limit.")

if __name__ == "__main__":
    main()
