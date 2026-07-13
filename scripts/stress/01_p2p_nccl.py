#!/usr/bin/env python
"""PCCD Day-1 stress — Part 1: P2P bandwidth + NCCL all-reduce across 2 GPUs.
Run: python scripts/stress/01_p2p_nccl.py 2>&1 | tee logs/stress_01.log
Verifies the two cards can talk (needed only for the online-cascade demo; the
main experiments keep one process per card and do NOT pool memory)."""
import os, time, torch

def p2p_bandwidth(src=0, dst=1, mb=512, iters=20):
    if torch.cuda.device_count() < 2:
        print("  <2 GPUs, skip P2P"); return
    n = mb * 1024 * 1024 // 4
    a = torch.empty(n, dtype=torch.float32, device=f"cuda:{src}")
    b = torch.empty(n, dtype=torch.float32, device=f"cuda:{dst}")
    for _ in range(3):
        b.copy_(a); torch.cuda.synchronize(dst)
    t0 = time.time()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize(dst)
    dt = time.time() - t0
    gb = mb / 1024 * iters
    print(f"  P2P cuda:{src}->cuda:{dst}  {gb/dt:.1f} GB/s  ({mb}MB x{iters})")

def nccl_allreduce():
    try:
        import torch.distributed as dist
        import torch.multiprocessing as mp
    except Exception as e:
        print("  torch.distributed unavailable:", e); return

    def worker(rank, world):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)
        x = torch.ones(1024 * 1024, device=f"cuda:{rank}") * (rank + 1)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  NCCL all-reduce OK  sum[0]={x[0].item():.0f} (expect {sum(range(1,world+1))})  "
                  f"{50/(time.time()-t0):.0f} it/s")
        dist.destroy_process_group()

    world = torch.cuda.device_count()
    if world < 2:
        print("  <2 GPUs, skip NCCL"); return
    mp.spawn(worker, args=(world,), nprocs=world, join=True)

if __name__ == "__main__":
    print("[P2P bandwidth]")
    p2p_bandwidth(0, 1)
    p2p_bandwidth(1, 0)
    print("[NCCL all-reduce]")
    nccl_allreduce()
    print("DONE stress 01.")
