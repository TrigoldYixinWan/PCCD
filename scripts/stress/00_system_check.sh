#!/usr/bin/env bash
# PCCD Day-1 stress test — Part 0: system / topology / driver check
# Run on AutoDL: bash scripts/stress/00_system_check.sh 2>&1 | tee logs/stress_00.log
set -uo pipefail
echo "==================================================================="
echo "PCCD STRESS 00 — SYSTEM & TOPOLOGY  ($(date))"
echo "==================================================================="

echo -e "\n[1] nvidia-smi (expect 2x RTX PRO 6000 96GB)"
nvidia-smi

echo -e "\n[2] GPU topology (expect P2P/NVLink or PCIe path)"
nvidia-smi topo -m

echo -e "\n[3] CUDA / driver versions"
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv

echo -e "\n[4] CPU / RAM"
nproc && free -h | head -2

echo -e "\n[5] Disk (expect data disk >=200GB free)"
df -h | grep -E "root|autodl|data" || df -h

echo -e "\n[6] Python / torch / CUDA visibility"
python - <<'PY'
import torch, platform
print("python     :", platform.python_version())
print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available())
print("gpu count  :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  {p.total_memory/1e9:.1f} GB  cc{p.major}.{p.minor}")
PY

echo -e "\n[7] Key libs importable"
python - <<'PY'
mods = ["transformers","trl","peft","vllm","mapie","netcal","datasets","accelerate"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK  {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:14s} {e}")
PY
echo -e "\nDONE stress 00."
