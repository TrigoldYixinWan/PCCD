#!/usr/bin/env bash
# PCCD Day-1 stress — master runner. Run from repo root on AutoDL:
#   bash scripts/stress/run_all.sh
# Paste the full logs/ back to the assistant for analysis.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
echo "### PCCD stress suite start $(date) ###"

echo -e "\n===== 00 system check ====="
bash scripts/stress/00_system_check.sh 2>&1 | tee logs/stress_00.log

echo -e "\n===== 01 P2P + NCCL ====="
python scripts/stress/01_p2p_nccl.py 2>&1 | tee logs/stress_01.log

echo -e "\n===== 04 NVMe I/O ====="
bash scripts/stress/04_nvme_io.sh 2>&1 | tee logs/stress_04.log

echo -e "\n===== 02 burn/thermal (20 min) ====="
python scripts/stress/02_burn_thermal.py --minutes 20 2>&1 | tee logs/stress_02.log

echo -e "\n===== 03 vLLM teacher probe (needs /root/models/qwen32b) ====="
if [ -d /root/models/qwen32b ]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/stress/03_vllm_teacher_probe.py \
    --model /root/models/qwen32b --n 64 2>&1 | tee logs/stress_03.log
else
  echo "SKIP 03: /root/models/qwen32b not present yet. Run after model download." | tee logs/stress_03.log
fi

echo -e "\n===== auto-verdict ====="
python scripts/stress/verdict.py 2>&1 | tee logs/stress_verdict.log

echo -e "\n### PCCD stress suite done $(date) ###"
echo "GATES: (a) 2x96GB visible  (b) no persistent throttle & temp<85C"
echo "       (c) NVMe >1GB/s  (d) 32B loads on 1 card, throughput>=500/hr, JSON>=99%"
echo "Paste logs/stress_*.log back to the assistant for analysis."
