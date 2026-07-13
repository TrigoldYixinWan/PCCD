#!/usr/bin/env bash
# PCCD Day-2 orchestration: sample pool -> dual-GPU teacher labeling -> static audit.
# Run from repo root:  bash scripts/day2/run_day2.sh
#
# Layout of teacher labeling (no cross-GPU comm; one 32B teacher per card):
#   GPU0 labels even-indexed items of each split (shard 0)
#   GPU1 labels odd-indexed  items of each split (shard 1)
# Splits labeled: train, calib, test, conflict. (audit split uses perturbation audit.)
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
mkdir -p outputs/pool outputs/labels logs

MODEL="${MODEL:-/root/models/qwen32b}"
DATA="${DATA:-/root/data}"

echo "### PCCD Day-2 start $(date) ###"

# 0) ensure datasets are present locally (download if missing)
if [ ! -d "$DATA/pku-saferlhf" ] || [ ! -d "$DATA/ultrafeedback" ]; then
  echo "[0] datasets missing under $DATA -> downloading ..."
  bash scripts/setup/download_data.sh
fi

# 1) sample the pool (CPU, once). Local-only: fails fast with a clear message if offline.
if [ ! -f outputs/pool/train.jsonl ]; then
  echo "[1] sampling pool ..."
  python src/sample_data.py --data_dir "$DATA" --out outputs/pool --seed 0
else
  echo "[1] pool already exists, skip sampling"
fi

# 2) dual-GPU labeling — launch one process per card, per split, in background.
label_split () {
  local split="$1"
  echo "[2] labeling split=$split on 2 GPUs ..."
  CUDA_VISIBLE_DEVICES=0 python src/gen_labels.py --model "$MODEL" \
    --in "outputs/pool/${split}.jsonl" \
    --out "outputs/labels/${split}.shardA.jsonl" \
    --shard 0 --num_shards 2 > "logs/label_${split}_A.log" 2>&1 &
  local pidA=$!
  CUDA_VISIBLE_DEVICES=1 python src/gen_labels.py --model "$MODEL" \
    --in "outputs/pool/${split}.jsonl" \
    --out "outputs/labels/${split}.shardB.jsonl" \
    --shard 1 --num_shards 2 > "logs/label_${split}_B.log" 2>&1 &
  local pidB=$!
  wait $pidA $pidB
  # merge shards
  cat "outputs/labels/${split}.shardA.jsonl" "outputs/labels/${split}.shardB.jsonl" \
    > "outputs/labels/${split}.jsonl"
  echo "    merged -> outputs/labels/${split}.jsonl ($(wc -l < outputs/labels/${split}.jsonl) lines)"
}

# Day-2 core: train + calib + test. (conflict labeled Day-3 alongside perturbation audit.)
for split in train calib test; do
  label_split "$split"
done

# 3) static audit on each produced file
echo "[3] static audit ..."
for split in train calib test; do
  echo "----- $split -----"
  python src/audit_labels.py static --labels "outputs/labels/${split}.jsonl" \
    2>&1 | tee "logs/audit_static_${split}.log"
done

echo "### PCCD Day-2 done $(date) ###"
echo "Next (Day-3): conflict split + perturbation audit:"
echo "  bash scripts/day2/run_day2.sh   # (train/calib/test)"
echo "  CUDA_VISIBLE_DEVICES=0 python src/audit_labels.py perturb \\"
echo "     --model $MODEL --audit outputs/pool/audit.jsonl \\"
echo "     --out outputs/labels/audit_perturb.jsonl | tee logs/audit_perturb.log"
