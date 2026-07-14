#!/usr/bin/env bash
# PCCD Day-2 orchestration: sample pool -> dual-GPU teacher labeling -> static audit.
# Run from repo root:  bash scripts/day2/run_day2.sh
#
# All heavy paths live on the DATA disk via scripts/setup/env.sh (models, datasets,
# pool, labels). Only small logs stay in the repo. One 32B teacher per card, no
# cross-GPU comm: GPU0 = shard 0, GPU1 = shard 1.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/setup/env.sh
mkdir -p logs "$PCCD_OUT/pool" "$PCCD_OUT/labels"

MODEL="${MODEL:-$MODELS_DIR/qwen32b}"
DATA="${DATA:-$DATA_DIR}"
POOL="$PCCD_OUT/pool"
LAB="$PCCD_OUT/labels"

echo "### PCCD Day-2 start $(date) ###"
echo "MODEL=$MODEL  DATA=$DATA  POOL=$POOL  LAB=$LAB"

# 0) ensure datasets present on data disk (download if missing)
if [ ! -d "$DATA/pku-saferlhf" ] || [ ! -d "$DATA/ultrafeedback" ]; then
  echo "[0] datasets missing under $DATA -> downloading ..."
  bash scripts/setup/download_data.sh
fi

# 1) sample the pool (CPU, once). Local-only: fails fast with a clear message if offline.
if [ ! -f "$POOL/train.jsonl" ]; then
  echo "[1] sampling pool ..."
  python src/sample_data.py --data_dir "$DATA" --out "$POOL" --seed 0
else
  echo "[1] pool already exists, skip sampling"
fi

# 2) dual-GPU labeling — one process per card, per split.
label_split () {
  local split="$1"
  echo "[2] labeling split=$split on 2 GPUs ..."
  CUDA_VISIBLE_DEVICES=0 python src/gen_labels.py --model "$MODEL" \
    --in "$POOL/${split}.jsonl" --out "$LAB/${split}.shardA.jsonl" \
    --shard 0 --num_shards 2 > "logs/label_${split}_A.log" 2>&1 &
  local pidA=$!
  CUDA_VISIBLE_DEVICES=1 python src/gen_labels.py --model "$MODEL" \
    --in "$POOL/${split}.jsonl" --out "$LAB/${split}.shardB.jsonl" \
    --shard 1 --num_shards 2 > "logs/label_${split}_B.log" 2>&1 &
  local pidB=$!
  wait $pidA $pidB
  cat "$LAB/${split}.shardA.jsonl" "$LAB/${split}.shardB.jsonl" > "$LAB/${split}.jsonl"
  echo "    merged -> $LAB/${split}.jsonl ($(wc -l < "$LAB/${split}.jsonl") lines)"
}

for split in train calib test; do
  label_split "$split"
done

# 3) static audit
echo "[3] static audit ..."
for split in train calib test; do
  echo "----- $split -----"
  python src/audit_labels.py static --labels "$LAB/${split}.jsonl" \
    2>&1 | tee "logs/audit_static_${split}.log"
done

echo "### PCCD Day-2 done $(date) ###"
echo "Next (Day-3): conflict split + perturbation audit:"
echo "  CUDA_VISIBLE_DEVICES=0 python src/audit_labels.py perturb \\"
echo "     --model $MODEL --audit $POOL/audit.jsonl \\"
echo "     --out $LAB/audit_perturb.jsonl | tee logs/audit_perturb.log"
