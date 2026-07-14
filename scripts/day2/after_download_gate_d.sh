#!/usr/bin/env bash
# Wait for the resumable model download, verify exact snapshots, then run Gate D.
# Deliberately stops after Gate D so its verdict is reviewed before Day-2 labeling.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/setup/env.sh
mkdir -p logs "$MODELS_DIR/qwen32b" "$MODELS_DIR/qwen7b"

echo "### waiting for model download $(date -Is) ###"
while pgrep -f "[d]ownload_models.sh" >/dev/null; do
  q32="$(find "$MODELS_DIR/qwen32b" -maxdepth 1 -type f -name "*.safetensors" 2>/dev/null | wc -l)"
  q7="$(find "$MODELS_DIR/qwen7b" -maxdepth 1 -type f -name "*.safetensors" 2>/dev/null | wc -l)"
  echo "$(date -Is) qwen32b=$q32/17 qwen7b=$q7/4"
  sleep 60
done

q32="$(find "$MODELS_DIR/qwen32b" -maxdepth 1 -type f -name "*.safetensors" 2>/dev/null | wc -l)"
q7="$(find "$MODELS_DIR/qwen7b" -maxdepth 1 -type f -name "*.safetensors" 2>/dev/null | wc -l)"
pending="$(find "$MODELS_DIR/qwen32b" "$MODELS_DIR/qwen7b" \
  -type f -name "*.incomplete" -print -quit 2>/dev/null || true)"
if [ "$q32" -ne 17 ] || [ "$q7" -ne 4 ] || [ -n "$pending" ]; then
  echo "BLOCKED: incomplete model snapshot qwen32b=$q32/17 qwen7b=$q7/4 pending=${pending:-none}"
  exit 1
fi

echo "### imports $(date -Is) ###"
python -c "import torch,datasets,transformers,trl,peft,vllm; print('imports OK')"

echo "### Gate D $(date -Is) ###"
CUDA_VISIBLE_DEVICES=0 python scripts/stress/03_vllm_teacher_probe.py \
  --model "$MODELS_DIR/qwen32b" --n 64 2>&1 | tee logs/stress_03.log
echo "### Gate D command completed $(date -Is); review verdict before Day-2 ###"
