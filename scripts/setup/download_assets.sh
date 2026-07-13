#!/usr/bin/env bash
# PCCD setup — download models + datasets to local NVMe. Run: bash scripts/setup/download_assets.sh
set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # AutoDL CN acceleration
export HF_HUB_ENABLE_HF_TRANSFER=1
MODELS="${MODELS_DIR:-/root/models}"
DATA="${DATA_DIR:-/root/data}"
mkdir -p "$MODELS" "$DATA"

echo "== models =="
huggingface-cli download Qwen/Qwen2.5-32B-Instruct  --local-dir "$MODELS/qwen32b"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct   --local-dir "$MODELS/qwen7b"
huggingface-cli download Qwen/Qwen2.5-14B-Instruct  --local-dir "$MODELS/qwen14b"

echo "== datasets =="
huggingface-cli download PKU-Alignment/PKU-SafeRLHF --repo-type dataset --local-dir "$DATA/pku-saferlhf"
huggingface-cli download openbmb/UltraFeedback      --repo-type dataset --local-dir "$DATA/ultrafeedback"
huggingface-cli download walledai/HarmBench          --repo-type dataset --local-dir "$DATA/harmbench" || \
  echo "  (HarmBench may need manual/gated access; safe to defer to Day 9)"

echo "DONE downloads. Disk usage:"
du -sh "$MODELS"/* "$DATA"/* 2>/dev/null || true
