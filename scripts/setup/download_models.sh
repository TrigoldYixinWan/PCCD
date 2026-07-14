#!/usr/bin/env bash
# PCCD — download models to the DATA disk (/root/autodl-tmp), serially, resumable.
# Run: bash scripts/setup/download_models.sh
# Only teacher(32B) + critic/policy(7B) are needed to start; 14B is optional (Day 8).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/setup/env.sh

# AutoDL academic acceleration (helps reach huggingface.co)
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
# NOTE: hf_transfer disabled on purpose — its parallel multi-file fetch spikes
# temp usage and was a factor in the earlier disk blowup. Serial is safer here.
export HF_HUB_ENABLE_HF_TRANSFER=0

dl_model () {   # dl_model <hub_id> <subdir>
  local hub="$1" sub="$2" dst="$MODELS_DIR/$2"
  if [ -d "$dst" ] && [ "$(ls "$dst"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "== $sub already present, verifying/ resuming =="
  fi
  echo "== downloading $hub -> $dst =="
  for endpoint in "https://huggingface.co" "https://hf-mirror.com"; do
    echo "  endpoint=$endpoint"
    HF_ENDPOINT="$endpoint" huggingface-cli download "$hub" \
      --local-dir "$dst" --max-workers 1 && { echo "  OK"; return 0; }
    echo "  failed, next endpoint..."
  done
  echo "  ERROR: $hub download failed"; return 1
}

# order: 32B teacher first (largest, most critical), then 7B, then optional 14B
dl_model "Qwen/Qwen2.5-32B-Instruct" "qwen32b"
dl_model "Qwen/Qwen2.5-7B-Instruct"  "qwen7b"
if [ "${WITH_14B:-0}" = "1" ]; then
  dl_model "Qwen/Qwen2.5-14B-Instruct" "qwen14b"
fi

echo -e "\nModels on data disk:"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
df -h "$PCCD_DISK" | tail -1
