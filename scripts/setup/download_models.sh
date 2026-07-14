#!/usr/bin/env bash
# PCCD — download models to the DATA disk (/root/autodl-tmp), serially, resumable.
# Run: bash scripts/setup/download_models.sh
# Only teacher(32B) + critic/policy(7B) are needed to start; 14B is optional (Day 8).
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/setup/env.sh

# AutoDL academic acceleration (helps reach huggingface.co)
[ -f /etc/network_turbo ] && source /etc/network_turbo || true
# AutoDL's academic proxy is needed for the Hub control plane, but it resets
# large Xet transfers. Keep the authenticated CAS control plane on the proxy,
# and bypass it only for the directly reachable transfer data plane.
_xet_transfer_host="transfer.xethub.hf.co"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_xet_transfer_host}"
export no_proxy="${no_proxy:+$no_proxy,}${_xet_transfer_host}"
# NOTE: hf_transfer disabled on purpose — its parallel multi-file fetch spikes
# temp usage and was a factor in the earlier disk blowup. Serial is safer here.
export HF_HUB_ENABLE_HF_TRANSFER=0

dl_model () {   # dl_model <hub_id> <subdir> <expected_safetensor_shards>
  local hub="$1" sub="$2" expected="$3" dst="$MODELS_DIR/$2"
  if [ -d "$dst" ] && [ "$(ls "$dst"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "== $sub already present, verifying/ resuming =="
  fi
  echo "== downloading $hub -> $dst =="
  for endpoint in "https://huggingface.co" "https://hf-mirror.com"; do
    echo "  endpoint=$endpoint"
    if HF_ENDPOINT="$endpoint" hf download "$hub" \
      --local-dir "$dst" --max-workers 1; then
      local incomplete shards
      incomplete="$(find "$dst" -type f -name "*.incomplete" -print -quit)"
      shards="$(find "$dst" -maxdepth 1 -type f -name "*.safetensors" | wc -l)"
      if [ -z "$incomplete" ] && [ "$shards" -eq "$expected" ]; then
        echo "  OK ($shards/$expected safetensor shards)"; return 0
      fi
      echo "  incomplete snapshot: shards=$shards/$expected pending=${incomplete:-none}"
    fi
    echo "  failed, next endpoint..."
  done
  echo "  ERROR: $hub download failed"; return 1
}

# order: 32B teacher first (largest, most critical), then 7B, then optional 14B
dl_model "Qwen/Qwen2.5-32B-Instruct" "qwen32b" 17
dl_model "Qwen/Qwen2.5-7B-Instruct"  "qwen7b" 4
if [ "${WITH_14B:-0}" = "1" ]; then
  dl_model "Qwen/Qwen2.5-14B-Instruct" "qwen14b" 8
fi

echo -e "\nModels on data disk:"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
df -h "$PCCD_DISK" | tail -1
