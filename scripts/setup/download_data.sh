#!/usr/bin/env bash
# PCCD — download ONLY the datasets (models handled separately). Robust to AutoDL
# network conditions: tries AutoDL academic acceleration, then hf-mirror, then hf.co.
# Run: bash scripts/setup/download_data.sh
set -uo pipefail
DATA="${DATA_DIR:-/root/data}"
mkdir -p "$DATA"

# 1) enable AutoDL academic acceleration if present (helps reach huggingface.co)
if [ -f /etc/network_turbo ]; then
  echo "[net] sourcing AutoDL /etc/network_turbo"
  source /etc/network_turbo || true
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

dl () {   # dl <hub_id> <local_subdir>
  local hub="$1" sub="$2" dst="$DATA/$2"
  echo "== downloading $hub -> $dst =="
  for endpoint in "https://huggingface.co" "https://hf-mirror.com"; do
    echo "  trying HF_ENDPOINT=$endpoint"
    HF_ENDPOINT="$endpoint" huggingface-cli download "$hub" --repo-type dataset \
      --local-dir "$dst" && { echo "  OK via $endpoint"; return 0; }
    echo "  failed via $endpoint, next..."
  done
  echo "  ERROR: could not download $hub from any endpoint"
  return 1
}

dl "PKU-Alignment/PKU-SafeRLHF" "pku-saferlhf"
dl "openbmb/UltraFeedback"      "ultrafeedback"

echo -e "\nDone. Local datasets:"
du -sh "$DATA"/* 2>/dev/null || true
echo "Then run: python src/sample_data.py --data_dir $DATA --out outputs/pool"
