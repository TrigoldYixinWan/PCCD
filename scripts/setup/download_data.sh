#!/usr/bin/env bash
# PCCD — download ONLY the datasets to the DATA disk, serially, robust to endpoints.
# Run: bash scripts/setup/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/setup/env.sh

[ -f /etc/network_turbo ] && source /etc/network_turbo || true
# Keep the proxy for Hub metadata, but bypass it for directly reachable Xet
# data-plane hosts; the AutoDL proxy resets sustained Xet transfers.
_xet_hosts="cas-server.xethub.hf.co,cas-bridge.xethub.hf.co,transfer.xethub.hf.co"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_xet_hosts}"
export no_proxy="${no_proxy:+$no_proxy,}${_xet_hosts}"
# hf_transfer disabled: its parallel fetch spiked temp usage in the earlier blowup.
export HF_HUB_ENABLE_HF_TRANSFER=0

dl () {   # dl <hub_id> <local_subdir>
  local hub="$1" sub="$2" dst="$DATA_DIR/$2"
  echo "== downloading $hub -> $dst =="
  for endpoint in "https://huggingface.co" "https://hf-mirror.com"; do
    echo "  endpoint=$endpoint"
    HF_ENDPOINT="$endpoint" hf download "$hub" --type dataset \
      --local-dir "$dst" --max-workers 1 && { echo "  OK"; return 0; }
    echo "  failed, next endpoint..."
  done
  echo "  ERROR: could not download $hub"; return 1
}

dl "PKU-Alignment/PKU-SafeRLHF" "pku-saferlhf"
dl "openbmb/UltraFeedback"      "ultrafeedback"

echo -e "\nDatasets on data disk:"
du -sh "$DATA_DIR"/* 2>/dev/null || true
echo "Then: python src/sample_data.py --data_dir $DATA_DIR --out $PCCD_OUT/pool"
