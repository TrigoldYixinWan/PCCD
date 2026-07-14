#!/usr/bin/env bash
# PCCD — download ONLY the datasets to the DATA disk, serially, robust to endpoints.
# Run: bash scripts/setup/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/setup/env.sh

[ -f /etc/network_turbo ] && source /etc/network_turbo || true
# Keep the proxy for Hub metadata, but bypass it for directly reachable Xet
# transfer data-plane host; the AutoDL proxy resets sustained transfers. The
# authenticated CAS control plane must continue to use the proxy.
_xet_transfer_host="transfer.xethub.hf.co"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_xet_transfer_host}"
export no_proxy="${no_proxy:+$no_proxy,}${_xet_transfer_host}"
# High performance applies within the one active Xet file; --max-workers 1
# continues to prevent the prior multi-file disk blowup.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
# hf_transfer disabled: its parallel fetch spiked temp usage in the earlier blowup.
export HF_HUB_ENABLE_HF_TRANSFER=0

dl () {   # dl <hub_id> <local_subdir> <expected_repo_files>
  local hub="$1" sub="$2" expected="$3" dst="$DATA_DIR/$2"
  echo "== downloading $hub -> $dst =="
  for endpoint in "https://huggingface.co" "https://hf-mirror.com"; do
    echo "  endpoint=$endpoint"
    if HF_ENDPOINT="$endpoint" python scripts/setup/hf_download.py "$hub" --repo-type dataset \
      --local-dir "$dst" --max-workers 1; then
      local incomplete files
      incomplete="$(find "$dst" -type f -name "*.incomplete" -print -quit)"
      files="$(find "$dst" -type f ! -path "$dst/.cache/*" | wc -l)"
      if [ -z "$incomplete" ] && [ "$files" -ge "$expected" ]; then
        echo "  OK ($files/$expected repository files)"; return 0
      fi
      echo "  incomplete snapshot: files=$files/$expected pending=${incomplete:-none}"
    fi
    echo "  failed, next endpoint..."
  done
  echo "  ERROR: could not download $hub"; return 1
}

dl "PKU-Alignment/PKU-SafeRLHF" "pku-saferlhf" 9
dl "openbmb/UltraFeedback"      "ultrafeedback" 8

echo -e "\nDatasets on data disk:"
du -sh "$DATA_DIR"/* 2>/dev/null || true
echo "Then: python src/sample_data.py --data_dir $DATA_DIR --out $PCCD_OUT/pool"
