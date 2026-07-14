#!/usr/bin/env bash
# PCCD setup — DEPRECATED. This used to download to /root (system disk) and filled it.
# Use the disk-safe scripts instead (they target the DATA disk /root/autodl-tmp):
#   bash scripts/setup/download_models.sh   # 32B teacher + 7B critic/policy
#   bash scripts/setup/download_data.sh     # PKU-SafeRLHF + UltraFeedback
set -euo pipefail
echo "download_assets.sh is deprecated (it targeted the small system disk)."
echo "Run instead:"
echo "  bash scripts/setup/download_models.sh"
echo "  bash scripts/setup/download_data.sh"
exit 1
