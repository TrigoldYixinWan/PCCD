#!/usr/bin/env bash
# PCCD — free the system disk after the earlier failed downloads filled /root.
# Removes half-finished HF downloads and misplaced model/data dirs on the SYSTEM disk.
# SAFE: only touches known PCCD download locations; asks before deleting large dirs.
# Run: bash scripts/setup/cleanup_systemdisk.sh
set -uo pipefail

echo "=== BEFORE ==="
df -h /root | tail -1

echo -e "\n[1] removing incomplete HF downloads under /root/models ..."
find /root/models -name "*.incomplete" -delete 2>/dev/null || true
rm -rf /root/models/*/.cache 2>/dev/null || true

echo "[2] misplaced dirs on system disk (these should live on /root/autodl-tmp):"
du -sh /root/models /root/data /root/.cache/huggingface 2>/dev/null || true

echo -e "\n[3] To reclaim space, remove the misplaced copies on the SYSTEM disk."
echo "    (Models/data will be re-downloaded to /root/autodl-tmp via download scripts.)"
read -r -p "    Delete /root/models, /root/data, /root/.cache/huggingface ? [y/N] " ans
if [ "${ans:-N}" = "y" ] || [ "${ans:-N}" = "Y" ]; then
  rm -rf /root/models /root/data /root/.cache/huggingface
  echo "    deleted."
else
  echo "    skipped (delete manually if you want the space back)."
fi

echo -e "\n=== AFTER ==="
df -h /root | tail -1
