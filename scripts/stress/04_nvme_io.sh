#!/usr/bin/env bash
# PCCD Day-1 stress — Part 4: NVMe sequential write/read (checkpoints + label shards).
# Run: bash scripts/stress/04_nvme_io.sh 2>&1 | tee logs/stress_04.log
set -uo pipefail
DIR="${1:-/root/autodl-tmp/iotest}"
mkdir -p "$DIR"
echo "NVMe I/O test in $DIR"

echo -e "\n[write 8GB]"
dd if=/dev/zero of="$DIR/testfile" bs=1M count=8192 oflag=direct 2>&1 | tail -1 || \
dd if=/dev/zero of="$DIR/testfile" bs=1M count=8192 2>&1 | tail -1

echo -e "\n[drop caches if permitted]"
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "  (cannot drop caches, non-root fs)"

echo -e "\n[read 8GB]"
dd if="$DIR/testfile" of=/dev/null bs=1M 2>&1 | tail -1

rm -f "$DIR/testfile"
echo -e "\nDONE stress 04. Expect >1 GB/s on local NVMe; if <200MB/s you are on a network drive."
