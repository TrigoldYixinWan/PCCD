#!/usr/bin/env bash
set -euo pipefail

cd /root/PCCD
source scripts/setup/env.sh
source /etc/network_turbo

PREP=/root/autodl-tmp/pccd/tmp/prepare_guard_aria2.py
TMP=/root/autodl-tmp/pccd/tmp
OUT="$PCCD_OUT/labelsource/prelock"
mkdir -p "$OUT"

download_one() {
  local key=$1 repo=$2 revision=$3 model_dir=$4 concurrent=$5
  local input="$TMP/${key}.aria2.txt"
  local manifest="$OUT/${key}_weight_manifest.json"

  /root/miniconda3/bin/python "$PREP" \
    --repo "$repo" \
    --revision "$revision" \
    --model-dir "$model_dir" \
    --aria-input "$input" \
    --manifest "$manifest"

  if [[ -s "$input" ]]; then
    aria2c \
      --continue=true \
      --max-concurrent-downloads="$concurrent" \
      --max-connection-per-server=4 \
      --split=4 \
      --min-split-size=4M \
      --file-allocation=none \
      --auto-file-renaming=false \
      --allow-overwrite=false \
      --summary-interval=30 \
      --input-file="$input"
  fi
  rm -f -- "$input"
}

case "${1:-}" in
  llama_guard)
    download_one \
      llama_guard \
      meta-llama/Llama-Guard-3-8B \
      7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 \
      "$MODELS_DIR/llama-guard-3-8b-7327bd9" \
      4
    ;;
  shield_2b)
    download_one \
      shieldgemma_2b \
      google/shieldgemma-2b \
      d1dffc9c8c9237a90aab09c61383791e718ef9e8 \
      "$MODELS_DIR/shieldgemma-2b-d1dffc9" \
      2
    ;;
  shield_9b)
    download_one \
      shieldgemma_9b \
      google/shieldgemma-9b \
      b8b636016df4540721a098c7aab91c97ec6ee508 \
      "$MODELS_DIR/shieldgemma-9b-b8b6360" \
      4
    ;;
  shields)
    /bin/bash "$0" shield_2b
    /bin/bash "$0" shield_9b
    ;;
  *)
    echo "usage: $0 {llama_guard|shield_2b|shield_9b|shields}" >&2
    exit 2
    ;;
esac
