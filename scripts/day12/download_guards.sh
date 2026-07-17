#!/usr/bin/env bash
set -euo pipefail

cd /root/PCCD
source scripts/setup/env.sh
source /etc/network_turbo
# AutoDL's academic proxy reaches the Hub HTTPS/LFS endpoints but the Xet
# transfer backend stalls on CAS.  Force the standard resumable HTTP path.
export HF_HUB_DISABLE_XET=1

download_llama_guard() {
  hf download meta-llama/Llama-Guard-3-8B \
    --revision 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 \
    --local-dir "$MODELS_DIR/llama-guard-3-8b-7327bd9" \
    --max-workers 4
}

download_shield_2b() {
  hf download google/shieldgemma-2b \
    --revision d1dffc9c8c9237a90aab09c61383791e718ef9e8 \
    --local-dir "$MODELS_DIR/shieldgemma-2b-d1dffc9" \
    --max-workers 4
}

download_shield_9b() {
  hf download google/shieldgemma-9b \
    --revision b8b636016df4540721a098c7aab91c97ec6ee508 \
    --local-dir "$MODELS_DIR/shieldgemma-9b-b8b6360" \
    --max-workers 4
}

case "${1:-all}" in
  llama_guard)
    download_llama_guard
    ;;
  shield_2b)
    download_shield_2b
    ;;
  shield_9b)
    download_shield_9b
    ;;
  shields)
    download_shield_2b
    download_shield_9b
    ;;
  all)
    download_llama_guard
    download_shield_2b
    download_shield_9b
    ;;
  *)
    echo "usage: $0 {llama_guard|shield_2b|shield_9b|shields|all}" >&2
    exit 2
    ;;
esac
