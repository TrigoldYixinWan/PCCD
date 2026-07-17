#!/usr/bin/env bash
set -euo pipefail

LLAMA_PID=${1:?llama aria PID required}
SHIELDS_PID=${2:?shields wrapper PID required}

cd /root/PCCD
source scripts/setup/env.sh
PRELOCK="$PCCD_OUT/labelsource/prelock"
TMP=/root/autodl-tmp/pccd/tmp
CASES="$PRELOCK/guard_sanity_cases.json"

while kill -0 "$LLAMA_PID" 2>/dev/null || kill -0 "$SHIELDS_PID" 2>/dev/null; do
  date -Is
  du -sh \
    "$MODELS_DIR/llama-guard-3-8b-7327bd9" \
    "$MODELS_DIR/shieldgemma-2b-d1dffc9" \
    "$MODELS_DIR/shieldgemma-9b-b8b6360" || true
  sleep 30
done

python "$TMP/verify_guard_weights.py" \
  --manifest "$PRELOCK/llama_guard_weight_manifest.json" \
  --out "$PRELOCK/llama_guard_weight_verification.json"
python "$TMP/verify_guard_weights.py" \
  --manifest "$PRELOCK/shieldgemma_2b_weight_manifest.json" \
  --out "$PRELOCK/shieldgemma_2b_weight_verification.json"
python "$TMP/verify_guard_weights.py" \
  --manifest "$PRELOCK/shieldgemma_9b_weight_manifest.json" \
  --out "$PRELOCK/shieldgemma_9b_weight_verification.json"
rm -f -- "$TMP/llama_guard.aria2.txt" "$TMP/shieldgemma_2b.aria2.txt" "$TMP/shieldgemma_9b.aria2.txt"

source /etc/network_turbo
export HF_HUB_DISABLE_XET=1

hf download meta-llama/Llama-Guard-3-8B \
  .gitattributes LICENSE README.md USE_POLICY.md config.json generation_config.json \
  model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer_config.json \
  --revision 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 \
  --local-dir "$MODELS_DIR/llama-guard-3-8b-7327bd9"
hf download google/shieldgemma-2b \
  .gitattributes README.md config.json generation_config.json \
  model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model \
  tokenizer_config.json \
  --revision d1dffc9c8c9237a90aab09c61383791e718ef9e8 \
  --local-dir "$MODELS_DIR/shieldgemma-2b-d1dffc9"
hf download google/shieldgemma-9b \
  .gitattributes README.md config.json generation_config.json \
  model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model \
  tokenizer_config.json \
  --revision b8b636016df4540721a098c7aab91c97ec6ee508 \
  --local-dir "$MODELS_DIR/shieldgemma-9b-b8b6360"

python "$TMP/inspect_guard_tokenizer.py" \
  --model-dir "$MODELS_DIR/llama-guard-3-8b-7327bd9" \
  --guard llama_guard \
  --repo meta-llama/Llama-Guard-3-8B \
  --revision 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 \
  --out "$PRELOCK/llama_guard_tokenizer_registry.json"
python "$TMP/inspect_guard_tokenizer.py" \
  --model-dir "$MODELS_DIR/shieldgemma-2b-d1dffc9" \
  --guard shieldgemma \
  --repo google/shieldgemma-2b \
  --revision d1dffc9c8c9237a90aab09c61383791e718ef9e8 \
  --out "$PRELOCK/shieldgemma_2b_tokenizer_registry.json"
python "$TMP/inspect_guard_tokenizer.py" \
  --model-dir "$MODELS_DIR/shieldgemma-9b-b8b6360" \
  --guard shieldgemma \
  --repo google/shieldgemma-9b \
  --revision b8b636016df4540721a098c7aab91c97ec6ee508 \
  --out "$PRELOCK/shieldgemma_9b_tokenizer_registry.json"

python "$TMP/freeze_guard_registry.py" \
  --prelock-dir "$PRELOCK" \
  --guard-score "$TMP/guard_score.py" \
  --cases "$CASES" \
  --out "$PRELOCK/guard_registry_frozen.json" \
  | tee "$PRELOCK/freeze_guard_registry.log"

CUDA_VISIBLE_DEVICES=0 python "$TMP/guard_score.py" sanity \
  --guard llama_guard \
  --model "$MODELS_DIR/llama-guard-3-8b-7327bd9" \
  --repo meta-llama/Llama-Guard-3-8B \
  --revision 7327bd9f6efbbe6101dc6cc4736302b3cbb6e425 \
  --cases "$CASES" \
  --out "$PRELOCK/sanity_llama_guard.json" \
  | tee "$PRELOCK/sanity_llama_guard.log"

CUDA_VISIBLE_DEVICES=0 python "$TMP/guard_score.py" sanity \
  --guard shieldgemma \
  --model "$MODELS_DIR/shieldgemma-2b-d1dffc9" \
  --repo google/shieldgemma-2b \
  --revision d1dffc9c8c9237a90aab09c61383791e718ef9e8 \
  --cases "$CASES" \
  --out "$PRELOCK/sanity_shieldgemma_2b.json" \
  | tee "$PRELOCK/sanity_shieldgemma_2b.log"

CUDA_VISIBLE_DEVICES=0 python "$TMP/guard_score.py" sanity \
  --guard shieldgemma \
  --model "$MODELS_DIR/shieldgemma-9b-b8b6360" \
  --repo google/shieldgemma-9b \
  --revision b8b636016df4540721a098c7aab91c97ec6ee508 \
  --cases "$CASES" \
  --out "$PRELOCK/sanity_shieldgemma_9b.json" \
  | tee "$PRELOCK/sanity_shieldgemma_9b.log"

sha256sum "$PRELOCK"/*.json > "$PRELOCK/prelock_json_sha256.txt"
echo GATE_11_2_SANITY_COMPLETE
