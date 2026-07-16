#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/setup/env.sh

G2="$PCCD_OUT/g2"
MODEL="$MODELS_DIR/qwen7b"
mkdir -p logs

run_point() {
  local gpu="$1"
  local point="$2"
  local adapter="$3"
  local -a adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python src/compute_kl.py \
    --model "$MODEL" \
    "${adapter_args[@]}" \
    --generations "$G2/${point}_responses.jsonl" \
    --out "$G2/${point}_kl_tokens_summary.json" \
    --tokens_out "$G2/${point}_kl_tokens.jsonl" \
    --reference_items "$G2/${point}_kl_items.jsonl" \
    --reproduction_tolerance 1e-6 \
    --bootstrap 10000 \
    --seed 20260716 \
    2>&1 | tee "logs/day8_${point}_token_recompute.log"
}

for point in D1 D2 D3_control D4 D5 D6; do
  if [[ -e "$G2/${point}_kl_tokens.jsonl" || -e "$G2/${point}_kl_tokens_summary.json" ]]; then
    echo "refusing to overwrite existing Day-8 artifact for $point" >&2
    exit 1
  fi
done

# Balance the frozen generated-token counts across the two independent GPUs.
(
  run_point 0 D6 "$PCCD_OUT/policy/g2_D6_dpo_r16"
  run_point 0 D2 "$PCCD_OUT/policy/g2_D2_r4"
  run_point 0 D5 "$PCCD_OUT/policy/g2_D5_r32"
) &
gpu0_pid=$!

(
  run_point 1 D3_control "$PCCD_OUT/policy/d3_lora_r8"
  run_point 1 D1 ""
  run_point 1 D4 "$PCCD_OUT/policy/g2_D4_r16"
) &
gpu1_pid=$!

set +e
wait "$gpu0_pid"
gpu0_status=$?
wait "$gpu1_pid"
gpu1_status=$?
set -e
if [[ "$gpu0_status" -ne 0 || "$gpu1_status" -ne 0 ]]; then
  echo "token recomputation failed: gpu0=$gpu0_status gpu1=$gpu1_status" >&2
  exit 1
fi

token_files=()
for point in D1 D2 D3_control D4 D5 D6; do
  file="$G2/${point}_kl_tokens.jsonl"
  [[ "$(wc -l < "$file")" -eq 3000 ]]
  token_files+=("$file")
done

sha256sum "${token_files[@]}" > "$G2/kl_tokens.sha256"
chmod 0444 "${token_files[@]}" "$G2/kl_tokens.sha256"
sha256sum -c "$G2/kl_tokens.sha256"
echo "frozen token manifest: $G2/kl_tokens.sha256"
