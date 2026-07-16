#!/usr/bin/env bash
# Locked independent P2/P3/P8 confirmation pipeline.
# Run one explicit phase at a time; "unseal" is never invoked automatically.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
source scripts/setup/env.sh

PHASE="${1:-}"
CONF="$PCCD_OUT/confirmation"
PROMPTS="$CONF/confirmation_prompts.jsonl"
CALIB_IDS="$CONF/confirmation_target_calib_ids.json"
TEST_IDS="$CONF/confirmation_test_ids.json"
OLD_D5="$PCCD_OUT/policy/g2_D5_r32"
NEW_D5="$PCCD_OUT/policy/confirm_D5_r32_seed20260723"
PAIRS="$PCCD_OUT/g2/hidden_pairs.jsonl"
CRITIC="$PCCD_OUT/critic/d0"
mkdir -p "$CONF" logs

response_path() { printf '%s/%s_responses.jsonl' "$CONF" "$1"; }
teacher_path() { printf '%s/%s_teacher.jsonl' "$CONF" "$1"; }
logit_path() { printf '%s/%s_logits.jsonl' "$CONF" "$1"; }

label_variant() {
  local name="$1"
  local input
  input="$(response_path "$name")"
  local final
  final="$(teacher_path "$name")"
  local shard_a="$CONF/${name}_teacher.shardA.jsonl"
  local shard_b="$CONF/${name}_teacher.shardB.jsonl"
  local combined="$CONF/.${name}_teacher.unordered.jsonl"
  if [ -e "$final" ]; then
    echo "refusing to overwrite $final" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES=0 python src/gen_labels.py \
    --model "$MODELS_DIR/qwen32b" --in "$input" --out "$shard_a" \
    --shard 0 --num_shards 2 --retries 0 --max_tokens 256 \
    > "logs/confirmation_label_${name}_A.log" 2>&1 &
  local pid_a=$!
  CUDA_VISIBLE_DEVICES=1 python src/gen_labels.py \
    --model "$MODELS_DIR/qwen32b" --in "$input" --out "$shard_b" \
    --shard 1 --num_shards 2 --retries 0 --max_tokens 256 \
    > "logs/confirmation_label_${name}_B.log" 2>&1 &
  local pid_b=$!
  wait "$pid_a" "$pid_b"
  cat "$shard_a" "$shard_b" > "$combined"
  python src/subset_jsonl_by_manifest.py \
    --in "$combined" --manifest "$PROMPTS" --out "$final" --require_exhaustive
  rm -f "$combined"
}

score_variant() {
  local name="$1"
  accelerate launch --num_processes 2 src/eval_critic.py \
    --checkpoint "$CRITIC" \
    --labels "$(teacher_path "$name")" \
    --logits "$(logit_path "$name")" \
    --logits_only --batch 4 \
    2>&1 | tee "logs/confirmation_score_${name}.log"
  python src/subset_jsonl_by_manifest.py \
    --in "$(teacher_path "$name")" --manifest "$TEST_IDS" \
    --out "$CONF/${name}_teacher_test.jsonl"
  python src/subset_jsonl_by_manifest.py \
    --in "$(logit_path "$name")" --manifest "$TEST_IDS" \
    --out "$CONF/${name}_logits_test.jsonl"
}

case "$PHASE" in
  lockbox)
    python src/build_confirmation_lockbox.py
    python src/freeze_source_bins.py \
      --logits "$PCCD_OUT/results/d0_calib_logits.jsonl" \
      --out "$CONF/source_base_calib_edges.json"
    ;;

  adapter)
    CUDA_VISIBLE_DEVICES=0 python src/adapt_hidden.py \
      --model "$MODELS_DIR/qwen7b" \
      --pairs "$PAIRS" \
      --out "$NEW_D5" \
      --point D5 --method sft --rank 32 --epochs 4 \
      --effective_batch 32 --per_device_batch 1 --max_len 1024 \
      --learning_rate 2e-4 --seed 20260723 \
      2>&1 | tee logs/confirmation_train_new_D5.log
    python src/validate_confirmation_adapter.py \
      --old "$OLD_D5/adaptation_metadata.json" \
      --new "$NEW_D5/adaptation_metadata.json" \
      --pairs "$PAIRS" \
      --out "$CONF/new_D5_adapter_validation.json"
    ;;

  generate)
    CUDA_VISIBLE_DEVICES=0 python src/gen_policy_responses.py \
      --model "$MODELS_DIR/qwen7b" --prompts "$PROMPTS" \
      --variant D0 --expected_count 4000 \
      --out "$(response_path D0)" --seed 20260723 \
      > logs/confirmation_generate_D0.log 2>&1 &
    pid_d0=$!
    CUDA_VISIBLE_DEVICES=1 python src/gen_policy_responses.py \
      --model "$MODELS_DIR/qwen7b" --prompts "$PROMPTS" \
      --adapter "$OLD_D5" --variant old_D5 --max_lora_rank 64 \
      --expected_count 4000 --out "$(response_path old_D5)" --seed 20260723 \
      > logs/confirmation_generate_old_D5.log 2>&1 &
    pid_old=$!
    wait "$pid_d0" "$pid_old"
    CUDA_VISIBLE_DEVICES=0 python src/gen_policy_responses.py \
      --model "$MODELS_DIR/qwen7b" --prompts "$PROMPTS" \
      --adapter "$NEW_D5" --variant new_D5 --max_lora_rank 64 \
      --expected_count 4000 --out "$(response_path new_D5)" --seed 20260723 \
      2>&1 | tee logs/confirmation_generate_new_D5.log
    ;;

  label)
    label_variant D0
    label_variant old_D5
    label_variant new_D5
    ;;

  score)
    score_variant D0
    score_variant old_D5
    score_variant new_D5
    ;;

  audit)
    python src/build_human_audit.py \
      --prompts "$PROMPTS" --test_manifest "$TEST_IDS" \
      --domain D0 "$(teacher_path D0)" "$(logit_path D0)" \
      --domain new_D5 "$(teacher_path new_D5)" "$(logit_path new_D5)" \
      --private_out "$CONF/human_audit_private.jsonl" \
      --blind_out "$CONF/human_audit_blind.jsonl" \
      --manifest "$CONF/human_audit_manifest.json"
    ;;

  freeze)
    mkdir -p "$CONF/dependencies"
    if ! compgen -G "$CONF/dependencies/probmetrics-1.3.0-*.whl" > /dev/null; then
      python -m pip download --no-deps --dest "$CONF/dependencies" probmetrics==1.3.0
    fi
    python src/freeze_confirmation_artifacts.py \
      --repo "$(pwd)" \
      --out "$CONF/confirmation_preunseal.sha256" \
      --metadata_out "$CONF/confirmation_preunseal_environment.json" \
      --path "$CONF" \
      --path "$CRITIC" \
      --path "$OLD_D5" \
      --path "$NEW_D5" \
      --path "$PAIRS" \
      --path "$MODELS_DIR/qwen7b/config.json" \
      --path "$MODELS_DIR/qwen32b/config.json"
    ;;

  unseal)
    test -f "$CONF/confirmation_preunseal.sha256"
    python src/analyze_confirmation.py \
      --test_manifest "$TEST_IDS" \
      --hash_manifest "$CONF/confirmation_preunseal.sha256" \
      --source_bin_edges "$CONF/source_base_calib_edges.json" \
      --d0_labels "$CONF/D0_teacher_test.jsonl" \
      --d0_logits "$CONF/D0_logits_test.jsonl" \
      --old_d5_labels "$CONF/old_D5_teacher_test.jsonl" \
      --old_d5_logits "$CONF/old_D5_logits_test.jsonl" \
      --new_d5_labels "$CONF/new_D5_teacher_test.jsonl" \
      --new_d5_logits "$CONF/new_D5_logits_test.jsonl" \
      --out "$PCCD_OUT/results/confirmation_p2_p3.json" \
      2>&1 | tee logs/confirmation_unseal_p2_p3.log
    core_verdict="$(
      python - "$PCCD_OUT/results/confirmation_p2_p3.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["reported_verdict"])
PY
    )"
    p2_status=FAIL
    if [ "$core_verdict" = "P2_ONLY" ] || [ "$core_verdict" = "P2_P3_CONFIRMED" ]; then
      p2_status=PASS
    fi
    python src/fit_g6.py fit \
      --mode confirmation \
      --labels "$(teacher_path new_D5)" \
      --logits "$(logit_path new_D5)" \
      --calib_manifest "$CALIB_IDS" \
      --test_manifest "$TEST_IDS" \
      --split_hash_manifest "$CONF/confirmation_preunseal.sha256" \
      --p2_status "$p2_status" \
      --out "$PCCD_OUT/results/g6_confirmation.json" \
      --plot "$PCCD_OUT/results/g6_confirmation.png" \
      2>&1 | tee logs/confirmation_unseal_p8.log
    ;;

  *)
    echo "usage: bash scripts/day9/run_confirmation.sh {lockbox|adapter|generate|label|score|audit|freeze|unseal}" >&2
    exit 2
    ;;
esac
