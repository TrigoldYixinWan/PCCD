#!/usr/bin/env bash
set -euo pipefail

source scripts/setup/env.sh

FORMAL_DIR="$PCCD_OUT/labelsource/formal"
PRELOCK_DIR="$PCCD_OUT/labelsource/prelock"
mkdir -p "$FORMAL_DIR" logs

LOCK_COMMIT="c10b67dd1dc718e34fa30d48c8975653753edc24"
PRELOCK_SHA="353d884d02d8daefce23b45d9a9e5997fd564c95af8a00ee17b8bbc4145d6d23"
BLIND_SHA="602de7160b536ae8567bda8a14ae6394a7413710c15b22157384c4d503a0c142"
QWEN_SCHEMA_SHA="3620d9c48a71ead307da7ec3a3c31d454051f2b4a83a8ef874589113a1c6e9aa"
START_MARKER="$FORMAL_DIR/RUN_STARTED.json"

if [[ ! -f "$START_MARKER" ]]; then
  python - "$START_MARKER" "$LOCK_COMMIT" "$PRELOCK_SHA" "$(git rev-parse HEAD)" <<'PY'
import datetime
import json
import sys
from pathlib import Path

path, lock_commit, prelock_sha, code_commit = sys.argv[1:]
payload = {
    "schema": "pccd.labelsource.formal_run_start.v1",
    "status": "FORMAL_RUN_CONSUMED_AND_STARTED",
    "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "formal_runs_authorized": 1,
    "formal_runs_consumed": 1,
    "lock_commit": lock_commit,
    "prelock_manifest_sha256": prelock_sha,
    "code_commit": code_commit,
    "outcomes_seen_before_start": False,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
else
  echo "[formal] existing RUN_STARTED marker: interruption recovery for the same run"
fi

QWEN_OUT="$FORMAL_DIR/qwen32b_proxy_labels.jsonl"
LLAMA_OUT="$FORMAL_DIR/llama_guard_3_8b_scores.jsonl"
SG2_OUT="$FORMAL_DIR/shieldgemma_2b_scores.jsonl"
SG9_OUT="$FORMAL_DIR/shieldgemma_9b_scores.jsonl"
BLIND="$PRELOCK_DIR/beavertails_330k_test_blind_items.jsonl"

run_qwen() {
  if [[ -f "$QWEN_OUT.meta.json" ]]; then
    echo "[formal] Qwen output already complete"
    return
  fi
  local resume=()
  [[ -f "$QWEN_OUT" ]] && resume=(--resume)
  CUDA_VISIBLE_DEVICES=0 python src/label_beavertails_qwen.py \
    --model "$MODELS_DIR/qwen32b" \
    --schema configs/beavertails_qwen32b_schema.json \
    --schema-sha256 "$QWEN_SCHEMA_SHA" \
    --in "$BLIND" --input-sha256 "$BLIND_SHA" \
    --out "$QWEN_OUT" --batch-size 128 --max-model-len 4096 \
    "${resume[@]}" 2>&1 | tee logs/labelsource_qwen32b.log
}

run_guard() {
  local guard_id="$1"
  local model="$2"
  local verification="$3"
  local out="$4"
  local batch_size="$5"
  if [[ -f "$out.meta.json" ]]; then
    echo "[formal] $guard_id output already complete"
    return
  fi
  local resume=()
  [[ -f "$out" ]] && resume=(--resume)
  CUDA_VISIBLE_DEVICES=1 python src/guard_score.py formal \
    --guard-id "$guard_id" --model "$model" \
    --weight-verification "$verification" \
    --blind "$BLIND" --blind-sha256 "$BLIND_SHA" \
    --out "$out" --batch-size "$batch_size" "${resume[@]}"
}

run_guards() {
  run_guard llama_guard_3_8b \
    "$MODELS_DIR/llama-guard-3-8b-7327bd9" \
    "$PRELOCK_DIR/llama_guard_weight_verification.json" \
    "$LLAMA_OUT" 32 2>&1 | tee logs/labelsource_llama_guard.log
  run_guard shieldgemma_2b \
    "$MODELS_DIR/shieldgemma-2b-d1dffc9" \
    "$PRELOCK_DIR/shieldgemma_2b_weight_verification.json" \
    "$SG2_OUT" 64 2>&1 | tee logs/labelsource_shieldgemma_2b.log
  run_guard shieldgemma_9b \
    "$MODELS_DIR/shieldgemma-9b-b8b6360" \
    "$PRELOCK_DIR/shieldgemma_9b_weight_verification.json" \
    "$SG9_OUT" 32 2>&1 | tee logs/labelsource_shieldgemma_9b.log
}

run_qwen &
qwen_pid=$!
run_guards &
guard_pid=$!
set +e
wait "$qwen_pid"
qwen_status=$?
wait "$guard_pid"
guard_status=$?
set -e
if [[ "$qwen_status" -ne 0 || "$guard_status" -ne 0 ]]; then
  echo "[formal] model stage failed: qwen=$qwen_status guards=$guard_status" >&2
  exit 2
fi

JOINED="$FORMAL_DIR/labelsource_joined.jsonl"
ANALYSIS="$FORMAL_DIR/labelsource_analysis.json"
REPORT="reports/day12_labelsource_confirmatory.md"

if [[ ! -f "$JOINED.meta.json" ]]; then
  python src/build_labelsource_eval.py \
    --blind "$BLIND" \
    --human "$PRELOCK_DIR/beavertails_330k_test_human_reference.jsonl" \
    --qwen "$QWEN_OUT" \
    --qwen-schema configs/beavertails_qwen32b_schema.json \
    --taxonomy reports/taxonomy_map.json \
    --llama-guard "$LLAMA_OUT" \
    --shieldgemma-2b "$SG2_OUT" \
    --shieldgemma-9b "$SG9_OUT" \
    --out "$JOINED" 2>&1 | tee logs/labelsource_join.log
fi

if [[ ! -f "$ANALYSIS" ]]; then
  python src/analyze_labelsource.py \
    --joined "$JOINED" --out "$ANALYSIS" --report "$REPORT" \
    2>&1 | tee logs/labelsource_analysis.log
fi

python - "$FORMAL_DIR/RUN_COMPLETE.json" "$FORMAL_DIR" "$REPORT" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

out, root, report = map(Path, sys.argv[1:])
if out.exists():
    raise FileExistsError(f"refusing to overwrite {out}")
names = [
    "qwen32b_proxy_labels.jsonl",
    "llama_guard_3_8b_scores.jsonl",
    "shieldgemma_2b_scores.jsonl",
    "shieldgemma_9b_scores.jsonl",
    "labelsource_joined.jsonl",
    "labelsource_analysis.json",
]
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
payload = {
    "schema": "pccd.labelsource.formal_run_complete.v1",
    "status": "FORMAL_COMPLETE",
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "artifacts": {name: sha(root / name) for name in names},
    "report": {"path": str(report), "sha256": sha(report)},
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
