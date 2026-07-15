# Day 3 conflict labeling, perturbation audit, and G1 report

Date: 2026-07-15

Execution revisions:

- conflict labeling: `0c57ad0d9dde9c0d9aa95e5d509432d316181b3e`;
- perturbation audit: `2c97407ca787ca60dd7801e6c6d92ec7602cc5fc`;
- final bootstrap and pairwise analysis: `b676fb688ec337fc8bae0f5dfa79c8485e8b6715`.

Status: **conflict-label artifact integrity PASS; teacher target-label heterogeneity is
strongly supported at the global level and for 44/45 policy pairs; the registered
order-swap and paraphrase reliability gates FAIL. D0 critic behavior and the
pre-registered per-policy F1 CV cannot be evaluated from the available artifacts.
Overall G1 is therefore PROVISIONAL/INCOMPLETE; no G1 PASS is declared. Training and the
optional one-step LoRA/DPO smoke run remain stopped for PaperGuru review.**

## 1. Git and execution scope

PR #2 (`day2/teacher-labeling` -> `main`) was strict-fast-forwarded to `0c57ad0` and
GitHub reports `merged=true`. Day 3 then started from that exact `main` revision on branch
`day3/g1-heterogeneity`.

The original perturbation implementation omitted the registered `repeat_sampling` call.
This Green implementation bug was fixed before the audit: the canonical prompt is now
submitted in a second independent `temperature=0` batch, all three registered
perturbations are persisted under their schema names, and policy order is guaranteed to
actually change. A mock test verified four independent inference batches, exact policy
schema, and a non-canonical order. No teacher prompt, label, data, threshold, or sampling
temperature was changed.

The AutoDL runtime was Python 3.12.3, torch 2.11.0/CUDA 13.0, transformers 5.13.1,
TRL 0.19.1, PEFT 0.19.1, vLLM 0.25.0, NumPy 2.3.2, and SciPy 1.18.0. Heavy artifacts
remained under `/root/autodl-tmp/pccd`; the data disk had 287 GB free.

## 2. Exact commands

### Conflict split labeling and static audit

```bash
cd /root/PCCD
source scripts/setup/env.sh

CUDA_VISIBLE_DEVICES=0 python src/gen_labels.py \
  --model "$MODELS_DIR/qwen32b" \
  --in "$PCCD_OUT/pool/conflict.jsonl" \
  --out "$PCCD_OUT/labels/conflict.shardA.jsonl" \
  --shard 0 --num_shards 2 > logs/label_conflict_A.log 2>&1 &
pidA=$!
CUDA_VISIBLE_DEVICES=1 python src/gen_labels.py \
  --model "$MODELS_DIR/qwen32b" \
  --in "$PCCD_OUT/pool/conflict.jsonl" \
  --out "$PCCD_OUT/labels/conflict.shardB.jsonl" \
  --shard 1 --num_shards 2 > logs/label_conflict_B.log 2>&1 &
pidB=$!
wait "$pidA" "$pidB"
cat "$PCCD_OUT/labels/conflict.shardA.jsonl" \
    "$PCCD_OUT/labels/conflict.shardB.jsonl" \
  > "$PCCD_OUT/labels/conflict.jsonl"
python src/audit_labels.py static \
  --labels "$PCCD_OUT/labels/conflict.jsonl" \
  2>&1 | tee logs/audit_static_conflict.log
```

The command ran under a disconnect-safe wrapper without changing its arguments. It ran
15:05:36--15:06:47 UTC+08:00. Shard A/B inference throughput was 47,093/48,583 items/hour.

### Registered perturbation audit

```bash
cd /root/PCCD
source scripts/setup/env.sh
CUDA_VISIBLE_DEVICES=0 python src/audit_labels.py perturb \
  --model "$MODELS_DIR/qwen32b" \
  --audit "$PCCD_OUT/pool/audit.jsonl" \
  --out "$PCCD_OUT/labels/audit_perturb.jsonl" \
  2>&1 | tee logs/audit_perturb.log
```

The first complete audit ran once, from 15:16:54--15:20:01 UTC+08:00. Its negative result
was retained without prompt edits, threshold changes, selective reruns, or resampling.

### Integrity and statistical analysis

```bash
python scripts/day3/check_conflict_integrity.py \
  --pool-dir "$PCCD_OUT/pool" \
  --label-dir "$PCCD_OUT/labels" --expected 400 \
  2>&1 | tee logs/day3_conflict_integrity.log

python scripts/day3/analyze_g1.py \
  --conflict "$PCCD_OUT/labels/conflict.jsonl" \
  --audit-pool "$PCCD_OUT/pool/audit.jsonl" \
  --perturb "$PCCD_OUT/labels/audit_perturb.jsonl" \
  --out "$PCCD_OUT/results/day3_g1_metrics.json" \
  --bootstrap 10000 --seed 20260715 \
  2>&1 | tee logs/day3_g1_analysis.log
```

All confidence intervals below are 95% item-cluster percentile-bootstrap intervals from
10,000 fixed-seed replicates. Each sampled item retains all ten correlated policy labels.

## 3. Conflict artifact completeness

| Check | Result |
|---|---:|
| Pool / merged labels | 400 / 400 |
| Shard A / B | 200 / 200 |
| JSON parse success | 400/400 (100.00%) |
| Attempts = 1 | 400/400 |
| Duplicate, missing, or extra IDs | 0 |
| Wrong even/odd shard assignment or shard overlap | 0 |
| Pool-to-label source/prompt/response/meta mismatch | 0 |
| Label-schema errors | 0 |
| Pairwise overlap among train/calib/test/audit/conflict | 0 for all 10 pairs |

The 400 records comprise 242 PKU-SafeRLHF, 119 UltraFeedback, and 39 soft-style items
(60.50% [55.75%, 65.25%], 29.75% [25.25%, 34.25%], and 9.75% [7.00%, 12.75%]).
The independent PKU safety anchor is balanced: 120 safe, 122 unsafe, and zero missing.

The file is named `conflict` by the fixed sampler, but it is a random held-out slice of
the same deduplicated pool, not a conflict-enriched construction. This report therefore
does not claim that the split was enriched or targeted after observing labels.

## 4. Conflict policy coverage

Every policy has both satisfied and violated examples. Percentages and intervals are over
all 400 items for applicability and over applicable items for the final column.

| Policy | Satisfied / violated / N/A | Applicable % [95% CI] | Violated among applicable % [95% CI] | Flag |
|---|---:|---:|---:|---|
| H1 | 72 / 41 / 287 | 28.25 [23.75, 32.75] | 36.28 [27.59, 45.37] | N/A 71.75% WARN |
| H2 | 269 / 101 / 30 | 92.50 [89.75, 95.00] | 27.30 [22.87, 31.93] | |
| H3 | 108 / 136 / 156 | 61.00 [56.25, 65.75] | 55.74 [49.59, 61.85] | |
| H4 | 290 / 95 / 15 | 96.25 [94.25, 98.00] | 24.68 [20.51, 29.02] | |
| H5 | 55 / 73 / 272 | 32.00 [27.25, 36.75] | 57.03 [48.41, 65.38] | N/A 68.00% WARN |
| S1 | 196 / 11 / 193 | 51.75 [46.75, 56.50] | 5.31 [2.44, 8.63] | |
| S2 | 194 / 44 / 162 | 59.50 [54.75, 64.25] | 18.49 [13.78, 23.46] | |
| S3 | 194 / 47 / 159 | 60.25 [55.50, 65.00] | 19.50 [14.75, 24.69] | |
| T1 | 238 / 140 / 22 | 94.50 [92.25, 96.75] | 37.04 [32.34, 41.98] | |
| T2 | 250 / 69 / 81 | 79.75 [75.75, 83.75] | 21.63 [17.25, 26.20] | |

H1's sparse applicability was accepted in Day 2. H5 uses the accepted broad-semantic
teacher ground truth. Neither policy was oversampled or relabeled for Day 3.

## 5. Teacher target-label distribution heterogeneity

This section compares the ten three-state teacher target-label marginals on the same 400
held-out responses. It does **not** measure D0 critic behavior, critic F1, or text generated
by an adapted policy model.

| Estimand/test | Estimate or statistic | 95% CI / adjusted p-value |
|---|---:|---:|
| Mean JSD over 45 policy pairs (base 2) | 0.139699 | [0.126815, 0.156415] |
| Mean total-variation distance over 45 pairs | 0.329333 | [0.314000, 0.347556] |
| Paired Cochran Q, satisfied indicator (df=9) | 964.722 | Bonferroni p=2.088e-201 |
| Paired Cochran Q, violated indicator (df=9) | 316.788 | Bonferroni p=2.141e-62 |
| Paired Cochran Q, N/A indicator (df=9) | 1050.777 | Bonferroni p=5.790e-220 |

The joint equal-marginal null is rejected. Pairwise Stuart-Maxwell tests with Holm control
over all 45 pairs reject equal marginals for **44/45 pairs**. The sole non-rejection is
S2--S3: Holm p=0.525788, JSD=0.000114 [0.000005, 0.001228], and TV=0.007500
[0.002500, 0.025000]. Thus every policy differs significantly from at least eight of the
other nine, but the stronger claim that every pair is statistically distinguishable is
not established.

## 6. Perturbation stability

The denominator for order-swap agreement is the 386 items for which both canonical and
order-swap outputs parsed. Repeat and paraphrase use all 400 paired items. The 90% gates
are the pre-existing whole-record exact-match thresholds in `teacher_schema.json`.

| Variant | Parse rate % [95% CI] | Whole-record exact % [95% CI] | Policy-cell micro % [95% CI] | Registered gate |
|---|---:|---:|---:|---|
| Canonical | 100.00 [100.00, 100.00] | -- | -- | JSON >=99%: PASS |
| Repeat sampling | 100.00 [100.00, 100.00] | 93.50 [91.00, 95.75] | 98.50 [97.83, 99.10] | descriptive; no separate threshold |
| Policy order swap | 96.50 [94.50, 98.25] | 10.36 [7.40, 13.47] | 74.38 [72.68, 76.07] | >=90%: **FAIL** |
| Policy paraphrase | 100.00 [100.00, 100.00] | 1.00 [0.25, 2.00] | 68.93 [67.55, 70.33] | >=90%: **FAIL** |

Per-policy canonical agreement is:

| Policy | Repeat % [95% CI] | Order swap % [95% CI] | Paraphrase % [95% CI] |
|---|---:|---:|---:|
| H1 | 99.00 [98.00, 99.75] | 70.21 [65.63, 74.74] | 75.00 [70.75, 79.25] |
| H2 | 99.25 [98.25, 100.00] | 64.25 [59.44, 68.91] | 10.75 [7.75, 14.00] |
| H3 | 99.50 [98.75, 100.00] | 77.20 [72.92, 81.32] | 80.50 [76.75, 84.25] |
| H4 | 100.00 [100.00, 100.00] | 75.13 [70.69, 79.37] | 71.75 [67.25, 76.25] |
| H5 | 98.75 [97.50, 99.75] | 72.80 [68.38, 77.23] | 78.50 [74.25, 82.50] |
| S1 | 97.25 [95.50, 98.75] | 76.42 [72.09, 80.62] | 86.00 [82.50, 89.25] |
| S2 | 97.50 [96.00, 99.00] | 76.42 [72.16, 80.57] | 82.75 [79.00, 86.25] |
| S3 | 97.50 [95.75, 99.00] | 81.09 [77.14, 84.94] | 47.00 [42.25, 52.00] |
| T1 | 98.25 [97.00, 99.50] | 79.79 [75.77, 83.80] | 80.25 [76.25, 84.00] |
| T2 | 98.00 [96.50, 99.25] | 70.47 [65.80, 74.94] | 76.75 [72.50, 80.75] |

The paraphrase failure is systematic rather than a rounding artifact. For H2, 263/266
canonical `satisfied` and 94/101 canonical `violated` labels become `not_applicable` under
the registered paraphrase. S3 also has only 47.00% agreement. Order swap has 14 parse
failures (9 PKU, 3 UltraFeedback, 2 soft-style). The current raw artifact preserves parsed
`null` for these failures but not the rejected generation text, so their syntax cannot be
diagnosed post hoc without a new run. No rerun was performed after observing these values.

Even the independent greedy repeat is not bitwise deterministic at whole-record level:
26/400 records have at least one changed policy cell, although 3940/4000 cells agree.

## 7. G1 verdict against BRIEF F.2

| Required evidence | Local result |
|---|---|
| Ten policies statistically distinguishable in output distribution | **Partial support only.** Global paired heterogeneity is decisive and 44/45 target-label pairs differ, but S2--S3 does not. These are teacher target labels, not D0 model-output behavior. |
| D0 critic behavior differs across policies | **NOT EVALUATED.** The repository has no critic training/inference implementation, D0 checkpoint, or D0 predictions. |
| Plan threshold: per-policy F1 CV >0.15 | **N/A.** F1 requires D0 predictions against teacher truth; violated-positive vs three-class F1 and N/A handling are also not pre-registered. |
| Teacher perturbation reliability prerequisite | **FAIL.** Both registered 90% whole-record gates fail by large margins. |

**Overall local status: PROVISIONAL/INCOMPLETE; no G1 PASS is declared.** Treating teacher
perturbation labels as critic predictions, replacing F1 CV with prevalence/distance, or
relaxing the perturbation threshold after seeing the result would change the scientific
gate and was not done. This Red-level blocker is recorded in
`reports/CHANGES/2026-07-15_g1_evidence_blocker.md`; work stops here for PaperGuru's
decision.

## 8. Pre-Day-4 smoke-test status

No one-step LoRA or DPO training was launched. A read-only import check found that the
installed TRL 0.19.1 cannot import `DPOTrainer` under transformers 5.13.1 because TRL
imports the removed `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`. `pip check` also reports that
the still-installed RewardBench 0.1.4 requires transformers 4.51.0. A minimal runtime
compatibility shim appears feasible, but implementing and running the smoke after this Red
G1 stop would exceed the current authorization sequence. The hard pre-Day-4 smoke gate
therefore remains pending.

## 9. Raw artifacts and checksums

Heavy artifacts remain on the AutoDL data disk:

- `/root/autodl-tmp/pccd/outputs/labels/conflict.jsonl`;
- `/root/autodl-tmp/pccd/outputs/labels/audit_perturb.jsonl`;
- `/root/autodl-tmp/pccd/outputs/results/day3_g1_metrics.json`.

Small committed logs are `logs/audit_static_conflict.log`, `logs/audit_perturb.log`,
`logs/day3_conflict_integrity.log`, `logs/day3_g1_analysis.log`, and
`logs/pre_day4_smoke_import_check.log`.

```text
05a92c5631bc79d24c8171e165ff3aa46ed85349d32986e01015b9c67087f55f  pool/conflict.jsonl
889b4ac4c313e2e9ff21d80d3545f064e2731ef5a571bf0756872dce8d49815b  labels/conflict.jsonl
f7f2a84a5a30c0411ed251ef1dc7a30fd2a6475d797ca25fc651206bc92df6a1  pool/audit.jsonl
dfefd97cfca1c676be8ef2883d394d17e64fae84cd33555e771722c9eb371218  labels/audit_perturb.jsonl
d7d77532225399e8de3347a2072307f1dd82c395318f7e4771aacd624ae49050  results/day3_g1_metrics.json
d5ca179456a2ab5aae918756b8d6221c3968e57620dab232fb22e5929d2c88b9  logs/audit_static_conflict.log
2c934aa43c9b55d9afcf73bbd0dfcabfef56eec97c112035a7cbcfeaf4e58968  logs/audit_perturb.log
1288fa11b4e50fa9ad5341f8864332bc04228535ff5b4a4fde2cdb048757ace1  logs/day3_conflict_integrity.log
7fd13b464d02fe9d57bdacecec0cdd187c9a43d113e60905e1d5fafd982af09e  logs/day3_g1_analysis.log
052b843164d43144db5c472bb70045c5c9cbbff43e63b7f7d5520b7c378bcd37  logs/pre_day4_smoke_import_check.log
```

Open questions for PaperGuru are: whether the S2--S3 non-separation is acceptable under
G1's wording; how to handle the failed registered perturbation gates without post hoc
metric/prompt changes; and what D0 critic implementation plus F1/N/A definition should be
pre-registered before any critic training.
