# Day 3 G1 diagnostic report (pre-confirmatory; no gate verdict)

Date: 2026-07-15

Branch: `day3/g1-heterogeneity`

Locked protocol: `reports/PREREG_G1.md` and `reports/day3_diag_plan.md` at `01a761f`

Teacher-execution code: `776261269d5f279bc0372ee8f1b4e262c4802131`

Final CPU-analysis code: `b9295d5`

Status: diagnostic complete; the confirmatory 400-item audit was **not run**; no G1/L1
PASS/FAIL/PARTIAL verdict is made here. L2 remains frozen at 44/45; L3 remains deferred.

## Exact commands

```bash
source scripts/setup/env.sh
python scripts/day3/build_diag100.py \
  --pool-dir "$PCCD_OUT/pool" --label-dir "$PCCD_OUT/labels" \
  --out-pool "$PCCD_OUT/pool/diag100.jsonl" \
  --out-reference "$PCCD_OUT/labels/diag100_reference.jsonl" \
  --manifest reports/artifacts/day3_diag100_manifest.json \
  2>&1 | tee logs/day3_diag100_build.log

CUDA_VISIBLE_DEVICES=0 python src/diag_structure_ablation.py \
  --model "$MODELS_DIR/qwen32b" \
  --diag "$PCCD_OUT/pool/diag100.jsonl" \
  --out "$PCCD_OUT/labels/diag100_structure_ablation.jsonl" \
  2>&1 | tee logs/day3_diag_structure.log

python scripts/day3/analyze_diag.py \
  --reference "$PCCD_OUT/labels/diag100_reference.jsonl" \
  --manifest reports/artifacts/day3_diag100_manifest.json \
  --ablation "$PCCD_OUT/labels/diag100_structure_ablation.jsonl" \
  --pool-dir "$PCCD_OUT/pool" --label-dir "$PCCD_OUT/labels" \
  --out "$PCCD_OUT/results/day3_diag_metrics.json" --force \
  2>&1 | tee logs/day3_diag_analysis.log
```

`--force` was used only for the deterministic CPU summary after adding partial-cell and
full-universe stratification. The 1,400 teacher calls were run once and were not repeated.

## diag100 specification and integrity

The diagnostic view was selected deterministically from all 10,800 mutually disjoint frozen
pool items with their frozen teacher labels. A SciPy/HiGHS MILP selected exactly 100 items,
fixed source/split counts to largest-remainder proportional quotas, required at least 20 of
each policy×state cell and at least 20 cross-policy items, then minimized absolute deviation
from 1/3 per state. HiGHS returned `Optimal` (SciPy 1.18.0; MIP gap
`5.065149155971285e-09`). This is a balanced diagnostic view only; it neither replaces a
frozen split nor enters training or gate scoring.

- Source counts: PKU 59, UltraFeedback 29, soft_style 12.
- Frozen-split overlap: train 74, calib 9, test 9, audit 4, conflict 4.
- Cross-policy conflict items: 36 (at least one satisfied and one violated label).
- Minimum observed policy×state count: 26.
- All 100 IDs, source/split membership, conflict flag, and checksums are frozen in
  `reports/artifacts/day3_diag100_manifest.json`.

| Policy | Satisfied | Violated | N/A |
|---|---:|---:|---:|
| H1 | 34 | 29 | 37 |
| H2 | 40 | 34 | 26 |
| H3 | 33 | 33 | 34 |
| H4 | 35 | 34 | 31 |
| H5 | 32 | 34 | 34 |
| S1 | 34 | 32 | 34 |
| S2 | 33 | 33 | 34 |
| S3 | 33 | 33 | 34 |
| T1 | 34 | 35 | 31 |
| T2 | 33 | 33 | 34 |

Pool SHA-256: `4b6460c5700eb550068142fe33c75df3db7c5d34372e4c95d2cbbf008e4520fb`

Reference SHA-256: `d98a92d88be4b6c380893678c1280378863efdad705bc0f457bb128422280e52`

## D-1: manual paraphrase equivalence audit

The comparison unit was the canonical line actually shown to the teacher, not an inferred
broader or narrower policy definition. Six old strings added qualifiers absent from the
canonical line; these were Green paraphrase bugs and were replaced before D-2. Canonical
strings, taxonomy, teacher label states, data, and frozen first-run outputs were unchanged.
The D-2 structures all use canonical policy lines, so this repair affects the later registered
paraphrase perturbation but does not alter the D-2 comparisons below.

| Policy | Verdict | Finding / registered replacement |
|---|---|---|
| H1 | EDIT | Old wording narrowed to “fields/internal restricted sources”; now “Keep confidential data undisclosed.” |
| H2 | EDIT | Old wording narrowed the role boundary to restricted actions for unauthorized roles; now “Honor role-based authorization limits.” |
| H3 | EDIT | Old wording added “operational,” “clearly,” and request/refusal qualifiers; now “Refuse harmful or dangerous assistance.” |
| H4 | EDIT | Old wording added “materially” and “on the task”; now “Be factually accurate and non-misleading.” |
| H5 | EDIT | Old wording narrowed the safeguard to operational help; now “Follow national-security safeguards.” |
| S1 | PASS | “Response formality should match what was requested.” |
| S2 | PASS | “Response length/conciseness should match what was requested.” |
| S3 | EDIT | Old wording required a structured-output format rather than matching structure generally; now “Response organization should match the requested structure.” |
| T1 | PASS | “The requested task/instruction should actually be completed.” |
| T2 | PASS | “Task content should be factually correct.” |

The first-run H2 `263/266 satisfied→N/A` transition remains frozen evidence about the old,
non-equivalent perturbation. It is not counted as teacher sensitivity under the repaired
meaning-preserving wording.

## D-2: prompt-structure ablation

All estimates compare the structure output with the frozen temperature-0 canonical label.
CIs are 95% item-cluster percentile bootstrap intervals (10,000 replicates, seed 20260715).
The primary diagnostic statistic is parsed policy-cell micro-agreement.

| Structure | Calls strictly parsed | Parsed cells | Cell-micro agreement, % [95% CI] |
|---|---:|---:|---:|
| single-policy | 1000/1000 | 1000/1000 | 58.40 [54.00, 62.70] |
| 5-policy block | 200/200 | 1000/1000 | 71.20 [67.30, 74.90] |
| 10-policy joint | 100/100 | 1000/1000 | 89.90 [87.10, 92.40] |
| Latin-square order | 90/100 | 990/1000 | 76.36 [72.51, 80.14] |

### Per-policy N/A rate, % [95% item-cluster bootstrap CI]

| Policy | Frozen ref | Single-policy | 5-block | 10-joint | Latin-square |
|---|---:|---:|---:|---:|---:|
| H1 | 37.0 | 55.0 [45.0, 65.0] | 37.0 [28.0, 47.0] | 33.0 [24.0, 42.0] | 21.0 [13.0, 29.0] |
| H2 | 26.0 | 13.0 [7.0, 20.0] | 6.0 [2.0, 11.0] | 20.0 [12.0, 28.0] | 11.0 [5.0, 17.0] |
| H3 | 34.0 | 47.0 [37.0, 57.0] | 33.0 [24.0, 42.0] | 32.0 [23.0, 41.0] | 29.0 [20.0, 38.0] |
| H4 | 31.0 | 15.0 [8.0, 22.0] | 27.0 [19.0, 36.0] | 20.0 [13.0, 28.0] | 10.0 [5.0, 16.0] |
| H5 | 34.0 | 62.0 [53.0, 71.0] | 70.0 [61.0, 79.0] | 47.0 [37.0, 57.0] | 34.0 [25.0, 43.0] |
| S1 | 34.0 | 3.0 [0.0, 7.0] | 11.0 [5.0, 17.0] | 38.0 [29.0, 48.0] | 48.0 [38.0, 58.0] |
| S2 | 34.0 | 2.0 [0.0, 5.0] | 12.0 [6.0, 19.0] | 36.0 [27.0, 46.0] | 49.0 [39.0, 59.0] |
| S3 | 34.0 | 1.0 [0.0, 3.0] | 13.0 [7.0, 20.0] | 35.0 [26.0, 45.0] | 43.3 [33.3, 53.4] |
| T1 | 31.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 18.0 [11.0, 26.0] | 9.0 [4.0, 15.0] |
| T2 | 34.0 | 5.0 [1.0, 10.0] | 37.0 [28.0, 47.0] | 29.0 [20.0, 38.0] | 30.0 [21.0, 39.0] |

### Localization

The preregistered “single stable, joint unstable” interference hypothesis is not supported.
Agreement rises monotonically from single (58.40%) to 5-block (71.20%) to canonical joint
(89.90%). The terse policy names are therefore not independently self-calibrating: removing
the other policies changes both applicability and polarity. Joint presentation supplies a
shared contrast/calibration context rather than merely adding interference.

This context effect is policy-specific, not a uniform N/A-under-load effect. For example,
single/block prompting under-calls N/A for S1-S3 and T1, but over-calls N/A for H5. H2 N/A
is 13% single, 6% block, 20% joint, versus 26% frozen; thus the old H2 paraphrase collapse
cannot be explained by joint load alone.

Latin-square ordering does not restore canonical agreement. Position-specific agreement
ranges from 67.78% (position 1; 90 parsed cells) to 88.00% (position 5; 100 cells). More
strongly, all 10 calls with S3 in position 1 omitted exactly the S3 key while returning the
other nine valid labels. Strict whole-call parsing is therefore 90/100; the cell-level metric
includes those 90 valid labels and excludes only the 10 unparsed S3 cells. No repair or
teacher rerun was performed. This is direct evidence of a position-dependent output-schema
failure plus broader label-position sensitivity.

## D-3: S2×S3 correlation and generator audit

Rows below are S2 and columns are S3, each ordered satisfied / violated / N/A.

Balanced diag100 contingency (`n=100`, Cramér's V `0.9702`):

| S2 \ S3 | Satisfied | Violated | N/A |
|---|---:|---:|---:|
| Satisfied | 32 | 1 | 0 |
| Violated | 1 | 32 | 0 |
| N/A | 0 | 0 | 34 |

Because balanced selection is diagnostic rather than representative, the same frozen labels
were also summarized over all 10,800 original items (read-only):

| S2 \ S3 | Satisfied | Violated | N/A |
|---|---:|---:|---:|
| Satisfied | 4461 | 82 | 22 |
| Violated | 62 | 1120 | 54 |
| N/A | 21 | 269 | 4709 |

Full-universe association is strong (`V=0.8912`). Crucially it remains strong outside the
self-built generator: PKU `n=6331, V=0.8275`; UltraFeedback `n=3183, V=0.8102`. Within
soft_style it is weak (`n=1286, V=0.0962`). Thus the global S2/S3 association is mainly a
teacher/applicability property of responses, not a soft-generator artifact.

The generator code and all 1,286 frozen soft records were audited. It chooses exactly one
registered axis per item: formality 443, verbosity 410, structure 433; explicit simultaneous
S2+S3 manipulation is `0/1286`. Structure templates do add layout and some length cues, so
433/1286 (33.67%) are conservatively flagged as a *potential* template confound. However,
the weak soft-only V and strong non-soft V show that this cannot explain the observed global
correlation. Chosen response poles are balanced (A 645, B 641). No template change is
indicated before confirmation.

## Anomalies and Green implementation notes

1. The 10 Latin S3-first key omissions are a measured position effect, not repaired data.
2. The initial CPU re-analysis failed before writing metrics with
   `ModuleNotFoundError: No module named 'scripts.day3'`; a one-line direct-script import
   fix (`b9295d5`) was applied and the deterministic CPU analysis rerun. Teacher outputs were
   untouched.
3. vLLM logged `SM 12.x requires CUDA >= 12.9` capability-detection warnings but completed
   all 1,400 calls on one RTX PRO 6000; this warning did not abort or change dtype.

## Raw artifacts and checksums

- diag pool: `$PCCD_OUT/pool/diag100.jsonl` —
  `4b6460c5700eb550068142fe33c75df3db7c5d34372e4c95d2cbbf008e4520fb`
- frozen reference: `$PCCD_OUT/labels/diag100_reference.jsonl` —
  `d98a92d88be4b6c380893678c1280378863efdad705bc0f457bb128422280e52`
- raw 1,400 calls: `$PCCD_OUT/labels/diag100_structure_ablation.jsonl` —
  `c9377648f591fd5c321ee6712962f2b1e967b682306b784578e89e77602bcfb9`
- full metrics: `$PCCD_OUT/results/day3_diag_metrics.json` (tracked copy:
  `reports/artifacts/day3_diag_metrics.json`) —
  `d4eb2c782de5747e2c263f9dd557a28f35b85c2279f04624e0db2e80c25ab7bf`
- selected IDs/manifest: `reports/artifacts/day3_diag100_manifest.json` —
  `4e233a5438d1b5a9b5630e6c2b85c2d0d33c35ab5fa0612b79772b424a403931`
- logs: `logs/day3_diag100_build.log`, `logs/day3_diag_structure.log`, and
  `logs/day3_diag_analysis.log`.

## Review boundary

Work stops here for PaperGuru review. Per the locked ordering, no confirmatory 400-item audit,
gate decision, Day-4 L3 critic, TRL shim/smoke test, or training was started.
