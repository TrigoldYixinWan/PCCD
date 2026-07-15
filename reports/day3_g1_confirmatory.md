# Day 3 G1 L1 confirmatory audit

Date: 2026-07-15

Branch: `day3/g1-heterogeneity`

Locked protocol: `reports/PREREG_G1.md` at `7770e93` (execution rules frozen)

Teacher-run code: `1fe7229e0e12eaed57f59b5a22e3f7e8776c49a4`

Report/analysis code: `5eeed79`


## Run status and boundary

AutoDL already contained the sealed single run when this turn began. Its run marker has
`status=complete`, `single_run_guard=true`, and a completed timestamp of
`2026-07-15T14:51:16.698865+00:00`; the preflight log confirms that the output and marker
paths were absent before the run. I verified the sealed artifacts and did **not** launch a
second teacher run. No threshold, audit item, model, or parser rule was changed after seeing
the result.

L1 verdict: **PARTIAL**. Repeat sampling passes the locked rule; order swap and paraphrase do
not. This is a genuine residual reliability finding under the repaired protocol, not a reason
to move a threshold. L2 remains frozen at 44/45; L3 remains deferred to Day 4; no training was
started.

## Exact commands in the sealed run

```bash
source scripts/setup/env.sh
python scripts/day3/preflight_confirmatory.py \
  --audit "$PCCD_OUT/pool/audit.jsonl" \
  --frozen-paraphrases reports/artifacts/l1_confirmatory_paraphrases.json \
  --out "$PCCD_OUT/labels/audit_confirmatory.jsonl" \
  --run-marker "$PCCD_OUT/results/l1_confirmatory_run.json" \
  2>&1 | tee logs/l1_confirmatory_preflight.log

CUDA_VISIBLE_DEVICES=0 python src/audit_labels.py confirmatory \
  --model "$MODELS_DIR/qwen32b" \
  --audit "$PCCD_OUT/pool/audit.jsonl" \
  --out "$PCCD_OUT/labels/audit_confirmatory.jsonl" \
  --frozen-paraphrases reports/artifacts/l1_confirmatory_paraphrases.json \
  --run-marker "$PCCD_OUT/results/l1_confirmatory_run.json" \
  --seed 20260715 --max-tokens 256 --gpu-mem-util 0.90 --max-model-len 4096 \
  2>&1 | tee logs/audit_confirmatory.log

python scripts/day3/analyze_confirmatory.py \
  --audit "$PCCD_OUT/pool/audit.jsonl" \
  --raw "$PCCD_OUT/labels/audit_confirmatory.jsonl" \
  --run-marker "$PCCD_OUT/results/l1_confirmatory_run.json" \
  --frozen-paraphrases reports/artifacts/l1_confirmatory_paraphrases.json \
  --out "$PCCD_OUT/results/l1_confirmatory_metrics.json" \
  --bootstrap 10000 --seed 20260715 \
  2>&1 | tee logs/l1_confirmatory_analysis.log
```

The sealed call used Qwen2.5-32B, temperature 0, seed 20260715, no retries, and independent
canonical/repeat/order/paraphrase calls. The frozen audit SHA-256 is
`f7f2a84a5a30c0411ed251ef1dc7a30fd2a6475d797ca25fc651206bc92df6a1`.

## Frozen paraphrases and schedule

The exact D-1 repaired map was frozen before inference. Its artifact SHA-256 is
`0b119118631fe8f41c39adbcdf9f4ad0b3dbee433bf6b4ae78e4e59188cbd3e5`.

The order perturbation uses the deterministic cyclic Latin square with base order
`H2,H1,H3,H4,H5,S1,S2,S3,T1,T2`; it has 10 unique rows and every policy appears in every
position exactly 40 times. Schedule SHA-256:
`5296f8bbfa6d442bee86afc05e62401394ab019b01530bace0be6c0320259bab`.

## L1 gate inputs

The primary metric is policy-cell micro-agreement over cells validly parsed in both variants.
The locked pass condition is point estimate ≥ threshold and bootstrap CI lower bound ≥
threshold − 2 percentage points. Bootstrap is the locked 10,000-replicate item-cluster
percentile bootstrap, seed 20260715.

| Perturbation | Agreement | 95% CI | Threshold | Required CI lower | Locked result |
|---|---:|---:|---:|---:|---|
| repeat_sampling | 98.850% | [98.275%, 99.350%] | 97% | 95% | PASS |
| policy_order_swap | 76.278% | [74.437%, 78.059%] | 90% | 88% | FAIL |
| policy_paraphrase | 73.559% | [72.037%, 75.094%] | 90% | 88% | FAIL |

Because repeat passes while another perturbation fails, the locked aggregate verdict is
**L1 PARTIAL**.

Strict whole-record exact match is descriptive only:

| Perturbation | Exact / jointly parsed | Exact-match |
|---|---:|---:|
| repeat_sampling | 375 / 400 | 93.750% |
| policy_order_swap | 58 / 392 | 14.796% |
| policy_paraphrase | 16 / 391 | 4.092% |

## Parsing and missing-key diagnostics (excluded from gate)

Cell-level parsing retained valid cells from records with one missing key. Strict 10-key JSON
success and valid-cell coverage were:

| Variant | Strict JSON | Valid cells |
|---|---:|---:|
| canonical | 400/400 (100.00%) | 4000/4000 (100.00%) |
| repeat_sampling | 400/400 (100.00%) | 4000/4000 (100.00%) |
| policy_order_swap | 392/400 (98.00%) | 3992/4000 (99.80%) |
| policy_paraphrase | 391/400 (97.75%) | 3990/4000 (99.75%) |

Nonzero missing-key cells were localized as follows; all omitted policies/positions had zero
missing cells:

| Variant | Policy missing | Position missing |
|---|---|---|
| policy_order_swap | S3: 8/400 (2.00%) | position 1: 8/400 (2.00%) |
| policy_paraphrase | H4: 9/400 (2.25%); T2: 1/400 (0.25%) | position 4: 9/400 (2.25%); position 10: 1/400 (0.25%) |

These tables are diagnostic and do not enter the L1 gate.

## N/A transitions (diagnostic only)

Counts below are over cells where both canonical and variant cells parsed. `from N/A` means
canonical N/A changed to an applicable label; `to N/A` means an applicable canonical label
changed to N/A.

| Perturbation | from N/A | to N/A | Notable policy transitions |
|---|---:|---:|---|
| repeat_sampling | 22 | 14 | all policies ≤4 in either direction |
| policy_order_swap | 373 | 411 | H2 to N/A=157; H5 from N/A=66 |
| policy_paraphrase | 373 | 516 | H2 to N/A=268; S3 to N/A=117 |

The repaired paraphrase removes the original non-equivalent wording bug, but substantial H2
and S3 applicability transitions remain. They are therefore teacher sensitivity findings,
not evidence that the D-1 text was still edited after observing this run.

## Per-policy cell agreement (descriptive)

Point estimate [95% item-cluster CI], with the locked gate still computed over all valid cells:

| Policy | Repeat | Order swap | Paraphrase |
|---|---:|---:|---:|
| H1 | 99.50 [98.75,100.00] | 72.00 [67.50,76.50] | 76.25 [72.00,80.50] |
| H2 | 99.25 [98.25,100.00] | 57.00 [52.25,61.75] | 32.00 [27.50,36.75] |
| H3 | 99.00 [98.00,99.75] | 78.75 [74.50,82.75] | 81.25 [77.50,85.00] |
| H4 | 99.50 [98.75,100.00] | 82.50 [78.50,86.25] | 82.35 [78.50,86.05] |
| H5 | 99.25 [98.25,100.00] | 74.50 [70.25,78.75] | 74.50 [70.24,78.75] |
| S1 | 98.75 [97.50,99.75] | 76.25 [72.00,80.25] | 81.50 [77.75,85.25] |
| S2 | 98.00 [96.50,99.25] | 78.75 [74.75,82.75] | 79.75 [75.75,83.75] |
| S3 | 98.50 [97.25,99.50] | 81.38 [77.38,85.13] | 66.75 [62.25,71.50] |
| T1 | 98.25 [97.00,99.50] | 84.25 [80.50,87.75] | 84.00 [80.25,87.50] |
| T2 | 98.50 [97.25,99.50] | 77.50 [73.25,81.50] | 77.44 [73.25,81.61] |

## Raw artifacts and checksums

- frozen paraphrases: `reports/artifacts/l1_confirmatory_paraphrases.json` —
  `0b119118631fe8f41c39adbcdf9f4ad0b3dbee433bf6b4ae78e4e59188cbd3e5`
- raw labels: `$PCCD_OUT/labels/audit_confirmatory.jsonl` —
  `2c5643c981493ef70e37b0cedd2b06fa9db7d17681de235c963ec28a7d5e5f40`
- run marker: `$PCCD_OUT/results/l1_confirmatory_run.json` —
  `e1598d864c2a8b4f0b541a78ffd2ad2085bf9d76a92956c40acf56495c44f67f`
- metrics: `$PCCD_OUT/results/l1_confirmatory_metrics.json` —
  `1fa43ff9691e80691c6c11c21220548a202d16afda242d74b958664d84fb75b3`
- logs: `logs/l1_confirmatory_preflight.log`, `logs/audit_confirmatory.log`, and
  `logs/l1_confirmatory_analysis.log`.

## Publication interpretation

The locked rerun removes the two diagnosed measurement bugs and recovers deterministic
repeat agreement (98.85%), but order and paraphrase sensitivity remain far below their frozen
90% thresholds. Therefore the reliability claim cannot be silently promoted to PASS. The
paper should report L1 as PARTIAL and retain the two earmarked findings: strong joint-context
consistency and genuine wording/order sensitivity. Any causal claims relying on perfect
teacher-label invariance should be narrowed accordingly.
