# Day 2 teacher labeling and static-audit report

Date: 2026-07-15

Execution revision: `2f3b21b5c3055daf07817b542579d7033b33e37f`

Status: **labeling complete and integrity checks passed; static policy-coverage audit
requires PaperGuru review because H5 lacks the required two-class coverage**

## 1. Commands and scope

The formally approved Gate-D runtime was used without changing the teacher, prompts,
sampling seed, data, label schema, or audit thresholds:

```bash
cd /root/PCCD
git pull --ff-only origin day2/teacher-labeling
source scripts/setup/env.sh
bash scripts/day2/run_day2.sh 2>&1 | tee logs/day2_full.log
```

The command generated the fixed five-way pool, labeled only `train`, `calib`, and
`test` with one independent 32B teacher process per GPU, merged both shards, and ran
the static audit. It started at 12:30:29 and finished at 12:39:17 UTC+08:00 (8m48s).
No conflict/perturbation audit, Day-3 work, D2-D6 adaptation, LoRA, DPO, SFT, or other
training was started.

After completion, a separate read-only integrity pass checked JSONL parsing, expected
row counts, unique IDs, even/odd shard assignment, missing/extra IDs, shard overlap,
record-content identity, schema states, and cross-split leakage. It also computed the
source and PKU safety composition reported below.

## 2. Artifact completeness and source composition

| Split | Expected/pool/labels | Shard A/B | PKU-SafeRLHF | UltraFeedback | Soft-style |
|---|---:|---:|---:|---:|---:|
| train | 8000/8000/8000 | 4000/4000 | 4709 | 2350 | 941 |
| calib | 1000/1000/1000 | 500/500 | 591 | 287 | 122 |
| test | 1000/1000/1000 | 500/500 | 555 | 301 | 144 |

All three splits had zero malformed pool, merged-label, shard-A, or shard-B lines.
For every split, all of the following integrity counts were exactly zero:

- pool duplicates and label duplicates;
- missing and extra label IDs;
- wrong or missing IDs in either even/odd shard;
- shard-A/shard-B overlap;
- source, prompt, response, or metadata mismatch against the pool.

The additional pool-only splits were also complete: `audit` had 400 unique rows
(234 PKU, 126 UltraFeedback, 40 soft-style), and `conflict` had 400 unique rows
(242 PKU, 119 UltraFeedback, 39 soft-style). Every pairwise overlap among train,
calib, test, audit, and conflict was zero.

## 3. PKU safe/unsafe balance

Safe/unsafe is taken only from the independent PKU `meta.is_safe` anchor, never
inferred from the teacher's H labels.

| Split | PKU safe | PKU unsafe | Missing anchor | Safe | Unsafe |
|---|---:|---:|---:|---:|---:|
| train | 2385 | 2324 | 0 | 50.65% | 49.35% |
| calib | 282 | 309 | 0 | 47.72% | 52.28% |
| test | 262 | 293 | 0 | 47.21% | 52.79% |

The source-level safety anchor is balanced in every labeled split and has no missing
values.

## 4. Teacher JSON parsing

| Split | Parse OK | Parse fail | Failure rate | Attempts = 1 | Attempts = 2 |
|---|---:|---:|---:|---:|---:|
| train | 8000 | 0 | 0.0000% | 7995 | 5 |
| calib | 1000 | 0 | 0.0000% | 998 | 2 |
| test | 1000 | 0 | 0.0000% | 999 | 1 |

All 10,000 records contain exactly H1-H5/S1-S3/T1-T2 with a valid schema state.
Eight records required one format-repair retry; all eight then parsed. No record was
dropped, truncated, or silently replaced.

Observed final per-GPU labeling rates from the process logs were:

| Split | GPU0 shard A | GPU1 shard B |
|---|---:|---:|
| train | 51,578/hour | 52,034/hour |
| calib | 46,279/hour | 49,292/hour |
| test | 46,913/hour | 53,865/hour |

## 5. Per-policy coverage

Counts below use only successfully parsed records. Positive/negative coverage is
reported without relabeling semantics: `satisfied` and `violated` must both be nonzero
for every hard policy H1-H5. N/A percentages use all parsed records in the split as the
denominator; values above 60% are marked WARN.

### Train

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 1724 | 739 | 5537 | 69.21% | WARN N/A >60% |
| H2 | 5353 | 1900 | 747 | 9.34% | |
| H3 | 2308 | 2615 | 3077 | 38.46% | |
| H4 | 5551 | 2248 | 201 | 2.51% | |
| H5 | 0 | 12 | 7988 | 99.85% | WARN lacks two-class coverage; N/A >60% |
| S1 | 4295 | 340 | 3365 | 42.06% | |
| S2 | 4135 | 1034 | 2831 | 35.39% | |
| S3 | 4116 | 1108 | 2776 | 34.70% | |
| T1 | 4702 | 3142 | 156 | 1.95% | |
| T2 | 4985 | 2031 | 984 | 12.30% | |

### Calibration

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 195 | 95 | 710 | 71.00% | WARN N/A >60% |
| H2 | 639 | 258 | 103 | 10.30% | |
| H3 | 263 | 353 | 384 | 38.40% | |
| H4 | 697 | 279 | 24 | 2.40% | |
| H5 | 0 | 1 | 999 | 99.90% | WARN lacks two-class coverage; N/A >60% |
| S1 | 528 | 31 | 441 | 44.10% | |
| S2 | 499 | 135 | 366 | 36.60% | |
| S3 | 498 | 142 | 360 | 36.00% | |
| T1 | 592 | 388 | 20 | 2.00% | |
| T2 | 621 | 256 | 123 | 12.30% | |

### Test

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 183 | 110 | 707 | 70.70% | WARN N/A >60% |
| H2 | 652 | 250 | 98 | 9.80% | |
| H3 | 264 | 333 | 403 | 40.30% | |
| H4 | 674 | 301 | 25 | 2.50% | |
| H5 | 0 | 0 | 1000 | 100.00% | WARN neither satisfied nor violated; N/A >60% |
| S1 | 521 | 61 | 418 | 41.80% | |
| S2 | 497 | 160 | 343 | 34.30% | |
| S3 | 489 | 170 | 341 | 34.10% | |
| T1 | 537 | 445 | 18 | 1.80% | |
| T2 | 584 | 276 | 140 | 14.00% | |

## 6. Static-audit verdict and anomalies

The artifact, parsing, source-balance, and integrity checks pass. H1-H4 each have both
`satisfied` and `violated` examples in all three splits. However, the stated hard-policy
coverage requirement is **not met** for H5:

- train: 0 satisfied, 12 violated, 7,988 N/A;
- calib: 0 satisfied, 1 violated, 999 N/A;
- test: 0 satisfied, 0 violated, 1,000 N/A.

H1 also exceeds the 60% N/A warning threshold in every split (69.21%-71.00%). H5 exceeds
it overwhelmingly (99.85%-100%). These are observed audit outcomes, not parse failures.
No data, labels, prompts, policy definitions, or metrics were modified to make the audit
pass.

Accordingly, Day-2 execution is complete, but the coverage audit is **blocked for
PaperGuru judgment** rather than declared passed. Day-3 and all training remain stopped.
Any proposed correction that changes sampling, policy prompts, or data allocation must
be classified under BRIEF section G before implementation.

## 7. Raw artifacts and checksums

Heavy artifacts remain on the AutoDL data disk:

- pool: `/root/autodl-tmp/pccd/outputs/pool/`;
- merged and shard labels: `/root/autodl-tmp/pccd/outputs/labels/`;
- full run log: `/root/PCCD/logs/day2_full.log`;
- per-GPU logs: `/root/PCCD/logs/label_{train,calib,test}_{A,B}.log`;
- integrity summary: `/root/PCCD/logs/day2_integrity.log`;
- committed static logs: `logs/audit_static_{train,calib,test}.log`.

SHA-256 checksums:

```text
807af2b60683005dfce8772dd8cda18528b13c2c4a5386406d4815634d63267a  train.jsonl
7d6826c950695441fa367f4b8d772a52f17f55a852ac0211bade5e125ea77923  calib.jsonl
3055a1b3df00d7a29e82aa56a9dea6cf055d443e5c5a30f2f33e69746793677f  test.jsonl
465ebb39357ea9da6b5d40b826836a18e856c028350d0fba0a03ea00bb9c1340  audit_static_train.log
1bcc4fbec96361b6493dbc4be26914cc370fe1e3dae9ed233bcf70d688924bb2  audit_static_calib.log
47eb3d733c8c16824f524465dee057b9b3dc5b251796025f9b31139f8ff00d2b  audit_static_test.log
```
