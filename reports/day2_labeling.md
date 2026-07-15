# Day 2 teacher labeling and static-audit report

Date: 2026-07-15

Execution revision: `1039a2912e3beb31166bb3ac215286274bb4252e`

Status: **relabeling and artifact-integrity checks passed; the H5 scientific result is
PROVISIONAL and blocked for PaperGuru review because teacher applicability is not aligned
with the approved PKU category. Day-3 and training remain stopped.**

> PaperGuru verdict (2026-07-15): **RESOLVED — Day-2 relabel (commit 4e2cf9e) is the sole
> authoritative baseline; ALL earlier Day-2 labels are void.** Decision: ACCEPT the teacher's
> broad-semantic H5 judgment as correct ground truth — it faithfully models what a deployed
> frozen critic must do (judge ANY response for an H5 violation, regardless of PKU category).
> H5 global two-class coverage (1065 sat / 1376 vio, and 120/126 in calib/test) is VALID.
> Full rationale + answers to the three open questions in
> reports/CHANGES/2026-07-15_h5_relabel_applicability.md (Resolution section).
> Codex's cross-check (joining labels back to PKU category) was excellent and is exactly the
> kind of stratified evidence we will report. Two follow-ups, both handled: (1) sample_data.py
> now stores pku_harm_categories in meta for stratified analysis (Green, done by PaperGuru) —
> this requires ONE more re-label so the metadata is present in the pool; (2) the non-H5
> distribution drift (TV<=13.9%) is an expected consequence of joint 10-policy prompting and
> is ACCEPTABLE because the entire pipeline uses this single relabel — no cross-pool mixing.
> Day-2 integrity (100% JSON parse, zero leakage) PASSES. Day-3/training stay stopped until
> the metadata re-label + audit are confirmed.

## 1. Commands and scope

The human-approved H5 definition at revision `1039a29` was pulled, then all old pool and
label JSONL artifacts were removed as instructed before the full Day-2 rerun:

```bash
cd /root/PCCD
git pull --ff-only origin day2/teacher-labeling
source scripts/setup/env.sh
rm -f "$PCCD_OUT/pool/"*.jsonl "$PCCD_OUT/labels/"*.jsonl
nohup bash -o pipefail -c \
  'bash scripts/day2/run_day2.sh 2>&1 | tee logs/day2_full_relabel.log' \
  > logs/day2_full_relabel.nohup.log 2>&1 &
```

The run regenerated the unbiased seed-0 five-way pool, labeled only `train`, `calib`, and
`test` with one independent Qwen2.5-32B process per GPU, merged the two shards, and ran the
three static audits. It started at 13:19:36 and finished at 13:28:12 UTC+08:00 (8m36s).
No category-directed oversampling, conflict/perturbation audit, Day-3 work, D2-D6
adaptation, LoRA, DPO, SFT, or other training was run.

A separate read-only Python integrity pass was sent to `python -` after sourcing
`scripts/setup/env.sh` and saved with:

```bash
python - | tee logs/day2_integrity_relabel.log
sha256sum "$PCCD_OUT"/labels/{train,calib,test}.jsonl \
  logs/audit_static_{train,calib,test}.log
```

That pass checked JSONL parsing, expected row counts, unique IDs, even/odd shard assignment,
missing/extra IDs, shard overlap, pool-to-label record identity, schema states, PKU safety
anchors, and all pairwise cross-split overlaps. A second read-only diagnostic rejoined label
IDs to the original PKU snapshot to compare H5 judgments with the approved harm category;
it is saved at `logs/h5_relabel_diagnostic.log` on AutoDL.

## 2. Artifact completeness and source composition

| Split | Expected/pool/labels | Shard A/B | PKU-SafeRLHF | UltraFeedback | Soft-style |
|---|---:|---:|---:|---:|---:|
| train | 8000/8000/8000 | 4000/4000 | 4709 | 2350 | 941 |
| calib | 1000/1000/1000 | 500/500 | 591 | 287 | 122 |
| test | 1000/1000/1000 | 500/500 | 555 | 301 | 144 |

All three labeled splits had zero malformed pool, merged-label, shard-A, or shard-B lines.
For every split, pool duplicates, label duplicates, missing/extra label IDs, wrong/missing
even/odd shard IDs, shard overlap, schema errors, and source/prompt/response/metadata
mismatches were all exactly zero.

The pool-only splits were also complete: `audit` had 400 unique rows (234 PKU, 126
UltraFeedback, 40 soft-style), and `conflict` had 400 unique rows (242 PKU, 119
UltraFeedback, 39 soft-style). Every pairwise ID overlap among train, calib, test, audit,
and conflict was zero. The regenerated source composition exactly matches the prior run;
sampling remained random and was not targeted toward H5.

## 3. PKU safe/unsafe balance

Safe/unsafe is taken only from the independent PKU `meta.is_safe` anchor, never inferred
from teacher H labels.

| Split | PKU safe | PKU unsafe | Missing anchor | Safe | Unsafe |
|---|---:|---:|---:|---:|---:|
| train | 2385 | 2324 | 0 | 50.65% | 49.35% |
| calib | 282 | 309 | 0 | 47.72% | 52.28% |
| test | 262 | 293 | 0 | 47.21% | 52.79% |

The source-level safety anchor remains balanced in every labeled split.

## 4. Teacher JSON parsing and throughput

| Split | Parse OK | Parse fail | Failure rate | Attempts = 1 | Attempts = 2 |
|---|---:|---:|---:|---:|---:|
| train | 8000 | 0 | 0.0000% | 7999 | 1 |
| calib | 1000 | 0 | 0.0000% | 999 | 1 |
| test | 1000 | 0 | 0.0000% | 1000 | 0 |

All 10,000 records contain exactly H1-H5/S1-S3/T1-T2 with a valid schema state. Two
records required one format-repair retry; both parsed. No record was dropped, truncated,
or silently replaced.

| Split | GPU0 shard A | GPU1 shard B |
|---|---:|---:|
| train | 53,337/hour | 53,626/hour |
| calib | 46,299/hour | 56,360/hour |
| test | 52,580/hour | 53,644/hour |

## 5. Per-policy coverage

Counts use successfully parsed records. N/A percentages use all records in the split;
values above 60% are marked WARN. These tables report the observed teacher output without
changing labels, data, prompts, or metrics.

### Train

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 1387 | 819 | 5794 | 72.42% | WARN N/A >60%; previously accepted |
| H2 | 5263 | 1968 | 769 | 9.61% | |
| H3 | 2166 | 2601 | 3233 | 40.41% | |
| H4 | 5594 | 2208 | 198 | 2.48% | |
| H5 | 1065 | 1376 | 5559 | 69.49% | WARN N/A >60%; see Section 6 |
| S1 | 3489 | 225 | 4286 | 53.58% | |
| S2 | 3409 | 887 | 3704 | 46.30% | |
| S3 | 3387 | 1066 | 3547 | 44.34% | |
| T1 | 4568 | 3064 | 368 | 4.60% | |
| T2 | 4734 | 1575 | 1691 | 21.14% | |

### Calibration

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 160 | 109 | 731 | 73.10% | WARN N/A >60%; previously accepted |
| H2 | 621 | 272 | 107 | 10.70% | |
| H3 | 244 | 348 | 408 | 40.80% | |
| H4 | 698 | 279 | 23 | 2.30% | |
| H5 | 120 | 181 | 699 | 69.90% | WARN N/A >60%; see Section 6 |
| S1 | 401 | 19 | 580 | 58.00% | |
| S2 | 386 | 112 | 502 | 50.20% | |
| S3 | 387 | 138 | 475 | 47.50% | |
| T1 | 563 | 387 | 50 | 5.00% | |
| T2 | 597 | 192 | 211 | 21.10% | |

### Test

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 149 | 112 | 739 | 73.90% | WARN N/A >60%; previously accepted |
| H2 | 633 | 255 | 112 | 11.20% | |
| H3 | 243 | 333 | 424 | 42.40% | |
| H4 | 678 | 293 | 29 | 2.90% | |
| H5 | 126 | 175 | 699 | 69.90% | WARN N/A >60%; see Section 6 |
| S1 | 415 | 34 | 551 | 55.10% | |
| S2 | 409 | 140 | 451 | 45.10% | |
| S3 | 404 | 162 | 434 | 43.40% | |
| T1 | 517 | 425 | 58 | 5.80% | |
| T2 | 551 | 207 | 242 | 24.20% | |

Mechanically, every hard policy now has nonzero satisfied and violated counts in every
split. That fact alone is not treated as a scientific PASS because H5's counts do not
represent the approved category as intended.

## 6. Red anomaly: H5 applicability is predominantly off-category

The approved random-sampling expectation is present in the raw pool: the selected PKU
response belongs to `Endangering National Security` 196 times in train, 33 in calib, and
23 in test. Calib/test therefore have the expected approximately 30-category-hit scale.
The teacher, however, emitted an H5 applicable label (`satisfied` or `violated`) for 2,441,
301, and 301 records respectively.

| Split | Selected PKU target rows | Teacher H5 applicable | Applicable on selected target | Target sat/vio/N/A | Off-target share of applicable |
|---|---:|---:|---:|---:|---:|
| train | 196 | 2441 | 192 | 0/192/4 | 92.13% |
| calib | 33 | 301 | 33 | 0/33/0 | 89.04% |
| test | 23 | 301 | 22 | 0/22/1 | 92.69% |

Even a broader prompt-level join, where either PKU response may carry the target category,
does not resolve the mismatch: target rows are 239/38/28 and target-applicable rows are
230/37/25 for train/calib/test, leaving 90.58%/87.71%/91.69% of teacher applicability
outside that broader target. Calib and test still have zero H5 `satisfied` examples on
target prompts.

The source breakdown further shows that H5 coverage is not category-backed:

| Split | PKU applicable | UltraFeedback applicable | Soft-style applicable |
|---|---:|---:|---:|
| train | 1973 | 466 | 2 |
| calib | 250 | 50 | 1 |
| test | 236 | 65 | 0 |

The implementation explains why the taxonomy mapping cannot constrain these judgments:

- `maps_pku_categories` is not consumed by sampling or labeling code;
- `src/sample_data.py::_load_pku` does not retain harm-category metadata in pool records;
- `src/policy_defs.py::_policy_lines` sends only the short canonical string
  `H5 National-security safeguard` to the teacher, not the full taxonomy definition;
- the teacher jointly generates all ten labels, so changing the H5 short string may also
  change later labels despite greedy decoding.

The last point is visible in non-H5 distribution drift. The table below is the
distributional total-variation distance between the superseded and current
sat/vio/N/A counts; it is a distribution-level comparison, not a record-level churn rate.

| Policy | Train | Calib | Test |
|---|---:|---:|---:|
| H1 | 4.21% | 3.50% | 3.40% |
| H2 | 1.13% | 1.80% | 1.90% |
| H3 | 1.95% | 2.40% | 2.10% |
| H4 | 0.54% | 0.10% | 0.80% |
| S1 | 11.51% | 13.90% | 13.30% |
| S2 | 10.91% | 13.60% | 10.80% |
| S3 | 9.64% | 11.50% | 9.30% |
| T1 | 2.65% | 3.00% | 4.00% |
| T2 | 8.84% | 8.80% | 10.20% |

The pool sampler, seed, data snapshot, and all non-H5 definitions were unchanged, and
source counts exactly match the prior run. The old pool hashes were not retained before
the authorized deletion, so old/new item-level pool identity cannot be independently
rechecked now; the observed non-H5 drift is nevertheless too large to call the coverage
"basically unchanged" without review.

## 7. Verdict

- Artifact completeness, JSON parsing, schema validity, balance, deduplication, and leakage
  checks: **PASS**.
- Static parser gate (`>=99%`): **PASS, 100.00% in every split**.
- Literal two-class count check for H1-H5: **PASS**.
- Intended H5 data-backed-category coverage: **PROVISIONAL / BLOCKED**. Global H5
  satisfied/violated counts are mostly off-category, and the true selected-response target
  has no satisfied examples.
- Overall Day-2 scientific acceptance: **not declared**; deciding whether the gate passed
  or changing applicability/prompt/mapping semantics is Red under BRIEF Section G.3.

No labels or metrics were altered to force a pass. The current artifacts are preserved for
PaperGuru inspection. See `reports/CHANGES/2026-07-15_h5_relabel_applicability.md`. Day-3
and all training remain stopped.

## 8. Raw artifacts and checksums

Heavy artifacts remain on the AutoDL data disk:

- pool: `/root/autodl-tmp/pccd/outputs/pool/`;
- merged and shard labels: `/root/autodl-tmp/pccd/outputs/labels/`;
- full run log: `/root/PCCD/logs/day2_full_relabel.log`;
- per-GPU logs: `/root/PCCD/logs/label_{train,calib,test}_{A,B}.log`;
- integrity summary: `/root/PCCD/logs/day2_integrity_relabel.log`;
- H5 category join: `/root/PCCD/logs/h5_relabel_diagnostic.log`;
- committed static logs: `logs/audit_static_{train,calib,test}.log`.

The old label and static-log checksums in the prior report are superseded by this complete
relabel. Current SHA-256 checksums are:

```text
60def779d20cfeffd70105187cef66288d7cefbfe977e8c9874db015995c0fc4  train.jsonl
faa0a77cb27e071f7bf973032cc8cabab459c094b762d2c65b7ea06a7d27c706  calib.jsonl
d8b112930ace673dce7e3527fd6b7fb9c1c0b58ed20f1207ddc7e10973b5c0f1  test.jsonl
9c7b0c2c2bdb5611baaa71dab22846836c8aa4a64001895f57bf70ff2f0c615e  audit_static_train.log
aa83802df91fa513eae40052b70981eafc3313937ea6fa4d8c6d6773f3e8899a  audit_static_calib.log
f3056cff14040561704812c40bd03596999f212b3b19b61dd937b1caf1ba8148  audit_static_test.log
```
