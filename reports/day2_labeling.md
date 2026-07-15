# Day 2 teacher labeling and static-audit report

Date: 2026-07-15

Execution revision: `4a6b5d336bf66b370c089804f99bb702ae363cce`

Status: **final metadata relabel complete; artifact integrity, metadata propagation,
JSON parsing, and policy coverage PASS. Ready for PaperGuru's final Day-2 confirmation.
Day-3 and training have not been started.**

> PaperGuru decision (2026-07-15): the teacher's broad-semantic H5 judgment is the
> deployment-faithful ground truth and its global two-class coverage is valid. PKU
> `Endangering National Security` membership is retained only for downstream stratified
> G2/FN-asymmetry analysis, not for category-gating H5 applicability. The earlier Red
> question is resolved in
> `reports/CHANGES/2026-07-15_h5_relabel_applicability.md`.

## 1. Commands and scope

The human-approved metadata revision was pulled, all superseded pool/label JSONL files
were removed, and the complete seed-0 Day-2 pipeline was rerun:

```bash
cd /root/PCCD
# Direct GitHub pull was attempted first but timed out without changing HEAD.
git fetch /root/autodl-tmp/pccd/tmp/day2-metadata-4a6b5d3.bundle \
  refs/heads/day2/teacher-labeling:refs/remotes/origin/day2/teacher-labeling
git merge --ff-only refs/remotes/origin/day2/teacher-labeling
source scripts/setup/env.sh
rm -f "$PCCD_OUT/pool/"*.jsonl "$PCCD_OUT/labels/"*.jsonl
bash scripts/day2/run_day2.sh 2>&1 | tee logs/day2_full_relabel2.log
```

The command was placed under a `nohup bash` wrapper so it would survive SSH disconnects;
the experiment command itself was unchanged. It regenerated all five pool splits, labeled
only train/calib/test with one Qwen2.5-32B process per GPU, merged both shards, and ran the
three static audits. The successful run started at 14:23:06 and finished at 14:31:52
UTC+08:00 (8m46s). An initial Windows-to-SSH wrapper attempt exited before invoking
`run_day2.sh` and wrote zero-byte logs; it created no pool or label records. The command was
then relaunched with an unambiguous base64-fed shell wrapper, producing the single complete
run reported here.

The local checkout had already pulled `4a6b5d3` from GitHub and verified the bundle before
transfer. AutoDL's direct GitHub pull exceeded its 90-second network timeout, so the same
Git object was transferred as a verified bundle and fast-forwarded from `4e2cf9e` to the
exact approved commit. Both checkouts subsequently reported the same full execution SHA.

No category-directed oversampling, category-gated teacher labeling, conflict/perturbation
audit, Day-3 work, D2-D6 adaptation, LoRA, DPO, SFT, or other training was run.

After completion, read-only checks were saved under `/root/PCCD/logs`:

- `day2_integrity_relabel2.log`: full JSONL, shard, schema, metadata, and leakage checks;
- `day2_metadata_pool_check.log`: field/type/category-count checks for all five splits;
- `day2_metadata_source_join.log`: exact join back to the original PKU response fields.

## 2. Artifact completeness and source composition

| Split | Expected/pool/labels | Shard A/B | PKU-SafeRLHF | UltraFeedback | Soft-style |
|---|---:|---:|---:|---:|---:|
| train | 8000/8000/8000 | 4000/4000 | 4709 | 2350 | 941 |
| calib | 1000/1000/1000 | 500/500 | 591 | 287 | 122 |
| test | 1000/1000/1000 | 500/500 | 555 | 301 | 144 |

All three labeled splits had zero malformed pool, merged-label, shard-A, or shard-B lines.
For every split, pool duplicates, label duplicates, missing/extra IDs, wrong/missing
even/odd shard IDs, shard overlap, schema errors, and source/prompt/response/metadata
mismatches were all exactly zero.

The pool-only splits were also complete: `audit` had 400 unique rows (234 PKU, 126
UltraFeedback, 40 soft-style), and `conflict` had 400 unique rows (242 PKU, 119
UltraFeedback, 39 soft-style). Every pairwise ID overlap among train, calib, test, audit,
and conflict was zero. Source composition is unchanged from the authoritative `4e2cf9e`
run, so sampling remained random and was not targeted toward H5.

## 3. PKU metadata propagation and source validation

`meta.pku_harm_categories` is a sorted string list for every PKU item and is absent from
non-PKU items. Counts below are from the regenerated pool:

| Split | PKU rows | Field missing | Type/order errors | Field on non-PKU | National-security subset |
|---|---:|---:|---:|---:|---:|
| train | 4709 | 0 | 0 | 0 | 196 |
| calib | 591 | 0 | 0 | 0 | 33 |
| test | 555 | 0 | 0 | 0 | 23 |
| audit | 234 | 0 | 0 | 0 | 10 |
| conflict | 242 | 0 | 0 | 0 | 9 |

All 6,331 PKU pool records were joined back to the original selected
`response_{resp_idx}_harm_category` field using prompt/response UID plus `resp_idx`:

| Checked | Unmatched | Category mismatch | Ambiguous source keys |
|---:|---:|---:|---:|
| 6331 | 0 | 0 | 0 |

The label records preserve the pool metadata exactly (`content_mismatch=0`) and every PKU
label has the new field. One national-security example from each labeled split was also
inspected directly:

| Split | ID | resp_idx | Stored categories | H5 | Pool meta equal |
|---|---|---:|---|---|---|
| train | `pku__9e9b349355023b37` | 1 | Cybercrime; Endangering National Security | violated | yes |
| calib | `pku__402e76796a9340aa` | 1 | Cybercrime; Endangering National Security; Endangering Public Health; Physical Harm | violated | yes |
| test | `pku__a96bedb6156f7d0f` | 0 | Cybercrime; Endangering National Security | violated | yes |

## 4. PKU safe/unsafe balance

Safe/unsafe is taken only from the independent PKU `meta.is_safe` anchor, never inferred
from teacher H labels.

| Split | PKU safe | PKU unsafe | Missing anchor | Safe | Unsafe |
|---|---:|---:|---:|---:|---:|
| train | 2385 | 2324 | 0 | 50.65% | 49.35% |
| calib | 282 | 309 | 0 | 47.72% | 52.28% |
| test | 262 | 293 | 0 | 47.21% | 52.79% |

The source-level safety anchor remains balanced in every labeled split.

## 5. Teacher JSON parsing and throughput

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
| train | 53,267/hour | 53,580/hour |
| calib | 46,232/hour | 56,261/hour |
| test | 52,550/hour | 54,024/hour |

## 6. Per-policy coverage

Counts use successfully parsed records. N/A percentages use all records in the split;
values above 60% are marked WARN. H1's high N/A rate was previously accepted by
PaperGuru. H5's broad-semantic labels are the accepted deployment ground truth.

### Train

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 1402 | 815 | 5783 | 72.29% | WARN N/A >60%; accepted |
| H2 | 5267 | 1965 | 768 | 9.60% | |
| H3 | 2183 | 2599 | 3218 | 40.23% | |
| H4 | 5600 | 2198 | 202 | 2.52% | |
| H5 | 1084 | 1372 | 5544 | 69.30% | WARN N/A >60%; two-class coverage valid |
| S1 | 3505 | 225 | 4270 | 53.38% | |
| S2 | 3406 | 891 | 3703 | 46.29% | |
| S3 | 3385 | 1067 | 3548 | 44.35% | |
| T1 | 4564 | 3062 | 374 | 4.67% | |
| T2 | 4726 | 1565 | 1709 | 21.36% | |

### Calibration

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 162 | 110 | 728 | 72.80% | WARN N/A >60%; accepted |
| H2 | 619 | 272 | 109 | 10.90% | |
| H3 | 243 | 349 | 408 | 40.80% | |
| H4 | 698 | 280 | 22 | 2.20% | |
| H5 | 122 | 181 | 697 | 69.70% | WARN N/A >60%; two-class coverage valid |
| S1 | 402 | 19 | 579 | 57.90% | |
| S2 | 386 | 112 | 502 | 50.20% | |
| S3 | 387 | 139 | 474 | 47.40% | |
| T1 | 558 | 393 | 49 | 4.90% | |
| T2 | 595 | 197 | 208 | 20.80% | |

### Test

| Policy | Satisfied | Violated | N/A | N/A % | Audit flag |
|---|---:|---:|---:|---:|---|
| H1 | 148 | 112 | 740 | 74.00% | WARN N/A >60%; accepted |
| H2 | 639 | 255 | 106 | 10.60% | |
| H3 | 240 | 333 | 427 | 42.70% | |
| H4 | 679 | 295 | 26 | 2.60% | |
| H5 | 121 | 175 | 704 | 70.40% | WARN N/A >60%; two-class coverage valid |
| S1 | 421 | 36 | 543 | 54.30% | |
| S2 | 410 | 141 | 449 | 44.90% | |
| S3 | 406 | 164 | 430 | 43.00% | |
| T1 | 518 | 425 | 57 | 5.70% | |
| T2 | 551 | 210 | 239 | 23.90% | |

Every hard policy has both satisfied and violated examples in every split. Compared with
the authoritative `4e2cf9e` relabel, the distributional total-variation distance is
0.24%/0.20%/0.50% for H5 on train/calib/test; across all policies and splits the maximum
is 0.80% (test S1). This is consistent with unchanged judgment semantics.

## 7. H5 national-security stratification for downstream G2

Global H5 is the accepted deployment ground truth. The metadata-defined PKU subset below
is a separate precomputed stratum for the downstream G2 FN-asymmetry report:

| Split | Global H5 sat/vio/N/A | Prior `4e2cf9e` global | Distribution TV | True-category N | True-category H5 sat/vio/N/A |
|---|---:|---:|---:|---:|---:|
| train | 1084/1372/5544 | 1065/1376/5559 | 0.24% | 196 | 0/192/4 |
| calib | 122/181/697 | 120/181/699 | 0.20% | 33 | 0/33/0 |
| test | 121/175/704 | 126/175/699 | 0.50% | 23 | 0/22/1 |

The genuine national-security subset is therefore an almost pure violation stress case,
as anticipated: 247/252 are H5 violations, 5/252 are N/A, and 0/252 are satisfied. These
labels were not category-gated or overwritten. This table is preparatory evidence only;
FN-asymmetry itself is not measured until the approved downstream G2 analysis.

## 8. Verdict

- Artifact completeness, schema validity, deduplication, shard assignment, and leakage:
  **PASS**.
- Static parser gate (`>=99%`): **PASS, 100.00% in every split**.
- PKU metadata presence/type/order, pool-to-label propagation, and original-source join:
  **PASS, 6,331 checked with zero mismatch**.
- Hard-policy two-class coverage: **PASS in every split**, including accepted broad H5.
- H5 metadata relabel stability: **PASS**, global distribution TV at most 0.50% versus
  `4e2cf9e`.
- Day-2 labeling deliverable: **PASS and ready for PaperGuru confirmation**.

No data, labels, prompts, policies, or metrics were changed to force a pass. Day-3/G1 and
all training remain stopped pending the requested human confirmation. The separate
TRL/PEFT-on-transformers-5 LoRA/DPO smoke test remains mandatory before D2-D6 training.

## 9. Raw artifacts and checksums

Heavy artifacts remain on the AutoDL data disk:

- pool: `/root/autodl-tmp/pccd/outputs/pool/`;
- merged and shard labels: `/root/autodl-tmp/pccd/outputs/labels/`;
- full run log: `/root/PCCD/logs/day2_full_relabel2.log`;
- per-GPU logs: `/root/PCCD/logs/label_{train,calib,test}_{A,B}.log`;
- integrity summary: `/root/PCCD/logs/day2_integrity_relabel2.log`;
- metadata checks: `/root/PCCD/logs/day2_metadata_{pool_check,source_join}.log`;
- committed static logs: `logs/audit_static_{train,calib,test}.log`.

All label, pool, and static-log checksums from earlier Day-2 runs are superseded. Current
SHA-256 checksums are:

```text
# pool
956f3e9fda07fb5f13dc29055fffd5a223885f54cbcf91c8faa2275665c0dcbb  pool/train.jsonl
81767e95765cb273bbdd25f570137f0c5db4948bd98e1b8ad3187df9c18436ac  pool/calib.jsonl
187e6a10af528ecb00a24d5efa6f82644d0ad138881ad7b531e8fe02807442ef  pool/test.jsonl
f7f2a84a5a30c0411ed251ef1dc7a30fd2a6475d797ca25fc651206bc92df6a1  pool/audit.jsonl
05a92c5631bc79d24c8171e165ff3aa46ed85349d32986e01015b9c67087f55f  pool/conflict.jsonl

# merged labels
5ac19915fedd61bcc86ff17428d4d2a5aceb3bb054621f1b97f58f4aaca37e31  labels/train.jsonl
7f6e26d3cd6b0ec23670e064f5631561d8ef2a4e6681d32d578a0d6798279641  labels/calib.jsonl
7f9da2d6d0a2476784c822f27519db12c5183dc34f2d5476f40ec500200ff3e3  labels/test.jsonl

# committed static logs
c1be8cb3851f33b3401bb1deab0060d3daf46c4b9e755bb2cd0294b5f88517ac  audit_static_train.log
3e81a29e3c3d838febdcca4407b95d281536ba5b3945ae95afdb466f5dba7d0d  audit_static_calib.log
00e5412d7b20444636f86d4bc0a4577605a40921e1e0bfbcf8f9f1edd9de0952  audit_static_test.log
```
