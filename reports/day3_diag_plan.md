# PCCD — G1 Diagnostic Plan (~100 items, run BEFORE the confirmatory rerun)

Pre-registered 2026-07-15 (human-approved). Goal: locate the SOURCE of the order-swap /
paraphrase instability and the N/A collapse seen in the first Day-3 run, so the single
confirmatory 400-item audit is designed correctly instead of relaxed after the fact. This
is a DIAGNOSTIC (measurement-design) experiment; it does not touch any proposition/gate
verdict and produces no G1 pass/fail.

## Sample: pre-registered balanced diagnostic set (KEEP the original random splits too)
Construct a NEW ~100-item balanced diagnostic set (call it `diag100`) that is separate from
train/calib/test/audit/conflict and does NOT replace them. Sampling rule (LOCKED):
- Draw from the existing dedup pool sources only; record ids so overlap with the five
  frozen splits is reported (overlap is allowed here because diag100 is diagnostic-only and
  never used to train or to score a gate).
- Balance target: for each of the 10 policies, include items that (per the frozen Day-2/3
  teacher labels) yield a mix of satisfied / violated / not_applicable, plus a handful of
  cross-policy conflict items (one policy satisfied while another violated on the same
  response). This is BALANCED SELECTION for diagnosis, NOT oversampling into a scored set.
- Size ~100 (document exact n and per-policy sat/vio/na counts).

## D-1 — Manual paraphrase semantic-equivalence check
For the registered `policy_paraphrase` strings (the _PARAPHRASE map in src/policy_defs.py),
have a human (you) confirm each paraphrase is TRULY meaning-preserving vs the canonical
policy line. The first run showed H2 collapsing 263/266 satisfied -> N/A under paraphrase
and S3 at 47%; a paraphrase that subtly narrows/broadens scope would explain that. Output:
a per-policy PASS/EDIT verdict on paraphrase equivalence. If a paraphrase is judged
non-equivalent, it is a paraphrase BUG (Green fix), not evidence of teacher instability —
fix the wording and note it; if paraphrases are equivalent yet labels still move, that is a
genuine teacher-sensitivity finding.

## D-2 — Prompt structure ablation (locate order/joint interference)
On diag100, label each item under four teacher-prompt structures and compare cell-micro
agreement vs a temperature-0 canonical reference:
  (a) single-policy   : ask the teacher about ONE policy at a time (10 calls/item).
  (b) 5-policy block   : two blocks (H1-H5) and (S1-S3,T1-T2).
  (c) 10-policy joint  : the current production prompt.
  (d) Latin-square order: 10-policy joint but with policy order permuted per a Latin square,
      to separate "position effect" from "presence-of-other-policies effect".
Report per-structure cell-micro agreement and per-policy N/A rate. Hypotheses being tested:
- If (a) single-policy is stable but (c) joint is not, the instability is INTERFERENCE from
  jointly judging 10 policies (a real, reportable property of multi-policy LLM critics).
- If (d) Latin-square restores agreement, the effect is POSITIONAL and can be mitigated by
  order-randomization/averaging.
- N/A-collapse source: compare N/A rates across (a)-(d) to see whether joint prompting
  inflates N/A (the H2 paraphrase collapse suggests the teacher defers to N/A under load).

## D-3 — S2/S3 correlation check
Quantify whether S2 (verbosity) and S3 (structure) are genuinely correlated soft axes
(expected) or an artifact of the self-built soft-pair generator. On diag100, report the
joint contingency of S2×S3 teacher labels and the fraction of items where the soft-pair
generator set both axes from the same template body. If correlation is intrinsic, S2--S3
non-separation in L2 is accepted (per PREREG_G1 L2). If it is a generator artifact, it is a
Green fix to the soft-pair templates (does not affect hard policies).

## Deliverables
- reports/day3_diag.md: diag100 spec + per-policy balance counts; D-1 verdicts; D-2 table
  (4 structures × cell-micro × N/A rate, with 95% bootstrap CIs); D-3 S2×S3 contingency.
- Any paraphrase/template fixes are Green and land with a note; they must be applied BEFORE
  the confirmatory rerun, and the confirmatory rerun then uses the FIXED prompts.
- After the diagnostic, the single confirmatory 400-item audit is run ONCE against the
  LOCKED PREREG_G1 thresholds. No second bite.

## Ordering (LOCKED)
1. Build diag100 (record ids + balance).            (CPU)
2. D-1 manual paraphrase check + any Green fixes.
3. D-2 structure ablation + D-3 correlation.        (GPU, teacher)
4. Write reports/day3_diag.md; PaperGuru reviews.
5. Only then: confirmatory 400-item audit vs PREREG_G1 (L1) + re-affirm L2.
6. L3 (D0 critic F1-CV) remains deferred to Day-4, definitions locked in PREREG_G1.
Training stays STOPPED throughout; the trl/peft@transformers-5 smoke test is handled
separately (see below) but no D2-D6 training runs until L3 definitions are in force.
