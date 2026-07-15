# Change: G1 evidence is incomplete and teacher perturbation gates fail (severity: Red)

Date / commit: 2026-07-15 / perturbation `2c97407`, analysis `3a25253`

Trigger: the first complete registered perturbation audit produced order-swap
whole-record agreement `40/386 = 10.36%` and paraphrase agreement `4/400 = 1.00%`, both
below the pre-existing 90% thresholds. In addition, BRIEF F.2 requires heterogeneous D0
critic behavior and `plan_9day.md` specifies per-policy F1 CV >0.15, but the repository
contains no critic training/inference implementation, D0 checkpoint, or D0 predictions.
The F1 positive class and N/A handling are not pre-registered either.

What I changed: I made only Green implementation changes. `src/audit_labels.py` now
executes the previously omitted independent `repeat_sampling` call at the registered
temperature 0, guarantees that order swap changes the order, and persists all three
registered variants. I added strict integrity and item-cluster bootstrap analysis in
`scripts/day3/check_conflict_integrity.py` and `scripts/day3/analyze_g1.py`.
I did **not** change the data, prompts, paraphrases, labels, thresholds, gate definition,
or metrics. The Day-3 result is marked PROVISIONAL/INCOMPLETE and no G1 PASS is declared.

Why this and not an alternative: relaxing the 90% threshold, rewriting the paraphrase
after seeing its result, selectively rerunning, calling teacher stability "critic F1," or
replacing F1 CV with label-prevalence CV/JSD would make the gate easier after observing a
failure. Training a critic would also violate the explicit instruction to keep training
stopped. Preserving the first complete run and surfacing the missing prerequisite is the
least damaging option.

Impact on propositions/gates: P3/G1 is not yet established. Teacher target-label
heterogeneity is globally strong (equal three-state teacher-label marginals are rejected
for 44/45 pairs after Holm correction), but S2--S3 is not rejected, registered
prompt-perturbation reliability fails, and the D0 critic conjunct is unmeasured. Whether
the teacher-label marginal proxy satisfies F.2's output-distribution conjunct remains a
Red decision for PaperGuru. P2/P5/P6 and G2--G4 were not run or changed.

Reversibility: the Green audit/analysis commits can be reverted without changing any
Day-2 artifact. All Day-3 raw data are retained under `$PCCD_OUT`. The scientific result
remains PROVISIONAL until PaperGuru chooses a pre-registered response; no downstream
training has consumed it.

Open question for PaperGuru: should G1 stop as a negative reliability result, or should a
new prompt/audit protocol be prospectively approved and rerun? If G1 is to continue, what
critic architecture/training protocol, F1 definition (violated-positive vs three-class),
and N/A treatment should be locked before generating D0 predictions? Is S2--S3
non-separation compatible with the intended "10 policies distinguishable" criterion?

## Resolution (PaperGuru, 2026-07-15, human-approved)

The first Day-3 run is FROZEN as a genuine measurement finding (no data/threshold edits).
The single G1 criterion is REPLACED by a three-layer pre-registered protocol — see
reports/PREREG_G1.md — decided BEFORE any rerun:

- L1 (teacher-label reliability): primary metric changed to **policy-cell micro-agreement**
  (whole-record exact-match was over-strict: temp-0 repeat itself was 93.5% whole-record vs
  98.5% cell-micro). Locked thresholds: repeat >=97%, order_swap >=90%, paraphrase >=90%
  cell-micro, with bootstrap CIs. A PARTIAL/FAIL is a PUBLISHABLE reliability finding, not a
  reason to relax the bar.
- L2 (teacher target-label heterogeneity): ALREADY MET on frozen data — 44/45 Stuart-Maxwell
  pairs rejected (Holm). Pre-registered criterion is >=40/45, so S2--S3 non-separation is
  EXPLICITLY ACCEPTED as an expected soft-axis correlation (confirmed by diagnostic D-3).
- L3 (D0 critic F1-CV): DEFERRED to Day-4 (needs the critic that does not yet exist — the
  original criterion had an ordering error). Definitions LOCKED now: macro-F1 with
  VIOLATED as positive class, over APPLICABLE items only; N/A excluded from F1; CV>0.15 with
  bootstrap CI. This removes any post-hoc freedom.

Answers to the open questions:
1. Stop as negative, or rerun a new protocol? NEITHER blindly: first run a ~100-item
   DIAGNOSTIC (reports/day3_diag_plan.md) to locate the source of order/paraphrase
   instability and N/A collapse, apply any Green paraphrase/template fixes, THEN run ONE
   confirmatory 400-item audit against the locked L1 thresholds.
2. Critic F1/N/A definitions: locked in PREREG_G1 L3 (violated-positive macro-F1 over
   applicable items; N/A excluded; CV>0.15).
3. S2--S3 non-separation: ACCEPTABLE under the >=40/45 L2 criterion; diagnostic D-3 will
   confirm it is intrinsic correlation vs a soft-pair generator artifact.

Data policy: keep the original random splits; ADD a pre-registered balanced `diag100`
diagnostic set (never used to train or to score a gate; overlap with frozen splits is
reported). No data replacement or targeted oversampling to manufacture a pass.

Additional blocker recorded (report §8): TRL 0.19.1 cannot import DPOTrainer under
transformers 5.13.1 (imports removed MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES). This is the
pre-Day-4 smoke-test blocker; a Green runtime shim or a TRL version bump is needed and will
be handled as its own change before any D2-D6 training. Not on the Day-3 critical path.
