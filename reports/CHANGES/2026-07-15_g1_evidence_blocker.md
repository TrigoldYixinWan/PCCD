# Change: G1 evidence is incomplete and teacher perturbation gates fail (severity: Red)

Date / commit: 2026-07-15 / perturbation `2c97407`, analysis `b676fb6`

Trigger: the first complete registered perturbation audit produced order-swap
whole-record agreement `40/386 = 10.36%` and paraphrase agreement `4/400 = 1.00%`, both
below the pre-existing 90% thresholds. In addition, BRIEF F.2 requires heterogeneous D0
critic behavior and `plan_9day.md` specifies per-policy F1 CV >0.15, but the repository
contains no critic training/inference implementation, D0 checkpoint, or D0 predictions.
The F1 positive class and N/A handling are not pre-registered either.

What I changed: I made only Green implementation changes. `src/audit_labels.py` now
executes the previously omitted independent `repeat_sampling` call at the registered
temperature 0, guarantees that order swap changes the order, and persists all three
registered variants. I added strict integrity and item-cluster bootstrap analysis scripts.
I did **not** change the data, prompts, paraphrases, labels, thresholds, gate definition,
or metrics. The Day-3 result is marked PROVISIONAL/INCOMPLETE and no G1 PASS is declared.

Why this and not an alternative: relaxing the 90% threshold, rewriting the paraphrase
after seeing its result, selectively rerunning, calling teacher stability "critic F1," or
replacing F1 CV with label-prevalence CV/JSD would make the gate easier after observing a
failure. Training a critic would also violate the explicit instruction to keep training
stopped. Preserving the first complete run and surfacing the missing prerequisite is the
least damaging option.

Impact on propositions/gates: P3/G1 is not yet established. Teacher target-label
heterogeneity is globally strong (44/45 pairs significant after Holm correction), but
S2--S3 is not distinguished, registered prompt-perturbation reliability fails, and the
D0 critic conjunct is unmeasured. P2/P5/P6 and G2--G4 were not run or changed.

Reversibility: the Green audit/analysis commits can be reverted without changing any
Day-2 artifact. All Day-3 raw data are retained under `$PCCD_OUT`. The scientific result
remains PROVISIONAL until PaperGuru chooses a pre-registered response; no downstream
training has consumed it.

Open question for PaperGuru: should G1 stop as a negative reliability result, or should a
new prompt/audit protocol be prospectively approved and rerun? If G1 is to continue, what
critic architecture/training protocol, F1 definition (violated-positive vs three-class),
and N/A treatment should be locked before generating D0 predictions? Is S2--S3
non-separation compatible with the intended "10 policies distinguishable" criterion?
