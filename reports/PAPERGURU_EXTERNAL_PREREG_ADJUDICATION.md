# PaperGuru Adjudication of PREREG_EXTERNAL_GUARD §12 + GAP_SURVIVES ruling

Reviewer: PaperGuru (independent supervisor). Date: 2026-07-16.
Scope: PR #9. Adjudicates the GAP_SURVIVES conclusion in ICLR2025_COLLISION_AUDIT.md and the
eight open items in PREREG_EXTERNAL_GUARD.md §12. Changes NO frozen verdict (P2-P8, G1-G6,
CORE_NOT_ESTABLISHED all frozen). Qwen reference is NOT human ground truth; old lockbox not
reanalyzed; P5/mean-degradation not revived. No external guard scoring authorized by this doc.

## Part A — GAP_SURVIVES ruling: ACCEPTED

The collision audit is rigorous (full read of ICLR 2025 Liu et al. incl. code at commit
b46b1a3 — line-level evidence that categories are collapsed to binary in eval.py:233/253-258).
Q1-Q5 answers are correct and evidence-backed: Liu et al. is per-benchmark binary, no signed
criterion drift vector, no cancellation, no mean-vs-worst GUARD rank reversal, no simultaneous
criterion certificate, no criterion-vector stability. NO-GO trigger does NOT fire. ACCEPTED.

Boundary the audit correctly drew and that I RATIFY as binding:
- Hansen et al. 2024 already own "max group-wise ECE + worst-group selection changes the chosen
  method." Our increment is therefore NOT "worst-criterion matters" but specifically:
  overlapping SAFETY criteria (not demographic groups) + GUARD-family selection consequence +
  cross-taxonomy/cross-family replication + signed criterion-shift cancellation as a registered
  audit statistic. The paper MUST cite Hansen 2024, Detommaso 2024, Wu 2024, and the 2026
  toxicity-subgroup work as the subgroup-calibration baseline and claim only the safety-guard
  increment. Cancellation and "small mean does not bound max" are audit arithmetic, never theorems.
- Claim boundary is distribution/response-source/benchmark shift ONLY. No policy-adaptation claim.

## Part B — §12 open-item adjudication

### Item 1 — BeaverTails 30k vs exclusive 330k as a cohort shift: REVISE (downgrade the label)
Verified dataset structure (HF card): one `default` subset, splits 330k_train (301k),
330k_test (33.4k), 30k_train (27.2k), 30k_test (3.02k); `category` is a 14-key dict; `is_safe`
bool. The 30k and 330k are SCALE VARIANTS of the same collection, not independently sourced
cohorts. RULING: `BT-COHORT` (330k_test minus all 30k, near-dup removed) may be used ONLY as a
labelled **sampling/annotation-round shift within one collection**, and must be named exactly
that. Do NOT advertise it as a cross-cohort or cross-population shift. It is a WEAK shift; the
paper's substantive shift weight must rest on the cross-benchmark comparisons (BT<->AEGIS) and
the AEGIS prompt->response (AG-RESPONSE) task shift, not on BT 30k/330k. Add this naming
constraint to §4.

### Item 2 — AEGIS 2.0 category attributable to the same human-labelled unit: BLOCKING-UNTIL-VERIFIED
This cannot be ruled from memory. AEGIS2.0 mixes human and jury/LLM-expanded labels and has
prompt- and response-level fields. RULING: a metadata-only provenance audit MUST establish, and
freeze in the lock, that (a) `violated_categories` attaches to the SAME annotated unit whose
`*_label_source` == human, and (b) the safe/unsafe decision and the category come from the same
human annotation pass. If either fails, AEGIS is `NOT_LOCKABLE` as a criterion benchmark and the
study drops to one criterion benchmark => does not meet the >=2 requirement => the pivot pauses
for a replacement human benchmark (candidate: recover WildGuardTest native categories, or add a
third). No guard scoring until this audit passes. Keep §3's fail-closed rule; make it explicit
that the provenance audit is a PRE-LOCK gate, not a post-hoc check.

### Item 3 — AEGIS human RESPONSE support for AG-RESPONSE: BLOCKING-UNTIL-VERIFIED
Same audit must count human-labelled RESPONSE items per criterion. If any primary criterion
lacks >=100 human response-positives AND >=100 human safe response-negatives (the §7 minima),
AG-RESPONSE is not primary and is not silently replaced (§4 already says return-to-PaperGuru;
ratified). Report the counts in the frozen support table before lock.

### Item 4 — common-taxonomy mapping + two-reviewer procedure: APPROVED with additions
The 6-category provisional map (violence/physical; hate/identity; sexual incl. minors nested;
self-harm; illegal/criminal w/ weapons+substances pooled+separate; privacy/PII) is defensible.
Additions REQUIRED before lock: (a) the exact native->common map for BOTH BeaverTails' 14 flags
and AEGIS's categories, written out and SHA-frozen; (b) two reviewers map from published
definitions ONLY, blind to any guard output, disagreements resolved before outcome access, signed;
(c) native fine-grained results reported beside the mapped ones (§5 already requires; ratified);
(d) >=4 common criteria must clear support or the cross-benchmark analysis is NON_EVALUABLE.

### Item 5 — exact guard revisions + verbalizers: APPROVED-IN-PRINCIPLE, must be filled before lock
Three families (Llama-Guard-3-8B / WildGuard-Mistral / ShieldGemma-2B) satisfy ">=3 families,
>=2 scales." Before lock, freeze per guard: exact HF repo + revision hash, license, chat template,
the safe/unsafe decision verbalizer token(s), multi-token sequence-logprob rule, and the fixed
task prompt. §6's "raw logits + normalized prob, deterministic, malformed retained not repaired"
is ratified.

### Item 6 — ShieldGemma omnibus-policy comparability to other guards: REVISE (add a comparability check)
ShieldGemma emits a per-policy safety probability given a supplied policy; Llama Guard / WildGuard
emit an overall unsafe decision. Feeding ShieldGemma a single frozen omnibus policy listing the
taxonomy is acceptable ONLY if its output is treated as the SAME estimand ("overall unsafe prob")
as the others. RULING: add a pre-registered COMPARABILITY DIAGNOSTIC on the SOURCE split only
(no target, no outcome peeking beyond source): report each guard's source reliability diagram +
support so that a guard whose probability is degenerate (e.g. near-constant) is flagged before
target analysis. If ShieldGemma's omnibus score cannot be defensibly read as overall-unsafe on
source, it becomes a sensitivity guard and a 4th family must be pre-named as primary. This
protects H-Rank from being driven by an incomparable score.

### Item 7 — support minima + ECE thresholds + practical constants: APPROVED with one tightening
100/100 per-class primary minimum, 50-99 = UNDERPOWERED_DIAGNOSTIC, <50 descriptive: APPROVED.
tau=0.02, epsilon=0.01, c0=0.01, r0=0.01 absolute ECE: APPROVED as pre-registered constants
(they are consistent with the observed effect scale and are set before external outcomes).
TIGHTENING: with 15 equal-mass bins, an ECE estimate at 100 positives is noisy; REQUIRE that any
criterion entering a PRIMARY vector reports its per-bin occupancy and that the simultaneous
max-|t| interval (not the point estimate) governs every H-Aggregate/H-Cancellation decision
(§10 already uses max-|t|; make it explicit that no point-estimate-only claim is primary).

### Item 8 — H-Rank one-third / two-domain / two-benchmark rule: REVISE (require pre-declared domain count + power note)
The rule "robust reversal in >=1/3 of eligible substantive domains, min 2 domains spanning both
benchmarks, bootstrap P(distinct winners)>=0.95, worst-criterion improvement > r0 with paired CI
excluding 0" is sound in structure. RISK: with only ~4 substantive domains (BT-COHORT[weak],
BT->AEGIS, AG-RESPONSE, AG->BT), "1/3" is 2 domains — the same as the floor, so the fraction adds
nothing. RULING: (a) pre-declare the EXACT list and count of substantive domains in the lock
(controls excluded; BT-COHORT flagged weak); (b) restate H-Rank as ">=2 substantive domains
spanning both benchmarks AND both a within-benchmark and a cross-benchmark shift type," dropping
the uninformative 1/3; (c) add an outcome-blind note that with 3 guards the reversal test is
low-powered and a null H-Rank must be reported as inconclusive-for-rank, not as evidence of no
reversal.

## Part C — Consolidated verdict
GAP_SURVIVES ACCEPTED. Study is REVISE -> LOCKABLE once ALL of these pre-lock, metadata-only gates
pass and are frozen:
1. AEGIS provenance audit (Items 2,3) PASSES with >=2 criterion benchmarks and >=4 common criteria
   at >=100/100 support; else pause for a replacement benchmark.
2. Full native->common taxonomy map, two-reviewer sign-off, SHA (Item 4).
3. Exact guard repo/revision/verbalizer registry (Item 5) + source-only comparability diagnostic
   incl. ShieldGemma (Item 6).
4. BT-COHORT relabelled as within-collection annotation-round shift (Item 1); substantive-domain
   list pre-declared (Item 8).
5. §11 verdict ladder, §10 multiplicity, tau/epsilon/c0/r0 (Item 7) unchanged.

Codex is authorized to perform ONLY the metadata/provenance/taxonomy/verbalizer freezing work
above and to fill the required registries. It is NOT authorized to score any guard, compute any
ECE/drift/ranking, or mark the prereg LOCKED until PaperGuru signs off on the completed registries.
When the registries are complete, PaperGuru will flip the prereg header to LOCKED in a signed
commit. Until then the header stays DRAFT.

## Part D — stop conditions (reaffirmed)
Terminate the pivot if: AEGIS provenance fails and no replacement human criterion benchmark exists;
<3 guards expose defensible probabilities; <4 common criteria clear support; any guard's source
probability is degenerate and no comparable replacement exists; or the design would require post-hoc
taxonomy grouping / domain selection. A null external result is publishable and does not reopen any
frozen PCCD gate.
