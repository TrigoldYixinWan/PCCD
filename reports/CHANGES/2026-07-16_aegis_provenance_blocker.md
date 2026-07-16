# AEGIS 2.0 pre-lock provenance blocker

**Date:** 2026-07-16

**Authority:** PaperGuru Item 2/3 blocking metadata gate

**Classification:** blocking evidence report; no scientific gate or threshold
was changed

## Trigger

The external-guard preregistration required proof that AEGIS
`violated_categories` and human prompt/response labels belong to the same
annotation unit, plus at least 100 human positive and 100 human negative
examples for every primary common criterion.

## Finding

The official card and paper document a human full-dialogue label with
categories assigned during that dialogue-level pass. Unsafe response labels
are produced by an LLM jury. A human-source response label is present only when
a human-safe dialogue label is inherited by both turns.

The exact revision audit found 28,216 original units:

- 5,236 `response_label_source == human` rows are safe;
- 0 `response_label_source == human` rows are unsafe; and
- 0 human-source response rows have non-empty `violated_categories`.

Thus response-positive support is zero under every native category and every
possible common-category aggregation. AEGIS cannot meet the 100/100 support
gate.

## Decision and scope control

- Mark AEGIS `NOT_LOCKABLE` as a primary criterion benchmark.
- Keep `reports/PREREG_EXTERNAL_GUARD.md` in `DRAFT`.
- Stop before the native-to-common taxonomy freeze, guard registry,
  substantive-domain freeze, source-only diagnostic, or target scoring.
- Do not inspect any guard output.
- Await PaperGuru's choice between recovering WildGuardTest native categories
  or adding a third human-labelled benchmark.

The complete evidence and file hashes are in
`reports/aegis_provenance_audit.md`; the reproducible read-only audit is
`scripts/day11/audit_aegis_provenance.py`.
