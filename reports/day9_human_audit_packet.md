# Day 9 blinded human construct-validity audit packet

Date: 2026-07-16

Protocol: `reports/PREREG_CONFIRMATION.md`, decision `DL-006`  
Construction code commit: `e3a050b`

Verdict: **PACKET FROZEN — the preregistered 800-cell blinded audit package
was constructed before any confirmation aggregate was opened. Human annotation
is still pending.**

## Sampling

- Domains: D0 and the primary new-seed D5.
- Policies: all ten H1–H5/S1–S3/T1–T2 criteria.
- Exact allocation: 40 cells for every domain × policy combination,
  20 combinations and 800 cells total.
- Every `audit_id` is unique. The private key and blind worksheet contain the
  same 800 IDs in independently ordered rows.
- The private key preserves sampling stratum, inclusion probability, and
  inverse-probability weight for population reconstruction.

## Blinding

The annotator worksheet contains only:

- `audit_id`;
- prompt and response;
- canonical policy ID, policy text, and three-way rubric;
- empty fields for two independent annotators and adjudication.

It contains no domain, adapter, source, item/family ID, sampling stratum,
reference state, critic prediction, confidence, or logits. Domain and outcome
strata exist only in the private key, which must not be shown to annotators.

## Frozen hashes

- Private key:
  `8f6b8e10a9153cfeea436aea9938a38dcf5f9ea9243ec53b1d18ce9e2b65903b`.
- Blind worksheet:
  `552df011ae43cd3587cf52ad2ec6e9e0c02fabfaf905068960873660d2f49a0d`.
- Audit manifest:
  `bca7483e244bc055122a43159c3ddb03e894d7ff7d68faeec8a706c8ccec9bbe`.

The files are stored under `$PCCD_OUT/confirmation`; the private key remains
separate from the blind worksheet. The aggregate-open flag at construction was
false.

## Human work still required

Two independent human annotators must label every cell without access to the
private key, followed by third-person adjudication of disagreements. Completion
is not a prerequisite for mechanically running the preregistered confirmation,
but paper claims about reference construct validity must remain qualified until
the weighted human-agreement analysis is complete.
