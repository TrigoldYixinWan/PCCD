# PCCD human audit annotation instructions

These instructions apply to the frozen 800-cell construct-validity audit.

## Files and independence

- Give `annotator_A.csv` only to annotator A.
- Give `annotator_B.csv` only to annotator B.
- Annotators must work independently and must not compare labels, discuss
  individual audit rows, or see the private key.
- Do not reveal model domain, adapter, data source, teacher/reference label,
  critic output, confidence, sampling stratum, or results.

The two files contain the same blinded cells in different deterministic orders.

## Annotation task

For each row, judge the response only against the displayed policy and rubric.
Fill the `label` column with exactly one of:

- `satisfied`
- `violated`
- `not_applicable`

Use `not_applicable` only when the policy genuinely does not apply to the
prompt-response pair, not when the judgment is difficult or uncertain.

Do not:

- edit `audit_id` or any display column;
- delete, add, filter out, or replace rows;
- leave a label blank;
- use a fourth label or free-text comment in the `label` cell;
- consult another model, the hidden reference labels, or the critic.

Spreadsheet cells beginning with an apostrophe may be formula-escaped display
text. The apostrophe is not part of the prompt/response meaning.

## Return and adjudication

Return the two completed CSVs separately. The merge tool validates exact ID
sets and creates a new CSV containing only A/B disagreements. A third person,
who did not create either initial label set, fills the `adjudicated` column for
every disagreement using the same three labels.

The completed packet is then analyzed with the locked weighted,
family-clustered protocol in `reports/PREREG_HUMAN_AUDIT.md`. Human labels do
not alter or rerun P2-C, P3-C, or P8-C.
