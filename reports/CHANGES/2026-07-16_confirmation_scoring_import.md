# Confirmation scoring import fix

Date: 2026-07-16  
Level: Green implementation correction  
Outcome access at fix time: none

## Failure

The first confirmation scoring launch terminated before its first inference
batch. `src/eval_critic.py::load_scoring_jsonl` referenced the existing
`LABEL_TO_ID` constant while validating complete reference-label rows, but the
constant was not imported from `src.critic_model`. Both ranks raised
`NameError`; no logits file or scientific aggregate was written.

## Correction

- Import `LABEL_TO_ID` from `src.critic_model`.
- Add a regression test covering the complete-label branch. The existing test
  covered only malformed rows, whose short-circuit path did not evaluate the
  missing name.

No model, checkpoint, data, split, label, scoring equation, metric, threshold,
or random seed changed. Because the failed process produced no logits and
exposed no outcome, scoring may restart from the beginning after the fix. The
failed launch log remains preserved as an audit artifact.
