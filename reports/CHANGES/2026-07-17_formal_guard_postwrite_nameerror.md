# Formal guard post-write NameError

**Date:** 2026-07-17

**Classification:** Green implementation fix; same formal run, no estimand or outcome change

## Failure

The Llama Guard formal scorer wrote all 11,088 registered rows and its
`FORMAL_COMPLETE` metadata, including output SHA-256
`4df952dd12981687578563310f1632c626c7bc2959f92c5c956a7eefa1018ef7`.
Immediately afterward it raised:

```text
NameError: name 'probabilities' is not defined
```

The driver therefore did not start ShieldGemma. Qwen-32B continued on the
other GPU. No guard probabilities, Qwen labels, ECE, F1, ranking, or hypothesis
direction was inspected; only process state, row counts, the completion
manifest, and traceback were read.

## Cause and fix

While adding the formal subcommand, the unchanged tail of the pre-lock
`sanity_verdict` function was accidentally left after `run_formal`. Python
therefore treated it as post-write formal code. The fix moves that exact block
back into `sanity_verdict` and removes it from `run_formal`; it changes no formal
score computation, model call, prompt, probability rule, row, hash, threshold,
or analysis logic.

## Recovery boundary

The completed, hash-frozen Llama output is retained and must not be regenerated.
Recovery starts with the missing ShieldGemma outputs under the existing
`RUN_STARTED.json` marker (`formal_runs_consumed=1`). This is interruption
recovery for the same authorized run, not a second run.
