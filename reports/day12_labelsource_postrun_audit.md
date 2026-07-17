# Day-12 label-source confirmatory post-run audit

**Audit date:** 2026-07-17

**Locked verdict:** `LABELSOURCE_ONLY`

**Scope:** integrity and preregistration-conformance audit only; no model was rerun, no threshold or category grouping was changed, and no frozen outcome was reclassified.

## Run identity and completeness

- The single authorized run was consumed before inference and completed at `2026-07-17T22:31:36.677619+00:00`.
- The run used 10,000 paired QA-pair bootstrap replicates with seed `20260901` and the single locked Holm family.
- All 11,088 independent BeaverTails QA pairs were joined.
- Qwen-32B strict-JSON failures were retained: 65/11,088 (`0.5862%`). Qwen is an LLM proxy label source, not human ground truth and not an evaluated guard.
- All 84 expected label-source x guard x category cells are present in the report. All six simultaneous-uncertainty groups completed 10,000 replicates.
- Mapped guard-score failures, excluding preregistered no-defensible-mapping cells, were zero for all three guards. ShieldGemma no-map categories remain explicit `N/E` cells.

## Locked hypothesis audit

### H1 — aggregate hides worst: not supported

On human labels, every guard's mean-category ECE was already far above the locked `eps=0.015`; therefore none looked acceptably calibrated in the aggregate, even though all had large worst-category ECE.

| Guard | Mean category ECE M (95% CI) | Worst category ECE W (95% CI) |
|---|---:|---:|
| Llama-Guard-3-8B | 0.2121 [0.2053, 0.2198] | 0.4562 [0.4342, 0.4784] |
| ShieldGemma-2B | 0.2255 [0.2212, 0.2305] | 0.4791 [0.4741, 0.4836] |
| ShieldGemma-9B | 0.2359 [0.2315, 0.2404] | 0.4821 [0.4779, 0.4858] |

This is a failure of the preregistered “acceptable mean hides a bad category” conjunction, not evidence that the category-level calibration errors are small.

### H2 — criterion cancellation: not supported

For all three guards, all eligible human-label signed deviations had the same negative sign, so the locked cancellation statistic was exactly `C=0`. The observed failure mode is systematic underprediction of unsafe probability under the balanced estimand, not positive/negative criterion errors cancelling in an aggregate.

### H3 — guard-selection reversal: not established

Llama-Guard-3-8B minimized both mean-category and worst-category ECE on the seven-category common evaluable intersection. The bootstrap probability of distinct mean/worst winners was `0.0`, and the worst-ECE improvement was `0.0` with CI `[0.0, 0.0]`. Per the locked low-power clause, this is `INCONCLUSIVE-FOR-RANK`, not affirmative evidence that reversals cannot occur more broadly.

### H4 — label-source sensitivity: supported

The preregistered human-vs-Qwen-proxy label-source effect survived the single Holm family for both ShieldGemma sizes:

| Guard | Subjective minus objective mean absolute ECE difference (95% CI) | Cell Holm-adjusted p |
|---|---:|---:|
| ShieldGemma-2B | 0.01087 [0.00337, 0.01889] | 0.02990 |
| ShieldGemma-9B | 0.01547 [0.00807, 0.02308] | 0.01120 |

The H4 top-level union test also survived Holm correction (`p_adj=0.02990`). Llama-Guard-3-8B showed the opposite preregistered axis direction and did not support H4; it is retained in the complete report. The supported claim is narrow: a guard's per-category calibration audit can depend systematically on whether the reference labels are human BeaverTails annotations or blind Qwen-32B proxy annotations. It does not establish that either source is universally correct, that Qwen is human-equivalent, or that all guard families share the effect.

## Verification performed

- Recomputed the H4 objective/subjective axis directly from the stored per-category ECE deltas; both values exactly match the analysis JSON.
- Checked the 14 raw and Holm-adjusted tests. The only adjusted tests below 0.05 are `H4:shieldgemma_9b`, `H4:shieldgemma_2b`, and `H4:top`.
- Confirmed the committed report and analysis hashes match the remote completion marker.
- Re-ran the six label-source pipeline regression tests (`6 passed`), Python byte-compilation, and shell syntax validation after the post-write control-flow fix.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| Qwen proxy labels | `974b5e669e179613b7632024022574a431ec995fd118be99de5a6615ecb6cee6` |
| Llama Guard scores | `4df952dd12981687578563310f1632c626c7bc2959f92c5c956a7eefa1018ef7` |
| ShieldGemma-2B scores | `1d870436346885feb09b50a60a94988c3536ad2fcaf0aead14c1d331b8fe6e91` |
| ShieldGemma-9B scores | `b1c3701818a1e80584cd30f606bb5a6375ba2a641277865b6dbdc59a766df9d9` |
| Strict joined evaluation rows | `f6785917160950f2bcf40bd78bef915592db85f3194a270ef7a007a313e40050` |
| Analysis JSON | `9eb62e14472cd31ef6b95d2e91d83be87a22d5a57c3968a53e24bdfaa1d930ee` |
| Generated confirmatory report | `1d691b305249d210f34e633ed84f2fefea404c28c0b2334be7b472d1ac05a076` |

The raw model outputs remain under the frozen AutoDL output directory. Exact analysis JSON and the completion manifest are mirrored in `reports/artifacts/` for review.
