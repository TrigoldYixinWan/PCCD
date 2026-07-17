# AEGIS 2.0 provenance and support audit

**Status:** `BLOCKING FAIL — AEGIS NOT_LOCKABLE`

**Audit date:** 2026-07-16

**Scope:** metadata, dataset card, and paper only. No guard inference, ECE,
ranking, or external outcome inspection was performed.

**Preregistration status:** `DRAFT` remains unchanged.

## 1. Decision

AEGIS 2.0 cannot serve as a primary human-labelled criterion benchmark under
the current external-guard preregistration.

The official documentation establishes a human annotation unit at the **full
dialogue level**, not separate human prompt-criterion and
response-criterion units. More decisively, among the 28,216 original
annotation units, all 5,236 rows whose `response_label_source` is `human` are
labelled `safe`; there are **zero** human-labelled unsafe responses. Therefore
response-positive support is zero for every native category and for every
possible native-to-common aggregation. The required response support of at
least 100 positive and 100 negative examples per primary criterion cannot be
met.

Because this is a blocking pre-lock metadata gate, the taxonomy map, guard
registry, substantive-domain freeze, source-only comparability diagnostic, and
all target scoring are stopped pending PaperGuru's benchmark-replacement
decision.

## 2. Frozen source identity

The audit used the official
[`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
repository at revision
[`d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/tree/d86bb8bedff51d25ac834ab7838f1cc61acb7a2c).

| Raw file | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `train.json` | 25,007 | 22,705,956 | `154fba82c71d9fa73abd2ca5588a198e693ddc816c83444df180a22f613e02f6` |
| `validation.json` | 1,245 | 1,137,821 | `a97200e226ad4f6ba6a639982f817f675909bc74513859df3e8fc9a92951dfcd` |
| `test.json` | 1,964 | 1,731,373 | `b0a6d602260524866053cb34105194f074f2c2906e3691b68d43b9e6e9318f35` |
| `refusals_train.json` | 5,000 | 2,758,729 | `ff948d3696c9da94cf2523f5d4ae7f16cad7b3c0fd3cb46a331dad0ed717fbe2` |
| `refusals_validation.json` | 200 | 109,404 | `aba81546da0108bc931ae7fb7662b687b42b0bb3e734cde34fbddfbe33cadfcf` |

There are 33,416 logical rows but 28,216 unique IDs. The 5,200 refusal rows
reuse original IDs and are response augmentations, so they are not counted as
independent human annotation units. The three original splits contain 28,216
unique IDs, with zero cross-split ID overlap.

The audit can be reproduced with:

```bash
python scripts/day11/audit_aegis_provenance.py \
  --data-dir /path/to/aegis_raw_files \
  --out /tmp/aegis_provenance_audit.json
```

The script refuses to run if any raw-file hash differs from the table above.

## 3. What the official sources establish

The
[dataset card at the audited revision](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/blob/d86bb8bedff51d25ac834ab7838f1cc61acb7a2c/README.md)
describes hybrid annotation: a human overall safety label at the full-dialogue
level, followed where needed by a multi-LLM jury for response safety. It states
that `prompt_label_source` is human, while `response_label_source` can be
`human`, `llm_jury`, or `refusal_data_augmentation`.

The
[NAACL 2025 AEGIS 2.0 paper](https://aclanthology.org/2025.naacl-long.306/)
states that annotators first give each dialogue an overall dialogue-level
label and, for unsafe dialogues, identify risk categories associated with the
response. It then explains the turn-level derivation:

- the human majority conversation label is used for prompt classification;
- an unsafe conversation receives a separate LLM-jury response label; and
- when the human conversation vote is safe, the dataset assumes that vote
  applies to both prompt and response.

Consequently, `violated_categories` can be linked to the same **dialogue-level
human annotation pass**, but the sources do not establish it as a separate
human prompt-category label or human response-category label. In particular,
`response_label_source == human` does not mean that an annotator independently
judged the response turn; it is the inherited safe-conversation assumption.

## 4. Empirical provenance audit

Counts below use only the 28,216 original rows. Refusal augmentations are
excluded.

| Split | Original units | Human prompt safe | Human prompt unsafe | Human response safe | Human response unsafe | Jury response safe | Jury response unsafe | Response missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 25,007 | 12,296 | 12,711 | 4,682 | **0** | 2,011 | 3,541 | 14,773 |
| validation | 1,245 | 572 | 673 | 196 | **0** | 81 | 245 | 723 |
| test | 1,964 | 905 | 1,059 | 358 | **0** | 100 | 394 | 1,112 |
| **total** | **28,216** | **13,773** | **14,443** | **5,236** | **0** | **2,192** | **4,180** | **16,608** |

Additional checks:

- every original prompt has `prompt_label_source == human`;
- all 5,236 human-source response rows are safe;
- zero human-source response rows have non-empty `violated_categories`;
- 2,850 human-safe prompts have non-empty categories, including 449 with a
  category other than `Needs Caution`; and
- no human-unsafe prompt has an empty category field.

These observations are consistent with the documented dialogue-level
annotation and derived turn labels. They do not support interpreting the
category field as independently human-labelled prompt and response outcomes.

## 5. Support under a favorable diagnostic aggregation

To ensure that the blocker is not caused by a narrow taxonomy choice, the
audit used a broad, **non-frozen diagnostic** aggregation:

- violence/physical: `Violence`, `Threat`;
- hate/identity: `Hate/Identity Hate`;
- sexual: `Sexual`, `Sexual (minor)`;
- self-harm: `Suicide and Self Harm`;
- illegal/criminal: criminal planning, weapons, controlled substances,
  illegal activity, malware, and fraud; and
- privacy/PII: `PII/Privacy`.

Conservative negatives are human-safe prompts or responses. Positive support
requires a human-source unsafe label and a mapped native category.

| Common criterion | Train prompt +/− | Validation prompt +/− | Test prompt +/− | Train response +/− | Validation response +/− | Test response +/− |
|---|---:|---:|---:|---:|---:|---:|
| violence/physical | 2,137 / 12,296 | 113 / 572 | 189 / 905 | **0 / 4,682** | **0 / 196** | **0 / 358** |
| hate/identity | 1,943 / 12,296 | 113 / 572 | 176 / 905 | **0 / 4,682** | **0 / 196** | **0 / 358** |
| sexual | 1,372 / 12,296 | 58 / 572 | 111 / 905 | **0 / 4,682** | **0 / 196** | **0 / 358** |
| self-harm | 749 / 12,296 | 39 / 572 | 52 / 905 | **0 / 4,682** | **0 / 196** | **0 / 358** |
| illegal/criminal | 6,829 / 12,296 | 388 / 572 | 576 / 905 | **0 / 4,682** | **0 / 196** | **0 / 358** |
| privacy/PII | 1,019 / 12,296 | 43 / 572 | 99 / 905 | **0 / 4,682** | **0 / 196** | **0 / 358** |

Even before resolving the prompt-category attribution problem, the test prompt
side reaches 100 positives for only four of six diagnostic common criteria.
The response side reaches 100 positives for zero criteria in every split. No
alternative native-to-common map can turn zero human unsafe responses into the
required positive support.

## 6. Blocking-gate verdict

| Requirement | Result |
|---|---|
| `violated_categories` belongs to a human annotation unit | **Partial only:** same full-dialogue human pass is documented |
| Separate human prompt criterion labels established | **No** |
| Separate human response criterion labels established | **No** |
| Human response positive support ≥100 per common criterion | **No: 0 for every criterion** |
| Human response negative support ≥100 per common criterion | Yes |
| AEGIS eligible as primary criterion benchmark | **No — NOT_LOCKABLE** |
| Overall requirement: ≥2 benchmarks × ≥4 common criteria × ≥100/100 | **FAIL**; only BeaverTails remains provisionally available |

This failure occurs before any guard output is observed and is independent of
model performance. Per the adjudicated protocol, work stops here. PaperGuru
must decide whether to recover native WildGuardTest categories or introduce a
third human-labelled benchmark before the external-guard preregistration can be
locked.
