# BeaverTails-to-guard taxonomy map

**Status:** `ADJUDICATED_BEFORE_OUTCOMES_PENDING_HUMAN_LOCK`

This is preregistration §11.3 metadata work. No BeaverTails guard score,
reliability statistic, Qwen proxy label, ECE, F1, AUROC, or guard ranking was
available to either reviewer or used in adjudication.

## Blind-review procedure

The definition packet
`configs/labelsource_taxonomy_definition_packet.json` contains only the official
BeaverTails-14, Llama Guard 3, and ShieldGemma policy definitions. Its SHA-256 is
`71ea53cb5e7a7f64b7b8e4fa5c5c1711b4b3db06bfae890d075bb61d07fe860d`.

Two isolated Codex reviewer instances independently mapped that packet without
access to category support, guard sanity outputs, hypotheses, or downstream
outcomes. Their signed artifacts are:

- reviewer A: `reports/taxonomy_map_reviewer_a.json`, SHA-256
  `e918c95f45b424521ff21e1bd86c2c32e7fabc2287c86ff1d6a6054c3c9c3593`;
- reviewer B: `reports/taxonomy_map_reviewer_b.json`, SHA-256
  `c6ed25530e0197f7ac348b9ffe3ebedc86b93c7cdc5eadc70e94e29441fb7e51`.

The reviewers independently agreed on every ShieldGemma target set and on 13
of 14 Llama Guard target sets. They differed on one substantive target-set
choice and on four relation labels despite agreeing on the target set. The
deterministic adjudication is recorded in the machine-readable map and remains
subject to project-owner or PaperGuru acceptance in the final lock commit.

## Adjudicated map

`NONE` means that no native policy mapping was defensible from the frozen
definitions. Relation labels explicitly preserve partial and asymmetric
coverage instead of pretending that native taxonomies are equivalent.

| BeaverTails category | Llama Guard 3 native policy | Relation | ShieldGemma native policy | Relation |
|---|---|---|---|---|
| animal_abuse | S1 | partial overlap | NONE | no defensible mapping |
| child_abuse | S1, S4 | partial overlap | dangerous_content, harassment, sexually_explicit_information | partial overlap |
| controversial_topics,politics | S13 | source broader | NONE | no defensible mapping |
| discrimination,stereotype,injustice | S10 | source broader | hate_speech | source broader |
| drug_abuse,weapons,banned_substance | S2 | source narrower | dangerous_content | partial overlap |
| financial_crime,property_crime,theft | S2 | source narrower | NONE | no defensible mapping |
| hate_speech,offensive_language | S10 | partial overlap | hate_speech, harassment | partial overlap |
| misinformation_regarding_ethics,laws_and_safety | S6, S13 | partial overlap | dangerous_content | partial overlap |
| non_violent_unethical_behavior | S2, S5 | partial overlap | harassment | partial overlap |
| privacy_violation | S2, S7 | partial overlap | harassment | partial overlap |
| self_harm | S11 | partial overlap | dangerous_content | partial overlap |
| sexually_explicit,adult_content | S12 | source broader | sexually_explicit_information | direct |
| terrorism,organized_crime | S1, S2 | partial overlap | dangerous_content | partial overlap |
| violence,aiding_and_abetting,incitement | S1, S2 | partial overlap | dangerous_content, harassment, hate_speech | partial overlap |

## Disagreements and conservative resolution

The only target-set disagreement concerned
`non_violent_unethical_behavior`: reviewer A selected Llama Guard S2 and S5;
reviewer B additionally selected S8 (intellectual property). The adjudicated
map excludes S8 because the BeaverTails definition does not name intellectual
property, and an open-ended “other unethical conduct” clause cannot justify
mapping every narrower policy. This is the conservative, outcome-blind choice.

For `animal_abuse`, `child_abuse`, `hate_speech,offensive_language`, and
`self_harm`, target sets agreed and only the direction/overlap relation differed.
The final relations are `partial_overlap`; the per-reviewer reasoning and exact
adjudication are retained in `reports/taxonomy_map.json`.

## Freeze artifact

The adjudicated machine-readable artifact is `reports/taxonomy_map.json`,
SHA-256
`443f837bcc265ddafc35dff60edc3a49eadeae8e11e91e2422b7f484576bb8ed`.
It must not change after formal outcome access. Reviewer identity is stated
accurately: these were independent AI definition reviews, not human domain
experts. The final human lock signature is still required.
