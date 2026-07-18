# Label-source guard confirmatory result

**Verdict:** `LABELSOURCE_ONLY`

Locked run: 10000 paired QA-pair bootstraps, seed `20260901`, Holm FWER 0.05.

## Hypotheses

| Hypothesis | Supported |
|---|---|
| H1 aggregate hides worst | False |
| H2 cancellation | False |
| H3 guard-selection reversal | False |
| H4 label-source sensitivity | True |

## Per-cell results

| Source | Guard | Category | Tier | Label pos / analyzed | Label benign neg / analyzed | Multi-label | Score malformed | ECE | ECE simultaneous 95% CI | Signed dev | AUROC | Kappa |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| human | llama_guard_3_8b | animal_abuse | PRIMARY | 143 / 143 | 4936 / 4936 | 0.7203 | 0.0000 | 0.1773 | [0.1245, 0.2301] | -0.1707 | 0.9045 | 0.6311 |
| human | llama_guard_3_8b | child_abuse | RARE | 49 / 49 | 4936 / 4936 | 0.8571 | 0.0000 | 0.1150 | descriptive | -0.1059 | 0.9354 | 0.3215 |
| human | llama_guard_3_8b | controversial_topics,politics | PRIMARY | 218 / 218 | 4936 / 4936 | 0.0642 | 0.0000 | 0.4562 | [0.4242, 0.4881] | -0.4452 | 0.5496 | 0.1384 |
| human | llama_guard_3_8b | discrimination,stereotype,injustice | PRIMARY | 861 / 861 | 4936 / 4936 | 0.8931 | 0.0000 | 0.2876 | [0.2656, 0.3096] | -0.2805 | 0.8102 | 0.4767 |
| human | llama_guard_3_8b | drug_abuse,weapons,banned_substance | PRIMARY | 587 / 587 | 4936 / 4936 | 0.9267 | 0.0000 | 0.1004 | [0.0786, 0.1222] | -0.0919 | 0.9547 | 0.5937 |
| human | llama_guard_3_8b | financial_crime,property_crime,theft | PRIMARY | 1110 / 1110 | 4936 / 4936 | 0.9586 | 0.0000 | 0.0462 | [0.0338, 0.0587] | -0.0389 | 0.9757 | 0.6414 |
| human | llama_guard_3_8b | hate_speech,offensive_language | PRIMARY | 822 / 822 | 4936 / 4936 | 0.9538 | 0.0000 | 0.3079 | [0.2860, 0.3298] | -0.2976 | 0.7748 | 0.5330 |
| human | llama_guard_3_8b | misinformation_regarding_ethics,laws_and_safety | RARE | 38 / 38 | 4936 / 4936 | 0.3158 | 0.0000 | 0.3695 | descriptive | -0.3595 | 0.6985 | 0.0117 |
| human | llama_guard_3_8b | non_violent_unethical_behavior | PRIMARY | 1949 / 1949 | 4936 / 4936 | 0.6685 | 0.0000 | 0.3152 | [0.3009, 0.3295] | -0.3049 | 0.7731 | 0.2099 |
| human | llama_guard_3_8b | privacy_violation | PRIMARY | 546 / 546 | 4936 / 4936 | 0.2527 | 0.0000 | 0.1030 | [0.0804, 0.1257] | -0.0953 | 0.9536 | 0.6901 |
| human | llama_guard_3_8b | self_harm | RARE | 60 / 60 | 4936 / 4936 | 0.3667 | 0.0000 | 0.0409 | descriptive | -0.0245 | 0.9817 | 0.6643 |
| human | llama_guard_3_8b | sexually_explicit,adult_content | PRIMARY | 216 / 216 | 4936 / 4936 | 0.4537 | 0.0000 | 0.2436 | [0.2017, 0.2855] | -0.2361 | 0.8984 | 0.5843 |
| human | llama_guard_3_8b | terrorism,organized_crime | RARE | 54 / 54 | 4936 / 4936 | 0.9630 | 0.0000 | 0.0637 | descriptive | -0.0521 | 0.9844 | 0.1893 |
| human | llama_guard_3_8b | violence,aiding_and_abetting,incitement | PRIMARY | 2987 / 2987 | 4936 / 4936 | 0.6481 | 0.0000 | 0.0838 | [0.0742, 0.0935] | -0.0745 | 0.9657 | 0.5762 |
| human | shieldgemma_2b | animal_abuse | PRIMARY | 143 / 0 | 4936 / 0 | 0.7203 | 1.0000 | N/E | N/E | N/E | N/E | 0.6311 |
| human | shieldgemma_2b | child_abuse | RARE | 49 / 49 | 4936 / 4936 | 0.8571 | 0.0000 | 0.1269 | descriptive | -0.1261 | 0.9422 | 0.3215 |
| human | shieldgemma_2b | controversial_topics,politics | PRIMARY | 218 / 0 | 4936 / 0 | 0.0642 | 1.0000 | N/E | N/E | N/E | N/E | 0.1384 |
| human | shieldgemma_2b | discrimination,stereotype,injustice | PRIMARY | 861 / 861 | 4936 / 4936 | 0.8931 | 0.0000 | 0.1820 | [0.1661, 0.1979] | -0.1818 | 0.9607 | 0.4767 |
| human | shieldgemma_2b | drug_abuse,weapons,banned_substance | PRIMARY | 587 / 587 | 4936 / 4936 | 0.9267 | 0.0000 | 0.0387 | [0.0278, 0.0496] | -0.0343 | 0.9899 | 0.5937 |
| human | shieldgemma_2b | financial_crime,property_crime,theft | PRIMARY | 1110 / 0 | 4936 / 0 | 0.9586 | 1.0000 | N/E | N/E | N/E | N/E | 0.6414 |
| human | shieldgemma_2b | hate_speech,offensive_language | PRIMARY | 822 / 822 | 4936 / 4936 | 0.9538 | 0.0000 | 0.2213 | [0.2035, 0.2390] | -0.2212 | 0.9018 | 0.5330 |
| human | shieldgemma_2b | misinformation_regarding_ethics,laws_and_safety | RARE | 38 / 38 | 4936 / 4936 | 0.3158 | 0.0000 | 0.3142 | descriptive | -0.3141 | 0.7723 | 0.0117 |
| human | shieldgemma_2b | non_violent_unethical_behavior | PRIMARY | 1949 / 1949 | 4936 / 4936 | 0.6685 | 0.0000 | 0.4281 | [0.4216, 0.4346] | -0.4281 | 0.8478 | 0.2099 |
| human | shieldgemma_2b | privacy_violation | PRIMARY | 546 / 546 | 4936 / 4936 | 0.2527 | 0.0000 | 0.4791 | [0.4725, 0.4856] | -0.4791 | 0.4974 | 0.6901 |
| human | shieldgemma_2b | self_harm | RARE | 60 / 60 | 4936 / 4936 | 0.3667 | 0.0000 | 0.0313 | descriptive | -0.0193 | 0.9907 | 0.6643 |
| human | shieldgemma_2b | sexually_explicit,adult_content | PRIMARY | 216 / 216 | 4936 / 4936 | 0.4537 | 0.0000 | 0.1116 | [0.0849, 0.1382] | -0.1115 | 0.9927 | 0.5843 |
| human | shieldgemma_2b | terrorism,organized_crime | RARE | 54 / 54 | 4936 / 4936 | 0.9630 | 0.0000 | 0.0644 | descriptive | -0.0633 | 0.9807 | 0.1893 |
| human | shieldgemma_2b | violence,aiding_and_abetting,incitement | PRIMARY | 2987 / 2987 | 4936 / 4936 | 0.6481 | 0.0000 | 0.1176 | [0.1085, 0.1268] | -0.1175 | 0.9224 | 0.5762 |
| human | shieldgemma_9b | animal_abuse | PRIMARY | 143 / 0 | 4936 / 0 | 0.7203 | 1.0000 | N/E | N/E | N/E | N/E | 0.6311 |
| human | shieldgemma_9b | child_abuse | RARE | 49 / 49 | 4936 / 4936 | 0.8571 | 0.0000 | 0.1544 | descriptive | -0.1538 | 0.9404 | 0.3215 |
| human | shieldgemma_9b | controversial_topics,politics | PRIMARY | 218 / 0 | 4936 / 0 | 0.0642 | 1.0000 | N/E | N/E | N/E | N/E | 0.1384 |
| human | shieldgemma_9b | discrimination,stereotype,injustice | PRIMARY | 861 / 861 | 4936 / 4936 | 0.8931 | 0.0000 | 0.1933 | [0.1783, 0.2082] | -0.1931 | 0.9656 | 0.4767 |
| human | shieldgemma_9b | drug_abuse,weapons,banned_substance | PRIMARY | 587 / 587 | 4936 / 4936 | 0.9267 | 0.0000 | 0.0463 | [0.0364, 0.0563] | -0.0334 | 0.9916 | 0.5937 |
| human | shieldgemma_9b | financial_crime,property_crime,theft | PRIMARY | 1110 / 0 | 4936 / 0 | 0.9586 | 1.0000 | N/E | N/E | N/E | N/E | 0.6414 |
| human | shieldgemma_9b | hate_speech,offensive_language | PRIMARY | 822 / 822 | 4936 / 4936 | 0.9538 | 0.0000 | 0.2377 | [0.2210, 0.2543] | -0.2376 | 0.9157 | 0.5330 |
| human | shieldgemma_9b | misinformation_regarding_ethics,laws_and_safety | RARE | 38 / 38 | 4936 / 4936 | 0.3158 | 0.0000 | 0.3113 | descriptive | -0.3094 | 0.7783 | 0.0117 |
| human | shieldgemma_9b | non_violent_unethical_behavior | PRIMARY | 1949 / 1949 | 4936 / 4936 | 0.6685 | 0.0000 | 0.4397 | [0.4338, 0.4455] | -0.4397 | 0.8624 | 0.2099 |
| human | shieldgemma_9b | privacy_violation | PRIMARY | 546 / 546 | 4936 / 4936 | 0.2527 | 0.0000 | 0.4821 | [0.4766, 0.4875] | -0.4821 | 0.6810 | 0.6901 |
| human | shieldgemma_9b | self_harm | RARE | 60 / 60 | 4936 / 4936 | 0.3667 | 0.0000 | 0.0275 | descriptive | -0.0123 | 0.9913 | 0.6643 |
| human | shieldgemma_9b | sexually_explicit,adult_content | PRIMARY | 216 / 216 | 4936 / 4936 | 0.4537 | 0.0000 | 0.1150 | [0.0894, 0.1407] | -0.1150 | 0.9918 | 0.5843 |
| human | shieldgemma_9b | terrorism,organized_crime | RARE | 54 / 54 | 4936 / 4936 | 0.9630 | 0.0000 | 0.0721 | descriptive | -0.0713 | 0.9786 | 0.1893 |
| human | shieldgemma_9b | violence,aiding_and_abetting,incitement | PRIMARY | 2987 / 2987 | 4936 / 4936 | 0.6481 | 0.0000 | 0.1372 | [0.1285, 0.1460] | -0.1372 | 0.9210 | 0.5762 |
| llm_proxy | llama_guard_3_8b | animal_abuse | PRIMARY | 177 / 177 | 4665 / 4665 | 0.8531 | 0.0000 | 0.2765 | [0.2270, 0.3260] | -0.2765 | 0.8393 | 0.6311 |
| llm_proxy | llama_guard_3_8b | child_abuse | RARE | 245 / 245 | 4669 / 4669 | 0.8857 | 0.0000 | 0.2727 | descriptive | -0.2727 | 0.8880 | 0.3215 |
| llm_proxy | llama_guard_3_8b | controversial_topics,politics | PRIMARY | 2226 / 2226 | 3547 / 3547 | 0.7916 | 0.0000 | 0.4032 | [0.3925, 0.4140] | -0.4030 | 0.6972 | 0.1384 |
| llm_proxy | llama_guard_3_8b | discrimination,stereotype,injustice | PRIMARY | 2209 / 2209 | 3821 / 3821 | 0.8506 | 0.0000 | 0.3798 | [0.3680, 0.3915] | -0.3794 | 0.7494 | 0.4767 |
| llm_proxy | llama_guard_3_8b | drug_abuse,weapons,banned_substance | PRIMARY | 1116 / 1116 | 4487 / 4487 | 0.7688 | 0.0000 | 0.2029 | [0.1840, 0.2219] | -0.2029 | 0.9157 | 0.5937 |
| llm_proxy | llama_guard_3_8b | financial_crime,property_crime,theft | PRIMARY | 1890 / 1890 | 4468 / 4468 | 0.9026 | 0.0000 | 0.1560 | [0.1417, 0.1703] | -0.1560 | 0.9428 | 0.6414 |
| llm_proxy | llama_guard_3_8b | hate_speech,offensive_language | PRIMARY | 1201 / 1201 | 4549 / 4549 | 0.8634 | 0.0000 | 0.3075 | [0.2893, 0.3257] | -0.3075 | 0.8504 | 0.5330 |
| llm_proxy | llama_guard_3_8b | misinformation_regarding_ethics,laws_and_safety | RARE | 2896 / 2896 | 4592 / 4592 | 0.9734 | 0.0000 | 0.1326 | descriptive | -0.1326 | 0.9572 | 0.0117 |
| llm_proxy | llama_guard_3_8b | non_violent_unethical_behavior | PRIMARY | 3489 / 3489 | 4249 / 4249 | 0.8937 | 0.0000 | 0.2063 | [0.1954, 0.2172] | -0.2063 | 0.9104 | 0.2099 |
| llm_proxy | llama_guard_3_8b | privacy_violation | PRIMARY | 873 / 873 | 4581 / 4581 | 0.5590 | 0.0000 | 0.1772 | [0.1559, 0.1984] | -0.1772 | 0.9389 | 0.6901 |
| llm_proxy | llama_guard_3_8b | self_harm | RARE | 105 / 105 | 4663 / 4663 | 0.6762 | 0.0000 | 0.1901 | descriptive | -0.1901 | 0.8917 | 0.6643 |
| llm_proxy | llama_guard_3_8b | sexually_explicit,adult_content | PRIMARY | 428 / 428 | 4583 / 4583 | 0.6495 | 0.0000 | 0.3100 | [0.2816, 0.3385] | -0.3100 | 0.8880 | 0.5843 |
| llm_proxy | llama_guard_3_8b | terrorism,organized_crime | RARE | 151 / 151 | 4664 / 4664 | 0.9801 | 0.0000 | 0.1769 | descriptive | -0.1769 | 0.8926 | 0.1893 |
| llm_proxy | llama_guard_3_8b | violence,aiding_and_abetting,incitement | PRIMARY | 2503 / 2503 | 4621 / 4621 | 0.8889 | 0.0000 | 0.1140 | [0.1028, 0.1252] | -0.1140 | 0.9620 | 0.5762 |
| llm_proxy | shieldgemma_2b | animal_abuse | PRIMARY | 177 / 0 | 4665 / 0 | 0.8531 | 1.0000 | N/E | N/E | N/E | N/E | 0.6311 |
| llm_proxy | shieldgemma_2b | child_abuse | RARE | 245 / 245 | 4669 / 4669 | 0.8857 | 0.0000 | 0.2644 | descriptive | -0.2640 | 0.8904 | 0.3215 |
| llm_proxy | shieldgemma_2b | controversial_topics,politics | PRIMARY | 2226 / 0 | 3547 / 0 | 0.7916 | 1.0000 | N/E | N/E | N/E | N/E | 0.1384 |
| llm_proxy | shieldgemma_2b | discrimination,stereotype,injustice | PRIMARY | 2209 / 2209 | 3821 / 3821 | 0.8506 | 0.0000 | 0.3112 | [0.3017, 0.3208] | -0.3110 | 0.9721 | 0.4767 |
| llm_proxy | shieldgemma_2b | drug_abuse,weapons,banned_substance | PRIMARY | 1116 / 1116 | 4487 / 4487 | 0.7688 | 0.0000 | 0.1343 | [0.1208, 0.1477] | -0.1341 | 0.9844 | 0.5937 |
| llm_proxy | shieldgemma_2b | financial_crime,property_crime,theft | PRIMARY | 1890 / 0 | 4468 / 0 | 0.9026 | 1.0000 | N/E | N/E | N/E | N/E | 0.6414 |
| llm_proxy | shieldgemma_2b | hate_speech,offensive_language | PRIMARY | 1201 / 1201 | 4549 / 4549 | 0.8634 | 0.0000 | 0.2366 | [0.2223, 0.2509] | -0.2366 | 0.9176 | 0.5330 |
| llm_proxy | shieldgemma_2b | misinformation_regarding_ethics,laws_and_safety | RARE | 2896 / 2896 | 4592 / 4592 | 0.9734 | 0.0000 | 0.2089 | descriptive | -0.2089 | 0.9250 | 0.0117 |
| llm_proxy | shieldgemma_2b | non_violent_unethical_behavior | PRIMARY | 3489 / 3489 | 4249 / 4249 | 0.8937 | 0.0000 | 0.4545 | [0.4506, 0.4584] | -0.4545 | 0.8099 | 0.2099 |
| llm_proxy | shieldgemma_2b | privacy_violation | PRIMARY | 873 / 873 | 4581 / 4581 | 0.5590 | 0.0000 | 0.4714 | [0.4652, 0.4775] | -0.4714 | 0.5927 | 0.6901 |
| llm_proxy | shieldgemma_2b | self_harm | RARE | 105 / 105 | 4663 / 4663 | 0.6762 | 0.0000 | 0.1620 | descriptive | -0.1614 | 0.9673 | 0.6643 |
| llm_proxy | shieldgemma_2b | sexually_explicit,adult_content | PRIMARY | 428 / 428 | 4583 / 4583 | 0.6495 | 0.0000 | 0.1866 | [0.1635, 0.2096] | -0.1864 | 0.9924 | 0.5843 |
| llm_proxy | shieldgemma_2b | terrorism,organized_crime | RARE | 151 / 151 | 4664 / 4664 | 0.9801 | 0.0000 | 0.1488 | descriptive | -0.1479 | 0.9779 | 0.1893 |
| llm_proxy | shieldgemma_2b | violence,aiding_and_abetting,incitement | PRIMARY | 2503 / 2503 | 4621 / 4621 | 0.8889 | 0.0000 | 0.1113 | [0.1026, 0.1200] | -0.1112 | 0.9606 | 0.5762 |
| llm_proxy | shieldgemma_9b | animal_abuse | PRIMARY | 177 / 0 | 4665 / 0 | 0.8531 | 1.0000 | N/E | N/E | N/E | N/E | 0.6311 |
| llm_proxy | shieldgemma_9b | child_abuse | RARE | 245 / 245 | 4669 / 4669 | 0.8857 | 0.0000 | 0.2582 | descriptive | -0.2581 | 0.8930 | 0.3215 |
| llm_proxy | shieldgemma_9b | controversial_topics,politics | PRIMARY | 2226 / 0 | 3547 / 0 | 0.7916 | 1.0000 | N/E | N/E | N/E | N/E | 0.1384 |
| llm_proxy | shieldgemma_9b | discrimination,stereotype,injustice | PRIMARY | 2209 / 2209 | 3821 / 3821 | 0.8506 | 0.0000 | 0.3200 | [0.3107, 0.3293] | -0.3198 | 0.9675 | 0.4767 |
| llm_proxy | shieldgemma_9b | drug_abuse,weapons,banned_substance | PRIMARY | 1116 / 1116 | 4487 / 4487 | 0.7688 | 0.0000 | 0.1323 | [0.1197, 0.1448] | -0.1321 | 0.9867 | 0.5937 |
| llm_proxy | shieldgemma_9b | financial_crime,property_crime,theft | PRIMARY | 1890 / 0 | 4468 / 0 | 0.9026 | 1.0000 | N/E | N/E | N/E | N/E | 0.6414 |
| llm_proxy | shieldgemma_9b | hate_speech,offensive_language | PRIMARY | 1201 / 1201 | 4549 / 4549 | 0.8634 | 0.0000 | 0.2562 | [0.2421, 0.2702] | -0.2562 | 0.9182 | 0.5330 |
| llm_proxy | shieldgemma_9b | misinformation_regarding_ethics,laws_and_safety | RARE | 2896 / 2896 | 4592 / 4592 | 0.9734 | 0.0000 | 0.2277 | descriptive | -0.2277 | 0.9216 | 0.0117 |
| llm_proxy | shieldgemma_9b | non_violent_unethical_behavior | PRIMARY | 3489 / 3489 | 4249 / 4249 | 0.8937 | 0.0000 | 0.4611 | [0.4575, 0.4648] | -0.4611 | 0.8199 | 0.2099 |
| llm_proxy | shieldgemma_9b | privacy_violation | PRIMARY | 873 / 873 | 4581 / 4581 | 0.5590 | 0.0000 | 0.4753 | [0.4698, 0.4808] | -0.4753 | 0.7173 | 0.6901 |
| llm_proxy | shieldgemma_9b | self_harm | RARE | 105 / 105 | 4663 / 4663 | 0.6762 | 0.0000 | 0.1566 | descriptive | -0.1543 | 0.9712 | 0.6643 |
| llm_proxy | shieldgemma_9b | sexually_explicit,adult_content | PRIMARY | 428 / 428 | 4583 / 4583 | 0.6495 | 0.0000 | 0.1730 | [0.1523, 0.1937] | -0.1730 | 0.9944 | 0.5843 |
| llm_proxy | shieldgemma_9b | terrorism,organized_crime | RARE | 151 / 151 | 4664 / 4664 | 0.9801 | 0.0000 | 0.1489 | descriptive | -0.1470 | 0.9788 | 0.1893 |
| llm_proxy | shieldgemma_9b | violence,aiding_and_abetting,incitement | PRIMARY | 2503 / 2503 | 4621 / 4621 | 0.8889 | 0.0000 | 0.1275 | [0.1189, 0.1361] | -0.1273 | 0.9610 | 0.5762 |

## Integrity

- Qwen strict-parse failures: 65 / 11088 (0.5862%).
- Joined input SHA-256: `f6785917160950f2bcf40bd78bef915592db85f3194a270ef7a007a313e40050`
- Analysis JSON SHA-256: `9eb62e14472cd31ef6b95d2e91d83be87a22d5a57c3968a53e24bdfaa1d930ee`
- Every guard/category/label-source cell is listed above; no null, reversed, malformed, rare, or no-map cell was removed.
- Qwen is an LLM proxy label source, not human ground truth and not an evaluated guard.
