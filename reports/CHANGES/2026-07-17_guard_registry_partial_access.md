# Guard registry access partially resolved

- Date: 2026-07-17
- Stage: `PREREG_LABELSOURCE_GUARD` §11.2
- Outcome access: none

The owner authenticated AutoDL to Hugging Face. `hf auth whoami` succeeds, and
both pinned ShieldGemma repositories allow file access. The pinned Llama Guard
3 repository still returns `Access denied. This repository requires approval`,
so the overall gate remains blocked.

The two ShieldGemma checkpoints may be downloaded and their tokenizer metadata
frozen while waiting. No distribution sanity, BeaverTails scoring, ECE,
ranking, taxonomy adjudication, or Qwen proxy annotation is authorized until
all three pinned guards are accessible.
