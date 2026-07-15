# Change: Upgrade TRL 0.19.1 to 1.8.0 for Transformers-5 training compatibility (severity: Green)

Date / commit: 2026-07-15 / `a2c73f35e2566dd4210cead6b4a5c84fefb4f3e9`

Trigger: The pre-Day-4 import check failed at `from trl import DPOTrainer` because
TRL 0.19.1 imports the Transformers symbol
`MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`, which was removed in Transformers 5.13.1.
The main environment also still contained `rewardbench==0.1.4`, whose strict
`transformers==4.51.0` requirement made `pip check` fail.

What I changed:

- Upgraded the main environment and `requirements.txt` from `trl==0.19.1` to
  `trl==1.8.0`; Transformers remains 5.13.1 and PEFT remains 0.19.1.
- Removed RewardBench 0.1.4 from the main environment and from
  `scripts/setup/install_env.sh`. It remains designated for an isolated Day-9
  environment, as already required by the Transformers-5 change report.
- Added `scripts/day4/smoke_trl_peft.py`, which exercises one LoRA SFT step and
  one LoRA DPO step on a randomly initialized tiny Qwen2 model and synthetic data.
- Saved before/after package freezes under `$PCCD_OUT/env/`.

Why this and not an alternative: `pip install --dry-run trl==1.8.0` showed that
all existing dependencies already met its requirements and only TRL itself would
change. The installed package then imported `SFTTrainer`, `DPOTrainer`,
`SFTConfig`, and `DPOConfig` directly under Transformers 5.13.1. This is cleaner
and easier to reproduce than injecting a removed Transformers mapping through a
runtime shim. RewardBench cannot coexist cleanly with the validated core stack
and is not used before its isolated Day-9 evaluation.

Impact on propositions/gates: This is API/runtime plumbing only. It changes no
P1-P6 definition, G1-G4 threshold, dataset, model architecture, or scientific
training protocol. The smoke model and data are synthetic and are not research
artifacts. D0 critic architecture/training remains undecided and Red-locked.

Reversibility: Pin TRL back to 0.19.1 and restore the pre-change freeze only if
Transformers is also downgraded or a compatibility shim is deliberately adopted.
No scientific result is provisional because no D0 or D2-D6 training was run.

Open question for PaperGuru: none; request confirmation of the smoke PASS before
defining or running D0.
