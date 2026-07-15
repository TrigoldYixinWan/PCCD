# Day 4 D0 critic implementation and sandbox validation

Date: 2026-07-15  
Pre-registration: `reports/PREREG_D0_CRITIC.md` at `bf26e0abd18b91a46cff16de553d5bdb69925051`  
Implementation revision tested: `2b4bd43ce2ed0c1c8ac4664a8f6272bc4c951d48`  
Status: **implementation self-test PASS; real D0 training NOT started; awaiting review**

## 1. Locked protocol mapping

| Pre-registered requirement | Implementation |
|---|---|
| Qwen2.5-7B-Instruct shared backbone | `train_d0.py` loads only the local `qwen7b` path through `AutoModel`; no teacher weights are loaded |
| Ten independent policy heads | `MultiPolicyCritic.heads` is a `ModuleDict` in fixed H1-H5/S1-S3/T1-T2 order |
| Linear -> GELU -> Linear, 3-way | Each head is `Linear(hidden,256) -> GELU -> Linear(256,3)`; 256 is a Green implementation width |
| Last non-pad-token pooling | `last_non_pad_hidden` selects the final mask-positive token and supports either padding side |
| 3-way target including N/A | Class order is exactly satisfied/violated/not_applicable; summed ten-head CE validates every class index and masks nothing |
| LoRA r16/alpha32/dropout0.05 | Applied to q/k/v/o and gate/up/down projections with base weights frozen |
| Fully trained heads | Optimizer asserts every head parameter is trainable |
| AdamW dual learning rates | Separate parameter groups: LoRA `1e-4`, heads `1e-3` |
| Cosine schedule + 3% warmup | Total optimizer steps derive from dataset size/effective batch; warmup uses `ceil(0.03 * steps)` |
| Effective batch 32 | Gradient accumulation is computed from per-device batch × explicit process count and must divide 32 exactly |
| bf16, max length 4096 | Training requires a bf16-capable CUDA device; collator truncates at the locked length |
| One to three epochs, calib early stop | CLI restricts epochs to 1/2/3; best adapter+heads checkpoint is selected by mean per-policy violated-positive F1 on calib |
| Natural label distribution | Day-2 JSONL records are read unchanged; no sampling or class weighting is implemented |
| Explicit multi-GPU only | More than one visible GPU with one process raises before model loading; Accelerate uses one process per GPU |
| D0 frozen checkpoint | Checkpoint contains the PEFT adapter, all ten heads, tokenizer, locked schema, and run metadata; it does not duplicate the 7B base |

The input collator uses `build_messages(prompt,response)` and the Qwen chat template
with a generation header but no teacher answer. Thus the classifier reads the frozen
canonical teacher prompt containing the same prompt/response pair and never receives a
teacher label at inference.

## 2. Evaluation implementation

`src/eval_critic.py` writes one JSONL record per test item containing the ten
three-way logit vectors and predictions. It also writes a metrics JSON and a combined
ten-panel reliability diagram.

Locked L3 is implemented as follows:

- for each policy, discard rows whose teacher label is N/A;
- treat violated as the positive event and satisfied as negative; a predicted N/A is
  non-positive and therefore can contribute a false negative;
- compute each policy's positive-class F1, then CV = population standard deviation / mean
  over the ten F1 values;
- item-cluster bootstrap with 10,000 replicates and seed 20260715;
- PASS only when CV > 0.15 and the 95% CI lower bound is also > 0.15.

P1 calibration uses the full three-way test target, including N/A. It reports per-policy
15-bin top-label multiclass ECE and equal-count adaptive-ECE, each with an item-bootstrap
95% CI, plus per-policy confidence-versus-accuracy reliability curves. These binning
choices are implementation-level and do not change an existing registered gate threshold.

## 3. Exact validation commands

No 7B weights were loaded and no optimizer step on research data was taken.

```bash
source scripts/setup/env.sh
CUDA_VISIBLE_DEVICES= python -m py_compile \
  src/critic_model.py src/train_d0.py src/eval_critic.py \
  scripts/day4/test_d0_impl_cpu.py
CUDA_VISIBLE_DEVICES= python scripts/day4/test_d0_impl_cpu.py \
  2>&1 | tee logs/day4_d0_impl_cpu.log

# Separate launcher-guard probes; neither loads a model nor trains.
CUDA_VISIBLE_DEVICES=0,1 python <two-visible-GPU guard probe>
CUDA_VISIBLE_DEVICES=0 python <single-visible-GPU guard probe>
```

The same four files also passed `python -m py_compile` in the local Windows checkout.

## 4. Sandbox results

| Check | Result |
|---|---|
| Dummy-backbone forward | PASS: logits `(4,10,3)`, pooled states `(4,16)` |
| Last-token pooling | PASS with right padding, left/right mixed padding, and unpadded input |
| Equal ten-head 3-way CE | PASS: finite summed loss equals direct recomputation; N/A labels retained |
| Backward through all heads | PASS: every head parameter received a gradient |
| Tiny random Qwen2 + PEFT | PASS: all seven registered target-module types accepted |
| Backbone trainability contract | PASS: every trainable backbone parameter is a LoRA parameter |
| AdamW parameter groups | PASS: LoRA `1e-4`, heads `1e-3` |
| Locked F1/CV | PASS against deterministic arrays, including N/A exclusion and bootstrap CI shape |
| ECE/adaptive-ECE | PASS against an analytic example with expected value 0.2; bootstrap paths executed |
| Real tokenizer + two Day-2 test rows | PASS: labels `(2,10)`, canonical tokens `(2,383)` |
| Two visible GPUs, one process | PASS: rejected with the explicit-strategy error before model loading |
| One visible GPU, one process | PASS: accepted |

Final CPU test output:

```text
PASS cpu-self-test: dummy+tiny-Qwen2 head_shape=(4,10,3), seven-target LoRA,
dual-lr optimizer, summed_3way_ce, backward, L3 F1/CV/bootstrap,
P1 ECE/adaptive-ECE/bootstrap, canonical_batch=(2, 383)
```

## 5. Integrity and raw artifacts

| Artifact | SHA-256 |
|---|---|
| `src/critic_model.py` | `a2de20abdb30b687a6b32c095c3946e3e6cbfa34b9f250959a6b05a44dd95c4d` |
| `src/train_d0.py` | `d17ccd417be649ace29b0f19e41cbb97855d2440807ac984ca1d066d82e4ed2a` |
| `src/eval_critic.py` | `102c5e7e30738f29e39afb574df1529117a4c6e73e18ab5b0e637d7056a23d3c` |
| `scripts/day4/test_d0_impl_cpu.py` | `13627f40685afb1bac57a6557e2995356eaf7b7170bc1c9a76f2154d7e7428f8` |
| `logs/day4_d0_impl_cpu.log` | `64f2c5f5298c2f25e71992b3f5c2761073050d2161495ec45dba1ec6fbd47e73` |
| `logs/day4_d0_impl_guard.log` | `105eb63353e930530fb338b12c8b0ef1f373bbed98cda0b5adc1f63e10d31af8` |

Raw logs remain on AutoDL at `/root/PCCD/logs/day4_d0_impl_cpu.log` and
`/root/PCCD/logs/day4_d0_impl_guard.log`.

## 6. Anomalies and boundary check

- The first invocation of the CPU test did not enter model code because executable
  scripts did not insert the repository root into `sys.path`. The three entry points were
  given the same root-path bootstrap; compilation and the complete tests then passed.
- Importing PEFT on CPU emitted an optional bitsandbytes message that the external
  `kernels` package was unavailable for CPU 4-bit GEMM. D0 uses neither quantization nor
  CPU training, and every requested non-quantized CPU test completed. No dependency was
  changed to suppress this irrelevant warning.
- No Red element was changed: backbone family/scale, per-policy head structure, 3-way N/A
  target, frozen-after-D0 rule, and L3 definition match the approved pre-registration.
- No D0 checkpoint, D0 logits, L3 result, P1 result, or D2-D6 training result exists yet.

**Implementation verdict: PASS for review. Stop here until PaperGuru authorizes the first
real D0 training run.**

## PaperGuru review verdict (2026-07-15, human-approved)

D0 critic implementation APPROVED. Reviewed against PREREG_D0_CRITIC.md line by line:
- Architecture EXACT: shared Qwen2.5-7B backbone + 10 independent Linear-GELU-Linear 3-way
  heads, last-non-pad pooling (robust to padding side), checkpoint saves only LoRA adapter +
  heads (no duplicate 7B).
- Loss EXACT: equal-weight sum of ten 3-way CEs, N/A retained as a real third class (not
  masked). LoRA r=16/alpha=32/dropout=0.05 on the 7 locked target modules; heads fully
  trained; dual lr 1e-4/1e-3.
- Leakage discipline VERIFIED: train_d0 consumes only --train and --calib; test is never
  loaded in training; early-stop is on calib macro-F1. Explicit multi-GPU guard (WORLD_SIZE
  vs Accelerate) correctly rejects implicit DataParallel.
- L3 metric EXACT: violated-positive F1, teacher-N/A rows excluded per policy, CV over the
  ten F1s, item-cluster bootstrap. P1: per-policy multiclass ECE + adaptive-ECE with
  bootstrap CIs, N/A kept as a real class (consistent with training).
- INDEPENDENT numerical re-check by PaperGuru (not just trusting the self-test): hand-worked
  cases reproduce exactly — F1=0.5, CV=0.4472, ECE=0.5. Self-test's analytic ECE=0.2 example
  and real-tokenizer (2,383)-token batch further confirm correctness.

CLEARED: authorize the FIRST real D0 training run on the Day-2 train split (early-stop on
calib), then eval on the base-distribution TEST split to produce P1 (base calibration) and
L3 (per-policy F1-CV). Hyperparameters (lr/rank/epochs) remain Green-tunable; the Red-locked
elements are unchanged. After D0: freeze the checkpoint + save logits for D1-D6 reuse.
