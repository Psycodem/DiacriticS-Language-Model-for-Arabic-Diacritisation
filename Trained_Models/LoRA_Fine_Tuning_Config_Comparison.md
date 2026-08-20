# LoRA Fine-Tuning Configuration — Gemma-4-E4B vs. Qwen3.5-4B

This document explains the fine-tuning configuration used for the two LoRA
Arabic-diacritization runs in this directory, and why the configuration makes
the two models comparable rather than an apples-to-oranges result.

- [`gemma4_lora_diacritization/`](gemma4_lora_diacritization/) — `google/gemma-4-E4B-it`
- [`qwen35_4b_lora_diacritization/`](qwen35_4b_lora_diacritization/) — `Qwen/Qwen3.5-4B`

Each folder contains:
- `fine_tuning_script/` — the self-contained training + evaluation script (data
  loading, LoRA setup, `Trainer`, inference, DER/WER scoring).
- `ibex_config/` — the final SLURM (`sbatch`) job files used to run that script
  on KAUST's Ibex HPC cluster, all 1-node / 3-GPU DDP runs via
  `torchrun --standalone --nproc_per_node=3`:
  - `run_<model>_3gpu.sbatch` — the full run, 1 epoch over 100% of the training split.
  - `run_<model>_10pct.sbatch`, `_30pct.sbatch`, `_50pct.sbatch` — the same
    fair-comparison config, trained on a nested prefix of the shuffled training
    split (seed 42) instead of the full corpus, for a 4-point (10/30/50/100%)
    data-scaling curve per model.

  Earlier single-GPU, smoke-test, and diagnostic job variants used while
  developing the scripts are not included here.

> **Note on redactions:** the copies here have the Hugging Face access token
> blanked out (`HF_TOKEN = ""`) and the `--mail-user` SLURM directive replaced
> with a placeholder. The original working files load the token from
> `$HOME/.hf_token` or the `HF_TOKEN` environment variable at submit time —
> nothing else changes at runtime.

## Why a fairness contract, not just "the same script twice"

Gemma-4-E4B (8.6B params, ~262k-token vocabulary) and Qwen3.5-4B (4.2–4.8B
params, ~248k-token vocabulary) are different sizes, so they cannot use
identical hardware settings — a batch size that fits the smaller model will
under-use the GPU, and one that fits the larger model may not fit the other at
all. Both training scripts encode an explicit **"FAIRNESS CONTRACT"**: every
setting that affects *what the optimizer sees and how it updates the model*
is held identical across both scripts, and only the settings that are purely
about *fitting the model in GPU memory* are allowed to differ — with the
downstream quantity (effective batch size) still forced equal.

## Identical across both models

| Setting | Value | Why it matters |
|---|---|---|
| Method | LoRA on a bf16 base (`USE_4BIT=False`) | Neither run uses QLoRA/4-bit — comparing LoRA-vs-LoRA, not LoRA-vs-QLoRA. |
| LoRA rank `r` | 16 | Adapter capacity. |
| LoRA `alpha` | 32 | Adapter scaling (`alpha/r = 2`). |
| LoRA `dropout` | 0.05 | Regularization on the adapters. |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Same 7 projections adapted in every decoder layer for both models. |
| LoRA `bias` | `"none"` | No bias terms trained. |
| Effective batch size | **96** (global, post-accumulation) | The quantity the optimizer actually steps on — see below. |
| Epochs | 1 full epoch over the training split | `MAX_STEPS=-1` so epoch count isn't silently overridden by a step cap. |
| Learning rate | `2e-4`, cosine schedule | Same LR and decay shape. |
| Warmup | **5% of total steps** (`WARMUP_RATIO = 0.05`) | Same warmup *shape*. See "Warmup as a ratio" below — a fixed step count is not equivalent across data fractions. |
| Weight decay | 0.01 | Same regularization. |
| Optimizer | `adamw_torch` | Same optimizer (only the 4-bit path would use `paged_adamw_8bit`, and neither run takes that path). |
| Gradient checkpointing | **On** for both | See "throughput confound" note below. |
| Max sequence length | 1024 tokens | Same truncation point for prompt+answer. |
| Random seed | 42 | Same data shuffling, sampling, and weight init noise. |
| Logging / eval / save cadence | `logging_steps=20`, `eval_steps=200`, `save_steps=200`, `save_total_limit=2` | Same monitoring granularity. |
| Final model selection | `load_best_model_at_end=False` — the **last** checkpoint is kept | See "Why the final checkpoint" below. Selecting on a noisy `eval_loss` made two runs incomparable. |
| Training data | `Misraj/Sadeed_Tashkeela` — train 1,042,698 rows / test 2,485 rows, used as-is (no extra preprocessing) | Same corpus, same split, same cleaning (none). |
| Benchmark data | `Misraj/SadeedDiac-25`, split `train`, 1,200 paragraphs, light preprocessing only (strip diacritics to build the prompt; gold text untouched) | Same held-out benchmark, same preprocessing (so numbers stay comparable to the Sadeed paper). |
| In-training eval subset | 500 rows of the test split | Same eval cost during training. |
| Post-training eval sample (train split) | 500 rows | Train-vs-test overfitting comparison uses the same sample size for both models. |
| Inference decoding | Greedy (`do_sample=False`), `max_new_tokens=512` | Deterministic, identical decoding strategy — no sampling-temperature confound. |
| Evaluation metrics | DER and WER, each with and without case-ending (i'rab), via the same `jiwer`-based word alignment | Identical scoring code, matching [`Evaluation_Functions_Corrected.py`](../Evaluation_Functions_Corrected.py). Verified numerically, not by inspection — see "Scoring parity is tested" below. |
| System prompt | Same Arabic instruction string | Identical task framing given to both models. |
| Hardware | Ibex, `a100` (80GB), 1 node, 3 GPUs, DDP via `torchrun --standalone --nproc_per_node=3` (`run_*_3gpu.sbatch`) | Same accelerator type and count for the comparison run. |

## What is allowed to differ, and why it stays fair

**`PER_DEVICE_TRAIN_BATCH_SIZE`** is the only training-relevant setting that
differs between the two scripts:

| | Gemma-4-E4B | Qwen3.5-4B |
|---|---|---|
| Per-device micro-batch | 4 | 8 |
| Gradient accumulation steps | 8 (derived) | 4 (derived) |
| GPUs (world size) | 3 | 3 |
| **Effective batch** | 4 × 8 × 3 = **96** | 8 × 4 × 3 = **96** |

Gemma-4-E4B is larger (8.6B params) and has a bigger vocabulary than
Qwen3.5-4B (4.2–4.8B params), so it fits a smaller micro-batch per GPU at the
same sequence length. Rather than hand-picking two unrelated batch sizes, each
script derives `GRADIENT_ACCUMULATION_STEPS` at runtime from a shared
constant, `EFFECTIVE_BATCH = 96`:

```python
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH // (PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE)
```

and both scripts raise a hard `SystemExit` if `EFFECTIVE_BATCH` isn't evenly
divisible — a run that would silently drift from 96 refuses to start rather
than quietly producing an unfair comparison. The result: each optimizer step
is still computed from the same number of examples (96) for both models, even
though the two models split that work across per-GPU batch vs. accumulation
differently. That is what "fair" means here — identical gradient statistics
per step, not identical micro-batch shapes.

**Gradient checkpointing is forced ON for both models**, even though it isn't
strictly required for Qwen3.5-4B at its batch size. Checkpointing trades
compute for memory (activations are recomputed on the backward pass instead of
stored), which changes *throughput*, not just memory headroom. Leaving it on
for Gemma (which needs it) and off for Qwen (which doesn't) would confound the
comparison with a hidden throughput difference; forcing it on for both removes
that variable.

## Deliberate deviation from `Confg_Info.md`, applied identically to both

The project's shared config notes (`Confg_Info.md`, "Option A") originally
specified 6 epochs, written for a smaller subset of the corpus. Both scripts
override this to **1 epoch** over the full 1,042,697-row corpus — at an
effective batch of 96 that is already ~10,800 optimizer steps, and 6 epochs
would be ~65,000 steps (multi-week runs on Ibex's time limits). This deviation
is applied identically to both models, so it does not introduce an asymmetry;
it changes the absolute amount of training both models receive, equally.

## Warmup as a ratio, not a fixed step count

Warmup is set as a **fraction of the run** (`WARMUP_RATIO = 0.05`) rather than a
fixed `WARMUP_STEPS = 300`. This matters because the same 300 steps means a
completely different schedule at each point of the data-scaling curve:

| Data fraction | Total steps | 300 fixed steps | 5% ratio |
|---|---|---|---|
| 10% | 1,086 | 27.6% of the run | 54 steps |
| 30% | 3,258 | 9.2% | 162 steps |
| 50% | 5,430 | 5.5% | 271 steps |
| 100% | 10,861 | 2.8% | 543 steps |

Under the fixed count the 10% model spent over a quarter of its training ramping
up while the 100% model spent under 3% — a 10x spread in how long each model sat
at peak learning rate. That is a confound in any comparison where dataset size is
supposed to be the only variable. A constant ratio gives every run the same
learning-rate curve *shape*, and still scales absolute warmup with data
(54 → 543 steps), because the step count already scales with the corpus.

One implementation note: **transformers v5 removed the `warmup_ratio` argument**
(only `warmup_steps` survives). Both scripts therefore resolve the ratio to an
absolute step count at runtime, just before constructing `TrainingArguments`:

```python
_total_steps = MAX_STEPS if (MAX_STEPS and MAX_STEPS > 0)     else steps_per_epoch * NUM_TRAIN_EPOCHS
warmup_steps_resolved = max(1, int(WARMUP_RATIO * _total_steps))
```

## Why the final checkpoint, not the "best" one

`load_best_model_at_end` is **off** in both scripts, so the model saved at the end
is the last one, not the one that minimised `eval_loss`.

This was not a stylistic choice. With it enabled, the 30%-data Gemma run saved
`checkpoint-200` out of 3,259 — confirmed by checksum, `final_adapter` was
byte-identical to that early checkpoint. The in-training eval uses only 500
examples, so `eval_loss` is noisy; it dipped at step 200 and never beat that, and
the trainer faithfully preserved a model that had seen ~19k examples and was
still inside warmup. The 10% run meanwhile kept its *last* checkpoint.

So "10% of data vs 30% of data" was really "a fully-trained model vs a
barely-started one", and the 30% run scored **worse** on DER (4.55 vs 3.85, both
re-scored with one implementation) despite better training *and* evaluation loss.
For a scaling curve, every point has to train to completion; checkpoint selection
on a noisy metric injects variance that has nothing to do with dataset size.

## Scoring parity is tested, not assumed

Both training scripts embed the DER/WER functions rather than importing them, so
"identical scoring code" is a claim that can rot silently — and it did once
before, when two of the benchmark scripts drifted apart on WER.

Parity is therefore checked by a **differential test**: every implementation is
run over the same 300 generated documents (including Eastern-Arabic numerals,
citation refs, deletions, substitutions and undiacritised words) and the four
metrics compared against `Evaluation_Functions_Corrected.py`. Both training
scripts and all four `Tested_Models_v2` benchmark scripts currently return
byte-identical results:

```
DER_ce 25.56   DER_noce 25.58   WER_ce 25.63   WER_noce 25.49
```

Re-run this whenever a scoring function is edited in any one script — a source
diff is not sufficient evidence, because formatting differences look like changes
and behavioural differences can hide in a helper.

## Net effect

Because every optimization-relevant hyperparameter (LoRA config, effective
batch size, learning-rate schedule *including warmup as a ratio*, epoch count,
final-checkpoint selection, sequence length, seed, data splits, decoding
strategy, and evaluation code) is identical, any difference
in the resulting DER/WER between Gemma-4-E4B and Qwen3.5-4B in
`eval_outputs/metrics_summary.csv` can be attributed to the models themselves
(architecture, pretraining, parameter count) rather than to the training
recipe.
