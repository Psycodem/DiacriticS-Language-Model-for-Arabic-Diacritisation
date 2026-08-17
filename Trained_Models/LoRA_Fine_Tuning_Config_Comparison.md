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
| Warmup steps | 300 | Same warmup length. |
| Weight decay | 0.01 | Same regularization. |
| Optimizer | `adamw_torch` | Same optimizer (only the 4-bit path would use `paged_adamw_8bit`, and neither run takes that path). |
| Gradient checkpointing | **On** for both | See "throughput confound" note below. |
| Max sequence length | 1024 tokens | Same truncation point for prompt+answer. |
| Random seed | 42 | Same data shuffling, sampling, and weight init noise. |
| Logging / eval / save cadence | `logging_steps=20`, `eval_steps=200`, `save_steps=200`, `save_total_limit=2` | Same monitoring granularity. |
| Training data | `Misraj/Sadeed_Tashkeela` — train 1,042,698 rows / test 2,485 rows, used as-is (no extra preprocessing) | Same corpus, same split, same cleaning (none). |
| Benchmark data | `Misraj/SadeedDiac-25`, split `train`, 1,200 paragraphs, light preprocessing only (strip diacritics to build the prompt; gold text untouched) | Same held-out benchmark, same preprocessing (so numbers stay comparable to the Sadeed paper). |
| In-training eval subset | 500 rows of the test split | Same eval cost during training. |
| Post-training eval sample (train split) | 500 rows | Train-vs-test overfitting comparison uses the same sample size for both models. |
| Inference decoding | Greedy (`do_sample=False`), `max_new_tokens=512` | Deterministic, identical decoding strategy — no sampling-temperature confound. |
| Evaluation metrics | DER and WER, each with and without case-ending (i'rab), via the same `jiwer`-based word alignment | Identical scoring code (copied verbatim into both scripts) so the numbers are computed the same way. |
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

## Net effect

Because every optimization-relevant hyperparameter (LoRA config, effective
batch size, learning-rate schedule, epoch count, sequence length, seed, data
splits, decoding strategy, and evaluation code) is identical, any difference
in the resulting DER/WER between Gemma-4-E4B and Qwen3.5-4B in
`eval_outputs/metrics_summary.csv` can be attributed to the models themselves
(architecture, pretraining, parameter count) rather than to the training
recipe.
