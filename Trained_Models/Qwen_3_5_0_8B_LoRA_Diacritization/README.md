# Qwen3.5-0.8B + LoRA — fairness-contract run (r=16, effective batch 96)

One full epoch of `Qwen/Qwen3.5-0.8B` on `Misraj/Sadeed_Tashkeela` under
[`LoRA_Fine_Tuning_Config_Comparison.md`](../LoRA_Fine_Tuning_Config_Comparison.md), scored on
`Misraj/SadeedDiac-25`.

**No data-scaling curve here.** The sibling Gemma-4 and Qwen3.5-4B folders carry `10pct/`,
`30pct/` and `50pct/` rungs; this model was trained once, on 100% of the corpus, so the two
subfolders below are *checkpoint-selection* variants of the **same single run**, not different
training fractions.

| | `final_checkpoint/` | `best_checkpoint/` |
|---|---|---|
| step | 10,862 (end of epoch) | 1,200 |
| selected by | last step — the contract's rule | lowest `eval_loss` |
| `eval_loss` | 0.13022 | **0.11341** |
| **macro DER_noce (corrected)** | **3.8** | 11.0 |

## Results — SadeedDiac-25, corrected scorer

`final_checkpoint/` (step 10,862):

| split | DER_ce | DER_noce | WER_ce | WER_noce | skeleton mismatch |
|---|---:|---:|---:|---:|---:|
| CA (600) | 1.61 | 1.01 | 4.94 | 1.94 | 0.67% |
| MSA (600) | 6.82 | 6.60 | 13.29 | 9.37 | 31.33% |
| **macro mean** | **4.2** | **3.8** | **9.1** | **5.7** | 16.00% |

`best_checkpoint/` (step 1,200):

| split | DER_ce | DER_noce | WER_ce | WER_noce | skeleton mismatch |
|---|---:|---:|---:|---:|---:|
| CA (600) | 3.48 | 2.83 | 8.96 | 4.92 | 5.83% |
| MSA (600) | 18.94 | 19.22 | 26.53 | 21.40 | 78.17% |
| **macro mean** | **11.2** | **11.0** | **17.8** | **13.2** | 42.00% |

Zero empty and zero truncated generations in both. Macro mean of CA and MSA, never pooled.

### Which columns to read

`outputs/metrics_summary.csv` here carries **both** scorer generations, which the sibling folders'
CSVs do not — theirs have only the frozen columns:

- `DER_*_corr` / `WER_*_corr` — [`Evaluation_Functions_Corrected.py`](../../Evaluation_Functions_Corrected.py). **Cite these.** The tables above use them.
- `DER_*` / `WER_*` unsuffixed — the frozen generation-1 scorer, kept only so this run can be diffed against the 4B numbers in the sibling folders, which are generation-1. Never mix the two in one table.

Parity of the embedded corrected scorer against `Evaluation_Functions_Corrected.py` was verified
numerically over 300 generated documents (Eastern-Arabic numerals, citation refs, deletions,
substitutions, insertions, undiacritised words, non-canonical combining-mark order) — identical on
all four metrics, as the contract requires. A source diff is not sufficient evidence.

## The checkpoint-selection result

`best_checkpoint/` won on `eval_loss` and lost on DER by **2.9×**. The two metrics rank the same
pair of checkpoints in opposite directions.

This reproduces the contract's "Why the final checkpoint, not the 'best' one" finding — the same
failure it records for the 30%-data Gemma run, which shipped `checkpoint-200` of 3,259 and scored
worse than the 10% run. Had `load_best_model_at_end` been left at the training script's shipped
`True`, this folder would have recorded **11.0** for Qwen3.5-0.8B: worse than the 4B LoRA's 8.48,
and indistinguishable from a genuine model result.

The mechanism is skeleton corruption, not diacritic difficulty. At step 1,200 the model rewrites
the consonantal skeleton on **78.2%** of MSA paragraphs; by step 10,862 that falls to 31.3%. A
fluent-but-wrong rewrite is cheap in cross-entropy and catastrophic in DER, which is why 500-row
`eval_loss` fails to see it.

## Where this sits

| run | macro DER_noce |
|---|---:|
| Qwen3.5-4B + LoRA | 8.5 |
| **this run — best checkpoint** | **11.0** |
| **this run — final checkpoint** | **3.8** |
| Qwen3.5-0.8B LoRA, r=32, effective batch 64 | 3.15 |
| Qwen3.5-0.8B full fine-tune | 2.58 |
| Qwen3.5-0.8B character tagger | 2.07 |

Losing to the r=32 / batch-64 run is expected, not a defect: that run used twice the adapter rank.
The contract fixes r=16 so this number is comparable to Gemma-4 and Qwen3.5-4B — it was never the
capacity-optimal choice for this model.

## Contract compliance

Held identical to the Gemma-4 / Qwen3.5-4B runs: LoRA r=16, alpha=32, dropout=0.05, bias `none`;
**effective batch 96**; 1 epoch with `MAX_STEPS=-1`; lr 2e-4 cosine; **warmup ratio 0.05**; weight
decay 0.01; `adamw_torch`; gradient checkpointing **on**; max sequence length 1024; seed 42;
logging 20 / eval 200 / save 200 with `save_total_limit=2`; `load_best_model_at_end=False`;
in-training eval subset 500; greedy decoding; pinned dataset revisions. Every value is recorded in
`*/adapter/run_config.json`.

### Four settings the upstream script did not expose

Each was a module-level constant with no environment hook, so **the Ibex sbatch violated all four
too** — `export EVAL_STEPS=200` in particular was read by nothing. All four are env-settable in
this folder's copy of the training script, upstream defaults unchanged so the r=32 run still
reproduces:

| | script shipped | contract |
|---|---|---|
| `WARMUP_RATIO` | 0.03 | **0.05** |
| `load_best_model_at_end` | forced `True` | **`False`** |
| `IN_TRAINING_EVAL_SUBSET` | 1,000 | **500** |
| `EVAL_STEPS` | never read (`N_EVALS=20` scaled to run length → every 543 steps) | **200** |

### Deliberate deviations

- **150 LoRA target modules, not the contract's 7-name list.** Qwen3.5 is a hybrid — 18 of 24
  layers are Gated-DeltaNet (`in_proj_qkv`, `in_proj_z`, `out_proj`). The conventional
  `q,k,v,o,gate,up,down` list adapts only the 6 full-attention layers and silently skips the rest.
  Asserted at exactly 150.
- **World size 1 (one H100), not 3× A100.** The contract lets per-device batch differ and forces
  the *effective* batch equal; that is held at 96, so per-step gradient statistics are identical.
- **Length-adaptive generation cap** (`min(input_len × ratio + 32, 1024)`) rather than a flat
  `max_new_tokens=512`. Greedy either way, so no sampling confound; the cap only bounds runaway
  generations, and truncated rows are scored as deletions rather than discarded.
- **`--splits benchmark` only.** `outputs/` therefore has no `train_predictions.csv` or
  `test_predictions.csv`, which the sibling folders do carry. Scoring the 500-row train sample and
  the 2,485-row test split was dropped to stay inside the run's GPU budget.

## Run facts

Executed on Modal, one H100, because Ibex job `50791419` was still `PENDING` behind
higher-priority work. One epoch = 10,862 optimizer steps over 1,042,693 usable rows (434M tokens);
5.14 h wall clock at 1.72 s/step; ~$23.7.

`ibex_config/run_qwen35_08b_lora_3gpu.sbatch` is transcribed from the queued job, with the four
contract-patch exports added and marked — it is the 3-GPU replication recipe, but **it has not been
run**. The numbers in this folder come from the Modal run.

## Files

```
Train_LoRA_Qwen_3_5_0_8B_Diacritization.py   training + evaluation + scoring, patched
modal_config/Modal_Train_Qwen3.5_0.8B.py     Modal image, Volume, CPU prep, GPU train/eval
ibex_config/run_qwen35_08b_lora_3gpu.sbatch  3-GPU Slurm recipe (not run)
final_checkpoint/adapter/                    metadata for step 10,862; weights on the Volume
final_checkpoint/outputs/                    benchmark predictions, metrics, loss curve, log
best_checkpoint/adapter/                     metadata for step 1,200
best_checkpoint/outputs/                     benchmark predictions, metrics
```

`HUGGINGFACE.md` explains where the adapter weights actually live — nothing here is on the Hub.
