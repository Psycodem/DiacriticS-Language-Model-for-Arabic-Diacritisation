# Qwen3.5-0.8B character tagger — double-pass

Reframes diacritisation as **per-character classification** instead of generation. The generative
LM head is replaced with a 16-class head over a frozen label vocabulary, and the whole body is
fine-tuned for one epoch on `Misraj/Sadeed_Tashkeela`.

**No `adapter/` folder and no `10pct`/`30pct`/`50pct` rungs.** This is a full fine-tune, not LoRA,
and it was trained once on 100% of the corpus — so it has `outputs/` directly, the same way
[`../Qwen_3_5_0_8B_Full_FineTune/`](../Qwen_3_5_0_8B_Full_FineTune/) does.

It **warm-starts from the 2.58-DER full fine-tune**, not from stock `Qwen/Qwen3.5-0.8B`. That is
the point of the experiment: a tagging head measured against a generative head on a shared body.

## Results — SadeedDiac-25, corrected scorer

| split | DER_ce | DER_noce | WER_ce | WER_noce | skeleton mismatch |
|---|---:|---:|---:|---:|---:|
| CA (600) | 1.22 | 0.65 | 4.17 | 1.43 | **0.00%** |
| MSA (600) | 3.92 | 3.49 | 12.02 | 8.21 | **0.00%** |
| **macro mean** | **2.6** | **2.1** | **8.1** | **4.8** | **0.00%** |

## It wins on DER and loses on WER — say which you are quoting

Against the generative full fine-tune it warm-started from:

| metric | generative (2.58) | tagger | winner |
|---|---:|---:|---|
| DER_ce | 3.06 | **2.57** | tagger |
| DER_noce | 2.58 | **2.07** | tagger |
| WER_ce | **7.34** | 8.09 | generative |
| WER_noce | **4.12** | 4.82 | generative |
| MSA skeleton corruption | 18.17% | **0.00%** | tagger |

Which metric you quote decides which model wins, so **do not cite 2.07 bare.** The tagger trades
character accuracy for word accuracy: fewer character errors spread over more words. CA wins on
all four metrics; the entire WER regression is MSA (12.02 / 8.21 vs 10.40 / 6.78). *Why the
regression is MSA-only is not established* — a per-word error-distribution analysis was attempted
and discarded because its counts did not reconcile with the scorer's own DER.

The 0.00% skeleton mismatch is structural, not learned: a tagger cannot alter the consonantal
skeleton because it only assigns marks to existing characters. That is the architectural case for
this formulation, and it is what the generative runs cannot match at any step count.

## The orphan-mark decode bug — read this before reusing these numbers

The numbers above are **post-fix**. `Train-Tagging-Qwen/eval_results_fullepoch/metrics_tagging_qwen.csv`
in the untracked tree still holds the **pre-fix** values (DER_ce 4.35, WER_ce 15.72) — do not use
that file.

The char head emits a label at *every* position, including spaces, punctuation and digits.
Training masks those with `-100`, so the model never learns them and the loss ignores them — but
at inference `argmax` still returns a class, and `labels.py::decode` rendered it. That produced
60,520 combining marks on non-letters across 1,200 rows (99.9% of rows) against 0 in gold. They
land at word boundaries, which is exactly where the case ending is scored:

| | before | after |
|---|---:|---:|
| DER_ce | 4.35 | **2.57** |
| WER_ce | 15.72 | **8.09** |
| DER_noce | 2.07 | 2.07 (unchanged) |

`DER_noce` never moved because it drops the word-final character, so it never saw the damage. The
fix is one condition — `and is_diacritizable(ch)` in `decode` — applied in **both** copies of
`labels.py`. The same bug inflated the AraBERT control's DER_ce (12.97 → 10.59).

A metric pair moving in opposite directions (DER_noce better, DER_ce much worse) is a bug
signature, not a capability finding.

## Training

One epoch, 1,040,468 rows, 16,258 steps, 2 h 13 m, ~$10 on one H100 at 0.49 s/step. Effective
batch 64 (16 × 4), lr 2e-5, warmup ratio 0.03, seed 42, 16 labels.

Best checkpoint was step **16,000 of 16,258** — i.e. the very end. That breaks the project's
"task saturates at ~25% of an epoch" finding: early saturation is a property of the **generative**
formulation, not of the task.

`scan_labels` → `verify_labels` is a hard gate before training: `decode()` must reconstruct the
gold text exactly, 100% round-trip. The label vocabulary at `runs/arabert-tagging/labels.json` is
frozen and shared with the AraBERT control — reused, never re-derived.

## Control arm

[`control_arabert/`](control_arabert/) holds the AraBERT character tagger that this run is
measured against — same task, same frozen label vocabulary, a non-Qwen body. It is the control
that separates "tagging helps" from "this body helps".

## Files

```
train_tagging_qwen.py          training loop
model_tagging_qwen.py          the 16-class per-character head
prepare_data_tagging_qwen.py   tokenisation + label alignment
labels.py                      label vocabulary and decode() — carries the is_diacritizable gate
scoring.py                     corrected DER/WER
evaluate_tagging_qwen.py       benchmark generation + scoring
diagnose_msa_coverage.py       the matched-frequency CA/MSA control
modal_config/                  Modal image, Volume, label scan/verify, train, evaluate
outputs/                       metrics, predictions, loss curve, training log, run config
control_arabert/               AraBERT control-arm metrics and predictions
```
