---
base_model: Qwen/Qwen3.5-0.8B
pipeline_tag: text-generation
tags:
- arabic
- diacritization
- tashkeel
- full-fine-tune
language:
- ar
datasets:
- Misraj/Sadeed_Tashkeela
- Misraj/SadeedDiac-25
---

# Qwen3.5-0.8B, fully fine-tuned for Arabic diacritisation

Every weight updated, on the full `Misraj/Sadeed_Tashkeela` corpus. Part of
**DiacriticS**, a contamination-controlled study of open-weights models on Arabic
diacritisation ([project site](https://diacritics.vercel.app/) ·
[code](https://github.com/Psycodem/DiacriticS-Language-Model-for-Arabic-Diacritisation)).

This is the one place in the study where the **adaptation regime** can be
compared with the base model, corpus and prompt held constant: the same
checkpoint was also adapted with LoRA, and both were scored the same way.

## Results

Scored with [`Evaluation_Functions_Corrected.py`](../../Evaluation_Functions_Corrected.py).
Percentages; lower is better. CE = case ending (*i'rab*).

| Split | n | DER (CE) | DER (no CE) | WER (CE) | WER (no CE) |
|---|---:|---:|---:|---:|---:|
| Train sample | 500 | 0.94 | 0.83 | 2.28 | 1.47 |
| Sadeed Tashkeela test | 2,485 | 2.52 | 1.91 | 5.62 | 2.75 |
| **SadeedDiac-25 benchmark** | **1,200** | **3.06** | **2.58** | **7.34** | **4.12** |

Per-register, on the benchmark:

| Track | n | DER (CE) | WER (CE) |
|---|---:|---:|---:|
| MSA | 600 | 4.79 | 10.40 |
| CA | 600 | 1.33 | 4.28 |

Unusually for this study, Classical Arabic is the *easier* register here — the
reverse of what every zero-shot model shows.

### Against the alternatives

| System | Benchmark DER (CE) | WER (CE) |
|---|---:|---:|
| Qwen3.5-0.8B, zero-shot | 92.18 | 99.67 |
| Qwen3.5-0.8B + LoRA | 3.60 | 8.28 |
| **Qwen3.5-0.8B, full fine-tune** | **3.06** | **7.34** |
| Qwen3.5-4B + LoRA, 50% of corpus | 4.38 | 8.18 |

Full fine-tuning beats LoRA by 0.54 DER points on the same checkpoint, and both
0.8B systems beat a five-times-larger LoRA-adapted model. For a task this narrow,
the trainable fraction matters more than the parameter count.

## Training

| | |
|---|---|
| Method | Full fine-tune (all weights) |
| Data | 100% of `Misraj/Sadeed_Tashkeela`, revision `c10bcbb` |
| Learning rate | 2e-5, cosine |
| Warmup | 3% of total steps |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Effective batch | 64 |
| Epochs | 1 |
| Max sequence length | 1024 |
| Precision | fp32 master weights, bf16 autocast |
| Seed | 42 |

**Two things to know before comparing this with the LoRA runs.**

The learning rate and warmup are *not* the same as the LoRA runs in the sibling
folders (which use 2e-5 → **2e-4** and 3% → **5%**). This is necessary rather than
sloppy: a rate tuned for low-rank adapters diverges when applied to every weight.
So the LoRA-versus-full comparison holds the model, corpus, prompt and scorer
constant, but not the optimiser settings — each regime uses a rate appropriate to
it.

The published weights are **not the last step**. `load_best_model_at_end`
selected `checkpoint-4070` of a planned 16,293 on an eval loss of 0.1073, i.e.
about a quarter of the way through the epoch. That is early stopping working as
intended, not the truncated-checkpoint artefact that affected an early Gemma run;
4,070 steps at effective batch 64 is roughly 260k examples. It does mean the run
did not consume the full epoch it was configured for.

## Files here

| Path | What |
|---|---|
| `Qwen3.5-0.8B_Full-FineTune_Diacritization.ipynb` | the training and evaluation notebook |
| `outputs/metrics_finetuned_fullft.csv` | scored splits, both scorer generations |
| `outputs/metrics_baseline_zeroshot.csv` | the untrained baseline |
| `outputs/predictions_sadeeddiac25_finetuned.csv` | all 1,200 benchmark predictions |
| `outputs/predictions_sadeeddiac25_baseline.csv` | the same, before training |
| `outputs/training_log.csv`, `outputs/loss_curve.png` | training trace |
| `outputs/corpus_composition.csv` | what the training mix contained |

The metrics CSVs carry **two sets of columns**: `*_corr` from the corrected
scorer (the figures quoted above and in the report) and unsuffixed columns from
the older frozen scorer, kept only for continuity. Cite the `_corr` values.

## Weights

The weights are **not in this repository and are not yet on the Hub** — they were
written to `/scratch/runs/qwen3.5-0.8b-fullft/` on the training cluster and have
not been retrieved. A full fine-tune of a 0.8B model is roughly 1.6 GB in bf16,
well past GitHub's 100 MB file limit, so the Hub is where they belong once
available. The LoRA adapters for the 4B models are already published:

- [Psycodem/gemma-4-e4b-lora-diacritization](https://huggingface.co/Psycodem/gemma-4-e4b-lora-diacritization)
- [Psycodem/qwen3.5-4b-lora-diacritization](https://huggingface.co/Psycodem/qwen3.5-4b-lora-diacritization)

## Prompt

Byte-identical to the 4B runs, which is what makes the numbers comparable:

```python
SYSTEM_PROMPT = (
    "أنت نظام متخصص في التشكيل الآلي للنصوص العربية. "
    "مهمتك إضافة الحركات (التشكيل) الصحيحة إلى النص العربي المُدخل دون تغيير الكلمات أو ترتيبها، "
    "مع مراعاة السياق النحوي والصرفي الكامل للجملة."
)
```

Decode greedily. Two details the notebook handles and a naive harness will not:
the checkpoint ships no `generation_config.json` and its declared EOS is
`<|endoftext|>` rather than the `<|im_end|>` that ends a chat turn, so both must
be passed as `eos_token_id` or every generation runs to the cap; and padding and
truncation must both be **left**, because right truncation removes the trailing
turn marker that tells the model to answer.
