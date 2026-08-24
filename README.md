# DiacriticS: Fine-Tuning an Open Language Model for Arabic Diacritisation
**Authors:** <a href="https://mo-youssef.github.io/"><span>Youssef&nbsp;S.&nbsp;Mohamed</span></a> (Mentor, KAUST), 
<a href="https://linktr.ee/psycodem"><span>Mahdi&nbsp;Alkhamis</span></a> (Team Leader, IAU), 
<a href="https://www.linkedin.com/in/mohammad-alali-9b992135a/"><span>Mohammad&nbsp;Alali</span></a> (KFUPM), 
<a href="https://www.linkedin.com/in/saadtalnafjan/"><span>Saad&nbsp;Alnafjan</span></a> (KAU)

<div align="center">

[![Website](https://img.shields.io/badge/Website-diacritics.vercel.app-000000.svg?logo=vercel)](https://diacritics.vercel.app/)
[![Benchmark](https://img.shields.io/badge/Benchmark-SadeedDiac--25-yellow.svg)](https://huggingface.co/datasets/Misraj/SadeedDiac-25)
[![Corpus](https://img.shields.io/badge/Corpus-Sadeed__Tashkeela-yellow.svg)](https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela)
[![Gemma adapters](https://img.shields.io/badge/%F0%9F%A4%97%20Adapters-gemma--4--e4b--lora-orange.svg)](https://huggingface.co/Psycodem/gemma-4-e4b-lora-diacritization)
[![Qwen adapters](https://img.shields.io/badge/%F0%9F%A4%97%20Adapters-qwen3.5--4b--lora-orange.svg)](https://huggingface.co/Psycodem/qwen3.5-4b-lora-diacritization)

<!-- To add a banner: drag an image into a GitHub issue/PR comment, copy the
     generated user-attachments URL, and paste it into the <img> tag below.
<img width="1188" alt="DiacriticS" src="PASTE_UPLOADED_IMAGE_URL_HERE" />
-->
 
 [**LoRA config notes**](Trained_Models/LoRA_Fine_Tuning_Config_Comparison.md)

</div>

## Highlights

- **Contamination-controlled by construction** — trained on Sadeed-Tashkeela, evaluated on SadeedDiac-25, with only **0.4% overlap** between the training corpus and the Fadel test set it is scored against
- **12 open-weights models benchmarked zero-shot**, scored on the mean of the Modern Standard and Classical Arabic halves of the benchmark
- **LoRA under an explicit fairness contract** that holds every optimisation-relevant hyperparameter identical across both 4B models — plus one full fine-tune of Qwen3.5-0.8B, so the adaptation regime itself can be compared with the base model, corpus and prompt held constant
- **4-point data-scaling curve** (zero-shot, then 10/30/50%) over *nested* subsets — the 10% slice sits inside the 30% and that inside the 50% — so the points form a curve rather than four unrelated samples
- **Corrected DER/WER scorer** — NFC normalisation, letter-anchored mark comparison, reference-fixed denominator, hallucinated words counted, dagger alef included
- **Bilingual project site** (English / Arabic)

## Introduction

Arabic Text Diacritisation (ATD) — restoring the short vowels and marks (*tashkeel*) that are almost always omitted from written Arabic — is a foundational preprocessing step for Text-to-Speech, Machine Translation, and Information Extraction. Without diacritics, one skeleton of letters maps to several unrelated words. The root **ع ل م** alone can read as:

| Word | Meaning |
|---|---|
| عَلَمٌ | a flag |
| عِلْمٌ | knowledge |
| عَلَّمَ | he taught |

The harder problem is *measuring* diacritisation honestly. Existing benchmarks suffer from **data contamination** — structural train/test overlap, notably between the Abbad and Fadel splits, both drawn from Tashkeela — and from **register bias** toward either Classical Arabic (CA) or Modern Standard Arabic (MSA). Both inflate reported accuracy.

**DiacriticS** builds the pipeline around that problem: a cleaned training corpus with verified minimal overlap against a balanced CA/MSA benchmark, used first to score open-weights models cold, then to fine-tune them with LoRA at several data scales.

<details>
<summary>The two datasets, precisely</summary>

**Training — [`Misraj/Sadeed_Tashkeela`](https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela)**
The cleaned split of the Tashkeela corpus. Diacritisation style is unified (sukūn dropped on elongation letters and on the definite article's lām before a sun letter; frequent stop words corrected; *iltiqāʾ as-sākinayn* resolved by phonological rule), while non-Arabic characters and symbols are deliberately preserved. Text is chunked into 50–60-word samples and filtered so few carry more than two undiacritised words. Every example overlapping the Fadel test set was removed, cutting overlap to **0.4%** (at most two shared words per sample). Result: **1,042,698 examples, ~53M words**.

**Benchmark — [`Misraj/SadeedDiac-25`](https://huggingface.co/datasets/Misraj/SadeedDiac-25)**
1,200 paragraphs, split 50/50 by register. The MSA half is 600 paragraphs of 40–50 words (454 curated web articles across sports, politics, religion and cooking, plus 146 from WikiNews), auto-diacritised then corrected by two independent experts who each re-reviewed the other's work. The CA half is 600 paragraphs from the Fadel test set. The modern half was diacritised in-house and never published in diacritised form, so no existing model had read it.

Both are introduced in Aldallal et al., *Sadeed: Advancing Arabic Diacritization Through Small Language Model* ([arXiv:2504.21635](https://arxiv.org/abs/2504.21635)).

</details>

### Method

The pipeline enforces a strict wall between what the model learns from and what it is judged on:

1. **Train on Sadeed-Tashkeela** — the cleaned split, with the 0.4% train/test wall built in.
2. **Test on SadeedDiac-25** — MSA and CA in one benchmark, 1,200 expert-reviewed paragraphs.
3. **Benchmark zero-shot, across scales** — score open-weights models cold, before any fine-tuning, for an honest baseline.
4. **Fine-tune with LoRA, and scale the data** — LoRA on a bf16 base (no QLoRA), repeated over 10%, 30% and 50% of the corpus.
5. **Score, then break the score apart** — DER and WER with and without sentence-final case endings, compared across CA vs. MSA to isolate where the errors come from.

```
DER = (Incorrect diacritics / Total characters carrying diacritics) × 100
WER = (Words with ≥1 diacritic error / Total words) × 100
```

Both metrics are reported with and without Case Endings (CE).

## Installation

### Requirements
- Python ≥ 3.9, CUDA GPU recommended (the fine-tuning runs assume 3× A100 80GB).
- A Hugging Face account and `HF_TOKEN` for the gated models and datasets.
- Pinned dependencies: [`Requirements.txt`](Requirements.txt). The pins are
  deliberate — `torch 2.10` + `transformers 5.15` cannot run `gemma-4-E4B-it` on
  CUDA at all, and `transformers < 5.15` does not recognise the `gemma4`
  architecture. Each pin is annotated with the failure it prevents.

### Setup

```bash
# Clone the repository
git clone https://github.com/Psycodem/DiacriticS-Language-Model-for-Arabic-Diacritisation.git
cd DiacriticS-Language-Model-for-Arabic-Diacritisation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (pinned — see the comments in the file)
pip install -r Requirements.txt

# Hugging Face token (needed for gated models / datasets)
export HF_TOKEN="hf_..."   # On Windows PowerShell: $env:HF_TOKEN="hf_..."
```

<details>
<summary>Repository layout</summary>

```
Evaluation_Functions_Corrected.py    # the scorer — every number in this repo uses it
Evaluation_Scorer_Comparison.py      # runnable diff vs the benchmark's own evaluator
Requirements.txt                     # pinned deps, annotated with what each pin prevents
Reports/
  DiacriticS_Final_Report.pdf        # the full write-up, also served from the site
Tested_Models/
  Tested Models Results.xlsx         # zero-shot results (MSA, CA, and mean)
  ibex_config/                       # the one benchmark that ran on Slurm
    Gemma_Qwen_test.py  run_gemma_qwen_test.sbatch
  modal_config/                      # the rest ran on Modal; these produced the numbers
    modal_small_models.py            # Fine-Tashkeel, Tashkeel-350M-v2
    modal_large_models.py            # aya-expanse-8b, Moonlight-16B-A3B-Instruct
    modal_fanar.py                   # Fanar-1-9B-Instruct
    modal_flan_byt5.py               # Flan-T5-Tashkeel-Small, Glonor-ByT5-Arabic
    modal_gemma3_1b.py               # gemma-3-1b-pt-10k-diacritization
    modal_qwen35_08b_plain.py        # Qwen3.5-0.8B, stock zero-shot baseline
Trained_Models/
  LoRA_Fine_Tuning_Config_Comparison.md   # the fairness contract, in full
  Gemma_4_LoRA_Diacritization/
    Train_LoRA_Gemma_4_Diacritization.py
    ibex_config/                     # 30pct / 50pct Slurm jobs
    modal_config/                    # the 10pct run
    HUGGINGFACE.md                   # where the published adapters live
    10pct/ 30pct/ 50pct/
      adapter/                       # config + model card (weights are on the Hub)
      outputs/                       # predictions, metrics, training log, loss curve
  Qwen_3_5_4B_LoRA_Diacritization/   # same shape
  Qwen_3_5_0_8B_Full_FineTune/       # full fine-tune: notebook + outputs (weights pending)
DiacriticS_Website/                  # Astro site with the live demo (deployed on Vercel)
  src/  public/  astro.config.mjs
```

Adapter weights are **not** in git — they are on the Hugging Face Hub (see below).
Everything needed to reproduce or audit a run — config, per-split predictions,
training log, model card — is tracked.

</details>

## Models on the Hub

| Repo | Contents | Load with |
|---|---|---|
| [`Psycodem/gemma-4-e4b-lora-diacritization`](https://huggingface.co/Psycodem/gemma-4-e4b-lora-diacritization) | LoRA adapters at 10%, 30%, 50% | `subfolder="50pct"` |
| [`Psycodem/qwen3.5-4b-lora-diacritization`](https://huggingface.co/Psycodem/qwen3.5-4b-lora-diacritization) | LoRA adapters at 10%, 30%, 50% | `subfolder="50pct"` |

```python
from peft import PeftModel
model = PeftModel.from_pretrained(
    model, "Psycodem/gemma-4-e4b-lora-diacritization", subfolder="50pct")
```

Each subfolder carries its own model card with that run's scores and
hyperparameters. The full fine-tune of Qwen3.5-0.8B is documented under
[`Trained_Models/Qwen_3_5_0_8B_Full_FineTune/`](Trained_Models/Qwen_3_5_0_8B_Full_FineTune/);
its weights have not yet been retrieved from the training cluster.

**Datasets** — [`Misraj/Sadeed_Tashkeela`](https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela)
for training, [`Misraj/SadeedDiac-25`](https://huggingface.co/datasets/Misraj/SadeedDiac-25)
for evaluation.

## Quick Start

### A) Benchmark open-weights models zero-shot

Each script loads `Misraj/SadeedDiac-25` (`split="train"`, 1,200 paragraphs), strips the diacritics to build the prompt, generates, and scores DER/WER with and without case endings.

Most of them run on [Modal](https://modal.com) — one GPU, checkpointed per
prediction so an interrupted run resumes rather than restarts:

```bash
modal run Tested_Models/modal_config/modal_small_models.py --probe 5 --wait   # smoke test
modal run --detach Tested_Models/modal_config/modal_small_models.py           # full 1,200
```

Swap in `modal_large_models.py`, `modal_fanar.py`, `modal_flan_byt5.py`,
`modal_gemma3_1b.py` or `modal_qwen35_08b_plain.py` for the other models. Use
`--detach` for real runs: the entrypoint spawns the job and returns, so killing
the client cannot cancel the work.

On KAUST's Ibex cluster:

```bash
sbatch Tested_Models/ibex_config/run_gemma_qwen_test.sbatch
```

### B) Score your own predictions

The corrected scorer is a drop-in replacement for the original — same function names, same return keys.

```python
from Evaluation_Functions_Corrected import compute_der_wer

metrics = compute_der_wer(predictions, references)   # both: list[str]

print(metrics["DER_ce"], metrics["WER_ce"])      # with sentence-final case endings
print(metrics["DER_noce"], metrics["WER_noce"])  # without
```

<details>
<summary>What the corrected scorer fixes</summary>

1. **No NFC normalisation** — shadda + fatha can be stored in either order and render identically; a positional comparison scored the two orders as different (worth ~2.4× on identical predictions).
2. **Marks compared without their host letter** — a flat per-word mark list makes "right mark, wrong letter" compare equal; with only ~8 distinct marks, those collisions are common.
3. **Moving DER denominator** — it was `max(len(ref_marks), len(pred_marks))`, so it moved with the prediction instead of being fixed by the reference.
4. **Hallucinated words scored as free** — inserted words with no reference counterpart were skipped by alignment, hiding the failure mode most worth catching in a generative model.
5. **Dagger alef (U+0670) missing** from the mark class, so a missing dagger alef was invisible to scoring.

</details>

### C) Reproduce a LoRA fine-tuning run

Both training scripts read their configuration from environment variables. `DIAC_TRAIN_FRACTION` picks the data-scaling point; `DIAC_RUN_TAG` keeps the four runs of one model from overwriting each other's adapter and metrics.

```bash
export DIAC_TRAIN_FRACTION=0.1     # 0.1 | 0.3 | 0.5 | 1.0 — subsets are nested (seed 42)
export DIAC_RUN_TAG=10pct

torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  Trained_Models/Gemma_4_LoRA_Diacritization/Train_LoRA_Gemma_4_Diacritization.py
```

On Ibex, the exact job files used for the published runs are checked in:

```bash
sbatch Trained_Models/Gemma_4_LoRA_Diacritization/ibex_config/run_gemma_lora_10pct.sbatch
sbatch Trained_Models/Qwen_3_5_4B_LoRA_Diacritization/ibex_config/run_qwen_lora_10pct.sbatch
```

Results land in `eval_outputs-<tag>/metrics_summary.csv`.

<details>
<summary>The fairness contract (why the two models are comparable)</summary>

Every setting that affects *what the optimizer sees and how it updates the model* is held identical across both scripts; only memory-fitting settings differ, with the downstream quantity forced equal.

| Setting | Value |
|---|---|
| Method | LoRA on a bf16 base (`USE_4BIT=False`) — never QLoRA |
| LoRA `r` / `alpha` / `dropout` | 16 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Effective batch | **96**, global and post-accumulation, for both models |
| Epochs | 1 full epoch over the training split |
| Learning rate | `2e-4`, cosine schedule, warmup = **5% of total steps** |
| Max sequence length | 1024 tokens |
| Seed | 42 |
| Decoding | Greedy, `max_new_tokens=512` |
| Hardware | 3× A100 80GB, DDP via `torchrun` — Ibex (Slurm) for the 30% and 50% runs, Modal for the 10% runs |

Only `PER_DEVICE_TRAIN_BATCH_SIZE` differs (Gemma 4, Qwen 8); gradient accumulation is derived at runtime so the effective batch stays 96, and the script exits rather than run if it cannot. Full rationale: [`Trained_Models/LoRA_Fine_Tuning_Config_Comparison.md`](Trained_Models/LoRA_Fine_Tuning_Config_Comparison.md).

</details>

### D) Run inference with a fine-tuned LoRA adapter

The training runs save a LoRA **adapter** (~165 MB), not a full model. Inference
loads the original base model from the Hub and applies the adapter on top:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE    = "google/gemma-4-E4B-it"        # or "Qwen/Qwen3.5-4B"
ADAPTER = "gemma-4-e4b-lora-diacritization-10pct/final_adapter"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()    # fold the adapter in — faster generation
model.eval()
```

> `merge_and_unload()` is optional. Skip it to keep the adapter detachable
> (useful for comparing several adapters against one base); keep it for speed.

**The prompt must match training exactly**, or quality drops sharply for reasons
that look like a bad model. Both scripts use this system prompt:

```python
SYSTEM_PROMPT = (
    "أنت نظام متخصص في التشكيل الآلي للنصوص العربية. "
    "مهمتك إضافة الحركات (التشكيل) الصحيحة إلى النص العربي المُدخل دون تغيير الكلمات أو ترتيبها، "
    "مع مراعاة السياق النحوي والصرفي الكامل للجملة."
)

def build_messages(text):
    # Both gemma-4-E4B-it and Qwen3.5-4B accept a system role. If you swap in a
    # model whose chat template rejects one (Gemma 3 does), fold the system
    # prompt into the user turn instead:
    #     f"{SYSTEM_PROMPT}\n\n{text}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text}]

def diacritize(text):
    msgs = build_messages(text)
    try:
        # Qwen3.5 is a reasoning model: without enable_thinking=False it emits a
        # <think> block that would be scored as if it were the diacritised text.
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    import re
    gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL)   # closed block
    if re.match(r"^\s*<think>", gen, flags=re.DOTALL):
        return ""                                                   # truncated mid-thought
    return gen.strip().strip('"“”').strip()

print(diacritize("ذهب الطالب إلى المدرسة"))
```

Three details that matter, each of which cost a run when it was missing:

- **Greedy decoding** (`do_sample=False`). The published numbers assume it;
  sampling makes results non-reproducible.
- **Strip `<think>` blocks** for Qwen. An *unclosed* block means generation hit
  `max_new_tokens` mid-reasoning — return an empty prediction rather than
  scoring reasoning text as if it were the answer.
- **Left padding** if you batch: `tok.padding_side = "left"`. Right padding
  silently corrupts batched generation.

To reproduce the published scores rather than run ad-hoc inference, use the
scoring functions in
[`Evaluation_Functions_Corrected.py`](Evaluation_Functions_Corrected.py) — see
section **B**.

### E) Run the project site locally

The site is an Astro app with a server-side inference proxy, so it needs Node rather than a static file server:

```bash
cd DiacriticS_Website && npm install && npm run dev
```

It serves on `http://localhost:4321` with English at `/` and Arabic at `/ar/`. The live demo at `/demo/` calls the same-origin `/api/diacritize` route, which forwards to a GPU endpoint; copy `.env.example` to `.env` and fill in `DIACRITICS_GPU_ENDPOINT` and `DIACRITICS_GPU_TOKEN` to enable it. Neither may carry a `PUBLIC_` prefix — Astro inlines `PUBLIC_` variables into the client bundle, which would publish the credential. Without them the rest of the site still runs; only the demo is inert.

## Report

The full write-up — methodology, the evaluation-protocol analysis, and the qualitative error breakdown — is at [`Reports/DiacriticS_Final_Report.pdf`](Reports/DiacriticS_Final_Report.pdf), and is linked from the [project site](https://diacritics.vercel.app/).

## Performance

All numbers below are produced by [`Evaluation_Functions_Corrected.py`](Evaluation_Functions_Corrected.py),
applied identically to every model and every split. Percentages; lower is better.
CE = case ending (*i'rab*).

### Zero-shot on SadeedDiac-25

Mean of the MSA and CA halves (600 paragraphs each), sorted by DER with case endings.

| Model | DER w/ CE (%) | WER w/ CE (%) | DER w/o CE (%) | WER w/o CE (%) |
|---|---|---|---|---|
| **Flan-T5-Tashkeel-Small** | **4.10** | **11.86** | **3.61** | **7.88** |
| gemma-4-E4B-it | 4.18 | 9.96 | 3.80 | 6.78 |
| Tashkeel-350M-v2 | 6.33 | 15.28 | 5.64 | 10.40 |
| Fine-Tashkeel | 17.91 | 22.61 | 17.73 | 20.35 |
| Fanar-1-9B-Instruct | 26.14 | 31.48 | 26.35 | 28.81 |
| Glonor-ByT5-Arabic | 33.62 | 50.68 | 33.22 | 44.77 |
| Qwen3.5-9B | 39.78 | 59.44 | 41.92 | 55.69 |
| gemma-3-1b-pt-10k-diacritization | 50.25 | 55.81 | 50.66 | 52.95 |
| Qwen3.5-4B | 58.29 | 79.00 | 59.43 | 75.22 |
| aya-expanse-8b | 66.55 | 85.78 | 67.08 | 84.04 |
| Moonlight-16B-A3B-Instruct | 86.60 | 99.01 | 87.10 | 98.84 |
| Qwen3.5-0.8B | 92.18 | 99.67 | 92.41 | 99.62 |

Per-register numbers (MSA and CA scored separately) are in
[`Tested_Models/Tested Models Results.xlsx`](Tested_Models/). The scripts that
produced them are in [`Tested_Models/`](Tested_Models/) — `modal_config/` for the
Modal runs, `ibex_config/` for the Slurm one.

### After LoRA fine-tuning

Both targets adapted under identical conditions (rank 16, alpha 32, effective
batch 96, one epoch, 5% warmup ratio, seed 42) on nested subsets of
`Misraj/Sadeed_Tashkeela` drawn with a common shuffle seed, so the 10% subset is
contained in the 30% and that in the 50%. Scored on the full 1,200-paragraph
SadeedDiac-25 benchmark.

| Base model | Training data | DER w/ CE (%) | WER w/ CE (%) | DER w/o CE (%) | WER w/o CE (%) |
|---|---|---|---|---|---|
| gemma-4-E4B-it | zero-shot | 4.18 | 9.96 | 3.80 | 6.78 |
| gemma-4-E4B-it | 10% (104,270 rows) | 3.15 | 7.28 | 2.68 | 4.47 |
| gemma-4-E4B-it | 30% (312,809 rows) | 2.98 | 6.82 | 2.54 | 4.12 |
| **gemma-4-E4B-it** | **50% (521,349 rows)** | **2.81** | **6.54** | **2.38** | **3.96** |
| Qwen3.5-4B | zero-shot | 58.29 | 79.00 | 59.43 | 75.22 |
| Qwen3.5-4B | 10% (104,270 rows) | 7.07 | 11.74 | 6.94 | 8.74 |
| Qwen3.5-4B | 30% (312,809 rows) | 5.27 | 9.28 | 5.03 | 6.54 |
| Qwen3.5-4B | 50% (521,349 rows) | 4.38 | 8.18 | 4.07 | 5.57 |

Adapters, per-split predictions and training logs for every row are under
[`Trained_Models/`](Trained_Models/), and the adapters are published on the
Hugging Face Hub:
[`Psycodem/gemma-4-e4b-lora-diacritization`](https://huggingface.co/Psycodem/gemma-4-e4b-lora-diacritization)
and
[`Psycodem/qwen3.5-4b-lora-diacritization`](https://huggingface.co/Psycodem/qwen3.5-4b-lora-diacritization),
with each training fraction as a subfolder:

```python
model = PeftModel.from_pretrained(
    model, "Psycodem/gemma-4-e4b-lora-diacritization", subfolder="50pct")
```

<details>
<summary>Published reference points on the same benchmark</summary>

Reported by Aldallal et al. (Sadeed, Table 8) on SadeedDiac-25 — **not re-run
here, and not directly comparable**: they were produced with the benchmark's own
reference evaluator, which discards any paragraph where the model changed the
word count (15.4% of the benchmark for one of our fine-tuned models) and is
sensitive to the Unicode ordering of combining marks. See
[`Evaluation_Scorer_Comparison.py`](Evaluation_Scorer_Comparison.py) for a
runnable demonstration. Listed for orientation only:

| Model | DER w/ CE (%) | WER w/ CE (%) | DER w/o CE (%) | WER w/o CE (%) |
|---|---|---|---|---|
| Claude-3-7-Sonnet-Latest | 1.39 | 4.67 | 0.77 | 2.31 |
| gemini-flash-2.0 | 3.19 | 7.99 | 2.38 | 5.50 |
| GPT-4 | 3.86 | 5.27 | 3.86 | 10.93 |
| Sadeed | 7.29 | 13.74 | 5.26 | 9.92 |
| aya-23-8B | 25.63 | 47.49 | 19.76 | 40.25 |
| ALLaM-7B-Instruct | 50.36 | 70.34 | 39.41 | 67.09 |
| Yehia-7B | 50.88 | 70.23 | 39.77 | 67.15 |
| jais-13B | 78.68 | 99.75 | 60.73 | 99.57 |
| gemma-2-9b | 78.86 | 99.79 | 60.92 | 99.59 |
| SILMA-9B-Instruct-v1.0 | 78.66 | 99.74 | 60.71 | 99.56 |

</details>

### Key findings

1. **The evaluation protocol is itself a confound.** The benchmark's reference
   evaluator drops any paragraph in which the model altered the word count, which
   removes exactly the failures that separate a fluent generator from a faithful
   diacritiser — 15.4% of the benchmark for our Qwen3.5-4B 50% model. It also
   treats two Unicode orderings of shadda + vowel as different, penalising text
   that NFC declares identical. Scores from different harnesses cannot be
   compared without reconciling them first.
2. **A weak starting point gains enormously from LoRA; a strong one has little
   room.** Qwen3.5-4B falls from 58.29% to 4.38% DER, most of it purchased by the
   first tenth of the corpus. gemma-4-E4B-it starts at 4.18% — already below
   where Qwen finishes — and reaches 2.81%. Gemma leads at every fraction, but
   the gap narrows from 54.1 points to 1.6.
3. **Sentence endings are the weak point.** Every model's error rate jumps on the
   sentence-final, case-marking diacritic (*i'rab*) compared with diacritics
   inside a word — these models resolve local spelling far better than
   sentence-level syntax.
4. **Classical Arabic is the harder domain.** Error rates rise markedly moving
   from MSA passages to Classical Arabic, showing that domain-balanced training —
   not just more data — is what closes the gap.

## References

1. Z. Aldallal, S. Chrouf, K. Hennara, M. M. Hamed, M. Hreden & S. AlModhayan, *"Sadeed: Advancing Arabic Diacritization Through Small Language Model"*, [arXiv:2504.21635](https://arxiv.org/abs/2504.21635), 2025.
2. A. Abbad et al., *"Character-based Arabic Tashkeel Transformer (CATT)"*, [github.com/abjadai/catt](https://github.com/abjadai/catt), 2023.
3. *"Arabic Text Diacritization In The Age Of Transfer Learning: Token Classification Is All You Need"*, [arXiv:2401.04848](https://arxiv.org/abs/2401.04848), 2024.
4. *"AraXLM: Evaluating Arabic Diacritization Tools for Cross-Language Plagiarism Detection"*, [annals-csis.org](https://annals-csis.org/Volume_43/drp/4862.html), 2024.
5. *"More Data, Fewer Diacritics: Scaling Arabic TTS"*, 2024.
6. M. Cherradi & H. El Mahajer, *"Arabic Text Diacritization Using Deep Neural Networks and Transformer-Based Architectures"*, Knowledge and Decision Systems with Applications, 2025.

---

<div align="center">

<!-- To show logos here, upload them to a GitHub issue/PR comment and paste the
     user-attachments URLs into the <img> tags below.
  <a href="https://academy.kaust.edu.sa/"><img src="PASTE_KAUST_ACADEMY_LOGO_URL" alt="KAUST Academy" height="150" width="150" /></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.kaust.edu.sa/"><img src="PASTE_KAUST_LOGO_URL" alt="KAUST" height="150" width="150" /></a>
-->

[**KAUST Academy**](https://academy.kaust.edu.sa/) &nbsp;·&nbsp; [**KAUST**](https://www.kaust.edu.sa/) &nbsp;·&nbsp; [**Project site**](https://diacritics.vercel.app/)

</div>
