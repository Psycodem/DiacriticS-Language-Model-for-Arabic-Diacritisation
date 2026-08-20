# DiacriticS: Fine-Tuning an Open Language Model for Arabic Diacritisation
**Authors:** <a href="https://mo-youssef.github.io/"><span>Youssef&nbsp;S.&nbsp;Mohamed</span></a> (Mentor, KAUST), 
<a href="https://linktr.ee/psycodem"><span>Mahdi&nbsp;Alkhamis</span></a> (Team Leader, IAU), 
<a href="https://www.linkedin.com/in/mohammad-alali-9b992135a/"><span>Mohammad&nbsp;Alali</span></a> (KFUPM), 
<a href="https://www.linkedin.com/in/saadtalnafjan/"><span>Saad&nbsp;Alnafjan</span></a> (KAU)

<div align="center">

[![Website](https://img.shields.io/badge/Website-diacritics.vercel.app-000000.svg?logo=vercel)](https://diacritics.vercel.app/)
[![Benchmark](https://img.shields.io/badge/Benchmark-SadeedDiac--25-yellow.svg)](https://huggingface.co/datasets/Misraj/SadeedDiac-25)
[![Corpus](https://img.shields.io/badge/Corpus-Sadeed__Tashkeela-yellow.svg)](https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela)

<!-- To add a banner: drag an image into a GitHub issue/PR comment, copy the
     generated user-attachments URL, and paste it into the <img> tag below.
<img width="1188" alt="DiacriticS" src="PASTE_UPLOADED_IMAGE_URL_HERE" />
-->
 
 [**LoRA config notes**](Trained_Models/LoRA_Fine_Tuning_Config_Comparison.md)

</div>

## Highlights

- **Contamination-controlled by construction** — trained on Sadeed-Tashkeela, evaluated on SadeedDiac-25, with only **0.4% overlap** between the training corpus and the Fadel test set it is scored against
- **11 open-weights models benchmarked zero-shot**, scored on the mean of the Modern Standard and Classical Arabic halves of the benchmark
- **LoRA only — no QLoRA, no full-weight retraining**, under an explicit fairness contract that holds every optimisation-relevant hyperparameter identical across models
- **4-point data-scaling curve** (10/30/50%) over *nested* subsets, so the points form a curve rather than four unrelated samples
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

### Setup

```bash
# Clone the repository
git clone https://github.com/Psycodem/DiacriticS-Language-Model-for-Arabic-Diacritisation.git
cd DiacriticS-Language-Model-for-Arabic-Diacritisation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers datasets accelerate peft jiwer tqdm pandas ipywidgets

# Hugging Face token (needed for gated models / datasets)
export HF_TOKEN="hf_..."   # On Windows PowerShell: $env:HF_TOKEN="hf_..."
```

<details>
<summary>Repository layout</summary>

```
Evaluation_Functions.py              # original DER/WER scorer (kept for reference)
Evaluation_Functions_Corrected.py    # corrected scorer — use this one
Tested_Models/
  Tested Models Results.xlsx         # raw benchmark results (MSA, CA, and mean)
  Tested_Models_v2/                  # current zero-shot benchmark scripts
    Small_Models_test.py             # < 500M: Fine-Tashkeel, Tashkeel-350M-v2
    Flan-T5-based_ByT5-based_test.ipynb
    Gemma_Qwen_test.py
    Large_Models_test.py
    Fanar_Model_test.py
    ibex_config/                     # SLURM job files for each test script
Trained_Models/
  LoRA_Fine_Tuning_Config_Comparison.md   # the fairness contract, in full
  Gemma_4_LoRA_Diacritization/
    Train_LoRA_Gemma_4_Diacritization.py
    ibex_config/                     # 10pct / 30pct / 50pct / full 3-GPU runs
  Qwen_3_5_4B_LoRA_Diacritization/
    Train_LoRA_Qwen_3_5_4B_Diacritization.py
    ibex_config/
DiacriticS_Website/                  # the bilingual project site (deployed on Vercel)
```

</details>

## Quick Start

### A) Benchmark open-weights models zero-shot

Each script loads `Misraj/SadeedDiac-25` (`split="train"`, 1,200 paragraphs), strips the diacritics to build the prompt, generates, and scores DER/WER with and without case endings.

```bash
python Tested_Models/Tested_Models_v2/Small_Models_test.py   # < 500M task-built models
python Tested_Models/Tested_Models_v2/Gemma_Qwen_test.py     # Gemma / Qwen instruction-tuned
python Tested_Models/Tested_Models_v2/Large_Models_test.py   # larger general LLMs
python Tested_Models/Tested_Models_v2/Fanar_Model_test.py    # Fanar-1-9B-Instruct
```

On KAUST's Ibex cluster, submit the matching job file instead:

```bash
sbatch Tested_Models/Tested_Models_v2/ibex_config/run_small_models_test.sbatch
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
| Learning rate | `2e-4`, cosine schedule, 300 warmup steps |
| Max sequence length | 1024 tokens |
| Seed | 42 |
| Decoding | Greedy, `max_new_tokens=512` |
| Hardware | Ibex, 1 node × 3× A100 80GB, DDP via `torchrun` |

Only `PER_DEVICE_TRAIN_BATCH_SIZE` differs (Gemma 4, Qwen 8); gradient accumulation is derived at runtime so the effective batch stays 96, and the script exits rather than run if it cannot. Full rationale: [`Trained_Models/LoRA_Fine_Tuning_Config_Comparison.md`](Trained_Models/LoRA_Fine_Tuning_Config_Comparison.md).

</details>

### D) Run the project site locally

```bash
python -m http.server 4173 --directory DiacriticS_Website
```

## Performance

Zero-shot on **SadeedDiac-25**, mean of the MSA and CA halves. Lower is better; sorted by DER with case endings.

| Model | DER w/ CE (%) | WER w/ CE (%) | DER w/o CE (%) | WER w/o CE (%) |
|---|---|---|---|---|
| **gemma-4-E4B-it** | **5.74** | **10.17** | **5.15** | **6.86** |
| Tashkeel-350M-v2 | 7.56 | 14.53 | 6.49 | 9.65 |
| Flan-T5-Tashkeel-Small | 11.52 | 19.94 | 10.56 | 14.12 |
| Fine-Tashkeel | 19.06 | 22.52 | 18.80 | 20.25 |
| Fanar-1-9B-Instruct | 24.98 | 29.58 | 24.72 | 26.78 |
| Glonor-ByT5-Arabic | 41.35 | 48.76 | 39.48 | 42.58 |
| Gemma-3-1B-pt-10k-diacritization | 41.58 | 47.98 | 42.26 | 44.26 |
| Qwen3.5-9B | 55.19 | 65.93 | 55.44 | 60.49 |
| Qwen3.5-4B | 73.74 | 81.86 | 73.49 | 77.25 |
| aya-expanse-8b | 92.35 | 92.78 | 92.02 | 91.85 |
| Moonlight-16B-A3B-Instruct | 99.38 | 98.68 | 99.31 | 98.50 |

Per-register numbers (MSA and CA scored separately) are in [`Tested_Models/Tested Models Results.xlsx`](Tested_Models/).

<details>
<summary>Published reference points on the same benchmark</summary>

Reported by Aldallal et al. (Sadeed, Table 8) on SadeedDiac-25 — not re-run here, listed for orientation:

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

1. **Sentence endings are the weak point.** Every model's error rate jumps on the sentence-final, case-marking diacritic (*i'rab*) compared to diacritics inside a word — these models resolve local spelling far better than sentence-level syntax.
2. **Classical Arabic is the harder domain.** Error rates rise markedly moving from MSA passages to Classical Arabic, showing that domain-balanced training — not just more data — is what closes the gap.

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
