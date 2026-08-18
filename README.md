# DiacriticS

**Fine-Tuning a Language Model for Arabic Diacritisation on the SadeedDiac-25 Benchmark**

A KAUST Academy research project on contamination-controlled Arabic diacritisation (tashkeel).

🔗 Project site: [diacritics.vercel.app](https://diacritics.vercel.app/)

---

## Overview

Arabic Text Diacritisation (ATD) — restoring the short vowels and diacritical marks (tashkeel) that are almost always omitted from written Arabic — is a foundational preprocessing step for downstream NLP systems such as Text-to-Speech (TTS), Machine Translation (MT), and Information Extraction. Without diacritics, a single skeleton of letters can map to multiple unrelated words and pronunciations. For example, the root **ع ل م** can read as:

| Word | Meaning |
|---|---|
| عَلَمٌ | a flag |
| عِلْمٌ | knowledge |
| عَلَّمَ | he taught |

Existing ATD benchmarks frequently suffer from **data contamination** — structural train/test overlaps (e.g. between the Abbad and Fadel evaluation splits) — and **domain bias** toward either Classical Arabic (CA) or Modern Standard Arabic (MSA), which can artificially inflate reported performance.

**DiacriticS** addresses this by building a contamination-controlled benchmarking and fine-tuning pipeline: a clean, normalized training corpus with verified minimal overlap against a balanced CA/MSA test benchmark (SadeedDiac-25), used to evaluate and then fine-tune open-weights language models for robust Arabic diacritisation.

## Motivation

- Diacritisation directly affects downstream tasks like TTS synthesis, MT, and cross-lingual semantic alignment.
- Prior benchmarks leaked between train and test sets, inflating reported accuracy.
- Task-specific small language models often outperform general instruction-tuned LLMs in structural stability, motivating a focused fine-tuning approach rather than relying on large general-purpose models.

## Method

The pipeline enforces a strict wall between what the model learns from and what it is judged on, in five stages:

1. **The training corpus** — Use the **Sadeed Tashkeela** dataset, a cleaned and normalized variant of the public Tashkeela corpus. A multi-stage cleaning pipeline removes duplicate entries, corrects missing initial diacritics, standardizes Alif-Lam (shadda/sukun) markers, and eliminates non-standard character anomalies. Result: ~1 million diacritized examples (~53 million words).
2. **Adopt a contamination-controlled test set** — **SadeedDiac-25**: 1,200 human-reviewed paragraphs, split evenly between MSA and CA across news, sports, politics, religion, and literature domains, with only **0.4% overlap** against the training corpus.
3. **Benchmark zero-shot, across scales** — Score open-weights models cold, before any fine-tuning, to establish an honest baseline.
4. **Fine-tune with LoRA / QLoRA** — Adapt the strongest candidate model on the cleaned Tashkeela set using parameter-efficient fine-tuning rather than full-weight retraining.
5. **Score, then break the score apart** — Compute Diacritic and Word Error Rates with and without sentence-final case endings, then compare across CA vs. MSA subsets to isolate where errors come from.

### Evaluation metrics

```
DER = (Incorrect Diacritics / Total Characters carrying Diacritics) × 100
WER = (Words with at least one diacritic error / Total Words) × 100
```

Both metrics are reported with and without Case Endings (CE) calculations.

## Baseline Results (Zero-Shot on SadeedDiac-25)

| Model | WER w/ CE (%) | DER w/ CE (%) | WER w/o CE (%) | DER w/o CE (%) |
|---|---|---|---|---|
| Moonlight-16B-A3B-Instruct | 99.38 | 99.31 | 98.68 | 98.5 |
| aya-expanse-8b | 84.81 | 84.39 | 86.23 | 84.4 |
| Glonor-ByT5-Arabic | 41.35 | 39.48 | 48.76 | 42.58 |
| Fanar-1-9B-Instruct | 24.98 | 24.72 | 29.58 | 26.78 |
| Fine-Tashkeel | 19.06 | 18.8 | 22.52 | 20.25 |
| Flan-T5-Tashkeel-Small | 11.52 | 10.56 | 19.94 | 14.12 |
| Tashkeel-350M-v2 | 7.56 | 6.49 | 14.53 | 9.65 |
| **Sadeed** | **7.29** | 13.74 | **5.26** | **9.93** |

### Key findings

1. **Sentence endings are the weak point.** Every model's error rate jumps on the sentence-final, case-marking diacritic (i'rab) compared to diacritics inside a word — evidence that these models resolve local spelling far better than sentence-level syntax.
2. **Classical Arabic is the harder domain.** Error rates rise markedly moving from MSA passages to Classical Arabic (CA) texts, showing that domain-balanced training — not just more data — is needed to close the gap.

## Team

| Name | Role | Affiliation |
|---|---|---|
| [Youssef S. Mohamed](https://mo-youssef.github.io/) | Mentor | KAUST |
| [Mahdi Alkhamis](https://linktr.ee/psycodem) | AI/ML Student | IAU |
| [Mohammad Alali](https://www.linkedin.com/in/mohammad-alali-9b992135a/) | Software Engineering Student | KFUPM |
| [Saad Alnafjan](https://www.linkedin.com/in/saadtalnafjan/) | Electrical Engineering Student | KAU |

## References

1. Abbad et al., *"Character-based Arabic Tashkeel Transformer (CATT)"*, [github.com/abjadai/catt](https://github.com/abjadai/catt), 2023.
2. Author et al., *"Arabic Text Diacritization In The Age Of Transfer Learning: Token Classification Is All You Need"*, 2024.
3. Author et al., *"Sadeed: Advancing Arabic Diacritization Through Small Language Model"*, 2025.
4. Author et al., *"AraXLM: Evaluating Arabic Diacritization Tools for Cross-Language Plagiarism Detection"*, 2024.
5. Author et al., *"More Data, Fewer Diacritics: Scaling Arabic TTS"*, 2024.
6. M. Cherradi & H. El Mahajer, *"Arabic Text Diacritization Using Deep Neural Networks and Transformer-Based Architectures"*, Knowledge and Decision Systems with Applications, 2025.
