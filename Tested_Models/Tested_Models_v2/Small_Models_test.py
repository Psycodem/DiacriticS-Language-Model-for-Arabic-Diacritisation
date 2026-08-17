# -*- coding: utf-8 -*-
"""
SMALL Models (< 500M Parameters)
Models: Fine-Tashkeel, Tashkeel-350M-v2
"""

import subprocess
import sys

packages = [
    "transformers",
    "datasets",
    "torch",
    "accelerate",
    "jiwer",
    "tqdm",
    "ipywidgets",
    "pandas",
]

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

import re
import sys
import gc
import unicodedata
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm.auto import tqdm
import jiwer

SAMPLE_SIZE = 600  # 600 CA + 600 MSA = 1200 samples total
CACHE_DIR = "./model_cache"  # directory where downloaded models/tokenizers are cached
skipped_models = []

"""# 2: Normalization & Evaluation Engine (CORRECTED scorer, copied from Evaluation_Functions_Corrected.py)

Replaces the project's old inline scorer, which had five confirmed defects:
no NFC normalization, diacritics compared as a flat per-word list with the
host letter discarded, a DER denominator that moved with the prediction
instead of being fixed by the reference, hallucinated/inserted words scored
as free, and dagger alef (U+0670) missing from the diacritic mark class. See
Evaluation_Functions_Corrected.py at the repo root for the full rationale.
"""

DIACRITICS = re.compile(r'[ً-ٰٟ]')
LETTER = re.compile(r'[ء-غف-يٱ-ۓ]')
TATWEEL = "ـ"

def strip_diacritics(text: str) -> str:
    """Consonantal skeleton: diacritics AND tatweel removed."""
    return DIACRITICS.sub('', text).replace(TATWEEL, '')

def clean_and_tokenize(text: str) -> list:
    """NFC-normalizes, strips tatweel, tokenizes."""
    text = unicodedata.normalize("NFC", text).replace(TATWEEL, '')
    cleaned = re.sub(r'[^\w\sً-ٰٟ]', '', text)
    return cleaned.split()

def segment_word(word: str) -> list:
    """Splits a word into [(base_letter, marks), ...] pairs, marks sorted so
    canonically equivalent orderings compare equal -- the fix for host-letter
    tracking."""
    units = []
    for ch in unicodedata.normalize("NFC", word):
        if DIACRITICS.match(ch):
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(base, "".join(sorted(marks))) for base, marks in units]

def scorable_units(units: list, ce: bool) -> list:
    """Diacritizable units, dropping the word-final one when ce=False."""
    idx = [i for i, (base, _) in enumerate(units) if LETTER.match(base)]
    if not ce and idx:
        idx = idx[:-1]
    return [units[i] for i in idx]

def is_digit_only(word: str) -> bool:
    return strip_diacritics(word).isdigit()

def _align_words(predictions: list, references: list) -> list:
    """Word alignment on consonantal skeletons. Returns (ref_or_None, pred_or_None)
    pairs; hallucinated/inserted words come back as (None, pred) instead of being
    dropped."""
    pairs = []
    for pred, ref in zip(predictions, references):
        pred_words = [w for w in clean_and_tokenize(pred) if strip_diacritics(w)]
        ref_words = [w for w in clean_and_tokenize(ref) if strip_diacritics(w)]
        alignment = jiwer.process_words(
            " ".join(strip_diacritics(w) for w in ref_words),
            " ".join(strip_diacritics(w) for w in pred_words))
        for chunk in alignment.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    pairs.append((ref_words[i],
                                  pred_words[chunk.hyp_start_idx + (i - chunk.ref_start_idx)]))
            elif chunk.type == "substitute":
                pairs += [(ref_words[i], None) for i in range(chunk.ref_start_idx, chunk.ref_end_idx)]
                pairs += [(None, pred_words[j]) for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)]
            elif chunk.type == "delete":
                pairs += [(ref_words[i], None) for i in range(chunk.ref_start_idx, chunk.ref_end_idx)]
            elif chunk.type == "insert":
                pairs += [(None, pred_words[j]) for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)]
    return pairs

def _compute_der_from_pairs(pairs: list, ce: bool = True) -> float:
    total, wrong = 0, 0
    for r_word, p_word in pairs:
        if r_word is None:  # hallucinated / inserted word
            if p_word is None or is_digit_only(p_word):
                continue
            n = len(scorable_units(segment_word(p_word), ce))
            total += n
            wrong += n
            continue
        if is_digit_only(r_word):
            continue
        r_units = scorable_units(segment_word(r_word), ce)
        total += len(r_units)
        if p_word is None:
            wrong += len(r_units)
            continue
        p_units = scorable_units(segment_word(p_word), ce)
        for i, (r_base, r_marks) in enumerate(r_units):
            if i >= len(p_units) or p_units[i] != (r_base, r_marks):
                wrong += 1
    return round(wrong / total * 100, 2) if total else 0.0

def _compute_wer_from_pairs(pairs: list, ce: bool = True) -> float:
    total, wrong = 0, 0
    for r_word, p_word in pairs:
        total += 1
        if r_word is None or p_word is None:
            wrong += 1
            continue
        if scorable_units(segment_word(r_word), ce) != scorable_units(segment_word(p_word), ce):
            wrong += 1
    return round(wrong / total * 100, 2) if total else 0.0

def compute_der(predictions: list, references: list, ce: bool = True) -> float:
    return _compute_der_from_pairs(_align_words(predictions, references), ce=ce)

def compute_wer(predictions: list, references: list, ce: bool = True) -> float:
    return _compute_wer_from_pairs(_align_words(predictions, references), ce=ce)

def compute_der_wer(predictions: list, references: list) -> dict:
    pairs = _align_words(predictions, references)
    return {
        "DER_ce": _compute_der_from_pairs(pairs, ce=True),
        "DER_noce": _compute_der_from_pairs(pairs, ce=False),
        "WER_ce": _compute_wer_from_pairs(pairs, ce=True),
        "WER_noce": _compute_wer_from_pairs(pairs, ce=False),
    }

"""# 3: Dataset Ingestion"""

print("Loading SadeedDiac-25 Dataset...")
try:
    dataset = load_dataset("Misraj/SadeedDiac-25", split="train")
except Exception as e:
    print(f"FATAL: could not load dataset: {e}")
    dataset = []

msa_samples, ca_samples = [], []

for item in dataset:
    try:
        text_input = item.get("input", "")
        text_output = item.get("output", "")
        source = item.get("filename", "").lower()
        sample = {"ground_truth": text_output, "raw_input": strip_diacritics(text_input)}
        if "fadel" in source or source in ["religion", "classical_poetry", "hadith"]:
            ca_samples.append(sample)
        else:
            msa_samples.append(sample)
    except Exception as e:
        print(f"Skipping malformed dataset row: {e}")
        continue

msa_test_set = msa_samples[:SAMPLE_SIZE]
ca_test_set = ca_samples[:SAMPLE_SIZE]

"""# 4: Inference Pipelines"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Fine-Tashkeel (seq2seq) ----------
def load_seq2seq(model_id, desc_tag):
    print(f"Loading Seq2Seq: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=CACHE_DIR).to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        skipped_models.append((desc_tag, str(e)))
        return None, None

def infer_seq2seq(model, tokenizer, test_dataset, desc_tag):
    predictions, references = [], [x["ground_truth"] for x in test_dataset]
    for item in tqdm(test_dataset, desc=f"Inference {desc_tag}"):
        try:
            inputs = tokenizer(item["raw_input"], return_tensors="pt",
                                max_length=1024, truncation=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=1024)
            predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        except Exception as e:
            print(f"  [warn] {desc_tag}: inference failed on a sample, using empty prediction ({e})")
            predictions.append("")
    return compute_der_wer(predictions, references)


# ---------- Tashkeel-350M-v2 (causal, chat-template based) ----------
def load_causal_chat(model_id, desc_tag):
    print(f"Loading CausalLM (chat template): {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="bfloat16",
            cache_dir=CACHE_DIR,
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        skipped_models.append((desc_tag, str(e)))
        return None, None

def infer_causal_chat(model, tokenizer, test_dataset, desc_tag):
    predictions, references = [], [x["ground_truth"] for x in test_dataset]
    for item in tqdm(test_dataset, desc=f"Inference {desc_tag}"):
        try:
            model_inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": "قم بتشكيل هذا النص" + ":\n" + item["raw_input"]}],
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            ).to(model.device)
            with torch.no_grad():
                # NOTE: max_new_tokens must be set explicitly -- without it, generate()
                # falls back to a ~20-token default which truncates diacritized output.
                output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
            input_len = model_inputs["input_ids"].shape[-1]
            pred_text = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
            predictions.append(pred_text.strip())
        except Exception as e:
            print(f"  [warn] {desc_tag}: inference failed on a sample, using empty prediction ({e})")
            predictions.append("")
    return compute_der_wer(predictions, references)

"""# 5: Model Execution Registry"""

model_registry = {
    "Fine-Tashkeel": {"repo": "basharalrfooh/Fine-Tashkeel", "type": "seq2seq"},
    "Tashkeel-350M-v2": {"repo": "Etherll/Tashkeel-350M-v2", "type": "causal_chat"},
}

domains = [("MSA (Modern)", msa_test_set), ("CA (Classical)", ca_test_set)]
benchmark_records = []

# Outer loop: models  ->  load once
# Inner loop: domains (MSA, then CA) -> run 600 + 600 = 1200 samples per model
# After a model finishes both domains, free the GPU before loading the next one.
for label, meta in model_registry.items():
    print(f"\n{'='*60}\n Loading model: {label}  ({meta['repo']})\n{'='*60}")

    try:
        if meta["type"] == "seq2seq":
            model, tokenizer = load_seq2seq(meta["repo"], label)
            infer_fn = infer_seq2seq
        elif meta["type"] == "causal_chat":
            model, tokenizer = load_causal_chat(meta["repo"], label)
            infer_fn = infer_causal_chat
        else:
            model, tokenizer = None, None
            infer_fn = None

        if model is None:
            print(f"Skipping {label}: model failed to load.\n")
            continue

        for domain, test_set in domains:
            metrics = infer_fn(model, tokenizer, test_set, label)
            if metrics:
                metrics.update({"Model": label, "Domain Track": domain})
                benchmark_records.append(metrics)
    except Exception as e:
        skipped_models.append((label, f"Unexpected failure: {e}"))
        print(f"Skipping {label}: unexpected failure ({e}).\n")
    finally:
        # --- Free the GPU before loading the next model, no matter what happened ---
        try:
            del model, tokenizer
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Finished {label} — GPU cache cleared.\n")

"""# 6: Results Report"""

performance_df = pd.DataFrame(benchmark_records)

if performance_df.empty:
    full_df = performance_df
    print("No models produced results — check the skip log below.")
else:
    # --- Mean row per model: average of MSA and CA (i.e. across domains) ---
    metric_cols = ["DER_ce", "DER_noce", "WER_ce", "WER_noce"]

    mean_df = (
        performance_df
        .groupby("Model")[metric_cols]
        .mean()
        .round(2)
        .reset_index()
    )
    mean_df.insert(1, "Domain Track", "Mean (MSA + CA)")

    full_df = pd.concat([performance_df, mean_df], ignore_index=True)
    full_df["Domain Track"] = pd.Categorical(
        full_df["Domain Track"],
        categories=["MSA (Modern)", "CA (Classical)", "Mean (MSA + CA)"],
        ordered=True,
    )
    full_df = full_df.sort_values(["Model", "Domain Track"]).reset_index(drop=True)

print("\n=== SMALL MODELS EVALUATION MATRIX (WITH MSA / CA MEAN ROW) ===")
print(full_df.to_markdown(index=False) if not full_df.empty else "No models processed.")

try:
    results_path = "small_models_results.csv"
    full_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
except Exception as e:
    print(f"\n[warn] Could not save results CSV: {e}")

if skipped_models:
    print("\n================ SKIPPED / FAILED MODELS (explicit) ================")
    for name, reason in skipped_models:
        print(f"- {name}: {reason}")
else:
    print("\nAll registered models ran successfully — no gaps in the table above.")
