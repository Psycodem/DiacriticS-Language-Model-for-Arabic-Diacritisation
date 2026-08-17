# -*- coding: utf-8 -*-
"""
Models: Fanar-1-9B-Instruct
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
    "pandas",
    "bitsandbytes",
]

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

import re, gc, unicodedata, torch, jiwer, pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

SAMPLE_SIZE = 600
CACHE_DIR = "./model_cache"  # directory where downloaded models/tokenizers are cached
skipped_models = []

if not torch.cuda.is_available():
    print("WARNING: no CUDA device detected. 4-bit bitsandbytes quantization requires a GPU; "
          "all large models will be skipped rather than crash the run.")

# Configure 4-Bit NormalFloat Quantization for high memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
)

"""# 2: Evaluation Suite (CORRECTED scorer \u2014 copied from Evaluation_Functions_Corrected.py)

Fixes five defects present in the original inline scorer this file used to carry:
no NFC normalization, diacritics compared as a flat per-word list with the host
letter discarded, a DER denominator that moved with the prediction instead of
being fixed by the reference, hallucinated/inserted words scored as free, and
dagger alef (U+0670) missing from the diacritic mark class. See
Evaluation_Functions_Corrected.py at the repo root for the full rationale.
"""

DIACRITICS = re.compile(r'[\u064B-\u0670\u065F]')
LETTER = re.compile(r'[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]')
TATWEEL = "\u0640"


def strip_diacritics(text: str) -> str:
    """Consonantal skeleton: diacritics AND tatweel removed."""
    return DIACRITICS.sub('', text).replace(TATWEEL, '')


def clean_and_tokenize(text: str) -> list:
    """NFC-normalizes, strips tatweel, tokenizes."""
    text = unicodedata.normalize("NFC", text).replace(TATWEEL, '')
    return re.sub(r'[^\w\s\u064B-\u0670\u065F]', '', text).split()


def segment_word(word: str) -> list:
    """Splits a word into [(base_letter, marks), ...] pairs, marks sorted so
    canonically equivalent orderings compare equal."""
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


def compute_der_wer(predictions: list, references: list) -> dict:
    """Returns all four headline metrics: DER/WER with & without case ending."""
    pairs = _align_words(predictions, references)
    return {
        "DER_ce": _compute_der_from_pairs(pairs, ce=True),
        "DER_noce": _compute_der_from_pairs(pairs, ce=False),
        "WER_ce": _compute_wer_from_pairs(pairs, ce=True),
        "WER_noce": _compute_wer_from_pairs(pairs, ce=False),
    }

"""
# 3: Data Parsing
"""

try:
    dataset = load_dataset("Misraj/SadeedDiac-25", split="train")
except Exception as e:
    print(f"FATAL: could not load dataset: {e}")
    dataset = []

msa_samples = [{"ground_truth": x["output"], "raw_input": strip_diacritics(x["input"])}
               for x in dataset if "fadel" not in x.get("filename","").lower()][:SAMPLE_SIZE]
ca_samples = [{"ground_truth": x["output"], "raw_input": strip_diacritics(x["input"])}
              for x in dataset if "fadel" in x.get("filename","").lower()][:SAMPLE_SIZE]

"""# 4: Heavy Model Runner"""

def load_large_model(model_id, label):
    print(f"Loading Quantized Heavy Model: {label} [{model_id}]...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        skipped_models.append((label, str(e)))
        print(f"Skipping {label}: {e}")
        return None, None

def infer_large_model(model, tokenizer, test_dataset, label):
    predictions, references = [], [x["ground_truth"] for x in test_dataset]

    for item in tqdm(test_dataset, desc=f"Evaluating {label}"):
        try:
            system_prompt = "قم بتشكيل النص العربي التالي تشكيلاً كاملاً ودقيقاً دون أي إضافة:"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["raw_input"]}
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{system_prompt}\n\nالنص: {item['raw_input']}\nالتشكيل:"

            # Use model.device rather than a hardcoded "cuda" -- with device_map="auto" the
            # model may be sharded, and model.device still resolves to the right input device.
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                # do_sample=False (greedy) makes temperature meaningless/contradictory, so it's omitted.
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            raw_pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            # Extract the line corresponding to the prediction output
            clean_pred = raw_pred.strip().split("\n")[0]
            predictions.append(clean_pred)
        except Exception as e:
            print(f"  [warn] {label}: inference failed on a sample, using empty prediction ({e})")
            predictions.append("")

    return compute_der_wer(predictions, references)

"""#5: Execution & Results"""

large_models = {
    "Fanar-1-9B-Instruct": "Fanar-1-9B-Instruct",
}

domains = [("MSA (Modern)", msa_samples), ("CA (Classical)", ca_samples)]
large_records = []

# Outer loop: models -> load once (these are 8B-16B, reloading per-domain would double
# download/VRAM cost). Inner loop: domains (MSA, then CA).
# GPU memory is freed after each model finishes both domains, before loading the next.
for label, path in large_models.items():
    print(f"\n{'='*60}\n Loading model: {label}  ({path})\n{'='*60}")
    model, tokenizer = None, None

    if not torch.cuda.is_available():
        skipped_models.append((label, "No CUDA device available for 4-bit quantized inference"))
        print(f"Skipping {label}: no CUDA device available.\n")
        continue

    try:
        model, tokenizer = load_large_model(path, label)
        if model is None:
            print(f"Skipping {label}: model failed to load.\n")
            continue

        for domain, test_set in domains:
            metrics = infer_large_model(model, tokenizer, test_set, label)
            if metrics:
                metrics.update({"Model": label, "Domain Track": domain})
                large_records.append(metrics)
    except Exception as e:
        skipped_models.append((label, f"Unexpected failure: {e}"))
        print(f"Skipping {label}: unexpected failure ({e}).\n")
    finally:
        # --- Free the GPU before loading the next (large!) model, no matter what happened ---
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Finished {label} — GPU cache cleared.\n")

"""# 6: Results Report"""

performance_df = pd.DataFrame(large_records)

if performance_df.empty:
    full_df = performance_df
    print("No models produced results \u2014 check the skip log below.")
else:
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

print("\n=== LARGE / MOE MODELS EVALUATION MATRIX ===")
print(full_df.to_markdown(index=False) if not full_df.empty else "No models processed.")

try:
    results_path = "large_models_results.csv"
    full_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
except Exception as e:
    print(f"\n[warn] Could not save results CSV: {e}")

if skipped_models:
    print("\n================ SKIPPED / FAILED MODELS (explicit) ================")
    for name, reason in skipped_models:
        print(f"- {name}: {reason}")
else:
    print("\nAll registered models ran successfully \u2014 no gaps in the table above.")