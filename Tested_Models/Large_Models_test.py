# -*- coding: utf-8 -*-
"""
Models tested:
- `CohereLabs/aya-expanse-8b` — causal LM, chat-template
- `moonshotai/Moonlight-16B-A3B-Instruct` — causal LM, chat-template, MoE, `trust_remote_code`
"""

import subprocess
import sys

# List of required packages
required_packages = [
    "datasets", "torch", "accelerate", "jiwer", 
    "tqdm", "ipywidgets", "pandas", "einops", 
    "huggingface_hub", "transformers==4.48.2"
]

def install_packages():
    print("Checking and installing required dependencies... Please wait.")
    # Calls the active Python interpreter to run pip safely
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + required_packages)

try:
    import transformers
    import datasets
except ImportError:
    install_packages()

# Your AI/Transformers code continues below safely
print("All dependencies are ready!")


"""
## 1. Hugging Face auth

`aya-expanse-8b` (Cohere, gated) and the Gemma-3-based diacritization model likely require you to accept the license on the model's Hugging Face page and authenticate here. `Moonlight-16B-A3B-Instruct` is usually open, but the token is wired through for all three regardless.

Paste your token below, or leave it as `""` to run unauthenticated (fine for open models; gated ones will show up in the skip log instead of crashing the whole run).
"""

HF_TOKEN = ""  # Hugging Face access token
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

"""## 2. Metrics & normalization — from `Models_Functions.py`

Defined directly here, unchanged except for the added `import jiwer`.
"""

import re
import jiwer  # added: required by _align_words()

# ── Existing diacritics regex ──────────────────────────────────────────
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652]')

# ── Numeral normalization ───────────────────────────────
_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_numerals(text: str) -> str:
    """Converts Eastern Arabic-Indic digits to Western Arabic digits."""
    return text.translate(_EASTERN_TO_WESTERN)

# ── Citation/reference stripping (Fadel-style "(41 / 251)") ──────
def strip_citation_refs(text: str) -> str:
    """Removes trailing parenthetical numeric citations like '(41 / 251)'."""
    text = re.sub(r'\(\s*\d+\s*/\s*\d+\s*\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def strip_diacritics(text: str) -> str:
    """Extracts raw consonantal skeleton text by eliminating vowels."""
    return ARABIC_DIACRITICS.sub('', text)

def clean_and_tokenize(text: str) -> list:
    """Normalizes numerals, strips citation refs, strips noise, tokenizes."""
    text = normalize_numerals(text)
    text = strip_citation_refs(text)
    cleaned = re.sub(r'[^\w\s\u064B-\u0652]', '', text)
    return cleaned.split()

def strip_case_ending(word: str) -> str:
    """
    Removes the diacritic(s) sitting on the word-final letter (i'rab /
    case ending), leaving any diacritics earlier in the word untouched.
    """
    last_base_idx = None
    for i, ch in enumerate(word):
        if not ARABIC_DIACRITICS.match(ch):
            last_base_idx = i
    if last_base_idx is None:
        return word
    return word[:last_base_idx + 1]

def _is_digit_only(word: str) -> bool:
    """True if the word (after stripping diacritics) is purely numeric —
    such words can never carry a diacritic, so they shouldn't be scored
    for DER."""
    skel = strip_diacritics(word)
    return skel.isdigit()

def _align_words(predictions: list, references: list):
    """
    Shared word-alignment step used by both DER and WER below, so a
    dropped/inserted word doesn't cascade into false mismatches for
    everything after it. Yields (ref_word, pred_word_or_None) pairs:
    pred is None for deletions.
    """
    for pred, ref in zip(predictions, references):
        pred_words = clean_and_tokenize(pred)
        ref_words = clean_and_tokenize(ref)
        pred_skel = [strip_diacritics(w) for w in pred_words]
        ref_skel = [strip_diacritics(w) for w in ref_words]

        alignment = jiwer.process_words(" ".join(ref_skel), " ".join(pred_skel))

        for chunk in alignment.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    r_word = ref_words[i]
                    p_word = pred_words[chunk.hyp_start_idx + (i - chunk.ref_start_idx)]
                    yield r_word, p_word
            elif chunk.type in ("substitute", "delete"):
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    yield ref_words[i], None
            # "insert" chunks (hallucinated extra words) are skipped —
            # no reference word to score diacritics against.

def compute_der(predictions: list, references: list, ce: bool = True) -> float:
    """
    Diacritic Error Rate via edit-distance word alignment.
    Pure-digit words (e.g. '25', '750') are EXCLUDED — they can never
    carry a diacritic, so including them dilutes/inflates DER with
    noise unrelated to diacritization quality.

    ce=True  -> standard DER (includes case ending).
    ce=False -> DER*, case ending stripped before comparing.
    """
    total_chars, wrong_chars = 0, 0

    for r_word, p_word in _align_words(predictions, references):
        if _is_digit_only(r_word):
            continue  # skip numeric tokens entirely for DER

        if ce is False:
            r_word = strip_case_ending(r_word)
            if p_word is not None:
                p_word = strip_case_ending(p_word)

        r_diacs = ARABIC_DIACRITICS.findall(r_word)
        n = max(len(r_diacs), 1)

        if p_word is None:
            total_chars += n
            wrong_chars += n
            continue

        p_diacs = ARABIC_DIACRITICS.findall(p_word)
        n = max(len(r_diacs), len(p_diacs), 1)
        total_chars += n
        if r_diacs != p_diacs:
            mism = sum(1 for a, b in zip(r_diacs, p_diacs) if a != b)
            mism += abs(len(r_diacs) - len(p_diacs))
            wrong_chars += max(mism, 1)

    return round((wrong_chars / total_chars) * 100, 2) if total_chars > 0 else 0.0

def compute_wer(predictions: list, references: list, ce: bool = True) -> float:
    """
    Word-level diacritization error rate. Numeric tokens ARE still
    scored here (unlike DER) since they're real content — the model
    should still reproduce '25', '750', etc. correctly; they just
    can't be judged on diacritics.

    ce=True  -> mismatch anywhere (incl. case ending) marks word wrong.
    ce=False -> WER*, case ending stripped before comparing.
    """
    total_words, wrong_words = 0, 0

    for r_word, p_word in _align_words(predictions, references):
        total_words += 1
        if p_word is None:
            wrong_words += 1
            continue
        if ce is False:
            r_word = strip_case_ending(r_word)
            p_word = strip_case_ending(p_word)
        if r_word != p_word:
            wrong_words += 1

    return round((wrong_words / total_words) * 100, 2) if total_words > 0 else 0.0

def compute_der_wer(predictions: list, references: list) -> dict:
    """Returns all four headline metrics: DER/WER with & without case ending."""
    return {
        "DER_ce": compute_der(predictions, references, ce=True),
        "DER_noce": compute_der(predictions, references, ce=False),
        "WER_ce": compute_wer(predictions, references, ce=True),
        "WER_noce": compute_wer(predictions, references, ce=False),
    }

"""## 3. Imports & config"""

import sys
import gc
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm.auto import tqdm

SAMPLE_SIZE = 600  # 600 CA + 600 MSA = 1200 samples total
CACHE_DIR = "./model_cache"
skipped_models = []

"""## 4. Dataset ingestion — `Misraj/SadeedDiac-25`"""

print("Loading SadeedDiac-25 Dataset...")
try:
    dataset = load_dataset("Misraj/SadeedDiac-25", split="train", token=HF_TOKEN or None)
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

"""## 5. Inference pipelines

`aya-expanse-8b` and `Moonlight-16B-A3B-Instruct` are general chat/instruct models (not fine-tuned for diacritization), so they're prompted zero-shot with an explicit Arabic instruction, following the chat-template pattern from `How_to_use.py`. `Bisher/gemma-3-1b-pt-10k-diacritization` is used exactly as in `How_to_use.py` — via `pipeline("text-generation", ...)`.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The used device is :{device}")
DIACRITIZE_INSTRUCTION = "قم بتشكيل هذا النص تشكيلاً كاملاً بدون أي إضافات أو شرح:\n{text}"

# ---------- Generic causal LM, chat-template based (aya-expanse-8b, Moonlight) ----------
def load_causal_chat(model_id, desc_tag, trust_remote_code=False):
    print(f"Loading CausalLM (chat template): {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, trust_remote_code=trust_remote_code,
            token=HF_TOKEN or None,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
            cache_dir=CACHE_DIR,
            token=HF_TOKEN or None,
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        skipped_models.append((desc_tag, str(e)))
        return None, None

def infer_causal_chat(model, tokenizer, test_dataset, desc_tag, system_message=None):
    predictions, references = [], [x["ground_truth"] for x in test_dataset]
    for item in tqdm(test_dataset, desc=f"Inference {desc_tag}"):
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": DIACRITIZE_INSTRUCTION.format(text=item["raw_input"])})

            model_inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            ).to(model.device)

            with torch.no_grad():
                # max_new_tokens set explicitly -- default generation length
                # is far too short for full diacritized output.
                output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)

            input_len = model_inputs["input_ids"].shape[-1]
            pred_text = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
            predictions.append(pred_text.strip())
        except Exception as e:
            print(f"  [warn] {desc_tag}: inference failed on a sample, using empty prediction ({e})")
            predictions.append("")
    return compute_der_wer(predictions, references)


# ---------- pipeline-based causal LM (Bisher/gemma-3-1b-pt-10k-diacritization) ----------
def load_pipeline_chat(model_id, desc_tag):
    print(f"Loading pipeline text-generation model: {model_id}...")
    try:
        generator = pipeline(
            "text-generation",
            model=model_id,
            device=device,
            token=HF_TOKEN or None,
        )
        return generator, None
    except Exception as e:
        skipped_models.append((desc_tag, str(e)))
        return None, None

def infer_pipeline_chat(generator, _tokenizer_unused, test_dataset, desc_tag):
    predictions, references = [], [x["ground_truth"] for x in test_dataset]
    for item in tqdm(test_dataset, desc=f"Inference {desc_tag}"):
        try:
            messages = [{"role": "user", "content": DIACRITIZE_INSTRUCTION.format(text=item["raw_input"])}]
            output = generator(messages, max_new_tokens=1024, return_full_text=False)[0]
            predictions.append(output["generated_text"].strip())
        except Exception as e:
            print(f"  [warn] {desc_tag}: inference failed on a sample, using empty prediction ({e})")
            predictions.append("")
    return compute_der_wer(predictions, references)

"""## 6. Model execution registry"""

model_registry = {
        "aya-expanse-8b": {
        "repo": "CohereLabs/aya-expanse-8b",
        "type": "causal_chat",
        "trust_remote_code": False,
        "system_message": None,
    },
    "Moonlight-16B-A3B-Instruct": {
        "repo": "moonshotai/Moonlight-16B-A3B-Instruct",
        "type": "causal_chat",
        "trust_remote_code": True,
        "system_message": "You are a helpful assistant provided by Moonshot-AI.",
    },
}

domains = [("MSA (Modern)", msa_test_set), ("CA (Classical)", ca_test_set)]
benchmark_records = []

# Outer loop: models  ->  load once
# Inner loop: domains (MSA, then CA) -> run 600 + 600 = 1200 samples per model
# After a model finishes both domains, free the GPU before loading the next one.
for label, meta in model_registry.items():
    print(f"\n{'='*60}\n Loading model: {label}  ({meta['repo']})\n{'='*60}")

    model, tokenizer, infer_fn = None, None, None
    try:
        if meta["type"] == "causal_chat":
            model, tokenizer = load_causal_chat(meta["repo"], label, trust_remote_code=meta["trust_remote_code"])
            infer_fn = lambda m, t, ds, lbl, _meta=meta: infer_causal_chat(m, t, ds, lbl, system_message=_meta["system_message"])
        elif meta["type"] == "pipeline_chat":
            model, tokenizer = load_pipeline_chat(meta["repo"], label)
            infer_fn = infer_pipeline_chat
        else:
            print(f"Skipping {label}: unknown model type '{meta['type']}'.")
            continue

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

"""## 7. Results report"""

performance_df = pd.DataFrame(benchmark_records)

if performance_df.empty:
    full_df = performance_df
    print("No models produced results — check the skip log below.")
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

print("\n=== INSTRUCT MODELS EVALUATION MATRIX (WITH MSA / CA MEAN ROW) ===")
print(full_df.to_markdown(index=False) if not full_df.empty else "No models processed.")

try:
    results_path = "instruct_models_results.csv"
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