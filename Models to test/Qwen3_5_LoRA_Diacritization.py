# -*- coding: utf-8 -*-
"""Qwen3_5_LoRA_Diacritization.py

Converted from Qwen3_5.ipynb (Colab) and restructured to match the
evaluation style of Mideum_models_v1.py.

# PIPELINE: Qwen3.5-4B — LoRA Fine-Tuning + DER/WER Evaluation
# Train dataset : Misraj/Sadeed_Tashkeela
# Test  dataset : Misraj/SadeedDiac-25
#
# Steps:
#   1) Environment setup
#   2) Load + preprocess training data
#   3) Load base model, attach LoRA adapters
#   4) Train, save adapter + merged model, save training log CSV
#   5) Load test set (MSA / CA split, same convention as Mideum_models_v1.py)
#   6) Run inference with the fine-tuned model
#   7) Compute DER / WER, each with case-ending (ce) and no-case-ending (noce)
#   8) Save test results CSV
"""

# ============================================================
# 1: ENVIRONMENT & DEPENDENCIES
# ============================================================
import subprocess
import sys


def _pip_install(packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=False)


_pip_install([
    "transformers", "datasets", "torch", "accelerate", "peft",
    "bitsandbytes", "pyarabic", "jiwer", "tqdm", "pandas",
])

import os
import re
import gc
import torch
import jiwer
import pandas as pd
import pyarabic.araby as araby
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel

# ---- Optional HF Hub login (set HF_TOKEN as an env var, never hardcode it) ----
# NOTE: this previously passed a literal token as the env var *name*, which both
# leaked the token into git history and always returned None. The leaked token
# must be revoked at huggingface.co/settings/tokens — deleting it here does not
# remove it from commit b6c05e6.
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login
    login(HF_TOKEN)

# ============================================================
# CONFIG
# ============================================================
CACHE_DIR = "./model_cache"          # shared cache dir for models/tokenizers/datasets
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"
LORA_ADAPTER_DIR = "./qwen35-tashkeel-lora"     # LoRA adapter checkpoint
MERGED_MODEL_DIR = "./qwen35-tashkeel-merged"   # full merged fine-tuned model

TRAIN_RESULTS_CSV = "train_results.csv"
TEST_RESULTS_CSV = "test_results.csv"

MAXLEN = 1024
TRAIN_SUBSET = 50000          # number of training examples to use (scale up as needed)
EVAL_SUBSET = 1000            # held-out split from training data used during training
TEST_SAMPLE_SIZE = 600        # samples per domain (MSA / CA) pulled from SadeedDiac-25 for final testing

ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652]')
device = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

SYS_PROMPT = (
    "You are a specialized Arabic diacritizer. Re-write the given text with "
    "complete Arabic diacritical marks (Tashkeel). Return ONLY the diacritized text."
)

skipped_models = []

# ============================================================
# 2: DER / WER EVALUATION FUNCTIONS (case-ending + no-case-ending)
# Reused from Mideum_models_v1.py so test results are directly comparable.
# ============================================================
def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS.sub('', text)


def clean_and_tokenize(text: str) -> list:
    return re.sub(r'[^\w\s\u064B-\u0652]', '', text).split()


def strip_case_ending(word: str) -> str:
    idx = [i for i, ch in enumerate(word) if not ARABIC_DIACRITICS.match(ch)]
    return word[:idx[-1] + 1] if idx else word


def _align_words(predictions: list, references: list):
    for pred, ref in zip(predictions, references):
        p_words, r_words = clean_and_tokenize(pred), clean_and_tokenize(ref)
        p_skel, r_skel = [strip_diacritics(w) for w in p_words], [strip_diacritics(w) for w in r_words]
        alignment = jiwer.process_words(" ".join(r_skel), " ".join(p_skel))
        for chunk in alignment.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    yield r_words[i], p_words[chunk.hyp_start_idx + (i - chunk.ref_start_idx)]
            elif chunk.type in ("substitute", "delete"):
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    yield r_words[i], None


def compute_der_wer(predictions: list, references: list) -> dict:
    total_chars, wrong_chars, total_words, wrong_words = 0, 0, 0, 0
    total_chars_noce, wrong_chars_noce, wrong_words_noce = 0, 0, 0

    for r_word, p_word in _align_words(predictions, references):
        total_words += 1
        r_word_noce = strip_case_ending(r_word)
        p_word_noce = strip_case_ending(p_word) if p_word else None

        if p_word is None:
            wrong_words += 1
            wrong_words_noce += 1
            n = max(len(ARABIC_DIACRITICS.findall(r_word)), 1)
            total_chars += n
            wrong_chars += n
            n_noce = max(len(ARABIC_DIACRITICS.findall(r_word_noce)), 1)
            total_chars_noce += n_noce
            wrong_chars_noce += n_noce
            continue

        if r_word != p_word:
            wrong_words += 1
        if r_word_noce != p_word_noce:
            wrong_words_noce += 1

        r_d, p_d = ARABIC_DIACRITICS.findall(r_word), ARABIC_DIACRITICS.findall(p_word)
        n = max(len(r_d), len(p_d), 1)
        total_chars += n
        if r_d != p_d:
            mism = sum(1 for a, b in zip(r_d, p_d) if a != b) + abs(len(r_d) - len(p_d))
            wrong_chars += max(mism, 1)

        r_d_n, p_d_n = ARABIC_DIACRITICS.findall(r_word_noce), ARABIC_DIACRITICS.findall(p_word_noce)
        n_n = max(len(r_d_n), len(p_d_n), 1)
        total_chars_noce += n_n
        if r_d_n != p_d_n:
            mism = sum(1 for a, b in zip(r_d_n, p_d_n) if a != b) + abs(len(r_d_n) - len(p_d_n))
            wrong_chars_noce += max(mism, 1)

    return {
        "DER_ce": round((wrong_chars / total_chars) * 100, 2) if total_chars else 0.0,
        "DER_noce": round((wrong_chars_noce / total_chars_noce) * 100, 2) if total_chars_noce else 0.0,
        "WER_ce": round((wrong_words / total_words) * 100, 2) if total_words else 0.0,
        "WER_noce": round((wrong_words_noce / total_words) * 100, 2) if total_words else 0.0,
    }


# ============================================================
# 3: LOAD + PREPROCESS TRAINING DATA (Misraj/Sadeed_Tashkeela)
# ============================================================
def bare(t):
    """Strip tashkeel/tatweel to get the raw (undiacritized) input text."""
    t = araby.strip_tatweel(araby.strip_tashkeel(t or ""))
    return " ".join(t.split())


def build_training_dataset(tokenizer):
    print("Loading training dataset Misraj/Sadeed_Tashkeela...")
    try:
        tash = load_dataset("Misraj/Sadeed_Tashkeela", split="train", cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"FATAL: could not load training dataset: {e}")
        return None

    col = next((c for c in ("text", "diacritized", "output", "sentence") if c in tash.column_names), None)
    if col is None:
        print("FATAL: could not find a diacritized text column in the training dataset.")
        return None

    def process(ex):
        diac = ex[col]
        inp = bare(diac)
        msgs = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": diac},
        ]
        full_txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_txt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer(full_txt, add_special_tokens=False)["input_ids"]
        prompt = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt) + full[len(prompt):]   # loss only on diacritized output
        return {
            "input_ids": full[:MAXLEN],
            "attention_mask": [1] * len(full[:MAXLEN]),
            "labels": labels[:MAXLEN],
        }

    tash = tash.filter(lambda ex: ex[col] and bare(ex[col]))
    tok_ds = tash.map(process, remove_columns=tash.column_names)
    tok_ds = tok_ds.train_test_split(test_size=0.02, seed=42)
    return tok_ds


# ============================================================
# 4: LOAD BASE MODEL + ATTACH LoRA
# ============================================================
def load_base_model_with_lora():
    print(f"Loading base model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ============================================================
# 5: TRAIN + SAVE MODEL (adapter, merged model, training log CSV)
# ============================================================
def train_and_save(model, tokenizer, tok_ds):
    train_ds = tok_ds["train"].select(range(min(TRAIN_SUBSET, len(tok_ds["train"]))))
    eval_ds = tok_ds["test"].select(range(min(EVAL_SUBSET, len(tok_ds["test"]))))

    args = TrainingArguments(
        output_dir=LORA_ADAPTER_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        bf16=BF16,
        fp16=not BF16,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train()

    # --- Save training log (loss/lr per logging step) to CSV ---
    train_log_df = pd.DataFrame(trainer.state.log_history)
    train_log_df.to_csv(TRAIN_RESULTS_CSV, index=False)
    print(f"Training log saved to {TRAIN_RESULTS_CSV}")

    # --- Save the LoRA adapter ---
    trainer.save_model(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR)
    print(f"LoRA adapter saved to {LORA_ADAPTER_DIR}")

    # --- Merge LoRA weights into the base model and save the full fine-tuned model ---
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    print(f"Merged fine-tuned model saved to {MERGED_MODEL_DIR}")

    return merged, tokenizer


# ============================================================
# 6: LOAD TEST DATA (Misraj/SadeedDiac-25), MSA / CA split
#     (same convention as Mideum_models_v1.py)
# ============================================================
def build_test_datasets():
    try:
        dataset = load_dataset("Misraj/SadeedDiac-25", split="train", cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"FATAL: could not load test dataset: {e}")
        return [], []

    msa_samples = [
        {"ground_truth": x["output"], "raw_input": strip_diacritics(x["input"])}
        for x in dataset if "fadel" not in x.get("filename", "").lower()
    ][:TEST_SAMPLE_SIZE]
    ca_samples = [
        {"ground_truth": x["output"], "raw_input": strip_diacritics(x["input"])}
        for x in dataset if "fadel" in x.get("filename", "").lower()
    ][:TEST_SAMPLE_SIZE]
    return msa_samples, ca_samples


# ============================================================
# 7: INFERENCE + EVALUATION (mirrors infer_medium_causal in Mideum_models_v1.py)
# ============================================================
def infer_and_evaluate(model, tokenizer, test_dataset, label):
    predictions, references = [], [x["ground_truth"] for x in test_dataset]
    for item in tqdm(test_dataset, desc=f"Evaluating {label}"):
        try:
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": item["raw_input"]},
            ]
            prompt = (
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if hasattr(tokenizer, "apply_chat_template") else item["raw_input"]
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            raw_pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            predictions.append(raw_pred.strip().split("\n")[0])
        except Exception as e:
            print(f"  [warn] {label}: inference failed on a sample, using empty prediction ({e})")
            predictions.append("")

    return compute_der_wer(predictions, references)


# ============================================================
# 8: MAIN EXECUTION
# ============================================================
def main():
    model, tokenizer = None, None
    try:
        # ---- Load base model + LoRA ----
        model, tokenizer = load_base_model_with_lora()

        # ---- Build training data and fine-tune ----
        tok_ds = build_training_dataset(tokenizer)
        if tok_ds is None:
            skipped_models.append(("training", "training dataset could not be built"))
            print("Skipping training — falling back to base model for evaluation.")
        else:
            model, tokenizer = train_and_save(model, tokenizer, tok_ds)

    except Exception as e:
        skipped_models.append(("training", f"Unexpected failure: {e}"))
        print(f"Training failed: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    if model is None:
        print("No model available for evaluation — aborting.")
        return

    # ---- Load held-out test set (MSA / CA) ----
    msa_samples, ca_samples = build_test_datasets()
    domains = [("MSA (Modern)", msa_samples), ("CA (Classical)", ca_samples)]

    records = []
    label = "Qwen3.5-4B-LoRA-Tashkeel"
    for domain, test_set in domains:
        if not test_set:
            print(f"Skipping {domain}: no test samples found.")
            continue
        metrics = infer_and_evaluate(model, tokenizer, test_set, f"{label} [{domain}]")
        metrics.update({"Model": label, "Domain Track": domain})
        records.append(metrics)

    # ---- Build results table with a mean row, same convention as Mideum_models_v1.py ----
    performance_df = pd.DataFrame(records)

    if performance_df.empty:
        full_df = performance_df
        print("No test results produced — check the skip log above.")
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

    print("\n=== FINE-TUNED MODEL TEST RESULTS (DER / WER) ===")
    print(full_df.to_markdown(index=False) if not full_df.empty else "No results.")

    try:
        full_df.to_csv(TEST_RESULTS_CSV, index=False)
        print(f"\nTest results saved to {TEST_RESULTS_CSV}")
    except Exception as e:
        print(f"\n[warn] Could not save test results CSV: {e}")

    if skipped_models:
        print("\n================ SKIPPED / FAILED STEPS ================")
        for name, reason in skipped_models:
            print(f"- {name}: {reason}")
    else:
        print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
