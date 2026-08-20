"""
Qwen3.5-4B — LoRA Fine-Tuning for Arabic Diacritization
========================================================

Self-contained: metrics, data prep, training, inference and scoring all live in
this one file. No imports from Models_Functions.py or the notebook at runtime.

Model         : Qwen/Qwen3.5-4B   (LoRA — bf16 base, adapters trained)
Training data : Misraj/Sadeed_Tashkeela  train 1,042,698 / test 2,485
                columns: filename, input, output — already clean, NO preprocessing
Benchmark     : Misraj/SadeedDiac-25     split "train", 1,200 paragraphs
                LIGHT preprocessing only (strip diacritics to build the prompt),
                per Arabic_Diacritization_Preprocessing_Minimal.ipynb §3
Evaluation    : DER / WER, each with and without case ending, on train, test and
                benchmark -> eval_outputs/metrics_summary.csv

Metric functions use the CORRECTED scorer (see Evaluation_Functions_Corrected.py
at the repo root): NFC normalization, per-letter mark tracking, a reference-fixed
DER denominator, hallucinated words counted as errors, and dagger alef (U+0670)
included in the diacritic mark class. This replaces the old scorer that used to
be copied verbatim from Models_Functions.py.

------------------------------------------------------------------------------
THREE THINGS TO SET OR CHECK BEFORE THE FIRST RUN
------------------------------------------------------------------------------
1. HF_TOKEN below is intentionally empty. Paste a token, or better, export it:
       export HF_TOKEN=hf_...
   Qwen/Qwen3.5-4B is not gated, but Misraj/Sadeed_Tashkeela IS — the token is
   needed for the dataset even though the model downloads freely.

2. NUM_TRAIN_EPOCHS defaults to 1, not the 6 in Confg_Info.md. See the note at
   that constant: 6 epochs over the full 1M-row corpus is ~195,000 optimizer
   steps. Confg_Info's 6 was written for a subset.

3. TRAIN_EVAL_SAMPLE_SIZE defaults to 2,485 rather than all 1,042,698. Training
   on the full corpus is the point; generating predictions for all of it just to
   report a train-set DER is not — see the note at that constant.
------------------------------------------------------------------------------
"""

import os
import re
import json
import random
import sys
import traceback

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless — required on a compute node
import matplotlib.pyplot as plt

# ==============================================================================================
# Configuration
# ==============================================================================================
MODEL_ID = "Qwen/Qwen3.5-4B"

# Paste your token here, or leave empty and export HF_TOKEN in the environment.
HF_TOKEN = ""

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")
# RUN_TAG keeps each data-scaling run in its OWN directories. Without it the
# 10/30/50/100% runs would all write the same adapter and metrics_summary.csv
# and silently overwrite one another.
RUN_TAG = os.environ.get("DIAC_RUN_TAG", "").strip()
_suffix = f"-{RUN_TAG}" if RUN_TAG else ""
OUTPUT_DIR = f"./qwen3.5-4b-lora-diacritization{_suffix}"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
EVAL_DIR = f"./eval_outputs{_suffix}"

# ---- datasets ----
TRAIN_DATASET = "Misraj/Sadeed_Tashkeela"
INPUT_COLUMN = "input"        # undiacritized source
OUTPUT_COLUMN = "output"      # fully diacritized target

BENCHMARK_DATASET = "Misraj/SadeedDiac-25"
BENCHMARK_SPLIT = "train"         # SadeedDiac-25 exposes its 1,200 paragraphs under "train"
BENCHMARK_TEXT_COLUMN = "output"  # diacritized ground truth

# None = use the whole split. Train/test are used in full for TRAINING.
TRAIN_FRACTION = None        # 0.1 / 0.3 / 0.5 / None(=all). Overridable via DIAC_TRAIN_FRACTION
TRAIN_SIZE = None          # 1,042,698 rows
TEST_SIZE = None              #     2,485 rows
BENCHMARK_SIZE = None         #     1,200 rows

# How many TRAIN rows to generate predictions for when scoring.
#
# Scoring is generation, not a forward pass: roughly 1-2 s per paragraph on one
# GPU. All 1,042,698 rows would be 300-500 GPU-hours for a diagnostic number.
# 500 matches from test split, so train-vs-test is a like-for-like
# comparison and the gap between them still reads as overfitting.
# Set to None to score the entire train split anyway.
TRAIN_EVAL_SAMPLE_SIZE = 500

# ---- LoRA (Confg_Info.md, Option A) ----
USE_4BIT = False              # False = LoRA on a bf16 base. True = QLoRA (4-bit NF4 base).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Leaf names only. The loader below resolves these to FULL dotted paths and keeps
# just the ones that are a real torch.nn.Linear.
#
# On Qwen3.5-4B all 128 matches are plain nn.Linear (verified: q/k/v/o_proj x8,
# gate/up/down_proj x32), so nothing is filtered out here. The isinstance check is
# kept anyway because it is what made this script survive gemma-4-E4B, whose
# vision/audio towers reuse these exact names for Gemma4ClippableLinear — a
# non-Linear wrapper that PEFT rejects outright (huggingface/peft#3129). Cheap
# insurance against the next architecture that does something similar.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---- training ----
# Raise this deliberately, ideally alongside MAX_STEPS, not by habit.
# 1 epoch over 1,042,697 rows at an effective batch of 64 is ~16,292 steps.
# At a realistic 8-12 s/step that is 36-54 h, plus ~2.5 h for model load,
# tokenising 1M rows on NFS, and the closing scoring pass (500 train + 2,485
# test + 1,200 benchmark = 4,185 generations). Hence --time=3-00:00:00 in
# run_qwen_lora.sbatch. Qwen3.5-4B is ~half the size of gemma-4-E4B (4.84B vs
# 8.61B), so expect materially faster steps — the smoke test MEASURES s/step and
# the real time limit is set from that, not guessed.
NUM_TRAIN_EPOCHS = 1
# -1 = disabled, so NUM_TRAIN_EPOCHS governs. Any positive value SILENTLY
# OVERRIDES the epoch count in HF Trainer — that is why this is off.
MAX_STEPS = -1

# ============================================================================
# FAIRNESS CONTRACT — must be IDENTICAL in the gemma-4-E4B script
# ============================================================================
# Qwen3.5-4B and gemma-4-E4B-it are being compared on the same benchmark, so
# every optimisation-relevant quantity has to match: same effective batch, same
# LR and schedule, same epochs, same LoRA rank/alpha/dropout/targets, same
# MAX_SEQ_LENGTH, same data and splits, same seed.
#
# What may legitimately differ is PER_DEVICE_TRAIN_BATCH_SIZE, because the models
# have different memory footprints (4.2B vs 8.6B params, and Gemma's vocabulary is
# larger still). GRADIENT_ACCUMULATION_STEPS is then DERIVED so the product stays
# fixed. That keeps the gradient each optimizer step is computed from identical in
# size, which is what "fair" actually means here — not identical micro-batches.
#
# Under DDP the world size multiplies in too:
#     effective batch = PER_DEVICE * ACCUM * NUM_GPUS
# so ACCUM is computed from EFFECTIVE_BATCH at runtime rather than hardcoded.
EFFECTIVE_BATCH = 96          # 8 x 4 x 3 GPUs. Divisible by 3 — 64 is not.

# Largest micro-batch that fits THIS model at MAX_SEQ_LENGTH on an 80GB A100.
# Measured: seq1024 bs=8 gc=on peaked at 36.0 GiB; bs=4/8 gc=off OOM'd.
PER_DEVICE_TRAIN_BATCH_SIZE = 8

# Derived below from EFFECTIVE_BATCH, PER_DEVICE and the world size — do not set
# by hand, or the two models stop being comparable.
GRADIENT_ACCUMULATION_STEPS = None

# ON, for two reasons.
#   1) Fairness: the gemma-4-E4B run needs it (8.6B, larger vocabulary), and a
#      comparison where one model recomputes activations and the other doesn't is
#      confounded — checkpointing changes throughput, not just memory.
#   2) It is also simply faster HERE at this batch size. Measured at seq 512:
#         bs=2  gc=off  34.23 s/step        bs=8  gc=on  20.83 s/step
#         bs=8  gc=off  OOM                 bs=16 gc=on  19.25 s/step
#      Turning it off caps the micro-batch at 2, and the small batch costs far
#      more than the recompute saves. The measured 11.08 s/step on 3 GPUs was
#      with this ON.
GRADIENT_CHECKPOINTING = True
PER_DEVICE_EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
# Warmup as a FRACTION of the run, not a fixed step count. A fixed 300 steps
# meant a different schedule at every data fraction:
#     10% =  1,086 steps -> 300 warmup = 27.6% of the run
#     30% =  3,258 steps -> 300 warmup =  9.2%
#     50% =  5,430 steps -> 300 warmup =  5.5%
#    100% = 10,861 steps -> 300 warmup =  2.8%
# a 10x spread in how long the model sat at peak LR, which is a confound in a
# data-scaling comparison. A constant ratio gives every fraction the same curve
# shape and still scales absolute warmup with data (54/162/271/543 steps at 5%).
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 20
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2
MAX_SEQ_LENGTH = 1024

# Eval during training runs on the 2,485-row test split. At batch size 2 that is
# ~1,250 forward passes every EVAL_STEPS — cap it so it doesn't dominate runtime.
IN_TRAINING_EVAL_SUBSET = 500

# ---- inference ----
INFER_BATCH_SIZE = 8
MAX_NEW_TOKENS = 512

# Row count of Misraj/Sadeed_Tashkeela's train split after the one null row is
# dropped. Used only so a smoke run can project the REAL job's duration.
FULL_TRAIN_ROWS = 1_042_697

RANDOM_SEED = 42

# ---- smoke-test overrides (environment only; defaults above are unchanged) ----
# Lets a 2-step trial validate the whole pipeline on a short queue slot without
# editing this file:
#     DIAC_MAX_STEPS=2 DIAC_TRAIN_SIZE=2000 DIAC_EVAL_SUBSET=8 python <this file>
# Unset in a real run, so the configured values above apply.
if os.environ.get("DIAC_MAX_STEPS"):
    MAX_STEPS = int(os.environ["DIAC_MAX_STEPS"])
    print(f"[SMOKE] MAX_STEPS overridden -> {MAX_STEPS}")
if os.environ.get("DIAC_TRAIN_FRACTION"):
    TRAIN_FRACTION = float(os.environ["DIAC_TRAIN_FRACTION"])
    print(f"[INFO] TRAIN_FRACTION -> {TRAIN_FRACTION:.0%} of the train split")
if os.environ.get("DIAC_TRAIN_SIZE"):
    TRAIN_SIZE = int(os.environ["DIAC_TRAIN_SIZE"])
    print(f"[SMOKE] TRAIN_SIZE overridden -> {TRAIN_SIZE}")
if os.environ.get("DIAC_TEST_SIZE"):
    TEST_SIZE = int(os.environ["DIAC_TEST_SIZE"])
    print(f"[SMOKE] TEST_SIZE overridden -> {TEST_SIZE}")
if os.environ.get("DIAC_BENCHMARK_SIZE"):
    BENCHMARK_SIZE = int(os.environ["DIAC_BENCHMARK_SIZE"])
    print(f"[SMOKE] BENCHMARK_SIZE overridden -> {BENCHMARK_SIZE}")
if os.environ.get("DIAC_EVAL_SUBSET"):
    IN_TRAINING_EVAL_SUBSET = int(os.environ["DIAC_EVAL_SUBSET"])
    print(f"[SMOKE] IN_TRAINING_EVAL_SUBSET overridden -> {IN_TRAINING_EVAL_SUBSET}")
if os.environ.get("DIAC_TRAIN_EVAL_SAMPLE"):
    TRAIN_EVAL_SAMPLE_SIZE = int(os.environ["DIAC_TRAIN_EVAL_SAMPLE"])
    print(f"[SMOKE] TRAIN_EVAL_SAMPLE_SIZE overridden -> {TRAIN_EVAL_SAMPLE_SIZE}")
# Batch shape and checkpointing are the two throughput levers — overridable so a
# smoke run can measure a candidate config without editing this file.
if os.environ.get("DIAC_BATCH"):
    PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ["DIAC_BATCH"])
    print(f"[SMOKE] PER_DEVICE_TRAIN_BATCH_SIZE overridden -> {PER_DEVICE_TRAIN_BATCH_SIZE}")
if os.environ.get("DIAC_ACCUM"):
    GRADIENT_ACCUMULATION_STEPS = int(os.environ["DIAC_ACCUM"])
    print(f"[SMOKE] GRADIENT_ACCUMULATION_STEPS overridden -> {GRADIENT_ACCUMULATION_STEPS}")
if os.environ.get("DIAC_GC"):
    GRADIENT_CHECKPOINTING = os.environ["DIAC_GC"].lower() in ("1", "true", "yes")
    print(f"[SMOKE] GRADIENT_CHECKPOINTING overridden -> {GRADIENT_CHECKPOINTING}")
if os.environ.get("DIAC_EFFECTIVE_BATCH"):
    EFFECTIVE_BATCH = int(os.environ["DIAC_EFFECTIVE_BATCH"])
    print(f"[SMOKE] EFFECTIVE_BATCH overridden -> {EFFECTIVE_BATCH}")

# ---- derive accumulation so the effective batch is exactly EFFECTIVE_BATCH ----
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_per_step_without_accum = PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE
if EFFECTIVE_BATCH % _per_step_without_accum != 0:
    raise SystemExit(
        f"FATAL: EFFECTIVE_BATCH={EFFECTIVE_BATCH} is not divisible by "
        f"PER_DEVICE({PER_DEVICE_TRAIN_BATCH_SIZE}) x WORLD_SIZE({WORLD_SIZE})"
        f"={_per_step_without_accum}. The two models would then train on different"
        f" effective batches and stop being comparable. Adjust PER_DEVICE."
    )
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH // _per_step_without_accum
print(f"[INFO] world_size={WORLD_SIZE}  per_device={PER_DEVICE_TRAIN_BATCH_SIZE}  "
      f"accum={GRADIENT_ACCUMULATION_STEPS}  -> effective batch "
      f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE}")

SYSTEM_PROMPT = (
    "أنت نظام متخصص في التشكيل الآلي للنصوص العربية. "
    "مهمتك إضافة الحركات (التشكيل) الصحيحة إلى النص العربي المُدخل دون تغيير الكلمات أو ترتيبها، "
    "مع مراعاة السياق النحوي والصرفي الكامل للجملة."
)

for d in (CACHE_DIR, OUTPUT_DIR, EVAL_DIR):
    os.makedirs(d, exist_ok=True)

random.seed(RANDOM_SEED)


# ==============================================================================================
# Evaluation metrics — CORRECTED scorer (copied from Evaluation_Functions_Corrected.py)
#
# Replaces the old scorer (previously copied verbatim from Models_Functions.py), which
# had five confirmed defects: no NFC normalization, diacritics compared as a flat
# per-word list with the host letter discarded, a DER denominator that moved with the
# prediction instead of being fixed by the reference, hallucinated/inserted words
# scored as free, and dagger alef (U+0670) missing from the diacritic mark class. See
# Evaluation_Functions_Corrected.py at the repo root for the full rationale.
# ==============================================================================================

import jiwer  # noqa: E402
import unicodedata  # noqa: E402

DIACRITICS = re.compile(r'[ً-ٰٟ]')
LETTER = re.compile(r'[ء-غف-يٱ-ۓ]')
TATWEEL = "ـ"


def normalize_numerals(text: str) -> str:
    """Converts Eastern Arabic-Indic digits to Western Arabic digits."""
    return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def strip_citation_refs(text: str) -> str:
    """Removes trailing parenthetical numeric citations like '(41 / 251)'."""
    text = re.sub(r'\(\s*\d+\s*/\s*\d+\s*\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def strip_diacritics(text: str) -> str:
    """Consonantal skeleton: diacritics AND tatweel removed."""
    return DIACRITICS.sub('', text).replace(TATWEEL, '')


def clean_and_tokenize(text: str) -> list:
    """NFC-normalizes, normalizes numerals, strips citations/tatweel, tokenizes."""
    text = unicodedata.normalize("NFC", text)
    text = strip_citation_refs(normalize_numerals(text)).replace(TATWEEL, '')
    return re.sub(r'[^\w\sً-ٰٟ]', '', text).split()


def segment_word(word: str) -> list:
    """Splits a word into [(base_letter, marks), ...] pairs, marks sorted so
    canonically equivalent orderings compare equal — the fix for host-letter
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
    """True if the word (after stripping diacritics) is purely numeric."""
    return strip_diacritics(word).isdigit()


def _align_words(predictions: list, references: list) -> list:
    """Shared word-alignment step used by DER and WER. Returns (ref_or_None,
    pred_or_None) pairs; hallucinated/inserted words come back as (None, pred)
    rather than being dropped."""
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
                # hallucinated extra words — now counted as errors, not skipped.
                pairs += [(None, pred_words[j]) for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)]
    return pairs


def _compute_der_from_pairs(pairs: list, ce: bool = True) -> float:
    """Diacritic Error Rate over pre-computed alignment pairs. Pure-digit words
    excluded. Denominator is fixed to the diacritizable characters in the
    REFERENCE only, and hallucinated words (ref is None) are added to both
    terms so they count as errors."""
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
    """Word-level diacritization error rate over pre-computed alignment pairs.
    A hallucinated or deleted word always counts as wrong."""
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
    """Diacritic Error Rate via edit-distance word alignment, corrected."""
    return _compute_der_from_pairs(_align_words(predictions, references), ce=ce)


def compute_wer(predictions: list, references: list, ce: bool = True) -> float:
    """Word-level diacritization error rate, corrected."""
    return _compute_wer_from_pairs(_align_words(predictions, references), ce=ce)


def compute_der_wer(predictions: list, references: list) -> dict:
    """Returns all four headline metrics: DER/WER with & without case ending.
    Aligns once and reuses the pairs for all four numbers."""
    pairs = _align_words(predictions, references)
    return {
        "DER_ce": _compute_der_from_pairs(pairs, ce=True),
        "DER_noce": _compute_der_from_pairs(pairs, ce=False),
        "WER_ce": _compute_wer_from_pairs(pairs, ce=True),
        "WER_noce": _compute_wer_from_pairs(pairs, ce=False),
    }


# ==============================================================================================
# Stage 1 — Hugging Face authentication
# ==============================================================================================

def hf_login():
    token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        print("[WARN] No HF token set. Misraj/Sadeed_Tashkeela is gated and the load "
              "will fail. Set HF_TOKEN at the top of this file or export it.")
        return
    try:
        from huggingface_hub import login
        login(token=token)
        print("[OK] Logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"[WARN] Hugging Face login failed, continuing anyway: {e}")


# ==============================================================================================
# Stage 2 — Prompt construction
#
# Gemma has NO system role. Passing {"role": "system"} to its chat template raises,
# so the system prompt is folded into the first user turn instead. This is detected
# once against the real tokenizer rather than hardcoded, so the script also works
# unchanged on a model that does support the role.
# ==============================================================================================

_SUPPORTS_SYSTEM_ROLE = None


def detect_system_role_support(tokenizer):
    global _SUPPORTS_SYSTEM_ROLE
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True,
        )
        _SUPPORTS_SYSTEM_ROLE = True
    except Exception:
        _SUPPORTS_SYSTEM_ROLE = False
    print(f"[INFO] chat template supports a system role: {_SUPPORTS_SYSTEM_ROLE}")
    return _SUPPORTS_SYSTEM_ROLE


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"^\s*<think>.*", re.DOTALL)


def clean_generation(text: str) -> str:
    """Strip any reasoning block Qwen emits before the diacritised text.

    enable_thinking=False should prevent these, but a reasoning model can still
    open a <think> block on its own. Left in, the block is scored as if it were
    the diacritised output and every such paragraph reads as ~100% error — which
    would look like a broken model rather than a formatting artefact.

    Two cases: a complete <think>...</think> pair is removed; an UNCLOSED block
    (generation hit max_new_tokens mid-thought) leaves nothing usable, so the
    prediction becomes empty and is counted in n_empty_predictions rather than
    silently scored as garbage.
    """
    text = _THINK_RE.sub("", text or "")
    if _OPEN_THINK_RE.match(text):
        return ""
    return text.strip().strip('"“”').strip()


def render_prompt(tokenizer, user_text: str) -> str:
    """Render the prompt, with Qwen's thinking mode OFF.

    Qwen3.5 is a reasoning model: by default its chat template opens a <think>
    block and the model emits a chain of thought before the answer. For
    diacritisation that is pure noise — it would be trained on as if it were the
    target, and at inference it would be scored as if it were the diacritised
    text. Passing enable_thinking=False makes the template emit a pre-closed
    "<think>\\n\\n</think>" so generation starts on the answer directly.

    The kwarg is accepted on this tokenizer (verified), but the try/except keeps
    the script working on templates that don't take it, rather than dying with
    an opaque TypeError deep inside a multi-hour run.
    """
    msgs = build_prompt_messages(user_text)
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def build_prompt_messages(user_text: str) -> list:
    """The prompt turns only — no assistant turn."""
    if _SUPPORTS_SYSTEM_ROLE:
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}]
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"}]


# ==============================================================================================
# Stage 3 — Data
#
# Sadeed_Tashkeela ships already cleaned, chunked and filtered: input/output pairs
# are used as-is, with no preprocessing beyond wrapping them in the chat format.
#
# SadeedDiac-25 gets LIGHT preprocessing only (notebook §3): the prompt is built by
# stripping diacritics off the gold text; the gold text itself is untouched. It is
# not cleaned, chunked, filtered or deduplicated — doing any of that would break
# comparability with the numbers in the Sadeed paper.
# ==============================================================================================

def load_training_data():
    from datasets import load_dataset

    ds = load_dataset(TRAIN_DATASET, cache_dir=CACHE_DIR)
    train_raw, test_raw = ds["train"], ds["test"]

    # ---- data-scaling subset: SHUFFLE FIRST, then take a prefix ----
    # select(range(n)) alone is a contiguous head slice. Sadeed_Tashkeela carries a
    # `filename` column and is ordered by source book, so a raw prefix would be
    # drawn from just the first few texts — the 10/30/50/100% points would then
    # differ in DOMAIN as well as size and the scaling curve would be worthless.
    #
    # Shuffling with a FIXED seed first gives a random sample, and because every
    # fraction shuffles identically and takes a prefix, the subsets are NESTED:
    #     10% subset  is contained in  30%  is contained in  50%  is contained in 100%
    # which is what makes the points comparable as a curve rather than four
    # unrelated samples.
    if TRAIN_FRACTION is not None and TRAIN_FRACTION < 1.0:
        n_frac = max(1, int(len(train_raw) * TRAIN_FRACTION))
        train_raw = train_raw.shuffle(seed=RANDOM_SEED).select(range(n_frac))
        print(f"[OK] TRAIN_FRACTION={TRAIN_FRACTION:.0%} -> {len(train_raw):,} rows "
              f"(shuffled seed={RANDOM_SEED}; nested across fractions)")
    elif TRAIN_SIZE:
        train_raw = train_raw.shuffle(seed=RANDOM_SEED).select(
            range(min(TRAIN_SIZE, len(train_raw))))
    if TEST_SIZE:
        test_raw = test_raw.select(range(min(TEST_SIZE, len(test_raw))))

    for col in (INPUT_COLUMN, OUTPUT_COLUMN):
        if col not in train_raw.column_names:
            raise KeyError(f"column '{col}' missing; found {train_raw.column_names}")

    # Drop rows with a null/blank input or output. This is NOT the "preprocessing"
    # the dataset doesn't need — Sadeed_Tashkeela really does contain rows where
    # `output` is None, and they cannot be turned into a training example at all.
    # Left in, they crash tokenisation with:
    #     TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
    # The count is printed so a suspiciously large number is visible rather than
    # silently swallowed.
    def usable(ex):
        i, o = ex.get(INPUT_COLUMN), ex.get(OUTPUT_COLUMN)
        return isinstance(i, str) and isinstance(o, str) and i.strip() != "" and o.strip() != ""

    nproc = max(1, min(8, (os.cpu_count() or 2) // 2))
    n_tr, n_te = len(train_raw), len(test_raw)
    train_raw = train_raw.filter(usable, num_proc=nproc, desc="dropping null rows (train)")
    test_raw = test_raw.filter(usable, num_proc=nproc, desc="dropping null rows (test)")

    d_tr, d_te = n_tr - len(train_raw), n_te - len(test_raw)
    if d_tr or d_te:
        print(f"[WARN] dropped {d_tr:,} train and {d_te:,} test rows with a null/blank "
              f"'{INPUT_COLUMN}' or '{OUTPUT_COLUMN}'")
    print(f"[OK] {TRAIN_DATASET}: train={len(train_raw):,} (of {n_tr:,})  "
          f"test={len(test_raw):,} (of {n_te:,})")
    if len(train_raw) == 0:
        raise ValueError("every train row was dropped — check the column names")
    return train_raw, test_raw


def load_benchmark_data():
    from datasets import load_dataset

    ds = load_dataset(BENCHMARK_DATASET, cache_dir=CACHE_DIR)
    if BENCHMARK_SPLIT not in ds:
        avail = list(ds.keys())
        print(f"[WARN] split '{BENCHMARK_SPLIT}' not found; available: {avail} — using '{avail[0]}'")
        raw = ds[avail[0]]
    else:
        raw = ds[BENCHMARK_SPLIT]

    if BENCHMARK_SIZE:
        raw = raw.select(range(min(BENCHMARK_SIZE, len(raw))))

    col = BENCHMARK_TEXT_COLUMN
    if col not in raw.column_names:
        raise KeyError(f"column '{col}' missing; found {raw.column_names}")

    print(f"[OK] {BENCHMARK_DATASET}: {len(raw)} paragraphs (light preprocessing only)")
    return raw


def as_pairs(dataset, input_col, output_col):
    """(prompt_text, gold_text) pairs, ready for generation and scoring."""
    return [(dataset[i][input_col], dataset[i][output_col]) for i in range(len(dataset))]


def benchmark_pairs(dataset):
    """Light preprocessing: the prompt is the gold text with diacritics removed."""
    golds = [g for g in dataset[BENCHMARK_TEXT_COLUMN]
             if isinstance(g, str) and g.strip() != ""]
    dropped = len(dataset) - len(golds)
    if dropped:
        print(f"[WARN] benchmark: skipped {dropped} paragraph(s) with null/blank "
              f"'{BENCHMARK_TEXT_COLUMN}'")
    return [(strip_diacritics(g), g) for g in golds]


# ==============================================================================================
# Stage 4 — Model + LoRA
# ==============================================================================================

def load_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"          # training; flipped to "left" for generation

    load_kwargs = dict(
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        # DDP: every rank must hold a FULL copy of the model on ITS OWN gpu.
        # device_map="auto" would shard one copy across all visible GPUs
        # (model parallel), which is not what torchrun/DDP expects and produces
        # either a hang or "expected all tensors on the same device". Under
        # torchrun, LOCAL_RANK is set, so pin the whole model to that device.
        device_map=({"": int(os.environ["LOCAL_RANK"])}
                    if os.environ.get("LOCAL_RANK") is not None else "auto"),
        dtype=torch.bfloat16,
    )
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    except Exception as e:
        # Gemma 4 ships a multimodal-capable architecture that AutoModelForCausalLM
        # sometimes refuses; the text tower loads fine through this class.
        print(f"[INFO] CausalLM load failed ({type(e).__name__}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs)

    model.config.use_cache = False
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Belt-and-braces, not a fix for a live bug: gradient checkpointing over a
        # frozen base can sever the autograd graph ("element 0 of tensors does not
        # require grad"), because the inputs to each checkpointed block don't
        # require grad. The 4-bit path is covered by prepare_model_for_kbit_training();
        # on transformers >= ~4.35 the plain-LoRA path is covered too, since
        # gradient_checkpointing_enable() calls enable_input_require_grads() itself
        # once a PEFT config is attached.
        #
        # Verified redundant on transformers 5.15.0 / peft 0.20.0 — backward
        # produces identical nonzero LoRA gradients with and without it. Kept only
        # so an older stack on the cluster degrades gracefully. Safe to delete.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Match by full dotted path, not just leaf name, and require an actual
    # torch.nn.Linear. Gemma 4's vision/audio towers wrap Linear in a custom
    # module (Gemma4ClippableLinear, for numerical-stability clamping) that
    # reuses the same leaf names (q_proj, k_proj, ...) but isn't nn.Linear;
    # PEFT's LoRA dispatch only supports nn.Linear/nn.Embedding/nn.Conv*/Conv1D,
    # so leaf-name-only matching lets it hit those wrapped modules too and
    # crash (https://github.com/huggingface/peft/issues/3129). Full paths plus
    # an isinstance check select only the text decoder's real Linear
    # projections and skip the wrapped ones automatically.
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in LORA_TARGET_MODULES
    ]
    if not targets:
        raise RuntimeError(
            f"none of {LORA_TARGET_MODULES} exist as nn.Linear in {MODEL_ID}. "
            f"Print model.named_modules() and set LORA_TARGET_MODULES by hand."
        )
    found_leaves = {t.split(".")[-1] for t in targets}
    if len(found_leaves) < len(LORA_TARGET_MODULES):
        missing = set(LORA_TARGET_MODULES) - found_leaves
        print(f"[WARN] no nn.Linear found for {missing} — present only as a non-Linear "
              f"wrapper (e.g. a vision/audio tower) or not present at all")

    model = get_peft_model(model, LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    ))
    model.print_trainable_parameters()
    detect_system_role_support(tokenizer)
    return model, tokenizer


# ==============================================================================================
# Stage 5 — Training
#
# Loss is computed only over the assistant turn. Rather than matching a literal
# response-template string (the Qwen script used "<|im_start|>assistant\n", which is
# not Gemma's), the prompt is tokenized separately and its tokens are masked with
# -100. Same completion-only loss, but it cannot silently mis-mask on a model whose
# template differs, and it needs no TRL version pinning.
# ==============================================================================================

def build_tokenizer_fn(tokenizer):
    def tokenize(ex):
        prompt_text = render_prompt(tokenizer, ex[INPUT_COLUMN])
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        # `or ""` on the row value as well as the eos token: load_training_data
        # already drops null rows, but this keeps a stray None from taking down a
        # multi-hour job at some arbitrary point mid-tokenisation.
        answer_ids = tokenizer((ex[OUTPUT_COLUMN] or "") + (tokenizer.eos_token or ""),
                               add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:MAX_SEQ_LENGTH]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_SEQ_LENGTH]
        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels}
    return tokenize


def train_model(model, tokenizer, train_raw, test_raw):
    from transformers import (DataCollatorForSeq2Seq, Trainer, TrainingArguments,
                              set_seed)
    import torch

    set_seed(RANDOM_SEED)
    tokenize = build_tokenizer_fn(tokenizer)

    eval_raw = test_raw
    if IN_TRAINING_EVAL_SUBSET and IN_TRAINING_EVAL_SUBSET < len(eval_raw):
        eval_raw = eval_raw.select(range(IN_TRAINING_EVAL_SUBSET))

    nproc = max(1, min(8, (os.cpu_count() or 2) // 2))
    train_tok = train_raw.map(tokenize, remove_columns=train_raw.column_names,
                              num_proc=nproc, desc="tokenising train")
    eval_tok = eval_raw.map(tokenize, remove_columns=eval_raw.column_names,
                            num_proc=nproc, desc="tokenising eval")

    # Drop rows whose answer was pushed entirely past MAX_SEQ_LENGTH — they carry
    # no trainable target and would contribute a NaN loss.
    has_target = lambda ex: any(l != -100 for l in ex["labels"])  # noqa: E731
    before = len(train_tok)
    train_tok = train_tok.filter(has_target, num_proc=nproc)
    eval_tok = eval_tok.filter(has_target, num_proc=nproc)
    if before != len(train_tok):
        print(f"[INFO] dropped {before - len(train_tok)} train rows with no target "
              f"within MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
    print(f"[OK] tokenised: train={len(train_tok)}  eval={len(eval_tok)}")

    # ---- how much of the corpus will actually be seen? ----
    # max_steps OVERRIDES num_train_epochs in HF Trainer whenever it is positive.
    # Setting both is the easy mistake: epochs is silently ignored and the run
    # covers however much max_steps happens to reach. Spell it out up front.
    # WORLD_SIZE matters: under DDP each of the N ranks consumes its own
    # per_device batch every micro-step, so one optimizer step sees N times more
    # data and an epoch takes N times fewer steps. Omitting it here would
    # overstate the step count (and the projected runtime) by exactly N.
    eff_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * WORLD_SIZE
    steps_per_epoch = max(1, len(train_tok) // eff_batch)
    print(f"[INFO] effective batch = {PER_DEVICE_TRAIN_BATCH_SIZE} x "
          f"{GRADIENT_ACCUMULATION_STEPS} x {WORLD_SIZE} gpu = {eff_batch}")
    print(f"[INFO] ~{steps_per_epoch:,} optimizer steps per epoch over {len(train_tok):,} rows")

    if MAX_STEPS and MAX_STEPS > 0:
        seen = MAX_STEPS * eff_batch
        frac = seen / max(1, len(train_tok))
        print(f"[WARN] MAX_STEPS={MAX_STEPS:,} OVERRIDES NUM_TRAIN_EPOCHS={NUM_TRAIN_EPOCHS}.")
        print(f"[WARN] This run will see ~{seen:,} examples = {frac:.2f} epochs "
              f"({frac * 100:.1f}% of one pass over the train split).")
        if frac < 1.0:
            print(f"[WARN] That is LESS THAN ONE FULL EPOCH. Set MAX_STEPS = -1 to "
                  f"train on the whole split for NUM_TRAIN_EPOCHS epochs "
                  f"(~{steps_per_epoch * NUM_TRAIN_EPOCHS:,} steps).")
    else:
        print(f"[INFO] MAX_STEPS disabled -> {NUM_TRAIN_EPOCHS} full epoch(s), "
              f"~{steps_per_epoch * NUM_TRAIN_EPOCHS:,} steps total")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # transformers v5 REMOVED the warmup_ratio argument (only warmup_steps
    # survives), so the ratio is resolved to an absolute step count here.
    _total_steps = MAX_STEPS if (MAX_STEPS and MAX_STEPS > 0)         else steps_per_epoch * NUM_TRAIN_EPOCHS
    warmup_steps_resolved = max(1, int(WARMUP_RATIO * _total_steps))
    print(f"[INFO] warmup {WARMUP_RATIO:.1%} of {_total_steps:,} steps "
          f"-> {warmup_steps_resolved:,} warmup steps")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=warmup_steps_resolved,
        weight_decay=WEIGHT_DECAY,
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs=({"use_reentrant": False}
                                        if GRADIENT_CHECKPOINTING else None),
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        # DISABLED. With this True the 30% gemma run saved checkpoint-200 of
        # 3,259 (verified by checksum: final_adapter == checkpoint-200). The
        # in-training eval uses only 500 examples so eval_loss is noisy; it
        # dipped at step 200 and never beat that, and the trainer faithfully
        # kept a model still inside warmup. The 10% run kept its LAST
        # checkpoint, so "10% vs 30%" was really "fully-trained vs
        # barely-started". For a scaling curve every point must train to
        # completion, so dataset size is the only variable.
        load_best_model_at_end=False,
        # group_by_length=True was removed in transformers v5 — it raises
        # TypeError: unexpected keyword argument. It only bucketed similar-length
        # sequences to cut padding waste, so losing it costs some throughput but
        # nothing in correctness. Do not re-add it on this stack.
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                             label_pad_token_id=-100),
    )

    # resume_from_checkpoint is safe on a fresh dir and lets a requeued job continue
    has_ckpt = any(d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR)) \
        if os.path.isdir(OUTPUT_DIR) else False
    result = trainer.train(resume_from_checkpoint=has_ckpt or None)

    # ---- measured throughput, and what a full epoch would actually cost ----
    # The smoke run exists partly to produce these two numbers: the real job's
    # --time is set from the measured s/step rather than an estimate.
    try:
        mt = result.metrics
        rt = mt.get("train_runtime", 0.0)
        done = max(1, int(trainer.state.global_step))
        s_per_step = rt / done
        print("\n" + "=" * 62)
        print(f"[THROUGHPUT] {done:,} steps in {rt/60:.1f} min "
              f"-> {s_per_step:.2f} s/step  (effective batch {eff_batch})")

        # Project the REAL corpus, not whatever subset this run used. Getting this
        # wrong once already produced "1 full epoch = 62 steps" from a 4k-row smoke
        # run — a 260x underestimate of the actual job.
        rows_here = len(train_tok)
        rows_full = FULL_TRAIN_ROWS if os.environ.get("DIAC_TRAIN_SIZE") else rows_here
        if rows_full != rows_here:
            print(f"[THROUGHPUT] NOTE: this run used {rows_here:,} rows (smoke override); "
                  f"projecting for the full {rows_full:,}")
        full_epoch_steps = max(1, rows_full // eff_batch)
        proj_h = full_epoch_steps * s_per_step / 3600.0
        print(f"[THROUGHPUT] 1 full epoch = {full_epoch_steps:,} steps "
              f"-> ~{proj_h:.1f} h ({proj_h/24:.1f} days) of training")
        print(f"[THROUGHPUT] + load/tokenise/scoring overhead (~2.5 h) "
              f"-> ~{proj_h + 2.5:.1f} h total")
        print("=" * 62 + "\n")
    except Exception as e:
        print(f"[warn] could not compute throughput: {type(e).__name__}: {e}")

    save_adapter(trainer, tokenizer)
    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(EVAL_DIR, "training_log.csv"), index=False)
    print(f"[OK] Saved training log -> {os.path.join(EVAL_DIR, 'training_log.csv')}")
    return trainer


def save_adapter(trainer, tokenizer):
    """LoRA adapter weights only — a few MB, reload onto the base with PeftModel."""
    try:
        os.makedirs(ADAPTER_DIR, exist_ok=True)
        trainer.save_model(ADAPTER_DIR)     # adapter_model.safetensors + adapter_config.json
        tokenizer.save_pretrained(ADAPTER_DIR)
        with open(os.path.join(ADAPTER_DIR, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "base_model": MODEL_ID, "use_4bit": USE_4BIT,
                "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
                "target_modules": LORA_TARGET_MODULES,
                "epochs": NUM_TRAIN_EPOCHS, "max_steps": MAX_STEPS,
                "lr": LEARNING_RATE, "max_seq_length": MAX_SEQ_LENGTH,
                "effective_batch": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "seed": RANDOM_SEED,
            }, f, indent=2)
        print(f"[OK] Saved LoRA adapter -> {ADAPTER_DIR}")
    except Exception as e:
        print(f"[WARN] Failed to save LoRA adapter: {e}")


def plot_training_curves(trainer):
    """Train loss vs. eval loss over training steps."""
    history = trainer.state.log_history
    train_steps, train_loss, eval_steps, eval_loss = [], [], [], []

    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step"))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step"))
            eval_loss.append(entry["eval_loss"])

    if not train_loss and not eval_loss:
        print("[WARN] No loss history found to plot.")
        return

    plt.figure(figsize=(8, 5))
    if train_loss:
        plt.plot(train_steps, train_loss, label="Train loss", marker="o", markersize=3)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Test (eval) loss", marker="s", markersize=3)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Qwen3.5-4B LoRA — Train vs. Test Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(EVAL_DIR, "train_test_loss_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved loss curve -> {out_path}")


# ==============================================================================================
# Stage 6 — Inference
# ==============================================================================================

def run_inference(model, tokenizer, pairs, name):
    """Greedy generation, checkpointed so a killed job doesn't lose the whole split."""
    import torch

    ckpt_path = os.path.join(EVAL_DIR, f"{name}_preds_checkpoint.csv")
    preds = [None] * len(pairs)

    if os.path.exists(ckpt_path):
        try:
            prev = pd.read_csv(ckpt_path, header=None, names=["idx", "pred"],
                               keep_default_na=False)
            for _, r in prev.iterrows():
                i = int(r["idx"])
                if 0 <= i < len(preds):
                    preds[i] = str(r["pred"])
            print(f"  resumed {sum(p is not None for p in preds)} cached predictions")
        except Exception as e:
            print(f"  could not read checkpoint ({type(e).__name__}); starting fresh")

    todo = [i for i in range(len(pairs)) if preds[i] is None]
    if not todo:
        print("  all predictions cached")
        return [p or "" for p in preds]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"      # right padding corrupts batched generation
    model.eval()

    def save_ckpt():
        done = [(i, preds[i]) for i in range(len(preds)) if preds[i] is not None]
        pd.DataFrame(done).to_csv(ckpt_path, index=False, header=False)

    # Length-bucketed so each batch pads to a similar length.
    order = sorted(todo, key=lambda i: len(pairs[i][0]))
    since_save = 0

    try:
        with torch.no_grad():
            for start in range(0, len(order), INFER_BATCH_SIZE):
                idxs = order[start:start + INFER_BATCH_SIZE]
                try:
                    # Same renderer as training — if these two ever diverge, the
                    # model is scored on a prompt format it was never fitted on.
                    prompts = [render_prompt(tokenizer, pairs[i][0]) for i in idxs]
                    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=MAX_SEQ_LENGTH
                                    ).to(model.device)
                    out_ids = model.generate(
                        **enc,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    gen_only = out_ids[:, enc["input_ids"].shape[1]:]
                    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
                    decoded = [clean_generation(d) for d in decoded]
                    for i, pred in zip(idxs, decoded):
                        preds[i] = pred.strip()
                except Exception as e:
                    print(f"  [warn] {name}: batch @ {start} failed "
                          f"({type(e).__name__}: {e}); leaving blank")
                    for i in idxs:
                        if preds[i] is None:
                            preds[i] = ""
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                since_save += len(idxs)
                if since_save >= 200:
                    save_ckpt(); since_save = 0
                if start % (INFER_BATCH_SIZE * 25) == 0:
                    done = sum(1 for p in preds if p is not None)
                    print(f"  {name}: {done}/{len(pairs)}")
        save_ckpt()
    finally:
        tokenizer.padding_side = prev_side

    return [p or "" for p in preds]


# ==============================================================================================
# Stage 7 — Evaluation, CSV export, performance plot
# ==============================================================================================

def evaluate_split(name, model, tokenizer, pairs):
    print(f"\n--- scoring {name} ({len(pairs)} examples) ---")
    inputs = [p[0] for p in pairs]
    references = [p[1] for p in pairs]
    predictions = run_inference(model, tokenizer, pairs, name)

    df = pd.DataFrame({"input": inputs, "reference": references, "prediction": predictions})
    csv_path = os.path.join(EVAL_DIR, f"{name}_predictions.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {len(df)} rows -> {csv_path}")

    blank = sum(1 for p in predictions if not p.strip())
    if blank:
        print(f"[WARN] {blank}/{len(predictions)} predictions are empty — each scores as "
              f"fully wrong and inflates the reported error rates.")

    metrics = compute_der_wer(predictions, references)
    metrics["split"] = name
    metrics["n_examples"] = len(df)
    metrics["n_empty_predictions"] = blank
    print(f"  DER_ce={metrics['DER_ce']}  DER_noce={metrics['DER_noce']}  "
          f"WER_ce={metrics['WER_ce']}  WER_noce={metrics['WER_noce']}")
    return metrics


def plot_performance_comparison(metrics_df):
    metric_cols = ["DER_ce", "DER_noce", "WER_ce", "WER_noce"]

    plt.figure(figsize=(8, 5))
    for _, row in metrics_df.iterrows():
        plt.plot(metric_cols, [row[c] for c in metric_cols], marker="o", label=row["split"])

    plt.xlabel("Metric")
    plt.ylabel("Error rate (%)")
    plt.title("Qwen3.5-4B LoRA — Train vs. Test vs. Benchmark")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(EVAL_DIR, "performance_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved performance comparison -> {out_path}")


# ==============================================================================================
# Main
# ==============================================================================================

def main():
    hf_login()

    # --- Data ---------------------------------------------------------------------------
    try:
        train_raw, test_raw = load_training_data()
    except Exception as e:
        print(f"[FATAL] Could not load {TRAIN_DATASET}: {e}")
        traceback.print_exc()
        # sys.exit(1), not return: a bare return exits 0, so SLURM records the job
        # as COMPLETED and the --mail-type=END,FAIL notification says it succeeded.
        # A crash must look like a crash to the scheduler.
        sys.exit(1)

    benchmark_raw = None
    try:
        benchmark_raw = load_benchmark_data()
    except Exception as e:
        print(f"[WARN] Could not load {BENCHMARK_DATASET}, continuing without it: {e}")

    # --- Model --------------------------------------------------------------------------
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"[FATAL] Could not load model/tokenizer: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Training -----------------------------------------------------------------------
    trainer = None
    try:
        trainer = train_model(model, tokenizer, train_raw, test_raw)
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        plot_training_curves(trainer)
    except Exception as e:
        print(f"[WARN] Failed to plot training curves: {e}")

    # ---- DDP: scoring and file writing happen on RANK 0 ONLY ----
    # Without this every rank runs the full generation pass over train+test+
    # benchmark and writes the SAME csv paths concurrently — 3x wasted GPU work
    # and a genuine risk of interleaved/corrupt output files. The smoke run made
    # this visible: every metric line printed three times.
    # Other ranks have already finished training (the adapter is saved by
    # Trainer on rank 0) and simply exit cleanly here.
    RANK = int(os.environ.get("RANK", "0"))
    if RANK != 0:
        print(f"[rank {RANK}] training complete; scoring runs on rank 0 only.")
        return

    # --- Evaluation: train, test, benchmark ----------------------------------------------
    all_metrics = []

    try:
        train_eval = train_raw
        if TRAIN_EVAL_SAMPLE_SIZE and TRAIN_EVAL_SAMPLE_SIZE < len(train_eval):
            train_eval = train_eval.shuffle(seed=RANDOM_SEED).select(
                range(TRAIN_EVAL_SAMPLE_SIZE))
        all_metrics.append(evaluate_split(
            "train", model, tokenizer, as_pairs(train_eval, INPUT_COLUMN, OUTPUT_COLUMN)))
    except Exception as e:
        print(f"[WARN] Train evaluation failed: {e}")
        traceback.print_exc()

    try:
        all_metrics.append(evaluate_split(
            "test", model, tokenizer, as_pairs(test_raw, INPUT_COLUMN, OUTPUT_COLUMN)))
    except Exception as e:
        print(f"[WARN] Test evaluation failed: {e}")
        traceback.print_exc()

    if benchmark_raw is not None:
        try:
            all_metrics.append(evaluate_split(
                "benchmark_sadeeddiac25", model, tokenizer, benchmark_pairs(benchmark_raw)))
        except Exception as e:
            print(f"[WARN] Benchmark evaluation failed: {e}")
            traceback.print_exc()

    # --- Metrics summary CSV --------------------------------------------------------------
    if all_metrics:
        try:
            metrics_df = pd.DataFrame(all_metrics)[
                ["split", "n_examples", "n_empty_predictions",
                 "DER_ce", "DER_noce", "WER_ce", "WER_noce"]
            ]
            metrics_df.insert(0, "model", MODEL_ID)
            metrics_df.insert(1, "method", "QLoRA" if USE_4BIT else "LoRA")

            summary_path = os.path.join(EVAL_DIR, "metrics_summary.csv")
            metrics_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"\n[OK] Saved metrics summary -> {summary_path}")
            print(metrics_df.to_string(index=False))

            plot_performance_comparison(metrics_df)
        except Exception as e:
            print(f"[WARN] Failed to save/plot metrics summary: {e}")
            traceback.print_exc()
    else:
        print("[WARN] No evaluation results were produced — skipping metrics summary.")

    print("\n[DONE] Run complete.")
    print(f"       adapter : {ADAPTER_DIR}")
    print(f"       metrics : {os.path.join(EVAL_DIR, 'metrics_summary.csv')}")


if __name__ == "__main__":
    main()
