"""
Qwen3.5-0.8B — FULL Fine-Tune for Arabic Diacritization
========================================================

Self-contained: metrics, data prep, training, inference and scoring all live in this one file.

Model         : Qwen/Qwen3.5-0.8B      (FULL fine-tune — every parameter trained, no LoRA)
Training data : Misraj/Sadeed_Tashkeela  train 1,042,698 / test 2,485
                columns: filename, input, output — already clean, NO preprocessing
Benchmark     : Misraj/SadeedDiac-25     split "train", 1,200 paragraphs
                columns: filename, input, output; CA/MSA split on `filename`
Evaluation    : DER / WER, with and without case ending, on train / test / benchmark,
                the benchmark scored THREE ways (CA, MSA, pooled) + a macro mean,
                and every split scored by BOTH scorers — see below

Ported from ../Train-Qwen/train_lora_qwen35_4b_diacritization.py. The metric functions, the
prompt, the NFC scoring, the generation settings and the effective batch are IDENTICAL, so the
benchmark numbers this produces are directly comparable to RUN_REPORT.md's 6.96 DER_noce.
Two variables change, deliberately and only two:

    1. the model      Qwen3.5-4B          -> Qwen3.5-0.8B
    2. the method     LoRA (0.50% params) -> full fine-tune (100% params)

------------------------------------------------------------------------------
WHY THIS IS FASTER AND WHY IT SHOULD STILL BE BETTER
------------------------------------------------------------------------------
RUN_REPORT.md §7 records eval_loss bottoming out at step 400 of 1500 and rising after, on
9.2% of one epoch. That is not a data-starved curve; it is a capacity/LR ceiling. LoRA r=16
gave 21.2M trainable parameters out of 4.23B. This run trades model size for trainable
fraction and for a full pass over the corpus:

                        4B LoRA (measured)        0.8B full FT (this file)
    trainable params    21,233,664  (0.50%)       ~751,000,000  (100%)
    corpus seen         96,000 rows (9.2%)        1,042,693 rows (100%)
    optimizer steps     1,500                     ~16,290
    wall clock          1h 29m                    ~3-5 h  (projected — VERIFY, see below)

The reference point for "is 0.8B big enough": the Sadeed paper reports DER_noce 5.26 from a
1.5B FULL fine-tune on this exact corpus, and Tashkeel-350M-v2 scores 6.49 zero-shot on this
benchmark. Both are smaller than 4B and both beat this project's fine-tuned 4B LoRA number.
Diacritization is a local, monotonic transduction; it needs orthographic and morphosyntactic
competence, not 4B of world knowledge.

DO NOT TRUST THE PROJECTED HOURS ABOVE. RUN_REPORT.md §5.6 records two premature cost
estimates from extrapolating the first few steps, which is meaningless here: the
length-grouped sampler sorts each megabatch longest-first, so early steps are the slowest in
their cycle, and step 1 additionally carries ~120s of Triton JIT. `--smoke-test` runs 60 steps
(more than one full 50-step megabatch cycle) and reports a steady-state figure with the first
20 steps discarded. That number is the estimate; this docstring is not.

------------------------------------------------------------------------------
WHAT CHANGED FROM THE 4B LoRA SCRIPT, AND WHY
------------------------------------------------------------------------------
1. fp32 MASTER WEIGHTS ARE MANDATORY, AND ARE ASSERTED BEFORE TRAINING.
   This is the single defect most likely to produce a run that completes, reports a
   plausible-looking loss, and has learned almost nothing. Load the model in bf16 and torch
   AdamW allocates exp_avg/exp_avg_sq via zeros_like(p) — i.e. also bf16. In bf16,
   1.0 + 1e-4 == 1.0 exactly, and at lr=2e-5 that is the scale of a typical update, so most
   of training silently rounds away. Train-Related-to-smaller-model/train_full_ft.py:178-187
   found this and solved it by REFUSING to run without DeepSpeed (ZeRO keeps fp32 master
   weights). At 9B that was the only option. At 0.75B it is not needed: load the model in
   fp32 and let `bf16=True` drive autocast, which is the stock HF mixed-precision recipe.
       fp32 weights 3.0GB + fp32 grads 3.0GB + AdamW m,v 6.0GB = 12.0GB
   on an 80GB H100, with the forward/backward still running in bf16. assert_master_dtype()
   checks this before a single step, because transformers 5.x resolves an unspecified `dtype`
   from the checkpoint's config (which says bfloat16) — the trap sets itself if you say nothing.

2. NO LoRA, SO NO peft. The 4B script's select_lora_targets() exists to keep adapters off the
   `visual.` and `mtp.` towers. Full FT does not need it, because AutoModelForCausalLM resolves
   to the text-only Qwen3_5ForCausalLM, which never ALLOCATES those towers
   (_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]). assert_text_only()
   verifies that empirically rather than trusting it, because if the composite class ever loads
   instead, ~100M vision+MTP parameters would silently receive AdamW state and gradients that
   contribute nothing to the loss.

3. GRADIENT CHECKPOINTING IS OFF BY DEFAULT. RUN_REPORT.md §6 identifies it as ~33% extra
   compute inherited from the gemma config, where it was needed. It is not needed here — see
   the memory budget at GRADIENT_CHECKPOINTING. This is the single largest free speed-up
   available and it costs nothing in accuracy.

4. FULL-FT HYPERPARAMETERS, not LoRA ones. lr 2e-4 -> 2e-5, warmup_steps 300 ->
   warmup_ratio 0.03, epochs -> exactly 1, max_steps -> -1 (the whole point is the full
   corpus). Values match Train-Related-to-smaller-model/config.py's TRAINING_ARGS so this is
   not a novel recipe.

5. eval/save INTERVALS ARE DERIVED FROM THE REAL STEP COUNT (config.schedule_for). The 4B
   script's fixed EVAL_STEPS=200 / SAVE_STEPS=200 over ~16,290 steps would be 81 evaluations
   and 81 checkpoints — and a full-FT checkpoint here is ~9GB (fp32 weights + fp32 optimizer
   state), not the 108MB a LoRA adapter costs. That is 81 x 9GB of Volume writes.

   A CHECKPOINT IS WRITTEN AT EVERY EVAL, and that is a correctness constraint rather than a
   preference. Evaluating more often than checkpointing silently breaks load_best_model_at_end
   in transformers 5.14.1 — Trainer records best_global_step at every eval but can only adopt
   a best checkpoint that exists on disk, so a minimum on an eval-only step is unreachable and
   the run ships the FINAL weights with no warning. schedule_for.__doc__ has the mechanism and
   the trainer.py lines; assert_best_model_selected() re-checks it after training.

6. THE BENCHMARK IS SCORED CA / MSA / POOLED FROM ONE GENERATION PASS. RUN_REPORT.md's
   headline finding is the 4x CA/MSA divergence (2.84 vs 11.07), but the 4B script only ever
   emitted a pooled number — the split was done by hand in the notebook afterwards. It is the
   number that decides what to do next, so it is computed here, natively, and the macro mean
   is written into the summary CSV alongside it.

9. EVERY SPLIT IS SCORED TWICE, BY TWO SCORERS, FROM ONE GENERATION PASS. The frozen metric
   block reproduces RUN_REPORT.md's 6.96 exactly; ../Train-Qwen/RESULTS.md then re-scored those
   same 4B predictions through a corrected scorer and retired 6.96 in favour of 8.48, with
   "use column (3) only against other column (3) numbers". Scoring only the frozen way would
   produce a number that cannot enter the corrected results table; scoring only the corrected
   way would break the 4B delta this package exists to measure. So both run, and the summary
   CSV carries DER_*/WER_* (frozen, for the delta) beside DER_*_corr/WER_*_corr (corrected,
   for citation). Costs CPU seconds, no GPU time. See the CORRECTED metrics block for the four
   defects — positional mark comparison, prediction-dependent denominator, free insertions, and
   U+0670 dagger alef being deleted before scoring rather than graded.

7. DETERMINISTIC EVAL ORDERING. transformers 5.14.1 drives BOTH samplers off the single
   `train_sampling_strategy` field, so "group_by_length" also makes _get_eval_sampler return a
   LengthGroupedSampler built with generator=None — a different dev ordering at every eval.
   eval_loss is a sample-weighted mean of per-batch token-mean losses, so regrouping moves the
   number for no modelling reason, and eval_loss is what load_best_model_at_end selects on.
   Fixed here (the 4B script has this bug); lifted from
   Train-Related-to-smaller-model/train_full_ft.py's DeterministicEvalTrainer.

8. DUAL PROMPT MODE. Defaults to the chat template with enable_thinking=False, byte-identical
   to the 4B run. If MODEL_ID has no chat template (e.g. Qwen/Qwen3.5-0.8B-Base) it falls back
   automatically to config.py's plain PROMPT_TMPL. Auto-detected, printed, and never mixed:
   train/eval prompt drift is a failure this project has already been bitten by.

Everything else — the metric functions, nfc(), the null-row guard, the materialized length
list, the fast-path assertion, the <|im_end|> EOS fix, the length-scaled max_new_tokens, the
--label namespacing, the complete-checkpoint resume — is carried over unchanged and is
commented at each site. See ../Train-Qwen/RUN_REPORT.md §5 for what each one cost to find.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python train_full_ft_qwen35_08b.py --analyze         # CPU: corpus + token economics
    python train_full_ft_qwen35_08b.py --smoke-test      # 60 steps, prints steady-state tok/s
    python train_full_ft_qwen35_08b.py --train-only
    python train_full_ft_qwen35_08b.py --eval-only --model <dir> --label ft
    python train_full_ft_qwen35_08b.py --eval-only --model none --label base   # zero-shot

The phase flags exist because Modal caps a function at 24h; see modal_app.py.
"""

import argparse
import dataclasses
import gc
import importlib
import inspect
import json
import math
import os
import random
import re
import time
import traceback
import unicodedata

import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless — required on a compute node
import matplotlib.pyplot as plt

# ==============================================================================================
# Configuration
# ==============================================================================================

# Qwen/Qwen3.5-0.8B (instruct) is the default rather than -Base, for ONE reason: comparability.
# The whole point of this run is to be diffable against RUN_REPORT.md's 4B LoRA numbers, and
# that run went through the instruct chat template with enable_thinking=False. -Base would
# change the prompt as well as the model and the method, confounding three variables instead of
# two. The 9B package chose -Base on the grounds that the instruct template's <think> block
# breaks prefix-based label masking — a real bug, but one the 4B script already fixed properly
# (render_prompt / assert_thinking_disabled), so it no longer argues for -Base.
#
# To run -Base anyway, it is one env var; the prompt mode switches itself:
#     MODEL_ID=Qwen/Qwen3.5-0.8B-Base python train_full_ft_qwen35_08b.py
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")

# LEAVE THIS EMPTY. Export HF_TOKEN in the environment instead (the Modal wrapper injects it
# from `modal secret create huggingface`). This file is committed; a token pasted here goes into
# git history and cannot be removed from it — that has already happened once in this repo
# (Models to test/instruct_models_v1.py ships a live hf_... string, which is why the top-level
# CLAUDE.md treats it as leaked).
HF_TOKEN = ""

# Qwen3.5 is natively supported by the pinned transformers (5.14.1 has Qwen3_5ForCausalLM), so
# nothing here needs to execute code downloaded from the Hub. Defaulting this to False closes
# that hole; set TRUST_REMOTE_CODE=1 only if you point MODEL_ID at an architecture this
# transformers does not know, and only for a repo you trust.
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "0") == "1"

# All read the environment so the Modal wrapper can redirect them onto the persistent Volume
# without forking this file.
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")
OUTPUT_DIR = os.environ.get("OUTPUT_ROOT", "./qwen3.5-0.8b-fullft-diacritization")
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
EVAL_DIR = os.environ.get("EVAL_ROOT", "./eval_outputs")

# ---- datasets ----
TRAIN_DATASET = "Misraj/Sadeed_Tashkeela"
INPUT_COLUMN = "input"        # undiacritized source
OUTPUT_COLUMN = "output"      # fully diacritized target
FILENAME_COLUMN = "filename"  # provenance; drives --include/--exclude-filename and --analyze

BENCHMARK_DATASET = "Misraj/SadeedDiac-25"
BENCHMARK_SPLIT = "train"         # SadeedDiac-25 exposes its 1,200 paragraphs under "train"
BENCHMARK_TEXT_COLUMN = "output"  # diacritized ground truth

# Pinned, and they must stay pinned. Identical to ../Train-Qwen and to
# Train-Related-to-smaller-model/config.py, so all three pipelines score the same test set.
# Unpinned, load_dataset() resolves whatever `main` points at on the day it runs — and a
# contamination-controlled study whose benchmark can move underneath it is not controlled.
TRAIN_DATASET_REVISION = os.environ.get(
    "TRAIN_DATASET_REVISION", "c10bcbb3b50dc96551f62c472389de666a8c1c4e"
)
BENCHMARK_DATASET_REVISION = os.environ.get(
    "BENCHMARK_DATASET_REVISION", "aa311213e44e4cab6cc3f2848daacd753adc1ce1"
)

# None = the whole split. THE DEFAULT IS THE WHOLE SPLIT, and that is the point of this run.
TRAIN_SIZE = None             # 1,042,698 rows
TEST_SIZE = None              #     2,485 rows
BENCHMARK_SIZE = None         #     1,200 rows

# How many TRAIN rows to generate predictions for when scoring. Scoring is generation, not a
# forward pass: all 1M rows would be hundreds of GPU-hours for a diagnostic number. 500 matches
# the 4B run so the train-vs-test gap stays comparable.
TRAIN_EVAL_SAMPLE_SIZE = 500

# ---- training ----
# Exactly one epoch, and max_steps disabled. RUN_REPORT.md's run saw 9.2% of one epoch; the
# reason this package exists is to see 100% of it. max_steps OVERRIDES num_train_epochs in HF
# Trainer whenever it is positive, so it is -1 here and the CLI flag exists only for rungs.
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))

# 8 x 8 = effective batch 64, IDENTICAL to the 4B LoRA run. Not tuned here on purpose: keeping
# the optimization geometry fixed is what makes the two benchmark numbers comparable.
#
# 8 and not 16, even though the model is 5.6x smaller: the ceiling is the LOGIT tensor, not the
# parameters. vocab_size is 248,320, so the logits are batch x seq x 248,320 and cross-entropy
# upcasts them to fp32. At batch 8 x seq 1024 that is ~4GB bf16 + ~8GB fp32 + gradient ~= 16-20GB,
# and it scales linearly with batch. 16 would put peak memory near 65GB on the longest
# length-grouped batches — and a length-grouped sampler puts the longest batch FIRST in every
# megabatch, so an OOM would not wait politely until the end of the run.
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8"))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "8"))

# Full-FT learning rate, NOT the LoRA 2e-4. LoRA adapters start at zero and need a large LR to
# move; every weight here is pretrained and 2e-4 would wreck them. 2e-5 + cosine + 3% warmup is
# Train-Related-to-smaller-model/config.py's full-FT recipe, unchanged, so this is not a novel
# choice.
#
# If the loss curve is still descending steeply at the end of the epoch, 3e-5 is the first knob
# to try — 0.75B tolerates more LR than the 9B that value was written for. Change it here, not
# mid-run; the cosine schedule is defined over the whole run.
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0     # HF default, spelled out: full FT can and does spike without it.
MAX_SEQ_LENGTH = 1024

# OFF, deliberately — RUN_REPORT.md §6 puts it at ~33% extra compute, inherited from a gemma
# config where it was needed. The budget on an 80GB H100 at 0.75B params, batch 8 x 1024:
#     fp32 weights + grads + AdamW m,v        12.0 GB
#     autocast bf16 weight copies            ~ 1.5 GB
#     logits + fp32 CE upcast + gradient     ~20.0 GB   <- the real consumer
#     activations, 24 layers, unchecked      ~ 6.0 GB
#                                            --------
#                                            ~40 GB of 80
# If a future change (longer sequences, bigger batch) breaks that, set GRADIENT_CHECKPOINTING=1
# rather than shrinking the batch — but re-measure, because it costs about a quarter of the run.
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1"

LOGGING_STEPS = 20
# Resident checkpoints, NOT total writes. Trainer rotates as it goes and protects the best one,
# so this bounds Volume usage at ~2 x 9GB regardless of how often checkpoints are written.
# Raise it (env) if you want to score intermediate checkpoints after the run — at the default
# only the best and the last survive to the end. See README "Variants".
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))

# How many evals across the WHOLE run, whatever its length. Fixed step intervals do not survive
# a 60-step smoke test and a 16,290-step epoch in the same file — see schedule_for().
N_EVALS = 20
# A checkpoint is written at EVERY eval. This MUST stay equal to N_EVALS: it is a correctness
# requirement, not a storage preference, and schedule_for() enforces it.
#
# It used to be 8, on the reasoning that a full-FT checkpoint here is ~9GB so saves should be
# rare. That reasoning was wrong twice over. SAVE_TOTAL_LIMIT already bounds resident
# checkpoints to 2 however often they are written, so saving less often bought no disk at all —
# only write time. And it silently broke load_best_model_at_end; see schedule_for().
N_SAVES = N_EVALS

# The in-training eval set. 1,000 rows of the 2,485-row test split — up from the 4B script's
# 500, because eval_loss here is doing more work: it selects the final checkpoint
# (load_best_model_at_end) across ~16,290 steps rather than 1,500. At batch 8 that is 125
# forward passes, ~15s, x20 evals = ~5 minutes over the whole run.
# select_eval_subset() picks WHICH 1,000, and documents what this signal can and cannot tell you.
IN_TRAINING_EVAL_SUBSET = 1000

# ---- smoke test ----
# 60 steps, not 20. transformers' LengthGroupedSampler builds megabatches of 50 x batch_size and
# sorts each longest-first, so step time oscillates on a ~50-step cycle; a 20-step measurement
# samples only the expensive half of one cycle. 60 steps covers a full cycle plus the 20 warmup
# steps that are discarded. This is the direct fix for RUN_REPORT.md §5.6's premature estimates.
SMOKE_MAX_STEPS = 60
SMOKE_WARMUP_STEPS_DISCARDED = 20
# 60 steps x effective batch 64 = 3,840 rows minimum. 12,000 gives headroom and a realistic
# length spread; drawn from a 60,000-row window so the smoke test never pays to filter 1M rows.
SMOKE_TRAIN_SIZE = 12000
SMOKE_SOURCE_WINDOW = 60000
# Usable rows in Sadeed_Tashkeela's train split after _drop_unusable_rows (1,042,698 minus the
# one null row — RUN_REPORT.md §5.3). Used ONLY so the smoke test can project the cost of a full
# epoch it is not itself running; the real run always counts its own rows. If the pinned corpus
# revision ever changes, this becomes stale and the smoke projection drifts — the real run's own
# "[INFO] ~N optimizer steps per epoch" line is authoritative.
FULL_CORPUS_ROWS = 1_042_693

# ---- inference ----
INFER_BATCH_SIZE = 8
# Not a flat 512. Output lengths on SadeedDiac-25 run p50=236 p90=355 p95=508 p99=607 max=708
# against the real Qwen3.5 tokenizer, so a flat 512 truncates the tail — and a truncated tail
# becomes deletions, which ARE penalized. Identical to the 4B run.
MAX_NEW_CAP = 1024
GEN_LEN_RATIO = 2.2

RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "أنت نظام متخصص في التشكيل الآلي للنصوص العربية. "
    "مهمتك إضافة الحركات (التشكيل) الصحيحة إلى النص العربي المُدخل دون تغيير الكلمات أو ترتيبها، "
    "مع مراعاة السياق النحوي والصرفي الكامل للجملة."
)

# Used only when the model has no chat template (e.g. -Base). Copied verbatim from
# Train-Related-to-smaller-model/config.py so a -Base run here matches a -Base run there.
PROMPT_TMPL = SYSTEM_PROMPT + "\n\nالنص:\n{inp}\n\nالنص المشكل:\n"

for _d in (CACHE_DIR, OUTPUT_DIR, EVAL_DIR):
    os.makedirs(_d, exist_ok=True)

random.seed(RANDOM_SEED)


# ==============================================================================================
# Evaluation metrics — copied verbatim from ../Train-Qwen and Models_Functions.py.
#
# DO NOT "improve" anything in this block. Every number in `Models to test & Results.xlsx` and
# in RUN_REPORT.md came out of these exact functions; changing one silently re-bases the whole
# results table. Scoring hygiene (what gets fed IN) lives in the next block, on purpose.
# ==============================================================================================

import jiwer  # noqa: E402

ARABIC_DIACRITICS = re.compile(r'[ً-ْ]')
_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_numerals(text: str) -> str:
    """Converts Eastern Arabic-Indic digits to Western Arabic digits."""
    return text.translate(_EASTERN_TO_WESTERN)


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
    cleaned = re.sub(r'[^\w\sً-ْ]', '', text)
    return cleaned.split()


def strip_case_ending(word: str) -> str:
    """Removes the diacritic(s) on the word-final letter (i'rab), leaving earlier ones intact."""
    last_base_idx = None
    for i, ch in enumerate(word):
        if not ARABIC_DIACRITICS.match(ch):
            last_base_idx = i
    if last_base_idx is None:
        return word
    return word[:last_base_idx + 1]


def _is_digit_only(word: str) -> bool:
    """True if the word (after stripping diacritics) is purely numeric."""
    return strip_diacritics(word).isdigit()


def _align_words(predictions: list, references: list):
    """Shared word-alignment step used by DER and WER."""
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
            # "insert" chunks (hallucinated extra words) are skipped.


def compute_der(predictions: list, references: list, ce: bool = True) -> float:
    """Diacritic Error Rate via edit-distance word alignment. Pure-digit words excluded."""
    total_chars, wrong_chars = 0, 0
    for r_word, p_word in _align_words(predictions, references):
        if _is_digit_only(r_word):
            continue
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
    """Word-level diacritization error rate. Numeric tokens are still scored here."""
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


# ==============================================================================================
# CORRECTED metrics — the second, independent scoring of the same predictions.
#
# WHY BOTH ARE COMPUTED, AND WHY NEITHER ONE ALONE IS ENOUGH.
#
# The block above is frozen for comparability: it is byte-identical to
# ../Train-Qwen/train_lora_qwen35_4b_diacritization.py, so the number it produces is directly
# diffable against RUN_REPORT.md's 6.96. That is the ONLY thing it is good for. ../Train-Qwen/
# RESULTS.md then re-scored those same saved 4B predictions through a corrected scorer and
# retired 6.96 in favour of 8.48:
#
#     SadeedDiac-25 macro DER_noce   | (1) old, no NFC | (2) old + NFC | (3) corrected |
#     4B LoRA                        |      18.33      |     6.96      |     8.48      |
#     Qwen3.5-4B zero-shot           |      69.80      |     65.90     |     59.57     |
#
# and states: "(3) is what to cite. Use column (3) only against other column (3) numbers."
#
# Scoring ONLY with the frozen block would produce a column-(2) number that cannot enter the
# corrected results table. Scoring ONLY with the corrected block would break the 4B delta this
# package exists to measure. So both run, over the SAME predictions, and both are written to
# metrics_summary_*.csv — frozen as DER_ce/DER_noce/WER_ce/WER_noce, corrected as *_corr.
# Generation is not repeated, so this costs CPU seconds and no GPU time at all.
#
# The four defects this block fixes, per RESULTS.md and verified against the 4B predictions:
#
#   1. Marks were compared as a flat positional list with the host letters discarded, so the
#      right marks on the wrong letters scored as correct ('رَسم' vs 'رسَم' -> DER_ce 0.00).
#      Here each word is segmented into (base char, marks) units and compared per base char.
#   2. The denominator counted MARKS and depended on the prediction
#      (n = max(len(r_diacs), len(p_diacs), 1)), so a model that emitted fewer marks got a
#      smaller denominator and a flattering score. Here it is the number of diacritizable
#      base characters in the REFERENCE — fixed per test set, and the Fadel/Sadeed definition
#      the published numbers use.
#   3. Insertions were free: `insert` chunks were skipped, so "correct answer + 50 junk words"
#      scored 0.00. Here inserted words count against both metrics.
#   4. The mark class [ً-ْ] stops at U+0652, so U+0670 dagger alef — ubiquitous in Classical
#      Arabic (هَٰذَا, الرَّحْمَٰن) — was deleted from BOTH sides before scoring and never graded.
#      Widened to [ً-ٰٟ], and tatweel is stripped from the skeleton.
#
# Ported verbatim from Train-Related-to-smaller-model/Models_Functions.py. Every name in here
# is prefixed `corr_` / `CORR_` because the frozen block above owns the unprefixed names and
# the two definitions genuinely differ — that collision is the whole point.
# ==============================================================================================

# Core tashkeel (U+064B-U+0652) + rarer marks through U+065F + U+0670 dagger alef.
CORR_DIACRITICS = re.compile(r'[ً-ٰٟ]')
# Base characters that can carry a diacritic. Excludes U+0640 tatweel (a stretching glyph,
# not a letter) and the diacritic ranges themselves.
CORR_LETTER = re.compile(r'[ء-غف-يٱ-ۓ]')
TATWEEL = "ـ"


def corr_strip_diacritics(text: str) -> str:
    """Consonantal skeleton: drops every diacritic AND tatweel.

    Dropping tatweel matters for alignment, not just tidiness: 'الرحمـن' and 'الرحمن' have
    different skeletons under the frozen scorer, so jiwer aligns them as a substitution and the
    reference word is scored as a deletion — a fabricated error.
    """
    return CORR_DIACRITICS.sub('', text).replace(TATWEEL, '')


def corr_clean_and_tokenize(text: str) -> list:
    """NFC, numerals, citation refs, tatweel, noise removal, whitespace tokenize.

    NFC happens HERE rather than at the call site (as the frozen path does it) so this scorer is
    correct no matter what it is handed.
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_numerals(text)
    text = strip_citation_refs(text)
    text = text.replace(TATWEEL, '')
    cleaned = re.sub(r'[^\w\sً-ٰٟ]', '', text)
    return cleaned.split()


def corr_segment(word: str) -> list:
    """Splits a word into [(base_char, marks), ...] units — the fix for defect 1.

    Marks are sorted so two canonically-equivalent orderings of the same cluster
    (shadda+damma vs damma+shadda) compare equal even where NFC did not reorder them.
    Leading marks with no base character are discarded.
    """
    units = []
    for ch in unicodedata.normalize("NFC", word):
        if CORR_DIACRITICS.match(ch):
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(base, "".join(sorted(marks))) for base, marks in units]


def corr_scorable(units: list, ce: bool) -> list:
    """Diacritizable units of a word, dropping the final one when ce=False.

    Note this is a different 'without case ending' convention from the frozen block's
    strip_case_ending(), which merely blanks the final letter's marks. Dropping the unit
    outright is Fadel's "excluding last character", which is what the published numbers use.
    """
    idx = [i for i, (base, _) in enumerate(units) if CORR_LETTER.match(base)]
    if not ce and idx:
        idx = idx[:-1]
    return [units[i] for i in idx]


def corr_is_digit_only(word: str) -> bool:
    """True if the word is purely numeric — it can never carry a diacritic."""
    return corr_strip_diacritics(word).isdigit()


def corr_align_words(predictions: list, references: list) -> list:
    """Aligns predictions to references on their skeletons, EMITTING INSERTIONS.

    Returns (ref_or_None, pred_or_None) pairs:
        (ref, pred) matched   (ref, None) deletion/substitution   (None, pred) INSERTION

    Empty skeletons are filtered from both lists in lockstep before joining, so a stray
    whitespace-separated bare diacritic cannot make jiwer's token count disagree with ours and
    shift every later pairing in the sentence.
    """
    pairs = []
    for pred, ref in zip(predictions, references):
        pred_words = [w for w in corr_clean_and_tokenize(pred) if corr_strip_diacritics(w)]
        ref_words = [w for w in corr_clean_and_tokenize(ref) if corr_strip_diacritics(w)]
        pred_skel = [corr_strip_diacritics(w) for w in pred_words]
        ref_skel = [corr_strip_diacritics(w) for w in ref_words]

        alignment = jiwer.process_words(" ".join(ref_skel), " ".join(pred_skel))

        for chunk in alignment.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    pairs.append(
                        (ref_words[i],
                         pred_words[chunk.hyp_start_idx + (i - chunk.ref_start_idx)])
                    )
            elif chunk.type == "substitute":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    pairs.append((ref_words[i], None))
                for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    pairs.append((None, pred_words[j]))
            elif chunk.type == "delete":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    pairs.append((ref_words[i], None))
            elif chunk.type == "insert":
                for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    pairs.append((None, pred_words[j]))
    return pairs


def corr_compute_der(pairs, ce: bool = True) -> float:
    """DER per diacritizable REFERENCE character. Insertions count against both terms."""
    total, wrong = 0, 0
    for r_word, p_word in pairs:
        if r_word is None:                       # insertion
            if p_word is None or corr_is_digit_only(p_word):
                continue
            n = len(corr_scorable(corr_segment(p_word), ce))
            total += n
            wrong += n
            continue

        if corr_is_digit_only(r_word):
            continue

        r_units = corr_scorable(corr_segment(r_word), ce)
        total += len(r_units)

        if p_word is None:                       # deletion / substitution
            wrong += len(r_units)
            continue

        p_units = corr_scorable(corr_segment(p_word), ce)
        for i, (r_base, r_marks) in enumerate(r_units):
            if i >= len(p_units):
                wrong += 1
            elif p_units[i][0] != r_base or p_units[i][1] != r_marks:
                wrong += 1

    return round((wrong / total) * 100, 2) if total > 0 else float("nan")


def corr_compute_wer(pairs, ce: bool = True) -> float:
    """Word-level rate. Deletions AND insertions count as wrong; insertions enlarge the
    denominator too, so the rate stays inside [0, 100]."""
    total, wrong = 0, 0
    for r_word, p_word in pairs:
        total += 1
        if r_word is None or p_word is None:     # insertion or deletion
            wrong += 1
            continue
        if corr_scorable(corr_segment(r_word), ce) != corr_scorable(corr_segment(p_word), ce):
            wrong += 1

    return round((wrong / total) * 100, 2) if total > 0 else float("nan")


def compute_der_wer_corrected(predictions: list, references: list) -> dict:
    """The four corrected metrics, suffixed `_corr` so they sit beside the frozen ones.

    Aligns ONCE and reuses the pairs for all four numbers; the frozen block realigns per call,
    i.e. twelve full jiwer passes per evaluation against this one.
    """
    pairs = corr_align_words(predictions, references)
    return {
        "DER_ce_corr": corr_compute_der(pairs, ce=True),
        "DER_noce_corr": corr_compute_der(pairs, ce=False),
        "WER_ce_corr": corr_compute_wer(pairs, ce=True),
        "WER_noce_corr": corr_compute_wer(pairs, ce=False),
    }


# ==============================================================================================
# Scoring hygiene — what gets fed INTO the metric functions above.
# ==============================================================================================

def nfc(text: str) -> str:
    """Canonical Unicode ordering.

    NOT cosmetic and NOT optional. ~90% of SadeedDiac-25's gold rows store their diacritics in
    non-canonical combining-mark order — a storage difference, not a diacritization difference —
    and compute_der compares the ARABIC_DIACRITICS.findall() lists POSITIONALLY. Any
    tokenizer-based model naturally emits NFC order, so without this a semantically identical
    word scores as wrong. RUN_REPORT.md §2a measures the gap on identical predictions:
    DER_noce 19.13 raw vs 7.62 NFC-normalized — 2.4x.

    Applied here rather than inside compute_der_wer so the metric functions stay byte-identical
    to Models_Functions.py.
    """
    return unicodedata.normalize("NFC", text)


# A leading reasoning block, if the model emits one despite enable_thinking=False. <think> and
# </think> are NOT special tokens in this tokenizer (added_tokens_decoder marks them
# special=false), so skip_special_tokens=True does NOT strip them — they arrive as literal text
# and would be scored as content.
_THINK_RE = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.DOTALL)

# The plain-prompt path only: a -Base model with no EOS training will happily echo its own
# prompt header back before answering.
_ECHOED_HEADER_RE = re.compile(r"^\s*النص المشكل\s*:?\s*", re.MULTILINE)


def clean_output(text: str, chat_mode: bool = True) -> str:
    """Strips whatever the model wrapped around its answer.

    Two modes, because the two prompt formats fail differently:

    chat_mode (Qwen3.5-0.8B, the default) — strip a leading reasoning block only. Do NOT
    truncate at the first blank line: this model emits <|im_end|>, and truncating on "\\n\\n"
    would silently amputate any target that legitimately contains a blank line.

    plain mode (-Base) — additionally strip an echoed "النص المشكل:" header and truncate at the
    first blank line, because a base model has no turn-end token and will generate to the cap
    forever, usually by inventing a new document. Same treatment as
    Train-Related-to-smaller-model/evaluate.py's clean_output.
    """
    out = _THINK_RE.sub("", text.strip()).strip()
    if not chat_mode:
        out = _ECHOED_HEADER_RE.sub("", out).strip()
        out = out.split("\n\n")[0].strip()
    return out


# ==============================================================================================
# Stage 0 — Preflight
# ==============================================================================================

def hf_login():
    token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        print("[WARN] No HF token set. Qwen/Qwen3.5-0.8B itself is UNGATED (verified), but "
              "Misraj/Sadeed_Tashkeela is gated:manual and WILL 401. Set HF_TOKEN at the top "
              "of this file or export it.")
        return
    try:
        from huggingface_hub import login
        login(token=token)
        print("[OK] Logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"[WARN] Hugging Face login failed, continuing anyway: {e}")


def assert_fast_path():
    """Fail before the GPU starts billing if Qwen3.5's linear-attention kernels are missing.

    Qwen3.5-0.8B is hybrid: config.json's layer_types gives 18 of its 24 layers as
    `linear_attention` (Gated-DeltaNet) and only 6 as `full_attention`
    (full_attention_interval=4). Without causal-conv1d and flash-linear-attention,
    modeling_qwen3_5's `is_fast_path_available` is False and those 18 layers fall back to
    `torch_chunk_gated_delta_rule`, a pure-PyTorch chunked scan. It announces itself with a
    single logger.warning_once that scrolls past in any real training log, so the run looks
    healthy while costing several times more GPU time.

    Note this is a LARGER share of the model than at 4B (18/24 = 75% vs 24/32 = 75% — same
    ratio, but here the full-attention layers are cheaper still), so the fallback hurts more.
    """
    missing = []
    for module_name, pip_name in (("causal_conv1d", "causal-conv1d"),
                                  ("fla", "flash-linear-attention")):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        raise RuntimeError(
            f"Qwen3.5 linear-attention kernels unavailable: missing {missing}. 18 of 24 layers "
            f"would silently run a pure-PyTorch fallback at several times the GPU cost. Install "
            f"with: pip install {' '.join(missing)}  (causal-conv1d is sdist-only and compiles "
            f"CUDA from source — set TORCH_CUDA_ARCH_LIST and MAX_JOBS). Set "
            f"REQUIRE_FAST_PATH=0 to override, but do not do that on a paid GPU."
        )

    # Second, independent check: the packages can import and still not be wired up, e.g. a
    # causal-conv1d built for a different arch. transformers exposes the resolved answer.
    try:
        mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except Exception:
        print("[INFO] could not import modeling_qwen3_5 to verify is_fast_path_available; "
              "the package imports above succeeded, continuing")
        return
    flag = getattr(mod, "is_fast_path_available", None)
    if callable(flag):
        flag = flag()
    if flag is False:
        raise RuntimeError(
            "causal-conv1d and flash-linear-attention import, but transformers reports "
            "is_fast_path_available=False — the kernels are present but not usable (usually a "
            "build for the wrong CUDA arch). Fix the build; do not pay for the fallback."
        )
    print(f"[OK] Qwen3.5 linear-attention fast path available: {flag}")


# ==============================================================================================
# Stage 1 — Prompt construction
#
# ONE render function for training and inference, both modes. Train/eval prompt drift is a
# failure this project has already been bitten by (Train-Related-to-smaller-model/config.py:90-95),
# and it is invisible in the loss curve — it only shows up as a bad DER at the very end.
# ==============================================================================================

_CHAT_MODE = None            # True = chat template, False = plain PROMPT_TMPL
_SUPPORTS_SYSTEM_ROLE = None


def detect_prompt_mode(tokenizer):
    """Chat template if the tokenizer has one, plain PROMPT_TMPL otherwise. Printed, not assumed.

    Qwen/Qwen3.5-0.8B ships chat_template.jinja; Qwen/Qwen3.5-0.8B-Base does not (verified
    against the HF file listing). Detecting rather than hardcoding means MODEL_ID can be swapped
    without editing this file, and means a future checkpoint that quietly gains or loses a
    template cannot silently change the prompt underneath a comparison.
    """
    global _CHAT_MODE, _SUPPORTS_SYSTEM_ROLE

    template = getattr(tokenizer, "chat_template", None)
    if not template:
        _CHAT_MODE, _SUPPORTS_SYSTEM_ROLE = False, False
        print(f"[INFO] {MODEL_ID} has NO chat template -> plain PROMPT_TMPL mode.")
        print("[WARN] This is NOT the prompt RUN_REPORT.md's 4B numbers were produced with. "
              "The benchmark score will still be valid, but the 4B-vs-0.8B delta will confound "
              "the prompt change with the model change. Say so wherever you report it.")
        return False

    _CHAT_MODE = True
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        _SUPPORTS_SYSTEM_ROLE = True
    except Exception:
        _SUPPORTS_SYSTEM_ROLE = False
    print(f"[INFO] chat-template mode; system role supported: {_SUPPORTS_SYSTEM_ROLE}")
    return True


def build_prompt_messages(user_text: str) -> list:
    """The prompt turns only — no assistant turn."""
    if _SUPPORTS_SYSTEM_ROLE:
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}]
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"}]


def render_prompt(tokenizer, user_text: str) -> str:
    """THE single prompt render. Training and inference both go through it.

    enable_thinking=False is the whole point of the chat branch. Without it Qwen3.5's template
    ends the prompt at "<|im_start|>assistant\\n<think>\\n" — an unclosed reasoning block.
    Training through that prompt teaches the model to emit the diacritized answer AS reasoning
    content, and inference then emits a think block that the metric scores as garbage. With it
    the template closes the block itself and the answer starts where the labels start.
    """
    if _CHAT_MODE is None:
        raise RuntimeError("detect_prompt_mode() must run before render_prompt()")
    if not _CHAT_MODE:
        return PROMPT_TMPL.format(inp=user_text)
    return tokenizer.apply_chat_template(
        build_prompt_messages(user_text),
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def assert_prompt_sane(tokenizer):
    """Check the rendered prompt before any GPU work, so an upstream template change is cheap."""
    rendered = render_prompt(tokenizer, "نص تجريبي")
    tail = rendered[-60:].replace("\n", "\\n")
    if _CHAT_MODE:
        if "<think>" in rendered and "</think>" not in rendered:
            raise RuntimeError(
                f"chat template left a <think> block OPEN despite enable_thinking=False. "
                f"Prompt tail: ...{tail!r}. Training through this would teach the model to "
                f"answer inside a reasoning block. Fix render_prompt before spending GPU time."
            )
        if rendered.rstrip().endswith("<think>"):
            raise RuntimeError(f"prompt ends on a bare <think> tag: ...{tail!r}")
    elif SYSTEM_PROMPT not in rendered:
        raise RuntimeError("plain prompt did not include SYSTEM_PROMPT — check PROMPT_TMPL")
    print(f"[OK] prompt renders; tail: ...{tail!r}")
    return rendered


def generation_eos_ids(tokenizer) -> list:
    """Every token id that should stop generation.

    Qwen3.5 ships NO generation_config.json, and its config eos_token_id is <|endoftext|>
    (248044) while a chat turn ends with <|im_end|> (248046) — which is what tokenizer.eos_token
    is, and therefore what build_tokenizer_fn appends to the labels and what training teaches the
    model to emit. Left to the config alone, generate() never sees its stop token and every
    single row runs to max_new_tokens: slow, and the trailing junk is scored. RUN_REPORT.md §3
    measured 280/1,200 zero-shot rows hitting the cap for exactly this reason.
    """
    vocab = tokenizer.get_vocab()
    ids = set()
    for tok in ("<|im_end|>", "<|endoftext|>"):
        if tok in vocab:
            ids.add(vocab[tok])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


# ==============================================================================================
# Stage 2 — Data
#
# Sadeed_Tashkeela ships already cleaned, chunked and filtered: input/output pairs are used
# as-is, with no preprocessing beyond wrapping them in the prompt format.
#
# NOTE ON DECONTAMINATION: this pipeline does NOT run an extra contamination pass, and that is
# deliberate rather than an omission. The 4B run it is being compared against did not run one
# either (the project's 0.4% overlap figure is a property of how SadeedDiac-25 was built against
# this corpus, not of a filter applied here). Adding a decontamination step here would confound
# the 4B-vs-0.8B comparison with a data change. Train-Related-to-smaller-model/decontaminate.py
# is the tool if you want that as a separate, deliberately-scoped experiment.
# ==============================================================================================

def _nproc():
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _drop_unusable_rows(ds, name, columns):
    """Drop rows where any required column is null or blank, and say how many.

    Sadeed_Tashkeela is documented as already cleaned and filtered, so this looks unnecessary.
    It is not: EXACTLY 1 row of 1,042,698 has a null text column (0.0001%), and it killed a
    real training run ~12 minutes in, with the GPU already rented, on
        TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
    A 200-row smoke test cannot find it — only the full corpus contains it. RUN_REPORT.md §5.3.

    Dropping rather than coercing to "" is deliberate. A null target coerced to an empty string
    is a training example that teaches the model to answer with nothing. The count is printed so
    a corpus that is suddenly 30% null cannot pass silently as a corpus with one bad row.
    """
    def ok(ex):
        return all(isinstance(ex[c], str) and ex[c].strip() for c in columns)

    before = len(ds)
    ds = ds.filter(ok, num_proc=_nproc(), desc=f"dropping unusable {name} rows")
    dropped = before - len(ds)
    if dropped:
        pct = 100 * dropped / max(before, 1)
        print(f"[WARN] {name}: dropped {dropped:,}/{before:,} rows ({pct:.3f}%) with a null or "
              f"blank {columns} column — this corpus is documented as pre-cleaned, so a large "
              f"number here means something changed upstream.")
    else:
        print(f"[OK] {name}: no null/blank rows in {columns}")
    return ds


def _apply_filename_filters(ds, name, include=None, exclude=None):
    """Optional regex filters on the `filename` column. Off by default; both are printed.

    This is the mechanism for RUN_REPORT.md §9.2 — "a domain-balanced rung, not a longer one".
    The deficit is MSA (11.07 vs CA 2.84) and it is created by training on a Tashkeela-derived,
    Classical-heavy corpus. No default mix is hardcoded here because guessing one without
    knowing the corpus composition would be worse than not offering it: run `--analyze` first,
    read the filename breakdown it prints, then come back and pass a regex.
    """
    if not include and not exclude:
        return ds
    if FILENAME_COLUMN not in ds.column_names:
        print(f"[WARN] {name}: no {FILENAME_COLUMN!r} column; filename filters ignored")
        return ds

    before = len(ds)
    if include:
        rx = re.compile(include)
        ds = ds.filter(lambda ex: bool(rx.search(ex[FILENAME_COLUMN] or "")),
                       num_proc=_nproc(), desc=f"include-filter {name}")
        print(f"[INFO] {name}: --include-filename {include!r} kept {len(ds):,}/{before:,}")
    if exclude:
        rx = re.compile(exclude)
        n0 = len(ds)
        ds = ds.filter(lambda ex: not rx.search(ex[FILENAME_COLUMN] or ""),
                       num_proc=_nproc(), desc=f"exclude-filter {name}")
        print(f"[INFO] {name}: --exclude-filename {exclude!r} kept {len(ds):,}/{n0:,}")

    if len(ds) == 0:
        raise RuntimeError(
            f"{name}: filename filters removed every row. Run --analyze to see the real "
            f"filename distribution before choosing a pattern."
        )
    return ds


def load_training_data(train_size=None, test_size=None, smoke=False,
                       include_filename=None, exclude_filename=None):
    from datasets import load_dataset

    ds = load_dataset(TRAIN_DATASET, cache_dir=CACHE_DIR, revision=TRAIN_DATASET_REVISION)
    print(f"[INFO] {TRAIN_DATASET} pinned revision: {TRAIN_DATASET_REVISION}")
    train_raw, test_raw = ds["train"], ds["test"]

    for col in (INPUT_COLUMN, OUTPUT_COLUMN):
        if col not in train_raw.column_names:
            raise KeyError(f"column '{col}' missing; found {train_raw.column_names}")

    # The smoke test must not pay to filter 1M rows just to train on 12k of them. Take a window
    # FIRST, then filter, then shuffle within it — the point of the smoke test is the throughput
    # measurement, and for that a 60k-row window has a realistic enough length distribution.
    if smoke:
        train_raw = train_raw.select(range(min(SMOKE_SOURCE_WINDOW, len(train_raw))))

    train_raw = _drop_unusable_rows(train_raw, "train", (INPUT_COLUMN, OUTPUT_COLUMN))
    test_raw = _drop_unusable_rows(test_raw, "test", (INPUT_COLUMN, OUTPUT_COLUMN))

    train_raw = _apply_filename_filters(train_raw, "train", include_filename, exclude_filename)

    if smoke:
        train_raw = train_raw.shuffle(seed=RANDOM_SEED).select(
            range(min(SMOKE_TRAIN_SIZE, len(train_raw))))

    # Size caps come AFTER filtering, so --train-size means "N usable rows" rather than
    # "N rows, some of which silently vanish".
    train_size = TRAIN_SIZE if train_size is None else train_size
    test_size = TEST_SIZE if test_size is None else test_size
    if train_size:
        train_raw = train_raw.select(range(min(train_size, len(train_raw))))
    if test_size:
        test_raw = test_raw.select(range(min(test_size, len(test_raw))))

    print(f"[OK] {TRAIN_DATASET}: train={len(train_raw):,}  test={len(test_raw):,}")
    return train_raw, test_raw


def load_benchmark_data():
    from datasets import load_dataset

    ds = load_dataset(BENCHMARK_DATASET, cache_dir=CACHE_DIR,
                      revision=BENCHMARK_DATASET_REVISION)
    print(f"[INFO] {BENCHMARK_DATASET} pinned revision: {BENCHMARK_DATASET_REVISION}")
    if BENCHMARK_SPLIT not in ds:
        avail = list(ds.keys())
        print(f"[WARN] split '{BENCHMARK_SPLIT}' not found; available: {avail} — using '{avail[0]}'")
        raw = ds[avail[0]]
    else:
        raw = ds[BENCHMARK_SPLIT]

    col = BENCHMARK_TEXT_COLUMN
    if col not in raw.column_names:
        raise KeyError(f"column '{col}' missing; found {raw.column_names}")

    # benchmark_pairs() calls strip_diacritics() on the gold text, which would raise on a null
    # exactly the way training did. This should print 0 — if it ever does not, say so loudly,
    # because rows silently vanishing from a 1,200-row benchmark changes the reported score and
    # breaks comparability with every other model in the results table.
    raw = _drop_unusable_rows(raw, "benchmark", (col,))

    if BENCHMARK_SIZE:
        raw = raw.select(range(min(BENCHMARK_SIZE, len(raw))))

    print(f"[OK] {BENCHMARK_DATASET}: {len(raw)} paragraphs (light preprocessing only)")
    return raw


def as_pairs(dataset, input_col, output_col):
    """(prompt_text, gold_text) pairs, ready for generation and scoring."""
    return [(dataset[i][input_col], dataset[i][output_col]) for i in range(len(dataset))]


def benchmark_pairs(dataset):
    """Light preprocessing: the prompt is the gold text with diacritics removed.

    SadeedDiac-25 DOES ship an `input` column, but the 4B run built the prompt by stripping the
    gold text instead (notebook §3), and so does this. Not because stripping is better — because
    it is what the number being compared against used. Do not switch to the `input` column
    without re-running the 4B baseline through the same change.
    """
    golds = dataset[BENCHMARK_TEXT_COLUMN]
    return [(strip_diacritics(g), g) for g in golds]


def benchmark_domains(dataset):
    """"CA" / "MSA" per row, from the filename. Returns None if the column is absent.

    Convention copied from Train-Related-to-smaller-model/config.is_ca: "fadel" in the filename
    means Classical Arabic (SadeedDiac-25's CA half is Fadel_test.txt). Verified to give exactly
    600/600 on the current dump — the assertion below catches it if that ever stops being true,
    because a silently lopsided split would make the CA/MSA gap unreadable.
    """
    if FILENAME_COLUMN not in dataset.column_names:
        print(f"[WARN] benchmark has no {FILENAME_COLUMN!r} column — CA/MSA split unavailable")
        return None
    domains = ["CA" if "fadel" in (f or "").lower() else "MSA"
               for f in dataset[FILENAME_COLUMN]]
    n_ca = domains.count("CA")
    n_msa = len(domains) - n_ca
    print(f"[OK] benchmark domains: CA={n_ca}  MSA={n_msa}")
    if min(n_ca, n_msa) == 0:
        print("[WARN] one domain is empty — the 'fadel' filename convention no longer holds on "
              "this dump. CA/MSA numbers below are not trustworthy.")
    return domains


# ==============================================================================================
# Stage 3 — Model
# ==============================================================================================

def _from_pretrained(cls, model_id, dtype, **kwargs):
    """from_pretrained with the transformers 4.x / 5.x dtype-kwarg rename handled once.

    transformers 5.x renamed `torch_dtype` to `dtype`. The Modal image pins 5.14.1, but this
    file is also meant to run in Colab on whatever is installed there.
    """
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def assert_text_only(model):
    """Confirm AutoModelForCausalLM did NOT pull in the vision tower or the MTP head.

    Qwen/Qwen3.5-0.8B's config declares architectures=["Qwen3_5ForConditionalGeneration"] and
    carries a 12-layer vision tower plus a 1-layer MTP head. AutoModelForCausalLM resolves to a
    DIFFERENT, text-only class — Qwen3_5ForCausalLM — whose __init__ constructs only
    self.model + self.lm_head and which lists
        _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]
    so those parameters are never allocated at all.

    Under LoRA, getting this wrong wasted a little VRAM. Under FULL fine-tuning it would be
    much worse: every one of those ~100M parameters would receive gradients and two AdamW
    moment tensors, would be updated by weight decay every step, and would be written into
    every 9GB checkpoint — all for parameters that are not in the causal-LM forward path and
    contribute nothing to the loss. Check, do not assume.
    """
    names = {n for n, _ in model.named_modules()}
    bad = sorted(n for n in names
                 if any(part in ("visual", "mtp") for part in n.split(".")))
    if bad:
        raise RuntimeError(
            f"non-text towers were allocated ({bad[:5]}{'...' if len(bad) > 5 else ''}). "
            f"AutoModelForCausalLM resolved to {type(model).__name__} instead of the text-only "
            f"Qwen3_5ForCausalLM. Full fine-tuning this would train, decay and checkpoint ~100M "
            f"parameters that never see the loss. Stop and fix the model class."
        )
    print(f"[OK] {type(model).__name__}: text decoder only, no vision/MTP submodules.")


def assert_master_dtype(model):
    """Refuse to train with bf16 master weights. THE most important guard in this file.

    With the model loaded in bf16, torch AdamW allocates exp_avg/exp_avg_sq via zeros_like(p),
    so the optimizer state is bf16 too and there are no fp32 master weights. In bf16,
    1.0 + 1e-4 == 1.0 EXACTLY, and at lr=2e-5 that is the scale of a typical update — so most
    of training rounds away and the run completes, logs a falling-looking loss, and has learned
    a fraction of what it should have. Train-Related-to-smaller-model/train_full_ft.py:178-187
    reached the same conclusion and solved it by requiring DeepSpeed ZeRO (which keeps fp32
    master weights); at 0.75B that is unnecessary machinery — fp32 weights + bf16 autocast is
    the stock recipe and costs 12GB.

    This has to be an assertion rather than a comment because transformers 5.x resolves an
    unspecified dtype from the checkpoint config, and Qwen3.5-0.8B's config.json says
    "dtype": "bfloat16". Say nothing and you get the broken path by default.
    """
    import torch
    dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    if dtypes != {torch.float32}:
        raise RuntimeError(
            f"trainable parameters are {sorted(str(d) for d in dtypes)}, not float32. Full "
            f"fine-tuning needs fp32 master weights: torch AdamW allocates its moments with "
            f"zeros_like(p), so bf16 weights give bf16 optimizer state, and at lr={LEARNING_RATE} "
            f"most updates round to zero in bf16. Load with dtype=torch.float32 and let "
            f"TrainingArguments(bf16=True) drive autocast for the forward/backward."
        )
    print("[OK] fp32 master weights confirmed (forward/backward still run in bf16 via autocast)")


def report_parameters(model):
    """Print the parameter breakdown, including the share that is the tied vocabulary table."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Embedding" and hasattr(module, "weight"):
            embed = max(embed, module.weight.numel())

    print(f"[INFO] total params      {total:,}")
    print(f"[INFO] trainable params  {trainable:,}  ({100 * trainable / max(total, 1):.2f}%)")
    if embed:
        # tie_word_embeddings=true in this config, so the input embedding and lm_head are ONE
        # tensor: ~34% of the trainable parameters are the 248,320-row vocabulary table, most of
        # whose rows are non-Arabic and will receive essentially no gradient on this corpus.
        # Harmless (decoupled weight decay over one epoch shrinks an ungradiented row by ~0.2%),
        # but it is why "751M trainable" overstates how much of the model is actually learning.
        print(f"[INFO] tied embedding    {embed:,}  "
              f"({100 * embed / max(trainable, 1):.1f}% of trainable; input embedding and "
              f"lm_head are the same tensor)")
    # Memory arithmetic, printed so an OOM later is diagnosable rather than mysterious.
    gb = trainable * 4 / 1e9
    print(f"[INFO] optimizer budget  fp32 weights {gb:.1f}GB + grads {gb:.1f}GB + "
          f"AdamW m,v {2 * gb:.1f}GB = {4 * gb:.1f}GB")
    return total, trainable


def load_model_for_training():
    """Tokenizer + fp32 model, with every precondition asserted before any step runs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=TRUST_REMOTE_CODE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"          # training; flipped to "left" for generation

    detect_prompt_mode(tokenizer)
    assert_prompt_sane(tokenizer)

    # fp32 EXPLICITLY — see assert_master_dtype. No device_map: on one GPU it only installs
    # accelerate hooks and sets hf_device_map, which makes Trainer think this is model-parallel.
    model = _from_pretrained(
        AutoModelForCausalLM, MODEL_ID, torch.float32,
        cache_dir=CACHE_DIR, trust_remote_code=TRUST_REMOTE_CODE,
    )

    assert_text_only(model)
    assert_master_dtype(model)
    report_parameters(model)

    model.config.use_cache = False            # incompatible with training; wastes memory
    return model, tokenizer


def load_model_for_eval(model_path: str):
    """Model + tokenizer for --eval-only. Returns (model, tokenizer, source_label).

    model_path == "none" scores the BASE model zero-shot through this identical pipeline — same
    prompt, same enable_thinking=False, same NFC scoring, same generation settings. That is the
    only baseline the fine-tuned number can honestly be compared against: the figures in
    `Models to test & Results.xlsx` were produced with a different prompt AND without NFC
    normalization, so a delta against them confounds three changes at once (RUN_REPORT.md §2).

    Loaded in bf16, not fp32: inference needs no master weights, and bf16 halves both the load
    time and the memory. This is not the training path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_only = str(model_path).lower() in ("none", "", "base")
    src = MODEL_ID if base_only else model_path
    if not base_only and not os.path.isdir(src):
        raise FileNotFoundError(f"{src} is not a directory. Pass --model none for the baseline.")

    tok_src = src
    if not base_only and not os.path.exists(os.path.join(src, "tokenizer_config.json")):
        print(f"[WARN] {src} has no tokenizer; falling back to {MODEL_ID}")
        tok_src = MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tok_src, cache_dir=CACHE_DIR,
                                              trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"           # right padding corrupts batched generation

    detect_prompt_mode(tokenizer)
    assert_prompt_sane(tokenizer)

    model = _from_pretrained(AutoModelForCausalLM, src, torch.bfloat16,
                             cache_dir=CACHE_DIR, trust_remote_code=TRUST_REMOTE_CODE)
    assert_text_only(model)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = True
    model.eval()
    print(f"[OK] {'BASE MODEL (zero-shot baseline)' if base_only else 'fine-tuned model'}: {src}")
    return model, tokenizer, ("base" if base_only else src)


# ==============================================================================================
# Stage 4 — Training
#
# Loss is computed only over the assistant turn. Rather than matching a literal response-template
# string, the prompt is tokenized separately and its tokens are masked with -100. Same
# completion-only loss, but it cannot silently mis-mask on a model whose template differs, and it
# needs no TRL version pinning.
# ==============================================================================================

def build_tokenizer_fn(tokenizer):
    eos = tokenizer.eos_token or ""

    def tokenize(ex):
        prompt_text = render_prompt(tokenizer, ex[INPUT_COLUMN])
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(ex[OUTPUT_COLUMN] + eos, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:MAX_SEQ_LENGTH]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_SEQ_LENGTH]
        # `length` feeds the length-grouped sampler. Without a real column the sampler
        # materializes every row in Python to recompute lengths before step 0 — ~1M row loads on
        # the full corpus, which looks exactly like a hang. See _make_training_trainer.
        return {"input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "length": len(input_ids),
                "n_prompt_tokens": len(prompt_ids)}
    return tokenize


class _CollatorDroppingExtras:
    """DataCollatorForSeq2Seq, minus the columns the sampler/report need but the model doesn't.

    `length` has to survive into the dataset (remove_unused_columns=False) for the
    length-grouped sampler to see it, which means it also reaches the collator — and
    Qwen3_5ForCausalLM.forward has **kwargs, so it would SWALLOW the extra key silently rather
    than raise. Strip explicitly.
    """

    MODEL_KEYS = ("input_ids", "attention_mask", "labels")

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        return self.base([{k: f[k] for k in self.MODEL_KEYS if k in f} for f in features])

    def __getattr__(self, item):
        # Trainer introspects the collator (e.g. for .tokenizer / .padding). Delegate rather
        # than shadowing it with a bare wrapper.
        return getattr(self.base, item)


def _length_grouping_kwargs(training_arguments_cls) -> dict:
    """Length-grouped batching, spelled the way the installed transformers spells it.

    transformers 5.x replaced the boolean `group_by_length` with the string-valued
    `train_sampling_strategy` (training_args.py:1314-1325); passing the old name to 5.x raises
    on an unexpected kwarg. Both are handled so this file runs on either stack.

    Not an optimization to skip: lengths run p50~380 / p99~900 against MAX_SEQ_LENGTH=1024 and
    the collator pads each batch to its own max, so a random sampler spends 35-42% of every
    forward/backward on pad tokens. Grouping drops that to ~0.5%.
    """
    fields = {f.name for f in dataclasses.fields(training_arguments_cls)}
    if "train_sampling_strategy" in fields:
        kw = {"train_sampling_strategy": "group_by_length"}
    elif "group_by_length" in fields:
        kw = {"group_by_length": True}
    else:
        print("[WARN] installed transformers has neither train_sampling_strategy nor "
              "group_by_length; batches will be padded to a random max length")
        return {}
    if "length_column_name" in fields:
        kw["length_column_name"] = "length"
    kw["remove_unused_columns"] = False
    return kw


def _make_training_trainer(Trainer):
    """Trainer with two fixes: a materialized length list, and deterministic eval ordering.

    FIX 1 — the apparent hang. Trainer._get_train_sampler does
    `lengths = train_dataset[self.args.length_column_name]`. Under datasets 5.x that returns a
    LAZY Arrow `Column`, not a list, and LengthGroupedSampler stores it as-is (it only converts
    torch.Tensor). get_length_grouped_indices then runs `sorted(megabatch, key=lambda i:
    lengths[i])` over every row, so each of ~1M lookups pays an Arrow chunk search. `.map()`
    writes 1,000-row chunks, so the cost is O(N x chunks):

        rows        chunks   per-read   total
        100,000        100     58 us     0.1 min
        400,000        400    157 us     1.0 min
        1,000,000    1,000    358 us     6.0 min      <- measured

    A 200-row smoke test never sees it (200 rows = 1 chunk). Materializing the column first
    costs ~26s at 1M rows and makes the sort take 0.05s. RUN_REPORT.md §5.4.

    Do NOT use `dataset.data.column(name).to_pylist()` instead: that reads the underlying Arrow
    table and IGNORES the indices map `.filter()` installs, so after the null-row filter it
    returns the wrong number of rows, silently misaligned with the dataset (measured: 300,000 vs
    150,000). Go through `list(dataset[name])`, which respects it.

    FIX 2 — deterministic eval. transformers 5.14.1 drives BOTH samplers off the single
    `train_sampling_strategy` field: with it set to "group_by_length", _get_eval_sampler
    (trainer.py:1051-1068) also returns a LengthGroupedSampler, constructed with generator=None
    — i.e. seeded from the global torch RNG and therefore a different dev ordering at every
    evaluation. Trainer's eval_loss is a sample-weighted mean of per-batch token-mean losses, so
    it depends on how examples are grouped into batches; an unseeded regrouping makes the number
    move between evals for no modelling reason. That matters here because eval_loss is what
    metric_for_best_model / load_best_model_at_end select the final weights on. The 4B LoRA
    script has this bug; this restores stock sequential ordering for eval only.
    """
    try:
        from transformers.trainer_pt_utils import LengthGroupedSampler
    except Exception as e:
        print(f"[WARN] could not import LengthGroupedSampler ({type(e).__name__}); "
              f"falling back to stock Trainer — expect a slow start on a large corpus")
        return Trainer

    from torch.utils.data import SequentialSampler

    class _FullFTTrainer(Trainer):
        def __init__(self, *args, train_lengths=None, **kwargs):
            self._train_lengths = train_lengths
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, *args, **kwargs):
            grouped = (getattr(self.args, "train_sampling_strategy", None) == "group_by_length"
                       or getattr(self.args, "group_by_length", False))
            if self._train_lengths is None or not grouped:
                return super()._get_train_sampler(*args, **kwargs)
            proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=self._train_lengths,
                model_input_name=proc.model_input_names[0] if proc is not None else None,
            )

        def _get_eval_sampler(self, eval_dataset=None, *args, **kwargs):
            if eval_dataset is None:
                return None
            return SequentialSampler(eval_dataset) if self.args.world_size <= 1 else None

    return _FullFTTrainer


def _make_loss_sanity_callback():
    from transformers import TrainerCallback

    class LossSanityCallback(TrainerCallback):
        """Catches inverted label masking at the first logged step, not at the end of the run.

        A loss near 0.0 at step 1 means the model is being scored on tokens it was handed —
        i.e. the mask is inverted or the prompt is being trained on. README_HANDOFF.md makes
        this the explicit smoke-test gate; enforcing it in code means every run gets it.
        """

        def __init__(self):
            self.checked = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.checked or not logs or "loss" not in logs:
                return
            self.checked = True
            loss = logs["loss"]
            if loss != loss or loss in (float("inf"), float("-inf")):
                raise RuntimeError(f"first training loss is not finite ({loss}) — stop.")
            if loss < 0.05:
                raise RuntimeError(
                    f"first training loss is {loss}, which is implausibly low for a model that "
                    f"has not been trained on this task. The label mask is almost certainly "
                    f"inverted (the model is being scored on tokens it was handed). Stop and "
                    f"inspect build_tokenizer_fn."
                )
            print(f"[OK] first logged loss = {loss} (finite, non-trivial)")

    return LossSanityCallback()


def _make_throughput_callback(tokens_per_step, run_steps, projection_steps, projection_label):
    """Steady-state tokens/s and a projected finish time, with the warmup steps discarded.

    This exists because of RUN_REPORT.md §5.6: "Extrapolating step time from the first few steps
    is meaningless here." Two reasons, both structural rather than incidental:

      - LengthGroupedSampler builds megabatches of 50 x batch_size and sorts each of them
        LONGEST-FIRST, so step time oscillates on a ~50-step cycle and the first steps of every
        cycle are the slowest in it.
      - Step 1 additionally carries ~120s of Triton JIT for the Gated-DeltaNet kernels.

    So the first SMOKE_WARMUP_STEPS_DISCARDED steps are thrown away and the mean is taken over
    everything after — which, at SMOKE_MAX_STEPS=60, is 40 steps, most of one full megabatch
    cycle. That is the number to budget from.

    `run_steps` is what THIS invocation will do; `projection_steps` is what to cost out. They
    differ for the smoke test, and that difference is the whole point of the smoke test:
    projecting its own 60 steps would report the cost of the smoke test, which nobody needs.
    Batch geometry (per-device batch, accumulation, MAX_SEQ_LENGTH) is identical between the two,
    so s/step carries across; only the number of steps changes.
    """
    from transformers import TrainerCallback

    class ThroughputCallback(TrainerCallback):
        def __init__(self):
            self.t_start = None
            self.step_start = None
            self.measured_steps = 0
            self.arm_at = SMOKE_WARMUP_STEPS_DISCARDED

        def on_train_begin(self, args, state, control, **kwargs):
            # Arm relative to where THIS invocation starts, not to absolute step 20. A run
            # resuming from checkpoint-8000 never passes through step 20, so keying off it left
            # t_start at None for the whole run and the final report printed "no trustworthy
            # measurement" — on the runs where a throughput number is most useful, because they
            # are the ones that were already interrupted once.
            self.arm_at = state.global_step + SMOKE_WARMUP_STEPS_DISCARDED
            if state.global_step:
                print(f"[INFO] resumed at step {state.global_step:,}; throughput measurement "
                      f"arms at step {self.arm_at:,}")

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step == self.arm_at:
                self.t_start = time.time()
                self.step_start = step
                return
            if self.t_start is None:
                return
            self.measured_steps = step - self.step_start
            if self.measured_steps and step % LOGGING_STEPS == 0:
                self.report(state, prefix="  ")

        def report(self, state, prefix=""):
            if not self.t_start or self.measured_steps <= 0:
                print(f"{prefix}[THROUGHPUT] only {state.global_step} steps ran, which is not "
                      f"past the {SMOKE_WARMUP_STEPS_DISCARDED}-step warmup discard — no "
                      f"trustworthy measurement. Do not extrapolate from the log lines above.")
                return None
            elapsed = time.time() - self.t_start
            s_per_step = elapsed / self.measured_steps
            tok_per_s = tokens_per_step / s_per_step
            print(f"{prefix}[THROUGHPUT] steady state over {self.measured_steps} steps "
                  f"(first {SMOKE_WARMUP_STEPS_DISCARDED} discarded): "
                  f"{s_per_step:.2f} s/step, {tok_per_s:,.0f} tok/s")
            if run_steps:
                remaining = max(0, run_steps - state.global_step)
                if remaining:
                    print(f"{prefix}[THROUGHPUT] this run: {remaining:,} of {run_steps:,} steps "
                          f"remaining = {remaining * s_per_step / 3600:.2f} h")
            if projection_steps:
                hours = projection_steps * s_per_step / 3600
                print(f"{prefix}[THROUGHPUT] {projection_label}: {projection_steps:,} steps = "
                      f"{hours:.2f} h wall clock, ~${hours * 4.58:.0f} at $4.58/hr "
                      f"(gpu=H100:1, cpu=8, memory=32GiB)")
                print(f"{prefix}[THROUGHPUT] excludes {N_SAVES} checkpoint writes of ~9GB each "
                      f"and {N_EVALS} in-training evals — add ~30-45 min to the figure above.")
            return s_per_step

    return ThroughputCallback()


def schedule_for(total_steps: int, n_evals: int = N_EVALS, n_saves: int = N_SAVES) -> dict:
    """eval/save intervals as a fraction of a run's real length. save_steps == eval_steps.

    Fixed intervals cannot serve both a 60-step smoke test and a 16,290-step epoch. The 4B
    script's EVAL_STEPS=SAVE_STEPS=200 over a full epoch here would be 81 evaluations and 81
    checkpoints. Guarantees at least one eval and one checkpoint however short the run.

    ---------------------------------------------------------------------------------------
    WHY EVERY EVAL MUST ALSO BE A SAVE. THIS IS A CORRECTNESS CONSTRAINT.
    ---------------------------------------------------------------------------------------
    The obvious-looking version of this helper — evaluate often, checkpoint rarely, keep
    save_steps a multiple of eval_steps — silently breaks load_best_model_at_end, and it breaks
    it in the direction that ships the WRONG WEIGHTS with no error and no warning.

    In transformers 5.14.1, Trainer._determine_best_metric runs at EVERY eval and updates both
    state.best_metric and state.best_global_step whenever the metric improves. Nothing requires
    a checkpoint to exist at that step. _save_checkpoint then runs only on save steps, and
    adopts the best only if its directory happens to be on disk (trainer.py):

        best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
        if os.path.exists(best_checkpoint_dir):
            self.state.best_model_checkpoint = best_checkpoint_dir

    So with N_EVALS=20 / N_SAVES=8 over a full epoch (eval every 814, save every 1628), ten of
    the twenty eval points had no checkpoint. If the best eval_loss landed on one of them —
    step 814, say, the FIRST eval — then:

        - best_metric pins to that value permanently, so no later eval can displace it;
        - best_global_step stays 814, whose directory is never written;
        - best_model_checkpoint therefore stays None for the whole run;
        - and trainer.py's `if load_best_model_at_end and best_model_checkpoint is not None`
          guard means _load_best_model() is simply skipped.

    The run then finishes, saves the FINAL-step weights as if they were the selected ones, and
    says nothing. This is not hypothetical for this project: RUN_REPORT.md §7 has the 4B run's
    eval_loss bottoming at step 400 of 1500, and RESULTS.md's headline number IS that step-400
    checkpoint rather than step 1500. An early minimum is the expected shape here, and step 814
    was exactly the unreachable one.

    Equal intervals cost ~10 extra checkpoint writes over a full epoch. SAVE_TOTAL_LIMIT bounds
    what is resident, so this is write time (~10 min), not storage.

    assert_best_model_selected() re-checks the outcome after training, because a future edit to
    N_SAVES would reintroduce this and the failure is invisible from the logs.
    """
    eval_steps = max(1, total_steps // max(1, n_evals))
    if n_saves != n_evals:
        print(f"[WARN] n_saves={n_saves} != n_evals={n_evals}; forcing save_steps=eval_steps. "
              f"Checkpointing less often than evaluating breaks load_best_model_at_end — see "
              f"schedule_for.__doc__.")
    return {"eval_steps": eval_steps, "save_steps": eval_steps}


def _pick_optim():
    """adamw_torch_fused if this transformers knows it, else adamw_torch.

    Not a micro-optimization at this scale: a full epoch is ~16,290 optimizer steps over 751M
    fp32 parameters, and the fused kernel replaces a few hundred small elementwise launches per
    step with one. It is the same maths — fused AdamW is not an approximation.
    """
    requested = os.environ.get("OPTIM", "adamw_torch_fused")
    try:
        from transformers.training_args import OptimizerNames
        if requested in {o.value for o in OptimizerNames}:
            return requested
        print(f"[WARN] optim {requested!r} unknown to this transformers; using adamw_torch")
    except Exception:
        pass
    return "adamw_torch"


def assert_best_model_selected(trainer):
    """After training: did load_best_model_at_end actually load anything?

    The failure this catches is silent by construction — see schedule_for.__doc__. Trainer skips
    _load_best_model() when best_model_checkpoint is None and logs nothing, so a run that
    selected nothing is indistinguishable in the output from one that selected the last
    checkpoint on purpose. Returns the provenance dict that goes into run_config.json, so the
    saved model always records which weights it actually holds.
    """
    state = trainer.state
    best_ckpt = getattr(state, "best_model_checkpoint", None)
    best_metric = getattr(state, "best_metric", None)
    best_step = getattr(state, "best_global_step", None)

    provenance = {
        "weights_are": None,
        "best_model_checkpoint": best_ckpt,
        "best_eval_loss": best_metric,
        "best_global_step": best_step,
        "final_global_step": state.global_step,
    }

    if not trainer.args.load_best_model_at_end:
        provenance["weights_are"] = "final_step"
        print(f"[INFO] load_best_model_at_end=False — final_model/ holds step {state.global_step}")
        return provenance

    if best_ckpt is None:
        provenance["weights_are"] = "final_step_UNSELECTED"
        print()
        print("=" * 94)
        print("[WARN] load_best_model_at_end=True but Trainer never recorded a best checkpoint.")
        print(f"       best_metric={best_metric} was reached at step {best_step}, and no")
        print(f"       checkpoint-{best_step} exists, so Trainer silently kept the FINAL weights")
        print(f"       from step {state.global_step} instead of the best ones.")
        print("       This means eval/save intervals drifted out of alignment — check")
        print("       schedule_for() and N_SAVES. The model in final_model/ is usable but it is")
        print("       NOT the selected checkpoint; say so wherever you report its score.")
        print("=" * 94)
        return provenance

    provenance["weights_are"] = "best_checkpoint"
    print(f"[OK] best checkpoint selected: {best_ckpt}")
    print(f"     eval_loss={best_metric} at step {best_step} of {state.global_step}")
    if best_step and state.global_step and best_step < 0.5 * state.global_step:
        print(f"     [NOTE] the best checkpoint is in the first half of the run. The 4B run had "
              f"the same shape (best at step 400 of 1500, RUN_REPORT.md §7) and it was read as "
              f"the LR/capacity saturating early rather than as overfitting. Worth checking the "
              f"loss curve before spending on a longer rung.")
    return provenance


def latest_checkpoint(out_dir: str):
    """Highest-numbered COMPLETE checkpoint under out_dir, or None.

    Deliberately not newest-by-mtime: a job preempted mid-write leaves a partial checkpoint
    holding the newest mtime, and resuming from it dies on a truncated shard. Rank by step
    number and require trainer_state.json, which Trainer writes last.
    """
    if not os.path.isdir(out_dir):
        return None
    candidates = []
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if not name.startswith("checkpoint-") or not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, "trainer_state.json")):
            print(f"[INFO] skipping incomplete checkpoint {path}")
            continue
        try:
            candidates.append((int(name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(candidates)[1] if candidates else None


def select_eval_subset(test_raw, n):
    """The in-training eval set. ONE definition, used by both train_model and prepare_only.

    Shuffled, not the first n rows. Both are deterministic and both hash the same under the
    datasets fingerprint cache, so the reason to prefer shuffling is representativeness: the
    test split is stored grouped by source file, so `select(range(1000))` of 2,485 rows takes a
    contiguous block of whichever sources happen to sort first. eval_loss over that block is
    what load_best_model_at_end selects the final weights on, and selecting on an unrepresentative
    slice of an already narrow distribution is a real risk for no gain.

    It stays a Sadeed_Tashkeela slice, so understand what it can and cannot do: the corpus is
    Tashkeela-derived and Classical-heavy, so this is an IN-DOMAIN (largely CA) selection signal,
    while the failure mode this package is chasing is MSA generalisation (RUN_REPORT.md §1). The
    honest fix would be an MSA dev set, and there isn't one here that is not SadeedDiac-25 —
    which is the benchmark, and using it to pick a checkpoint would contaminate the only number
    that matters. So: selection is in-domain, deliberately, and the benchmark stays untouched.
    Say so when reporting.

    Kept as one function because train_model and prepare_only MUST slice identically or the
    tokenization fingerprint differs and `prepare` saves the GPU job nothing.
    """
    if not n or n >= len(test_raw):
        return test_raw
    return test_raw.shuffle(seed=RANDOM_SEED).select(range(n))


def tokenize_splits(tokenizer, train_raw, eval_raw):
    """Tokenize + drop rows with no trainable target. Returns (train_tok, eval_tok)."""
    tokenize = build_tokenizer_fn(tokenizer)
    nproc = _nproc()

    t0 = time.time()
    train_tok = train_raw.map(tokenize, remove_columns=train_raw.column_names,
                              num_proc=nproc, desc="tokenising train")
    eval_tok = eval_raw.map(tokenize, remove_columns=eval_raw.column_names,
                            num_proc=nproc, desc="tokenising eval")
    print(f"[OK] tokenised in {time.time() - t0:.0f}s")

    # Drop rows whose answer was pushed entirely past MAX_SEQ_LENGTH — they carry no trainable
    # target and would contribute a NaN loss.
    has_target = lambda ex: any(l != -100 for l in ex["labels"])  # noqa: E731
    before = len(train_tok)
    train_tok = train_tok.filter(has_target, num_proc=nproc)
    eval_tok = eval_tok.filter(has_target, num_proc=nproc)
    if before != len(train_tok):
        print(f"[INFO] dropped {before - len(train_tok):,} train rows with no target within "
              f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
    print(f"[OK] tokenised: train={len(train_tok):,}  eval={len(eval_tok):,}")
    return train_tok, eval_tok


def train_model(model, tokenizer, train_raw, test_raw, smoke_test=False, max_steps=None):
    from transformers import (DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    import torch

    set_seed(RANDOM_SEED)

    eval_raw = select_eval_subset(test_raw, 64 if smoke_test else IN_TRAINING_EVAL_SUBSET)

    train_tok, eval_tok = tokenize_splits(tokenizer, train_raw, eval_raw)

    # Materialize the length column ONCE, here, and reuse it for both the token accounting below
    # and the grouped sampler further down. ~26s at 1M rows, and it is the fix for the apparent
    # 6-minute hang at step 0 described in _make_training_trainer. Doing it twice (once for a
    # mean, once for the sampler) would pay that twice.
    train_lengths = None
    if "length" in train_tok.column_names:
        t0 = time.time()
        train_lengths = list(train_tok["length"])
        print(f"[OK] materialized {len(train_lengths):,} lengths in {time.time() - t0:.1f}s")

    # ---- how much of the corpus will actually be seen, and what will it cost? ----
    # max_steps OVERRIDES num_train_epochs in HF Trainer whenever it is positive. Setting both
    # is the easy mistake: epochs is silently ignored and the run covers however much max_steps
    # happens to reach. Spell it out up front, every run.
    effective_max_steps = MAX_STEPS if max_steps is None else max_steps
    eff_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = max(1, math.ceil(len(train_tok) / eff_batch))

    mean_len = (sum(train_lengths) / max(1, len(train_lengths))) if train_lengths \
        else MAX_SEQ_LENGTH
    tokens_per_step = mean_len * eff_batch

    print(f"[INFO] effective batch = {PER_DEVICE_TRAIN_BATCH_SIZE} x "
          f"{GRADIENT_ACCUMULATION_STEPS} = {eff_batch}")
    print(f"[INFO] ~{steps_per_epoch:,} optimizer steps per epoch over {len(train_tok):,} rows")
    print(f"[INFO] mean sequence = {mean_len:.1f} tokens -> {tokens_per_step:,.0f} tokens/step, "
          f"{steps_per_epoch * tokens_per_step / 1e6:,.0f}M tokens/epoch")

    if effective_max_steps and effective_max_steps > 0:
        total_steps = effective_max_steps
        seen = effective_max_steps * eff_batch
        frac = seen / max(1, len(train_tok))
        print(f"[WARN] MAX_STEPS={effective_max_steps:,} OVERRIDES "
              f"NUM_TRAIN_EPOCHS={NUM_TRAIN_EPOCHS}.")
        print(f"[WARN] This run will see ~{seen:,} examples = {frac:.2f} epochs "
              f"({frac * 100:.1f}% of one pass over the train split).")
        if frac < 1.0 and not smoke_test:
            print("[WARN] That is LESS THAN ONE FULL EPOCH — which is the thing this package "
                  "exists to avoid (RUN_REPORT.md §1: the 4B run saw 9.2%). Set MAX_STEPS=-1.")
    else:
        total_steps = int(steps_per_epoch * NUM_TRAIN_EPOCHS)
        print(f"[INFO] MAX_STEPS disabled -> {NUM_TRAIN_EPOCHS} full epoch(s), "
              f"~{total_steps:,} steps total")

    # What to COST OUT, which is not always what this invocation will run. The smoke test's job
    # is to price the full epoch it is deliberately not running; pricing its own 60 steps would
    # answer a question nobody asked. Batch geometry is identical between the two, so s/step
    # carries over and only the step count changes.
    if smoke_test:
        projection_steps = math.ceil(FULL_CORPUS_ROWS / eff_batch)
        projection_label = (f"PROJECTED full epoch over {FULL_CORPUS_ROWS:,} rows "
                            f"(this smoke run is {len(train_tok):,} rows)")
    else:
        projection_steps, projection_label = total_steps, "this run"

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not bf16_ok:
        print("[WARN] bf16 unavailable — falling back to fp16 autocast. fp16 with an fp32 "
              "master copy needs loss scaling, which Trainer handles, but bf16 is what this "
              "recipe was measured on.")

    args_kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=effective_max_steps,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=0.0 if smoke_test else WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim=_pick_optim(),
        # bf16=True with an fp32 model is mixed precision via autocast: master weights and
        # optimizer state stay fp32, the matmuls run in bf16. This is the pairing
        # assert_master_dtype() exists to protect.
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if GRADIENT_CHECKPOINTING else None,
        logging_steps=1 if smoke_test else LOGGING_STEPS,
        eval_strategy="no" if smoke_test else "steps",
        save_strategy="no" if smoke_test else "steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=not smoke_test,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=RANDOM_SEED,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    if not smoke_test:
        sched = schedule_for(total_steps)
        args_kwargs.update(sched)
        print(f"[INFO] {total_steps:,} steps -> eval every {sched['eval_steps']:,}, "
              f"save every {sched['save_steps']:,} "
              f"(each checkpoint ~9GB: fp32 weights + fp32 optimizer state)")
    args_kwargs.update(_length_grouping_kwargs(TrainingArguments))

    args = TrainingArguments(**args_kwargs)

    # The list materialized above is handed to the sampler only if grouping is actually on —
    # see _make_training_trainer for what it costs not to.
    grouped = (getattr(args, "train_sampling_strategy", None) == "group_by_length"
               or getattr(args, "group_by_length", False))
    if not grouped:
        print("[WARN] length grouping is OFF; batches will pad to a random max length "
              "(measured 35-42% of every batch tensor as pad tokens on this corpus)")
        train_lengths = None

    throughput = _make_throughput_callback(tokens_per_step, total_steps,
                                           projection_steps, projection_label)

    TrainerCls = _make_training_trainer(Trainer)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=None if smoke_test else eval_tok,
        data_collator=_CollatorDroppingExtras(
            DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)),
        callbacks=[_make_loss_sanity_callback(), throughput],
    )
    # Only the subclass accepts train_lengths; if its sampler import failed it degrades to the
    # stock Trainer, which would reject the kwarg outright.
    if TrainerCls is not Trainer:
        trainer_kwargs["train_lengths"] = train_lengths
    # transformers 5.x replaced Trainer(tokenizer=...) with processing_class=. Passing it means
    # intermediate checkpoints carry a tokenizer, so `--eval-only --model <checkpoint>` works
    # without falling back to the hub. Detected rather than assumed, so 4.x still runs.
    _params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in _params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = TrainerCls(**trainer_kwargs)

    ckpt = None if smoke_test else latest_checkpoint(OUTPUT_DIR)
    if ckpt:
        print(f"[INFO] resuming from {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)

    print()
    throughput.report(trainer.state)

    if not smoke_test:
        provenance = assert_best_model_selected(trainer)
        save_full_model(trainer, tokenizer, total_steps, provenance)
        pd.DataFrame(trainer.state.log_history).to_csv(
            os.path.join(EVAL_DIR, "training_log.csv"), index=False)
        print(f"[OK] Saved training log -> {os.path.join(EVAL_DIR, 'training_log.csv')}")
    return trainer


def save_full_model(trainer, tokenizer, total_steps, provenance=None):
    """Full weights + tokenizer + provenance.

    ~3.0GB in fp32, not the 108MB a LoRA adapter costs — and unlike an adapter it is
    self-contained: no base model needed alongside it at eval time.

    `provenance` records WHICH weights these are (best checkpoint vs final step) so the
    directory answers that on its own, months later, without the training log.
    """
    try:
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        trainer.save_model(FINAL_MODEL_DIR)
        tokenizer.save_pretrained(FINAL_MODEL_DIR)
        with open(os.path.join(FINAL_MODEL_DIR, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "base_model": MODEL_ID,
                "checkpoint_provenance": provenance or {},
                "method": "full_fine_tune",
                "chat_mode": _CHAT_MODE,
                "epochs": NUM_TRAIN_EPOCHS,
                "max_steps": MAX_STEPS,
                "planned_total_steps": total_steps,
                "lr": LEARNING_RATE,
                "lr_scheduler": LR_SCHEDULER_TYPE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "max_seq_length": MAX_SEQ_LENGTH,
                "effective_batch": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "gradient_checkpointing": GRADIENT_CHECKPOINTING,
                "master_dtype": "float32",
                "autocast": "bfloat16",
                "seed": RANDOM_SEED,
                "train_dataset": TRAIN_DATASET,
                "train_dataset_revision": TRAIN_DATASET_REVISION,
                "benchmark_dataset_revision": BENCHMARK_DATASET_REVISION,
                "enable_thinking": False,
            }, f, indent=2)
        print(f"[OK] Saved full model -> {FINAL_MODEL_DIR}")
    except Exception as e:
        print(f"[WARN] Failed to save the final model: {e}")
        traceback.print_exc()


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
        plt.plot(train_steps, train_loss, label="Train loss", marker="o", markersize=2, lw=1)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Test (eval) loss", marker="s", markersize=4)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"{MODEL_ID} full FT — Train vs. Test Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(EVAL_DIR, "train_test_loss_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved loss curve -> {out_path}")


# ==============================================================================================
# Stage 5 — Inference
# ==============================================================================================

def prepare_model_for_generation(model):
    """Convert a just-trained model from training configuration to inference configuration.

    Only needed on the end-to-end path (train and score in one process). `--eval-only` loads
    from disk through load_model_for_eval(), which is already correct.

    Two changes, and skipping either is expensive rather than merely untidy:

    1. use_cache=True. load_model_for_training() sets it False, because a KV cache is useless
       during teacher-forced training and only wastes memory. Leaving it False through
       generate() disables the KV cache, so every new token re-runs the forward pass over the
       whole prefix — quadratic instead of linear. On 4,185 paragraphs of a few hundred tokens
       each that is the difference between under an hour and not finishing.

    2. bf16. The trained weights are fp32 master weights, which is the whole point during
       training, but inference needs no master copy: fp32 generation is ~2x the memory
       bandwidth for identical greedy output. This also matches what load_model_for_eval() does
       when it reads the saved checkpoint back, so the end-to-end path and the two-phase path
       score the same numbers rather than differing in the last decimal.

    Gradient checkpointing is disabled too — it is a no-op under torch.no_grad(), but leaving it
    on has been known to interact badly with cache-enabled forwards on some architectures.
    """
    import torch

    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    try:
        model = model.to(torch.bfloat16)
    except Exception as e:
        print(f"[WARN] could not cast the model to bf16 for generation ({type(e).__name__}); "
              f"scoring in fp32, which is slower but gives the same answers: {e}")
    model.eval()
    print("[OK] model switched to inference configuration (use_cache=True, bf16, eval mode)")
    return model


def run_inference(model, tokenizer, pairs, name):
    """Greedy generation, checkpointed so a killed job doesn't lose the whole split.

    Returns (predictions, n_truncated).
    """
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
        return [p or "" for p in preds], 0

    prev_side = tokenizer.padding_side
    prev_trunc = tokenizer.truncation_side
    tokenizer.padding_side = "left"      # right padding corrupts batched generation
    # Truncate from the LEFT. The tokenizer call below passes truncation=True to bound the
    # prompt at MAX_SEQ_LENGTH, and the default truncation_side="right" would cut the END of
    # the rendered chat template — i.e. "<|im_end|>\n<|im_start|>assistant\n", the part that
    # tells the model to answer. The model would then continue the user's text instead of
    # diacritizing it, and the row would score as a bad prediction rather than as a bug.
    # Prompts do not reach 1024 tokens on this corpus, so this is a latent trap rather than an
    # active one — but it is silent, and it arms itself if MAX_SEQ_LENGTH is ever lowered or a
    # longer-form corpus is swapped in.
    tokenizer.truncation_side = "left"
    model.eval()
    eos_ids = generation_eos_ids(tokenizer)
    print(f"  stopping generation on token ids {eos_ids}")

    def save_ckpt():
        done = [(i, preds[i]) for i in range(len(preds)) if preds[i] is not None]
        pd.DataFrame(done).to_csv(ckpt_path, index=False, header=False)

    # Length-bucketed so each batch pads to a similar length.
    order = sorted(todo, key=lambda i: len(pairs[i][0]))
    since_save = 0
    truncated = 0

    try:
        with torch.no_grad():
            for start in range(0, len(order), INFER_BATCH_SIZE):
                idxs = order[start:start + INFER_BATCH_SIZE]
                in_len = -1
                try:
                    prompts = [render_prompt(tokenizer, pairs[i][0]) for i in idxs]
                    # add_special_tokens=False: the chat template already emitted every special
                    # token this prompt needs, as text. Letting the tokenizer add more would
                    # prepend a second BOS-equivalent that training never saw.
                    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=MAX_SEQ_LENGTH,
                                    add_special_tokens=False).to(model.device)
                    in_len = enc["input_ids"].shape[1]
                    max_new = min(int(in_len * GEN_LEN_RATIO) + 32, MAX_NEW_CAP)
                    out_ids = model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_ids,
                    )
                    gen_only = out_ids[:, in_len:]

                    # Truncation is counted PER ROW, by asking whether each row actually
                    # produced a stop token. The obvious test — comparing out_ids.shape[1] to
                    # max_new — is a per-BATCH test: generate() returns one rectangular tensor
                    # whose width is set by the longest-running row, so a single row hitting the
                    # cap marks all INFER_BATCH_SIZE rows as truncated. n_truncated is reported
                    # per row in metrics_summary_*.csv, so that over-counts by up to 8x, and it
                    # over-counts precisely on the batches where a real truncation happened.
                    #
                    # A row that stopped early is right-padded with pad_token_id, which is in
                    # eos_ids here, so "contains no eos id" is exactly "ran to the cap".
                    eos_tensor = torch.tensor(eos_ids, device=gen_only.device)
                    hit_eos = (gen_only.unsqueeze(-1) == eos_tensor).any(-1).any(-1)
                    truncated += int((~hit_eos).sum().item())

                    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
                    for i, pred in zip(idxs, decoded):
                        preds[i] = clean_output(pred, chat_mode=bool(_CHAT_MODE))
                except torch.cuda.OutOfMemoryError:
                    # Do NOT swallow this. Writing "" for the whole batch and checkpointing it
                    # means a resumed run treats the blanks as completed work -- permanently.
                    # Blanks score as 100% error, and rows are sorted by length ascending, so
                    # memory peaks at the END: an OOM there silently subtracts real points from
                    # the final number.
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM on {name} at batch {start} (in_len={in_len}, "
                        f"batch={len(idxs)}). Lower INFER_BATCH_SIZE and re-run; the "
                        f"{sum(1 for p in preds if p is not None)} rows already generated are "
                        f"cached in {ckpt_path} and will be reused."
                    )
                except Exception as e:
                    # Anything else: leave the rows PENDING rather than caching a fake blank,
                    # so a dtype or tokenizer bug cannot masquerade as "the model is bad".
                    print(f"  [warn] {name}: batch @ {start} failed "
                          f"({type(e).__name__}: {e}); left pending")
                    gc.collect()
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
        tokenizer.truncation_side = prev_trunc

    pending = [i for i in range(len(preds)) if preds[i] is None]
    if pending:
        raise RuntimeError(
            f"{len(pending)} {name} rows failed to generate and were left pending (indices "
            f"{pending[:10]}{'...' if len(pending) > 10 else ''}). Refusing to score a partial "
            f"split -- fix the error above and re-run; cached rows will be reused."
        )
    if truncated:
        print(f"[WARN] {truncated}/{len(pairs)} generations hit max_new_tokens. Truncated "
              f"tails become deletions, which ARE penalized -- these rows fabricate errors.")
    return preds, truncated


# ==============================================================================================
# Stage 6 — Evaluation, CSV export, plots
# ==============================================================================================

def score_predictions(name, inputs, references, predictions, truncated=0):
    """The scoring half of an evaluation, separated so one generation pass can be scored N ways.

    The benchmark needs CA, MSA and pooled numbers from the SAME 1,200 generations; regenerating
    for each would triple the GPU cost for zero information. The 4B script fused generation and
    scoring, which is why its CA/MSA split had to be done by hand in a notebook afterwards.
    """
    blank = sum(1 for p in predictions if not p.strip())
    if blank:
        print(f"[WARN] {blank}/{len(predictions)} predictions are empty — each scores as "
              f"fully wrong and inflates the reported error rates.")

    # NFC on BOTH sides before scoring — see nfc(). Applied here rather than inside
    # compute_der_wer so the frozen metric functions stay byte-identical to the 4B script.
    predictions_nfc = [nfc(p) for p in predictions]
    references_nfc = [nfc(r) for r in references]
    inputs_nfc = [nfc(i) for i in inputs]

    # BOTH scorers, over the SAME predictions. See the CORRECTED metrics block for why neither
    # alone is sufficient: the frozen numbers are what compare to RUN_REPORT.md's 6.96, the
    # corrected ones are what compare to RESULTS.md's 8.48 and are what should be cited.
    # The corrected scorer NFCs internally too; feeding it NFC text is idempotent.
    metrics = compute_der_wer(predictions_nfc, references_nfc)
    metrics.update(compute_der_wer_corrected(predictions_nfc, references_nfc))

    # Does the prediction say the same WORDS as the prompt, ignoring diacritics? Diacritization
    # must not change the consonantal skeleton, so a non-zero rate here means the model is
    # echoing, rambling or paraphrasing -- and that is invisible in DER/WER alone. Compared as
    # TOKEN LISTS, not raw strings: comparing raw strings makes any whitespace or punctuation
    # difference count as a mismatch (measured: 94% flagged, of which 413/500 were whitespace
    # only). A diagnostic that fires on 94% of rows for a non-reason is one nobody will act on
    # when it fires for a real one.
    def _skeleton(text):
        return [strip_diacritics(w) for w in clean_and_tokenize(text)]

    mismatch = sum(1 for p, i in zip(predictions_nfc, inputs_nfc)
                   if _skeleton(p) != _skeleton(i))
    metrics["skeleton_mismatch_rate"] = round(100 * mismatch / max(len(inputs), 1), 2)
    metrics["split"] = name
    metrics["n_examples"] = len(predictions)
    metrics["n_empty_predictions"] = blank
    metrics["n_truncated"] = truncated
    print(f"  {name}:")
    print(f"      frozen    DER_ce={metrics['DER_ce']}  DER_noce={metrics['DER_noce']}  "
          f"WER_ce={metrics['WER_ce']}  WER_noce={metrics['WER_noce']}")
    print(f"      CORRECTED DER_ce={metrics['DER_ce_corr']}  DER_noce={metrics['DER_noce_corr']}  "
          f"WER_ce={metrics['WER_ce_corr']}  WER_noce={metrics['WER_noce_corr']}   <- cite this")
    print(f"      skeleton_mismatch={metrics['skeleton_mismatch_rate']}%")
    return metrics


def _write_predictions(name, inputs, references, predictions, domains=None):
    df = pd.DataFrame({"input": inputs, "reference": references, "prediction": predictions})
    if domains is not None:
        df.insert(0, "domain", domains)
    csv_path = os.path.join(EVAL_DIR, f"{name}_predictions.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {len(df)} rows -> {csv_path}")


def evaluate_split(name, model, tokenizer, pairs, label="ft"):
    """Generate and score one split.

    `label` namespaces EVERY file this writes, and that is load-bearing. run_inference() resumes
    from `<name>_preds_checkpoint.csv`. Without a label in that filename, scoring the base model
    after the fine-tuned one would "resume" the fine-tuned model's cached predictions and report
    them as the baseline — a silent result-swap that looks exactly like "fine-tuning changed
    nothing" (RUN_REPORT.md §5.5).
    """
    name = f"{label}_{name}" if label else name
    print(f"\n--- scoring {name} ({len(pairs)} examples) ---")
    inputs = [p[0] for p in pairs]
    references = [p[1] for p in pairs]
    predictions, truncated = run_inference(model, tokenizer, pairs, name)
    _write_predictions(name, inputs, references, predictions)
    return score_predictions(name, inputs, references, predictions, truncated)


def evaluate_benchmark(model, tokenizer, benchmark_raw, label="ft"):
    """Generate SadeedDiac-25 once, score it CA / MSA / pooled, and add the macro mean.

    Why the macro mean and not just the pooled number: pooling is diacritic-weighted and CA
    paragraphs are longer, so a pooled figure is CA-weighted — and CA is the easy half. The
    results spreadsheet's "Mean" column is the macro average of the two domains, so that is what
    goes in a comparison. Both are emitted; RUN_REPORT.md §3 explains the difference.

    Every row is scored twice, by both scorers, from this one generation pass — see the
    CORRECTED metrics block. The `_corr` columns are the ones to cite.
    """
    name = f"{label}_benchmark_sadeeddiac25" if label else "benchmark_sadeeddiac25"
    pairs = benchmark_pairs(benchmark_raw)
    domains = benchmark_domains(benchmark_raw)

    print(f"\n--- scoring {name} ({len(pairs)} examples, one generation pass) ---")
    inputs = [p[0] for p in pairs]
    references = [p[1] for p in pairs]
    predictions, truncated = run_inference(model, tokenizer, pairs, name)
    _write_predictions(name, inputs, references, predictions, domains)

    rows = [score_predictions(f"{name}_pooled", inputs, references, predictions, truncated)]

    if domains:
        for dom in ("CA", "MSA"):
            idx = [i for i, d in enumerate(domains) if d == dom]
            if not idx:
                continue
            rows.append(score_predictions(
                f"{name}_{dom}",
                [inputs[i] for i in idx],
                [references[i] for i in idx],
                [predictions[i] for i in idx],
                # Truncations are not attributed per-domain (run_inference counts per batch, and
                # batches mix domains). Reported only on the pooled row so it is never wrong.
                truncated=0,
            ))

        ca = next((r for r in rows if r["split"].endswith("_CA")), None)
        msa = next((r for r in rows if r["split"].endswith("_MSA")), None)
        if ca and msa:
            macro = {"split": f"{name}_MACRO_MEAN",
                     "n_examples": ca["n_examples"] + msa["n_examples"],
                     "n_empty_predictions": ca["n_empty_predictions"] + msa["n_empty_predictions"],
                     "n_truncated": truncated}
            for m in ("DER_ce", "DER_noce", "WER_ce", "WER_noce",
                      "DER_ce_corr", "DER_noce_corr", "WER_ce_corr", "WER_noce_corr",
                      "skeleton_mismatch_rate"):
                macro[m] = round((ca[m] + msa[m]) / 2, 2)
            rows.append(macro)

            # Both anchors are printed, each against the 4B number computed the SAME way.
            # Mixing them is the error this exists to prevent: 6.96 and 8.48 are the same 4B
            # predictions under two scorers, not two different runs.
            #
            # (A macro mean can differ by 0.01 from RESULTS.md's, which averaged the two
            # already-rounded domain figures: (2.67 + 14.28) / 2 = 8.475, which rounds to 8.47
            # here and is quoted as 8.48 there. Same numbers, not a scorer discrepancy.)
            print(f"\n  MACRO MEAN, frozen scorer     DER_noce={macro['DER_noce']}   "
                  f"(CA {ca['DER_noce']} / MSA {msa['DER_noce']}, "
                  f"gap {abs(msa['DER_noce'] - ca['DER_noce']):.2f})")
            print("      ^ compares to RUN_REPORT.md's 6.96 for 4B LoRA. Comparison only — "
                  "RESULTS.md retired this scorer; do not put it in the results table.")
            print(f"\n  MACRO MEAN, CORRECTED scorer  DER_noce={macro['DER_noce_corr']}   "
                  f"(CA {ca['DER_noce_corr']} / MSA {msa['DER_noce_corr']}, "
                  f"gap {abs(msa['DER_noce_corr'] - ca['DER_noce_corr']):.2f})")
            print("      ^ CITE THIS. Compares to RESULTS.md's 8.48 for 4B LoRA, and only to "
                  "other corrected-scorer numbers.")

            # Judged on the CORRECTED numbers: the frozen scorer understates MSA specifically
            # (11.07 -> 14.28 on the 4B predictions, +29% relative), because MSA is where the
            # model paraphrases and the frozen scorer lets insertions through for free. Running
            # this check on the flattering metric would be checking the wrong number.
            if msa["DER_noce_corr"] > 2 * max(ca["DER_noce_corr"], 0.01):
                print("\n  [NOTE] MSA error is >2x CA error on the corrected scorer, the same "
                      "asymmetry the 4B run showed. Sadeed_Tashkeela is Tashkeela-derived and "
                      "Classical-heavy, so the model over-applies CA conventions to MSA. Run "
                      "--analyze, then use --include-filename / --exclude-filename to build a "
                      "balanced rung. More steps on this corpus will not fix it.")
    return rows


def plot_performance_comparison(metrics_df):
    # Corrected scorer only. Plotting both would put two different definitions of the same
    # metric name on one axis, which is how a chart ends up quoted as the frozen number.
    metric_cols = ["DER_ce_corr", "DER_noce_corr", "WER_ce_corr", "WER_noce_corr"]

    plt.figure(figsize=(9, 5))
    for _, row in metrics_df.iterrows():
        plt.plot(metric_cols, [row[c] for c in metric_cols], marker="o", label=row["split"])

    plt.xlabel("Metric (corrected scorer)")
    plt.ylabel("Error rate (%)")
    plt.title(f"{MODEL_ID} full FT — by split (corrected scorer)")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)

    out_path = os.path.join(EVAL_DIR, "performance_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved performance comparison -> {out_path}")


def run_evaluation(model, tokenizer, train_raw, test_raw, benchmark_raw,
                   splits=("train", "test", "benchmark"), label="ft", model_source=""):
    all_metrics = []

    if train_raw is not None and "train" in splits:
        try:
            train_eval = train_raw
            if TRAIN_EVAL_SAMPLE_SIZE and TRAIN_EVAL_SAMPLE_SIZE < len(train_eval):
                train_eval = train_eval.shuffle(seed=RANDOM_SEED).select(
                    range(TRAIN_EVAL_SAMPLE_SIZE))
            all_metrics.append(evaluate_split(
                "train", model, tokenizer,
                as_pairs(train_eval, INPUT_COLUMN, OUTPUT_COLUMN), label))
        except Exception as e:
            print(f"[WARN] Train evaluation failed: {e}")
            traceback.print_exc()

    if test_raw is not None and "test" in splits:
        try:
            all_metrics.append(evaluate_split(
                "test", model, tokenizer,
                as_pairs(test_raw, INPUT_COLUMN, OUTPUT_COLUMN), label))
        except Exception as e:
            print(f"[WARN] Test evaluation failed: {e}")
            traceback.print_exc()

    if benchmark_raw is not None and "benchmark" in splits:
        try:
            all_metrics.extend(evaluate_benchmark(model, tokenizer, benchmark_raw, label))
        except Exception as e:
            print(f"[WARN] Benchmark evaluation failed: {e}")
            traceback.print_exc()

    if not all_metrics:
        print("[WARN] No evaluation results were produced — skipping metrics summary.")
        return

    try:
        # Corrected columns FIRST — they are the ones to cite. The frozen columns follow, and
        # exist only to be diffed against RUN_REPORT.md's 4B numbers.
        metrics_df = pd.DataFrame(all_metrics)[
            ["split", "n_examples", "n_empty_predictions", "n_truncated",
             "DER_ce_corr", "DER_noce_corr", "WER_ce_corr", "WER_noce_corr",
             "DER_ce", "DER_noce", "WER_ce", "WER_noce", "skeleton_mismatch_rate"]
        ]
        metrics_df.insert(0, "model", model_source or MODEL_ID)
        metrics_df.insert(1, "method", label)

        summary_path = os.path.join(
            EVAL_DIR, f"metrics_summary_{label}.csv" if label else "metrics_summary.csv")
        metrics_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Saved metrics summary -> {summary_path}")
        print(metrics_df.to_string(index=False))

        plot_performance_comparison(metrics_df)
    except Exception as e:
        print(f"[WARN] Failed to save/plot metrics summary: {e}")
        traceback.print_exc()


# ==============================================================================================
# Stage 7 — Corpus analysis (CPU only)
#
# Cheap, and it answers the two questions RUN_REPORT.md leaves open:
#   §9.2  "a domain-balanced rung, not a longer one" — needs the filename composition
#   §6    "19% of all compute spent re-encoding the same Arabic instruction" — needs the split
# ==============================================================================================

def analyze_corpus(sample_rows=5000):
    from collections import Counter
    from transformers import AutoTokenizer

    train_raw, test_raw = load_training_data()

    print("\n" + "=" * 90)
    print("CORPUS COMPOSITION BY FILENAME")
    print("=" * 90)
    if FILENAME_COLUMN in train_raw.column_names:
        counts = Counter(train_raw[FILENAME_COLUMN])
        total = sum(counts.values())
        print(f"{len(counts):,} distinct filenames over {total:,} rows. Top 40:\n")
        print(f"{'rows':>12}  {'share':>7}  filename")
        for fn, n in counts.most_common(40):
            print(f"{n:>12,}  {100 * n / total:>6.2f}%  {fn}")
        pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]),
                     columns=["filename", "rows"]).to_csv(
            os.path.join(EVAL_DIR, "corpus_composition.csv"), index=False,
            encoding="utf-8-sig")
        print(f"\n[OK] full breakdown -> {os.path.join(EVAL_DIR, 'corpus_composition.csv')}")
        print("\nUse this to build a domain-balanced rung:")
        print("    --exclude-filename '<regex matching the CA-heavy sources>'")
        print("The MSA deficit (RUN_REPORT.md: 11.07 vs CA 2.84) is created by training on a "
              "Classical-heavy corpus, and no amount of extra steps fixes it.")
    else:
        print(f"[WARN] no {FILENAME_COLUMN!r} column on this split.")

    print("\n" + "=" * 90)
    print(f"TOKEN ECONOMICS  (measured on {sample_rows:,} rows with the real tokenizer)")
    print("=" * 90)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR,
                                              trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    detect_prompt_mode(tokenizer)

    sample = train_raw.shuffle(seed=RANDOM_SEED).select(
        range(min(sample_rows, len(train_raw))))
    overhead = len(tokenizer(render_prompt(tokenizer, ""), add_special_tokens=False)["input_ids"])

    n_in = n_out = n_words = 0
    for row in sample:
        n_in += len(tokenizer(row[INPUT_COLUMN], add_special_tokens=False)["input_ids"])
        n_out += len(tokenizer(row[OUTPUT_COLUMN], add_special_tokens=False)["input_ids"])
        n_words += len(row[INPUT_COLUMN].split())
    n = len(sample)
    mean_in, mean_out, mean_words = n_in / n, n_out / n, n_words / n
    total = overhead + mean_in + mean_out

    print(f"\n{'component':<34}{'tokens/row':>12}{'share':>9}")
    print(f"{'diacritised target':<34}{mean_out:>12.1f}{100 * mean_out / total:>8.0f}%"
          f"   ({mean_out / max(mean_words, 1):.1f} tokens/word)")
    print(f"{'undiacritised input':<34}{mean_in:>12.1f}{100 * mean_in / total:>8.0f}%"
          f"   ({mean_in / max(mean_words, 1):.1f} tokens/word)")
    print(f"{'prompt template + system prompt':<34}{overhead:>12.1f}"
          f"{100 * overhead / total:>8.0f}%   (fixed, paid once per example)")
    print(f"{'TOTAL':<34}{total:>12.1f}{100:>8.0f}%")

    rows_full = len(train_raw)
    print(f"\nFull epoch: {rows_full:,} rows x {total:.0f} tokens = "
          f"{rows_full * total / 1e6:,.0f}M tokens")
    print(f"Of which {rows_full * overhead / 1e6:,.0f}M ({100 * overhead / total:.0f}%) is the "
          f"same instruction re-encoded {rows_full:,} times.")
    print("\nShortening SYSTEM_PROMPT is a direct saving of that share — but it changes the "
          "prompt, so the run stops being comparable to RUN_REPORT.md's 4B numbers. Worth it "
          "for a production run, not for the comparison this package exists to make.")
    print(f"\ntest split: {len(test_raw):,} rows")


def prepare_only(train_raw, test_raw):
    """Tokenize the full corpus on a CPU box, so the GPU job never pays for it.

    Two reasons this is a separate phase rather than something the training job just does:

    1. COST. Tokenizing ~1M rows takes 15-25 minutes on 8 cores. Done inside the GPU container
       that is $4.58/hr of H100 sitting idle (~$1.70); done here it is $0.63/hr (~$0.25).

    2. IT IS WHERE THE CORPUS BITES. RUN_REPORT.md §5.3: exactly 1 row of 1,042,698 has a null
       text column, and it killed a real training run ~12 minutes in with the GPU already
       rented. A 200-row smoke test cannot find it — only a pass over the full corpus can, and
       this is the cheapest possible pass over the full corpus.

    The GPU job reuses this work through the datasets `.map()` fingerprint cache, which lives on
    the same Volume (cache_dir=CACHE_DIR). If the fingerprint ever fails to match — a different
    transformers version hashing the tokenizer differently, say — the GPU job simply re-tokenizes
    and the only loss is the saving. It cannot produce wrong data.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR,
                                              trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"          # must match load_model_for_training()
    detect_prompt_mode(tokenizer)
    assert_prompt_sane(tokenizer)

    # Same helper as train_model(), so the slicing cannot drift and cause a cache miss.
    eval_raw = select_eval_subset(test_raw, IN_TRAINING_EVAL_SUBSET)

    train_tok, eval_tok = tokenize_splits(tokenizer, train_raw, eval_raw)

    lengths = list(train_tok["length"])
    n_prompt = list(train_tok["n_prompt_tokens"])
    total = sum(lengths)
    eff_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps = math.ceil(len(train_tok) / eff_batch)

    print("\n" + "=" * 90)
    print("WHAT THE GPU JOB WILL SEE")
    print("=" * 90)
    print(f"  train rows            {len(train_tok):,}")
    print(f"  eval rows             {len(eval_tok):,}")
    print(f"  total tokens/epoch    {total / 1e6:,.1f}M")
    print(f"  mean / p50 / p99 len  {total / len(lengths):.0f} / "
          f"{sorted(lengths)[len(lengths) // 2]} / {sorted(lengths)[int(len(lengths) * 0.99)]}"
          f"  (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH})")
    print(f"  at cap ({MAX_SEQ_LENGTH})         {sum(1 for l in lengths if l >= MAX_SEQ_LENGTH):,} rows "
          f"— these had their target truncated")
    print(f"  prompt overhead       {sum(n_prompt) / len(n_prompt):.0f} tokens/row "
          f"({100 * sum(n_prompt) / total:.0f}% of all tokens; masked out of the loss but still "
          f"forwarded)")
    print(f"  optimizer steps       {steps:,} at effective batch {eff_batch}")
    print("\n[OK] tokenization cached on the Volume. Run smoke_test next — its [THROUGHPUT] "
          "line, not this, is the cost estimate.")


# ==============================================================================================
# Main
# ==============================================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Full fine-tune of Qwen3.5-0.8B for Arabic "
                                            "diacritization")
    p.add_argument("--analyze", action="store_true",
                   help="CPU only: corpus composition by filename + token economics. Run this "
                        "before building a domain-balanced rung.")
    p.add_argument("--prepare-only", action="store_true",
                   help="CPU only: tokenize the full corpus into the datasets cache, then stop. "
                        "The GPU job reuses it via the .map() fingerprint.")
    p.add_argument("--smoke-test", action="store_true",
                   help=f"{SMOKE_TRAIN_SIZE:,} rows, {SMOKE_MAX_STEPS} steps, no checkpoints. "
                        f"Proves the plumbing AND measures steady-state throughput.")
    p.add_argument("--train-only", action="store_true", help="train, then stop (no scoring)")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; score --model against train/test/benchmark")
    p.add_argument("--model", default=FINAL_MODEL_DIR,
                   help=f"model dir for --eval-only, or 'none' to score the BASE model zero-shot "
                        f"through this identical pipeline (default: {FINAL_MODEL_DIR})")
    p.add_argument("--label", default="ft",
                   help="namespaces every output file. MUST differ between the base and "
                        "fine-tuned runs — see evaluate_split().")
    p.add_argument("--splits", default="train,test,benchmark",
                   help="comma list of splits to score. 'benchmark' alone is the cheap way to "
                        "get the only number that goes in the results table.")
    p.add_argument("--max-steps", type=int, default=None,
                   help=f"override MAX_STEPS={MAX_STEPS} (-1 = one full epoch, the default)")
    p.add_argument("--train-size", type=int, default=None, help="cap the train split")
    p.add_argument("--include-filename", default=None,
                   help="regex; keep only train rows whose `filename` matches. Run --analyze "
                        "first to see what the values actually are.")
    p.add_argument("--exclude-filename", default=None,
                   help="regex; drop train rows whose `filename` matches.")
    return p.parse_args()


def main():
    args = parse_args()
    hf_login()

    print(f"[INFO] MODEL_ID={MODEL_ID}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] EVAL_DIR={EVAL_DIR}")

    # --- Analysis only (CPU) ---------------------------------------------------------------
    if args.analyze:
        analyze_corpus()
        print("\n[DONE] Analysis complete.")
        return

    # The kernel check needs a CUDA build to be meaningful, and --analyze/--prepare-only run on
    # CPU boxes where it would fail for the wrong reason.
    if not args.prepare_only:
        if os.environ.get("REQUIRE_FAST_PATH", "1") != "0":
            assert_fast_path()
        else:
            print("[WARN] REQUIRE_FAST_PATH=0 — skipping the linear-attention kernel check. "
                  "18 of 24 layers may be running the slow PyTorch fallback.")

    # --- Data --------------------------------------------------------------------------------
    # Skip the training corpus entirely when nothing being scored needs it. Loading it means
    # downloading ~583MB and running the null-row filter over 1,042,698 rows — several minutes
    # with an H100 idle at $4.58/hr. `--eval-only --splits benchmark`, which is how the zero-shot
    # baseline is scored, needs none of it.
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    need_corpus = (not args.eval_only) or bool({"train", "test"} & set(splits))

    train_raw = test_raw = None
    if need_corpus:
        try:
            train_raw, test_raw = load_training_data(
                train_size=args.train_size,
                test_size=256 if args.smoke_test else None,
                smoke=args.smoke_test,
                include_filename=args.include_filename,
                exclude_filename=args.exclude_filename,
            )
        except Exception as e:
            print(f"[FATAL] Could not load {TRAIN_DATASET}: {e}")
            traceback.print_exc()
            raise
    else:
        print(f"[INFO] --splits {args.splits} needs neither train nor test — skipping the "
              f"{TRAIN_DATASET} load entirely (saves several minutes of GPU-idle time)")

    # --- Tokenize only (CPU) -----------------------------------------------------------------
    if args.prepare_only:
        prepare_only(train_raw, test_raw)
        return

    benchmark_raw = None
    if not (args.smoke_test or args.train_only) and "benchmark" in splits:
        try:
            benchmark_raw = load_benchmark_data()
        except Exception as e:
            print(f"[WARN] Could not load {BENCHMARK_DATASET}, continuing without it: {e}")

    # --- Evaluation only ---------------------------------------------------------------------
    if args.eval_only:
        try:
            model, tokenizer, source = load_model_for_eval(args.model)
        except Exception as e:
            print(f"[FATAL] Could not load model: {e}")
            traceback.print_exc()
            raise
        run_evaluation(model, tokenizer, train_raw, test_raw, benchmark_raw,
                       splits=splits, label=args.label, model_source=source)
        print(f"\n[DONE] Evaluation complete. metrics: "
              f"{os.path.join(EVAL_DIR, f'metrics_summary_{args.label}.csv')}")
        return

    # --- Model -------------------------------------------------------------------------------
    try:
        model, tokenizer = load_model_for_training()
    except Exception as e:
        print(f"[FATAL] Could not load model/tokenizer: {e}")
        traceback.print_exc()
        raise

    # --- Training ----------------------------------------------------------------------------
    try:
        trainer = train_model(model, tokenizer, train_raw, test_raw,
                              smoke_test=args.smoke_test,
                              max_steps=SMOKE_MAX_STEPS if args.smoke_test else args.max_steps)
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        traceback.print_exc()
        raise

    if args.smoke_test:
        losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
        print(f"\n[SMOKE] first 10 losses: {losses[:10]}")
        print(f"[SMOKE] last 5 losses:    {losses[-5:]}")
        print("[SMOKE] THE GATE: finite, non-zero, and trending down. A loss near 0.0 at step 1 "
              "means the masking is inverted — stop.")
        print("[SMOKE] The [THROUGHPUT] line above is your cost estimate. Nothing else is.")
        print("[DONE] Smoke test complete.")
        return

    try:
        plot_training_curves(trainer)
    except Exception as e:
        print(f"[WARN] Failed to plot training curves: {e}")

    if args.train_only:
        print(f"\n[DONE] Training complete (--train-only). model: {FINAL_MODEL_DIR}")
        return

    # --- Evaluation: train, test, benchmark --------------------------------------------------
    # The in-memory model is still configured for TRAINING. Scoring it as-is would be wrong in
    # two expensive ways, so convert it first — see prepare_model_for_generation().
    model = prepare_model_for_generation(model)
    run_evaluation(model, tokenizer, train_raw, test_raw, benchmark_raw,
                   splits=splits, label="ft", model_source=FINAL_MODEL_DIR)

    print("\n[DONE] Run complete.")
    print(f"       model   : {FINAL_MODEL_DIR}")
    print(f"       metrics : {os.path.join(EVAL_DIR, 'metrics_summary_ft.csv')}")


if __name__ == "__main__":
    main()
