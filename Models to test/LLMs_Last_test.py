# -*- coding: utf-8 -*-
"""
Models: Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, google/gemma-4-E4B-it
"""

import subprocess
import sys


def pip_install(*packages, upgrade=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if upgrade:
        cmd.append("-U")
    cmd.extend(packages)
    subprocess.run(cmd, check=True)


pip_install("transformers", "accelerate", "datasets", "huggingface_hub", upgrade=True)
pip_install("pyarabic", "prettytable", "tqdm", "sentencepiece", "jiwer")

"""## Step 2 - Configuration
- **`QUICK_TEST = False`** runs the whole benchmark. Flip it to `True` (with `LIMIT`) only if you want a fast sanity check.
- **`CHUNK_SIZE`** = how many paragraphs between disk checkpoints (kept below 100).
- Generation **batch sizes** are small (also below 100); lower them further if you still hit out-of-memory.
"""

MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-E4B-it",
]

DATASET_ID = "Misraj/SadeedDiac-25"

QUICK_TEST = False    # False -> full 1,200 paragraphs
LIMIT      = 20       # only used when QUICK_TEST is True

CHUNK_SIZE = 50       # checkpoint to disk every 50 paragraphs (< 100)

BATCH_SIZE = {
    "Qwen/Qwen3.5-4B": 16,
    "Qwen/Qwen3.5-9B": 8,
    "google/gemma-4-E4B-it": 16,
}
DEFAULT_BATCH = 8
MAX_BATCH     = 64    # hard cap so a batch is always < 100
MAX_NEW_CAP   = 1024  # ceiling on generated tokens per paragraph

import os
OUT_DIR = "sadeed_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Where Hugging Face model/tokenizer weights get cached. Point this at a
# persistent disk (e.g. a mounted Google Drive path) if you want re-runs
# across sessions to skip re-downloading multi-GB model weights.
CACHE_DIR = "hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

"""## Step 3 - Load the benchmark and build inputs
Loads SadeedDiac-25 and **auto-detects** the diacritized-gold column (the one with the most Arabic diacritics), so we don't depend on an exact field name. For each paragraph we remove *tatweel*, strip the diacritics to make the **model input**, and keep the tatweel-stripped gold as the **reference**.
"""

import gc, json, re, subprocess
import pandas as pd
import torch
from tqdm.auto import tqdm
from pyarabic import araby


def _diac_ratio(s):
    if not isinstance(s, str) or not s:
        return 0.0
    d = sum(1 for c in s if araby.is_haraka(c) or araby.is_shadda(c) or araby.is_tanwin(c))
    return d / len(s)


def load_gold_texts(dataset_id, limit=None):
    from datasets import load_dataset
    ds = load_dataset(dataset_id)
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    scores = {c: df[c].astype(str).head(50).map(_diac_ratio).mean() for c in df.columns}
    gold_col = max(scores, key=scores.get)
    print(f"split='{split}', rows={len(df)}, gold column: '{gold_col}' "
          f"(diacritic ratio {scores[gold_col]:.3f})")

    gold = df[gold_col].astype(str).tolist()
    if limit:
        gold = gold[:limit]
    gold = [araby.strip_tatweel(g) for g in gold]        # reference
    inputs = [araby.strip_tashkeel(g) for g in gold]      # model input
    return gold, inputs


golds, inputs = load_gold_texts(DATASET_ID, limit=LIMIT if QUICK_TEST else None)
print(f"Evaluating {len(golds)} paragraphs ({'QUICK TEST' if QUICK_TEST else 'FULL'})")

"""## Step 4 - Prompt and output cleaning
Everything goes in a **single user turn** (Gemma has no system role, so this is the one format that loads across all three models). `clean_output` removes any `<think>` reasoning block and stray quotes/labels.
"""

INSTRUCTION = (
    "أضِف التشكيل (الحركات) الكامل للنص العربي التالي. "
    "لا تُغيّر الكلمات ولا ترتيبها، ولا تحذف أو تُضِف أيّ كلمة، "
    "وأعِد النصّ المُشكَّل فقط دون أي شرح أو مقدمة.\n\nالنص:\n"
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_messages(text):
    return [{"role": "user", "content": INSTRUCTION + text}]


def clean_output(text):
    text = _THINK_RE.sub("", text)
    text = text.strip().strip('"“”').strip()
    text = re.sub(r"^\s*النص(\s+المشكّ?ل)?\s*:\s*", "", text)
    return text.strip()

"""## Step 5 - Run one model, with checkpointing + error handling
Generates diacritized text in small batches. After every `CHUNK_SIZE` paragraphs it **saves a checkpoint** (`..._preds.csv`). On re-run it **loads that checkpoint and skips finished rows**, so a crash costs you at most one chunk.

Safety details: left padding for batched decoding; the tokenized chat template adds each model's special tokens correctly; tries `AutoModelForCausalLM` then falls back to `AutoModelForImageTextToText`; **any batch that errors is caught**, its rows are left blank (they get skipped in scoring), and the run continues; VRAM is cleared before the next model.
"""

def run_model(model_id, inputs_texts, batch_size, chunk_size, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = os.path.join(out_dir, model_id.replace("/", "__") + "__preds.csv")
    preds = [None] * len(inputs_texts)

    # ---- resume from checkpoint if present ----
    if os.path.exists(ckpt):
        try:
            prev = pd.read_csv(ckpt, header=None, names=["idx", "pred"], keep_default_na=False)
            for _, r in prev.iterrows():
                i = int(r["idx"])
                if 0 <= i < len(preds):
                    preds[i] = str(r["pred"])
            print(f"  resumed {sum(p is not None for p in preds)} cached predictions")
        except Exception as e:
            print(f"  could not read checkpoint ({type(e).__name__}); starting fresh")

    todo = [i for i in range(len(inputs_texts)) if preds[i] is None]
    if not todo:
        print("  all predictions cached; skipping generation")
        return preds

    print(f"\nLoading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
            cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"  CausalLM failed ({type(e).__name__}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
            cache_dir=CACHE_DIR)
    model.eval()

    def save_ckpt():
        done = [(i, preds[i]) for i in range(len(preds)) if preds[i] is not None]
        pd.DataFrame(done).to_csv(ckpt, index=False, header=False)

    order = sorted(todo, key=lambda i: len(inputs_texts[i]))
    since_save = 0
    for start in tqdm(range(0, len(order), batch_size), desc=model_id.split("/")[-1]):
        idxs = order[start:start + batch_size]
        try:
            convs = [build_messages(inputs_texts[i]) for i in idxs]
            try:
                enc = tok.apply_chat_template(
                    convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", padding=True, enable_thinking=False)
            except TypeError:
                enc = tok.apply_chat_template(
                    convs, add_generation_prompt=True, tokenize=True, return_dict=True,
                    return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            in_len = enc["input_ids"].shape[1]
            max_new = min(int(in_len * 2.2) + 32, MAX_NEW_CAP)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     num_beams=1, pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            for j, i in enumerate(idxs):
                preds[i] = clean_output(dec[j])
        except Exception as e:
            print(f"  batch @ {start} failed ({type(e).__name__}: {e}); leaving blank")
            for i in idxs:
                if preds[i] is None:
                    preds[i] = ""
            gc.collect(); torch.cuda.empty_cache()

        since_save += len(idxs)
        if since_save >= chunk_size:
            save_ckpt(); since_save = 0

    save_ckpt()
    del model
    gc.collect(); torch.cuda.empty_cache()
    return preds

"""## Step 6 - DER/WER metrics (self-contained, no external file)
Word-level edit-distance alignment (via `jiwer`) rather than requiring
predictions/gold to have the same word count, so a hallucinated or dropped
word no longer causes an entire paragraph to be discarded from scoring --
it just gets scored as an error for that word. Also normalizes Eastern
Arabic-Indic numerals, strips Fadel-style citation refs like "(41 / 251)",
and excludes pure-digit tokens from DER (they can't carry a diacritic).
"""

import jiwer

ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652]')

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
    """True if the word (after stripping diacritics) is purely numeric --
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
            # "insert" chunks (hallucinated extra words) are skipped --
            # no reference word to score diacritics against.


def compute_der(predictions: list, references: list, ce: bool = True) -> float:
    """
    Diacritic Error Rate via edit-distance word alignment.
    Pure-digit words (e.g. '25', '750') are EXCLUDED -- they can never
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
    scored here (unlike DER) since they're real content -- the model
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


def evaluate(preds, golds):
    """
    Computes DER/WER (with & without case ending) using compute_der_wer
    above. Its jiwer-based alignment tolerates length mismatches between
    preds and golds internally, so nothing needs to be dropped
    sentence-by-sentence the way a positional-alignment evaluator would.
    """
    import warnings
    warnings.filterwarnings("ignore")

    scores = compute_der_wer(preds, golds)

    return {
        "DER% (with case)": scores["DER_ce"],
        "DER% (no case)":   scores["DER_noce"],
        "WER% (with case)": scores["WER_ce"],
        "WER% (no case)":   scores["WER_noce"],
        "scored":  len(preds),
        "skipped": 0,
    }

"""## Step 7 - Run everything and save results
For each model: generate (with checkpointing) → save its `(gold, pred)` CSV → score. **Each model is wrapped in try/except**; if one fails, its error is recorded and the others still run. `results.csv` is re-written after every model, so you always have whatever finished.
"""

rows = []
results_path = os.path.join(OUT_DIR, "results.csv")

for model_id in MODELS:
    try:
        bs = min(BATCH_SIZE.get(model_id, DEFAULT_BATCH), MAX_BATCH)  # always < 100
        preds = run_model(model_id, inputs, bs, CHUNK_SIZE, OUT_DIR)
        preds = [p if p is not None else "" for p in preds]

        pd.DataFrame({"gold": golds, "pred": preds}).to_csv(
            os.path.join(OUT_DIR, model_id.replace("/", "__") + ".csv"),
            index=False, header=False)

        metrics = {"model": model_id, **evaluate(preds, golds)}
        print(f"  -> {model_id}: DER(with)={metrics['DER% (with case)']:.3f}  "
              f"WER(with)={metrics['WER% (with case)']:.3f}")
    except Exception as e:
        print(f"[ERROR] {model_id} failed: {type(e).__name__}: {e}")
        metrics = {"model": model_id, "error": f"{type(e).__name__}: {e}"}

    rows.append(metrics)
    # save after every model so partial results are never lost
    pd.DataFrame(rows).set_index("model").to_csv(results_path)
    print(f"  saved results -> {results_path}")

results = pd.DataFrame(rows).set_index("model")
print("\n================ SadeedDiac-25 results ================")
results.round(3)

"""### Notes
- **Full run is long.** Checkpointing means you can stop/restart anytime, re-running Step 7 resumes from the last saved chunk.
- **MSA vs CA split:** in the full set the first 600 paragraphs are MSA and the last 600 are CA. To score each separately: `evaluate(preds[:600], golds[:600])` and `evaluate(preds[600:], golds[600:])`.
- **Add your own model** (e.g. CATT or your Qwen3.5 LoRA): append its id to `MODELS`; for a LoRA adapter, load the base model then `PeftModel.from_pretrained`.
- **Fresh start for one model:** delete its `sadeed_outputs/<model>__preds.csv` to force re-generation.
- **Metrics are self-contained:** all DER/WER functions live in Step 6 of this script -- no external `Models_Functions.py` or Sadeed repo clone required.
"""
