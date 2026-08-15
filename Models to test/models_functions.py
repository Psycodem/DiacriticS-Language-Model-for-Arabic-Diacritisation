# -*- coding: utf-8 -*-
"""models_functions.py — the one scoring implementation for every benchmark run.

Canonical DER/WER for this folder. Every *_models_v2.py script imports from here
so a number in the results sheet can be traced to exactly one implementation.

WHY THIS FILE EXISTS
--------------------
The v1 scripts each carried their own inline copy of these metrics, and the
copies had drifted:

    LLMs_Last_test.py, instruct_models_v1.py   numeral normalisation + citation
                                               stripping + digit-token exclusion
    Small_models_v1.py                         ce=/noce= split, but none of the above
    Mideum_models_v1.py, Large_models_v1.py    none of the above

That drift is visible in the results sheet: aya-expanse-8b appears twice on the
MSA track, at 96.28 (Large_models_v1.py) and 86.21 (instruct_models_v1.py) —
a ten-point spread produced by the scoring code, not the model. Numbers from
different implementations cannot be ranked against each other, so the v2 scripts
re-score every affected model through this module.

Kept behaviourally identical to `Train Related/Models_Functions.py`; if you edit
one, edit the other.
"""

import re

import jiwer

# ── Diacritics ─────────────────────────────────────────────────────────
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652]')

# ── Numeral normalization ──────────────────────────────────────────────
# Maps Eastern Arabic-Indic digits -> Western Arabic digits so numeral-
# system differences between prediction and reference aren't scored as
# diacritization errors.
_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_numerals(text: str) -> str:
    """Converts Eastern Arabic-Indic digits to Western Arabic digits."""
    return text.translate(_EASTERN_TO_WESTERN)


# ── Citation/reference stripping (Fadel-style "(41 / 251)") ────────────
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
