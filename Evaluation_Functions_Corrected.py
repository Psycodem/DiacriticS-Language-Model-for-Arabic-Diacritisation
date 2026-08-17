# -*- coding: utf-8 -*-
"""
Corrected DER/WER scorer for Arabic diacritization.

Drop-in replacement for Evaluation_Functions.py: same public function names
(compute_der, compute_wer, compute_der_wer) and the same return-dict keys
(DER_ce, DER_noce, WER_ce, WER_noce), so any script that does

    from Evaluation_Functions import compute_der_wer

can switch to

    from Evaluation_Functions_Corrected import compute_der_wer

with no other changes. Internally it fixes five defects confirmed present in
Evaluation_Functions.py (and in the byte-identical copy inside
train_lora_gemma4_diacritization.py / train_lora_qwen35_4b_diacritization.py):

  1. No NFC normalization. Arabic combining marks (e.g. shadda + fatha on one
     letter) can be stored in either order and render identically, but a
     positional comparison scores the two orders as different. Worth ~2.4x
     on identical predictions in the SadeedDiac-25 corpus.
  2. Diacritics were pulled into a flat per-word list with the host letter
     discarded, so the "right mark, wrong letter" case can accidentally
     compare equal — Arabic only has ~8 distinct diacritic symbols, so
     positional collisions are common, not rare.
  3. The DER denominator was max(len(ref_marks), len(pred_marks)), so it
     moved with the prediction instead of being fixed by the reference.
  4. Hallucinated / inserted words (no corresponding reference word) were
     skipped by _align_words, so a model that pads its output with invented
     text scored as if it hadn't. This is the single most dangerous defect:
     it hides exactly the failure mode most worth catching in a generative
     model, and it is scored as literally 0% error rather than an error.
  5. The diacritic-mark regex [\\u064B-\\u0652] excludes dagger alef
     (U+0670), common in Qur'anic/Classical spelling, so a missing dagger
     alef was invisible to scoring.

Ported from the corrected scorer in the DiacriticS Qwen3.5 fine-tuning
findings notebooks (Fine-Tune-qwen-9-0.8-Findings), where it is the scorer
whose numbers are cited in the published results.
"""

import re
import unicodedata

import jiwer

# ── Diacritics ─────────────────────────────────────────────────────────
# U+064B-U+065F: the full "Arabic combining marks" block (fatha, damma,
# kasra, sukun, shadda, the tanwin marks, Qur'anic annotation marks, etc.)
# plus U+0670 (dagger alef / superscript alef), which sits outside that
# contiguous block and was missing from the old [ً-ْ] class.
DIACRITICS = re.compile(r'[ً-ٰٟ]')

# Arabic base letters that can carry a diacritic: the main alphabet
# (U+0621-U+063A hamza..ghain, U+0641-U+064A feh..yeh) plus extended
# letters (U+0671-U+06D3: alef wasla, teh marbuta variants, Persian/Urdu
# extensions, etc.) that appear in Sadeed_Tashkeela / SadeedDiac-25.
LETTER = re.compile(r'[ء-غف-يٱ-ۓ]')

TATWEEL = "ـ"  # ـ — a stretching glyph, not a letter; drop it from both sides

_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_numerals(text: str) -> str:
    """Converts Eastern Arabic-Indic digits to Western Arabic digits."""
    return text.translate(_EASTERN_TO_WESTERN)


def strip_citation_refs(text: str) -> str:
    """Removes trailing parenthetical numeric citations like '(41 / 251)'."""
    text = re.sub(r'\(\s*\d+\s*/\s*\d+\s*\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def strip_diacritics(text: str) -> str:
    """Consonantal skeleton: diacritics AND tatweel removed.

    Dropping tatweel here too matters: 'الرحمـن' (with tatweel) and
    'الرحمن' (without) would otherwise align as a substitution and the
    reference word would be scored as a deletion — a fabricated error over
    a purely cosmetic stretching glyph.
    """
    return DIACRITICS.sub('', text).replace(TATWEEL, '')


def clean_and_tokenize(text: str) -> list:
    """NFC-normalizes, normalizes numerals, strips citations/tatweel, tokenizes.

    NFC runs here, inside tokenization, rather than requiring every caller
    to remember to normalize before scoring — the old scorer's NFC handling
    was a separate call site that was easy to forget (see defect 1).
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_citation_refs(normalize_numerals(text)).replace(TATWEEL, '')
    return re.sub(r'[^\w\sً-ٰٟ]', '', text).split()


def segment_word(word: str) -> list:
    """Splits a word into [(base_letter, marks), ...] pairs — the fix for
    defect 2 (host-letter tracking).

    Marks on the same letter are sorted before comparison, so two
    canonically equivalent orderings of the same cluster (e.g. shadda
    before vs. after the vowel) compare equal even in cases NFC alone
    doesn't reorder.
    """
    units = []
    for ch in unicodedata.normalize("NFC", word):
        if DIACRITICS.match(ch):
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(base, "".join(sorted(marks))) for base, marks in units]


def scorable_units(units: list, ce: bool) -> list:
    """Diacritizable (base_letter, marks) units, dropping the word-final one
    when ce=False (i'rab / case-ending stripped). Dropping the unit outright
    — not just its marks — matches the standard "exclude last character"
    convention used for DER* / WER* in the literature.
    """
    idx = [i for i, (base, _) in enumerate(units) if LETTER.match(base)]
    if not ce and idx:
        idx = idx[:-1]
    return [units[i] for i in idx]


def is_digit_only(word: str) -> bool:
    """True if the word (after stripping diacritics) is purely numeric —
    such words can never carry a diacritic, so they're excluded from DER."""
    return strip_diacritics(word).isdigit()


def align_words(predictions: list, references: list) -> list:
    """Word-alignment on consonantal skeletons, shared by DER and WER.

    Returns (ref_word_or_None, pred_word_or_None) pairs:
      - (ref, pred)  : matched (equal or substituted) words
      - (ref, None)  : a reference word with nothing aligned to it (deletion)
      - (None, pred) : a hallucinated/inserted word with no reference — the
                        fix for defect 4. The old scorer dropped these
                        entirely; here they are returned so the caller can
                        score them as errors instead of ignoring them.
    """
    pairs = []
    for pred, ref in zip(predictions, references):
        pred_words = [w for w in clean_and_tokenize(pred) if strip_diacritics(w)]
        ref_words = [w for w in clean_and_tokenize(ref) if strip_diacritics(w)]

        alignment = jiwer.process_words(
            " ".join(strip_diacritics(w) for w in ref_words),
            " ".join(strip_diacritics(w) for w in pred_words),
        )

        for chunk in alignment.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    pairs.append((ref_words[i],
                                  pred_words[chunk.hyp_start_idx + (i - chunk.ref_start_idx)]))
            elif chunk.type == "substitute":
                pairs += [(ref_words[i], None)
                          for i in range(chunk.ref_start_idx, chunk.ref_end_idx)]
                pairs += [(None, pred_words[j])
                          for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)]
            elif chunk.type == "delete":
                pairs += [(ref_words[i], None)
                          for i in range(chunk.ref_start_idx, chunk.ref_end_idx)]
            elif chunk.type == "insert":
                pairs += [(None, pred_words[j])
                          for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)]
    return pairs


def _compute_der_from_pairs(pairs: list, ce: bool = True) -> float:
    """DER over pre-computed alignment pairs. Denominator is fixed to the
    diacritizable characters in the REFERENCE only — the fix for defect 3
    (it no longer moves with what the model happened to predict).
    Hallucinated words (ref is None) are added to both terms, so they
    count as errors instead of being free (the fix for defect 4).
    """
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
        if p_word is None:  # deletion or substitution with nothing usable
            wrong += len(r_units)
            continue
        p_units = scorable_units(segment_word(p_word), ce)
        for i, (r_base, r_marks) in enumerate(r_units):
            if i >= len(p_units) or p_units[i] != (r_base, r_marks):
                wrong += 1
    return round(wrong / total * 100, 2) if total else 0.0


def _compute_wer_from_pairs(pairs: list, ce: bool = True) -> float:
    """WER over pre-computed alignment pairs. A hallucinated word or a
    deleted word always counts as wrong."""
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
    """Diacritic Error Rate via edit-distance word alignment, corrected.

    ce=True  -> standard DER (includes case ending).
    ce=False -> DER*, case ending stripped before comparing.
    """
    return _compute_der_from_pairs(align_words(predictions, references), ce=ce)


def compute_wer(predictions: list, references: list, ce: bool = True) -> float:
    """Word-level diacritization error rate, corrected.

    ce=True  -> mismatch anywhere (incl. case ending) marks word wrong.
    ce=False -> WER*, case ending stripped before comparing.
    """
    return _compute_wer_from_pairs(align_words(predictions, references), ce=ce)


def compute_der_wer(predictions: list, references: list) -> dict:
    """Returns all four headline metrics: DER/WER with & without case ending.

    Aligns ONCE and reuses the pairs for all four numbers, rather than
    re-running word alignment four times like separate compute_der/
    compute_wer calls would.
    """
    pairs = align_words(predictions, references)
    return {
        "DER_ce": _compute_der_from_pairs(pairs, ce=True),
        "DER_noce": _compute_der_from_pairs(pairs, ce=False),
        "WER_ce": _compute_wer_from_pairs(pairs, ce=True),
        "WER_noce": _compute_wer_from_pairs(pairs, ce=False),
    }


if __name__ == "__main__":
    # Sanity check, mirroring the notebooks' own verification cell: each
    # case should show the corrected scorer catching what the old one let
    # through silently. Run directly with `python Evaluation_Functions_Corrected.py`.
    print("1. Identical word, different combining-mark order (no NFC needed here):")
    print("   DER (no CE) =", compute_der(
        ["الرَّحْمَنِ"],
        ["الرَّحْمَنِ"], ce=False))
    print("   expected: 0.0 (old scorer gives 50.0 on this pair)\n")

    ref = ["رَسَمَ الطَّالِبُ"]
    print("2. Correct answer plus hallucinated extra words:")
    pred_halluc = ["رَسَمَ الطَّا"
                   "لِبُ وزيادة كل"
                   "مات مخترعة"]
    print("   DER (CE) =", compute_der(pred_halluc, ref, ce=True))
    print("   expected: > 0 (old scorer gives 0.0 — hallucinations were free)\n")

    print("3. Missing dagger alef (U+0670):")
    print("   DER (CE) =", compute_der(
        ["هَذا"], ["هَذٰا"], ce=True))
    print("   expected: > 0 (old scorer gives 0.0 — dagger alef wasn't in its mark class)")
