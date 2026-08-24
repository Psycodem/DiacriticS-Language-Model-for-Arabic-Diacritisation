"""
Copied verbatim from Train-Related-to-smaller-model/Models_Functions.py (the CORRECTED scorer —
see CLAUDE.md "Scorers: three generations"). Copy, not import, per this repo's convention that
no package imports another. Keep the two files identical; if you fix a bug in one, fix it here
too.

DER / WER metrics for Arabic diacritization.

REWRITTEN. The previous implementation had four defects that made its numbers
non-comparable to published DER and, in places, insensitive to real errors. All four are
fixed here, and every number produced by the old version must be regenerated -- they are
not on the same scale.

  1. POSITION WAS NOT TRACKED. The old code compared `findall(diacritics)` sequences with
     the host letters discarded, so a mark that moved to a different base letter was
     invisible whenever the resulting mark sequence was unchanged:
         pred='رَسم'  ref='رسَم'  ->  old DER_ce = 0.00
     Fixed by segmenting each word into (base character, marks) units and comparing
     per base character.

  2. THE DENOMINATOR COUNTED MARKS, NOT CHARACTERS, and depended on the prediction
     (`n = max(len(r_diacs), len(p_diacs), 1)`). Two models were scored against different
     totals, and a word carrying shadda+fatha counted 2 where the standard counts 1.
     Fixed: the denominator is the number of diacritizable base characters in the
     REFERENCE, which is fixed per test set and matches the Fadel/Sadeed definition that
     the literature numbers in README.md use.

  3. INSERTIONS WERE FREE. `_align_words` skipped `insert` chunks, so
     `correct answer + 50 junk words` scored DER 0.00 / WER 0.00. A base model with no
     EOS training generates to the token cap, so this flattered exactly the runs it
     should have punished. Fixed: inserted words count against both metrics.

  4. THE DIACRITIC CLASS WAS TOO NARROW. `[ً-ْ]` omits U+0670 (dagger alef,
     ubiquitous in Classical Arabic: هَٰذَا, الرَّحْمَٰن) and the rarer Quranic marks, so
     هَذَا and هَٰذَا scored identically. Widened below.

Conventions, stated explicitly because they must match the README formula:
  - DER denominator: diacritizable base characters (Arabic letters) in the reference.
    A reference character carrying NO diacritic is still scored -- predicting a mark
    where the gold has none is an error.
  - "without case ending" (noce) EXCLUDES the word-final diacritizable character
    entirely from both numerator and denominator (Fadel's "excluding last character"),
    rather than merely blanking its marks.
  - WER: a reference word is wrong if its diacritized form differs from the prediction's.
    Deletions and insertions both count as wrong, and insertions also enlarge the
    denominator, so the rate stays within [0, 100].
  - Both sides are NFC-normalized before segmentation. ~95% of SadeedDiac-25 gold rows
    store combining marks in non-canonical order; without this a correct prediction
    scores ~12.85 DER_ce against raw gold.
"""

import re
import unicodedata

import jiwer

# Core tashkeel (U+064B-U+0652) + rarer marks through U+065F + U+0670 dagger alef.
# U+0653-U+0655 (maddah, hamza above/below) are included for completeness; NFC usually
# composes them into آ/أ/إ before we ever see them.
ARABIC_DIACRITICS = re.compile(r'[ً-ٰٟ]')

# Base characters that can carry a diacritic. Excludes U+0640 tatweel (a stretching
# glyph, not a letter) and the diacritic ranges themselves.
ARABIC_LETTER = re.compile(r'[ء-غف-يٱ-ۓ]')

TATWEEL = "ـ"

_EASTERN_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_numerals(text: str) -> str:
    """Converts Eastern Arabic-Indic digits to Western Arabic digits."""
    return text.translate(_EASTERN_TO_WESTERN)


def strip_citation_refs(text: str) -> str:
    """Removes trailing parenthetical numeric citations like '(41 / 251)'."""
    text = re.sub(r'\(\s*\d+\s*/\s*\d+\s*\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def strip_diacritics(text: str) -> str:
    """Consonantal skeleton: drops every diacritic and tatweel.

    Note this now also removes U+0670. The old version left it in, and because U+0670 is
    not a `\\w` character the downstream punctuation substitution turned it into a SPACE,
    splitting 'هَٰذَا' into two words ('ه ذا') before shingling or alignment.
    """
    return ARABIC_DIACRITICS.sub('', text).replace(TATWEEL, '')


def clean_and_tokenize(text: str) -> list:
    """Normalizes numerals, strips citation refs and noise, tokenizes on whitespace.

    Diacritics are preserved (they are what we score); tatweel is dropped; anything that
    is neither word-character, whitespace, nor diacritic is REMOVED rather than replaced
    with a space, so a stray combining mark can no longer split a word in two and desync
    the alignment for the rest of the sentence.
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_numerals(text)
    text = strip_citation_refs(text)
    text = text.replace(TATWEEL, '')
    cleaned = re.sub(r'[^\w\sً-ٰٟ]', '', text)
    return cleaned.split()


def segment(word: str) -> list:
    """Splits a word into [(base_char, marks), ...] units.

    Marks are sorted so that two canonically-equivalent orderings of the same cluster
    (shadda+damma vs damma+shadda) compare equal even if NFC did not reorder them.
    Leading marks with no base character are discarded.
    """
    units = []
    for ch in unicodedata.normalize("NFC", word):
        if ARABIC_DIACRITICS.match(ch):
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(base, "".join(sorted(marks))) for base, marks in units]


def _scorable(units: list, ce: bool) -> list:
    """Diacritizable units of a word, dropping the final one when ce=False."""
    idx = [i for i, (base, _) in enumerate(units) if ARABIC_LETTER.match(base)]
    if not ce and idx:
        idx = idx[:-1]
    return [units[i] for i in idx]


def strip_case_ending(word: str) -> str:
    """Removes the diacritics on the word-final base letter (i'rab / case ending).

    Retained for backwards compatibility and for display purposes. The metrics below no
    longer use it -- they drop the final unit outright, which is the Fadel convention.
    """
    units = segment(word)
    if not units:
        return word
    last = len(units) - 1
    return "".join(b + ("" if i == last else m) for i, (b, m) in enumerate(units))


def _is_digit_only(word: str) -> bool:
    """True if the word is purely numeric -- it can never carry a diacritic."""
    return strip_diacritics(word).isdigit()


def align_words(predictions: list, references: list) -> list:
    """Aligns predicted to reference words on their diacritic-stripped skeletons.

    Returns a list of (ref_word_or_None, pred_word_or_None) pairs:
      (ref, pred) matched      (ref, None) deletion/substitution      (None, pred) INSERTION

    Insertions are now emitted. The old `_align_words` dropped them, which is what let a
    model append arbitrary text at no cost. Empty skeletons are filtered from BOTH lists
    in lockstep before joining, so a stray whitespace-separated bare diacritic can no
    longer make jiwer's token count disagree with ours and shift every later pairing.
    """
    pairs = []
    for pred, ref in zip(predictions, references):
        pred_words = [w for w in clean_and_tokenize(pred) if strip_diacritics(w)]
        ref_words = [w for w in clean_and_tokenize(ref) if strip_diacritics(w)]
        pred_skel = [strip_diacritics(w) for w in pred_words]
        ref_skel = [strip_diacritics(w) for w in ref_words]

        alignment = jiwer.process_words(" ".join(ref_skel), " ".join(pred_skel))

        for chunk in alignment.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    pairs.append(
                        (ref_words[i], pred_words[chunk.hyp_start_idx + (i - chunk.ref_start_idx)])
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


# Backwards-compatible generator name used by older scripts in "Models to test/".
def _align_words(predictions: list, references: list):
    for ref, pred in align_words(predictions, references):
        if ref is not None:
            yield ref, pred


def compute_der(predictions: list, references: list, ce: bool = True, pairs=None) -> float:
    """Diacritic Error Rate, per diacritizable reference character.

    Pure-digit words are excluded (they cannot carry a diacritic). Deleted reference
    words count every one of their characters as wrong. Inserted (hallucinated) words add
    their diacritizable characters to both numerator and denominator.
    """
    if pairs is None:
        pairs = align_words(predictions, references)

    total, wrong = 0, 0
    for r_word, p_word in pairs:
        if r_word is None:                       # insertion
            if p_word is None or _is_digit_only(p_word):
                continue
            n = len(_scorable(segment(p_word), ce))
            total += n
            wrong += n
            continue

        if _is_digit_only(r_word):
            continue

        r_units = _scorable(segment(r_word), ce)
        total += len(r_units)

        if p_word is None:                       # deletion / substitution
            wrong += len(r_units)
            continue

        p_units = _scorable(segment(p_word), ce)
        for i, (r_base, r_marks) in enumerate(r_units):
            if i >= len(p_units):
                wrong += 1
            elif p_units[i][0] != r_base or p_units[i][1] != r_marks:
                wrong += 1

    return round((wrong / total) * 100, 2) if total > 0 else float("nan")


def compute_wer(predictions: list, references: list, ce: bool = True, pairs=None) -> float:
    """Word-level diacritization error rate.

    A word is wrong if its scorable units differ from the reference's. Deletions and
    insertions both count as wrong words; insertions also enlarge the denominator.
    """
    if pairs is None:
        pairs = align_words(predictions, references)

    total, wrong = 0, 0
    for r_word, p_word in pairs:
        total += 1
        if r_word is None or p_word is None:     # insertion or deletion
            wrong += 1
            continue
        if _scorable(segment(r_word), ce) != _scorable(segment(p_word), ce):
            wrong += 1

    return round((wrong / total) * 100, 2) if total > 0 else float("nan")


def compute_der_wer(predictions: list, references: list) -> dict:
    """All four headline metrics. Aligns once and reuses it (the old version realigned
    four times per call, i.e. twelve full passes per evaluation)."""
    pairs = align_words(predictions, references)
    return {
        "DER_ce": compute_der(None, None, ce=True, pairs=pairs),
        "DER_noce": compute_der(None, None, ce=False, pairs=pairs),
        "WER_ce": compute_wer(None, None, ce=True, pairs=pairs),
        "WER_noce": compute_wer(None, None, ce=False, pairs=pairs),
    }
