# -*- coding: utf-8 -*-
"""Differential comparison: Evaluation_Functions_Corrected.py vs the Sadeed evaluator.

This is the script that produced Tables III and IV of Progress Report 5. It exists
so those numbers are reproducible rather than asserted.

WHAT IT COMPARES
----------------
`Evaluation_Functions_Corrected.py` (this repo) against the reference
`ArabicDiacritizationEvaluator` distributed with the SadeedDiac-25 benchmark as a
notebook. The reference implementation is NOT vendored here -- point --sadeed at
your own copy of `Eval_code.ipynb` (or at a .py containing the class).

THE TWO SUBSTANTIVE DIFFERENCES IT DEMONSTRATES
-----------------------------------------------
1. SENTENCE-LEVEL DISCARDING. The reference evaluator aligns prediction to
   reference positionally. It raises RuntimeError when the two token lists differ
   in length, and again when the consonantal skeleton of any aligned pair differs.
   `caculate_errors_on_sentences` catches that, warns, and `continue`s -- so the
   paragraph leaves the benchmark entirely rather than being penalised. Any
   paragraph where the model inserted, dropped or reordered a word contributes to
   neither numerator nor denominator. A model that hallucinates is scored only on
   the paragraphs where it did not.

   The corrected scorer aligns on consonantal skeletons and keeps every reference
   word: insertions become (None, pred) pairs and are charged in full, deletions
   become (ref, None), and the DER denominator is fixed by the reference so it
   cannot move with the prediction.

2. COMBINING-MARK ORDER. Arabic gemination is shadda + a short vowel on one
   consonant, and the two marks may appear in either order in the byte stream.
   Unicode treats the orderings as equivalent -- both normalise to the same NFC
   form. The reference `extract_harakat` walks left to right keeping only the
   first mark per letter unless that mark was the shadda, so shadda-then-fatha is
   read as gemination-with-vowel while fatha-then-shadda is read as a bare fatha
   with the shadda silently dropped. A model emitting the second ordering is
   charged an error on every geminated letter it got right. `segment_word` in the
   corrected scorer sorts the marks on each letter, so canonically equivalent
   spellings compare equal by construction.

Crucially, on inputs where the word count is preserved the two agree to the
second decimal (cases A and B below). The corrected scorer is the same metric
extended to cover length mismatch and mark order -- not a different metric.

USAGE
-----
    # the controlled example + the mark-ordering probe (Table III)
    python Evaluation_Scorer_Comparison.py --sadeed /path/to/Eval_code.ipynb

    # additionally score a real predictions CSV under both scorers (Table IV)
    python Evaluation_Scorer_Comparison.py --sadeed /path/to/Eval_code.ipynb \
        --predictions Trained_Models/.../benchmark_sadeeddiac25_predictions.csv

The CSV needs a reference column and a prediction column; the names are
auto-detected (reference/gold/target/ground_truth/output/label and *pred*), or
pass --ref-col / --pred-col explicitly.

Requires: pyarabic, prettytable, tqdm, pandas, jiwer (the first three only
because the reference evaluator imports them).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import unicodedata
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
CORRECTED = os.path.join(HERE, "Evaluation_Functions_Corrected.py")

# A four-word sentence with one geminated word, so a single case-ending change is
# easy to reason about and the word count is trivially checkable by eye.
REFERENCE = "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ"
CASES = [
    ("A. nothing (exact match)",          "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ"),
    ("B. one wrong case ending",          "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةُ"),
    ("C. one word dropped",               "ذَهَبَ الطَّالِبُ الْمَدْرَسَةِ"),
    ("D. two words hallucinated",         "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ فِي الصَّبَاحِ"),
    ("E. explanatory commentary appended", "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ هَذَا هُوَ النَّصُّ"),
]

SHADDA, FATHA, TAH = "ّ", "َ", "ط"


# --------------------------------------------------------------------- loading
def load_corrected(path: str = CORRECTED):
    if not os.path.exists(path):
        sys.exit(f"FATAL: corrected scorer not found at {path}")
    spec = importlib.util.spec_from_file_location("evalfns", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evalfns"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_sadeed(path: str):
    """Load ArabicDiacritizationEvaluator from a .ipynb or a .py.

    The class is executed rather than imported because it ships inside a notebook
    cell. Only the cell that defines the class is run -- the notebook's other
    cells install packages and run a demo, neither of which we want.
    """
    if not os.path.exists(path):
        sys.exit(f"FATAL: Sadeed evaluator not found at {path}\n"
                 f"       Pass --sadeed /path/to/Eval_code.ipynb")

    if path.lower().endswith(".ipynb"):
        nb = json.load(io.open(path, encoding="utf-8"))
        sources = ["".join(c.get("source", "")) for c in nb["cells"]
                   if c.get("cell_type") == "code"]
        src = next((s for s in sources if "class ArabicDiacritizationEvaluator" in s), None)
        if src is None:
            sys.exit(f"FATAL: no cell in {path} defines ArabicDiacritizationEvaluator")
    else:
        src = io.open(path, encoding="utf-8").read()

    ns: dict = {}
    exec(compile(src, "sadeed_evaluator", "exec"), ns)
    if "ArabicDiacritizationEvaluator" not in ns:
        sys.exit("FATAL: ArabicDiacritizationEvaluator missing after exec")
    return ns["ArabicDiacritizationEvaluator"]


@contextlib.contextmanager
def quiet():
    """The reference evaluator warns on every skipped paragraph and prints a tqdm
    bar per call; both would bury the table."""
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stderr(buf):
        warnings.simplefilter("ignore")
        yield


def sadeed_score(S, preds, refs):
    """Returns (DER_ce, DER_noce, WER_ce, WER_noce) or None if everything was discarded.

    Total_* include the word-final letter (with case ending); Morph_* exclude it
    (without case ending) -- the same distinction as the corrected scorer's
    ce / noce.
    """
    with quiet():
        try:
            twer, mwer, tder, mder, _nvw = S.caculate_errors_on_sentences(preds, refs)
        except ZeroDivisionError:
            return None            # every paragraph was discarded
    return tder, mder, twer, mwer


def sadeed_discarded(S, preds, refs) -> int:
    """How many paragraphs the reference evaluator refuses to score."""
    n = 0
    for p, r in zip(preds, refs):
        with quiet():
            try:
                S.caculate_error_on_single_sentence(p.strip(), r.strip())
            except RuntimeError:
                n += 1
    return n


# ---------------------------------------------------------------------- checks
def controlled_example(S, ev) -> None:
    print("=" * 96)
    print("TABLE III  scorer behaviour on a controlled four-word example")
    print("=" * 96)
    print(f"reference: {REFERENCE}\n")
    hdr = f"{'prediction differs from reference by':38} {'Sadeed DER':>11} {'Sadeed WER':>11} {'corr. DER':>11} {'corr. WER':>11}"
    print(hdr)
    print("-" * len(hdr))
    for label, pred in CASES:
        s = sadeed_score(S, [pred], [REFERENCE])
        if s is None:
            sad = f"{'discarded':>11} {'discarded':>11}"
        else:
            sad = f"{s[0]:11.2f} {s[2]:11.2f}"
        m = ev.compute_der_wer([pred], [REFERENCE])
        print(f"{label:38} {sad} {m['DER_ce']:11.2f} {m['WER_ce']:11.2f}")
    print("\nRows A and B preserve the word count and agree to the second decimal.")
    print("Rows C-E are discarded by the reference evaluator: they leave the benchmark")
    print("rather than being penalised.\n")


def mark_order_probe(S, ev) -> None:
    print("=" * 96)
    print("combining-mark order: two NFC-identical spellings of one geminated letter")
    print("=" * 96)
    a = TAH + SHADDA + FATHA        # shadda then fatha
    b = TAH + FATHA + SHADDA        # fatha then shadda
    print(f"  shadda+fatha : {[unicodedata.name(c) for c in a]}")
    print(f"  fatha+shadda : {[unicodedata.name(c) for c in b]}")
    same_nfc = unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)
    print(f"  identical after NFC normalisation: {same_nfc}\n")

    sa, sb = S.extract_harakat(a)[0], S.extract_harakat(b)[0]
    ca, cb = ev.segment_word(a), ev.segment_word(b)
    print(f"  Sadeed    extract_harakat: {sa!r}  vs  {sb!r}   -> {'SAME' if sa == sb else 'DIFFERENT'}")
    print(f"  corrected segment_word   : {ca!r}  vs  {cb!r}   -> {'SAME' if ca == cb else 'DIFFERENT'}")
    if same_nfc and sa != sb:
        print("\n  => The reference evaluator drops the shadda in the second ordering, so a")
        print("     model emitting it is charged an error on every geminated letter it got")
        print("     right. The corrected scorer sorts marks per letter and is order-free.\n")


def _pick(cols, wanted, contains=None):
    for c in cols:
        if c.lower().lstrip("﻿") in wanted:
            return c
    if contains:
        for c in cols:
            if contains in c.lower():
                return c
    return None


def real_predictions(S, ev, path, ref_col=None, pred_col=None) -> None:
    rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
    if not rows:
        sys.exit(f"FATAL: {path} has no rows")
    cols = list(rows[0])
    ref_col = ref_col or _pick(cols, {"reference", "gold", "target", "ground_truth",
                                      "output", "label"})
    pred_col = pred_col or _pick(cols, {"prediction", "pred"}, contains="pred")
    if not ref_col or not pred_col:
        sys.exit(f"FATAL: could not identify columns in {cols}. "
                 f"Pass --ref-col and --pred-col.")

    refs = [r[ref_col] for r in rows]
    preds = [r[pred_col] for r in rows]
    n = len(rows)
    print("=" * 96)
    print(f"TABLE IV  the same {n:,} predictions under both scorers")
    print("=" * 96)
    print(f"file: {os.path.basename(path)}   reference='{ref_col}'  prediction='{pred_col}'\n")

    dropped = sadeed_discarded(S, preds, refs)
    kept = n - dropped
    print(f"the reference evaluator discards {dropped:,} of {n:,} paragraphs "
          f"({dropped / n * 100:.1f}%) and scores {kept:,}\n")

    # the subset BOTH scorers can handle, so rule differences are isolated from
    # sample differences
    keep_p, keep_r = [], []
    for p, r in zip(preds, refs):
        with quiet():
            try:
                S.caculate_error_on_single_sentence(p.strip(), r.strip())
            except RuntimeError:
                continue
        keep_p.append(p)
        keep_r.append(r)

    s_all = sadeed_score(S, preds, refs)
    m_sub = ev.compute_der_wer(keep_p, keep_r) if keep_p else None
    m_all = ev.compute_der_wer(preds, refs)

    hdr = f"{'scorer / coverage':36} {'scored':>12} {'DER_ce':>8} {'DER_noce':>9} {'WER_ce':>8} {'WER_noce':>9}"
    print(hdr)
    print("-" * len(hdr))
    if s_all:
        print(f"{'Sadeed evaluator':36} {f'{kept:,} of {n:,}':>12} "
              f"{s_all[0]:8.2f} {s_all[1]:9.2f} {s_all[2]:8.2f} {s_all[3]:9.2f}")
    if m_sub:
        print(f"{'corrected scorer, same subset':36} {f'{kept:,}':>12} "
              f"{m_sub['DER_ce']:8.2f} {m_sub['DER_noce']:9.2f} "
              f"{m_sub['WER_ce']:8.2f} {m_sub['WER_noce']:9.2f}")
    print(f"{'corrected scorer, full benchmark':36} {f'{n:,}':>12} "
          f"{m_all['DER_ce']:8.2f} {m_all['DER_noce']:9.2f} "
          f"{m_all['WER_ce']:8.2f} {m_all['WER_noce']:9.2f}")
    print("\nRow 1 vs row 2 isolates the scoring rules (identical coverage).")
    print("Row 2 vs row 3 isolates the cost of the discarded paragraphs, which are")
    print("the hardest in the set.\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Differential comparison of the corrected scorer against the "
                    "Sadeed reference evaluator.")
    ap.add_argument("--sadeed", required=True,
                    help="path to Eval_code.ipynb (or a .py holding "
                         "ArabicDiacritizationEvaluator)")
    ap.add_argument("--corrected", default=CORRECTED,
                    help="path to Evaluation_Functions_Corrected.py")
    ap.add_argument("--predictions",
                    help="optional CSV of real predictions to score under both")
    ap.add_argument("--ref-col", help="reference column name (else auto-detected)")
    ap.add_argument("--pred-col", help="prediction column name (else auto-detected)")
    args = ap.parse_args()

    ev = load_corrected(args.corrected)
    S = load_sadeed(args.sadeed)
    print(f"corrected scorer : {args.corrected}")
    print(f"Sadeed evaluator : {args.sadeed}\n")

    controlled_example(S, ev)
    mark_order_probe(S, ev)
    if args.predictions:
        real_predictions(S, ev, args.predictions, args.ref_col, args.pred_col)
    else:
        print("(pass --predictions <csv> to reproduce Table IV on real model output)")


if __name__ == "__main__":
    main()
