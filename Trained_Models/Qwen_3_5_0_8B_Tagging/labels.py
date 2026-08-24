"""
Character-level label set for the tagging reframe of Arabic diacritization.

Diacritization-as-generation re-emits the whole consonantal skeleton and can therefore corrupt
it (measured: 109/600, 18.17% of MSA rows in the 0.8B full-FT run — see
Fine-Tune-qwen-9-0.8-Findings/02_Qwen3.5-0.8B_Full-FineTune). Diacritization-as-tagging assigns
one label per INPUT character and never re-emits a letter, so skeleton corruption is structurally
impossible: `decode()` below can only ever attach a mark to the character that was already there.

Do not hand-write the label set. Derive it by scanning the corpus (`--scan`) and freeze the
result to labels.json. `encode`/`decode` must round-trip 100% before anything downstream runs —
that gate is enforced by `--verify`, not assumed.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

# Copied, not imported — this repo's convention (see CLAUDE.md "repo is chronological layers,
# not modules"). Source of truth: Train-Related-to-smaller-model/Models_Functions.py:55-59.
ARABIC_DIACRITICS = re.compile(r'[ً-ٰٟ]')
ARABIC_LETTER = re.compile(r'[ء-غف-يٱ-ۓ]')
TATWEEL = "ـ"

NONE_LABEL = ""  # canonical spelling for "no diacritic" -- always present, always id 0


def segment_chars(text: str):
    """Splits an ENTIRE string (not a single word) into [(base_char, marks), ...] units.

    Unlike Models_Functions.segment() (which operates on one pre-tokenized word), this walks
    the raw string so spaces and punctuation survive in order — required for a tagger that
    labels every character position, not just letters inside words.

    A leading combining mark with no preceding base character is discarded (matches
    Models_Functions.segment()'s documented behavior; verified not to occur in practice — 1,200
    /1,200 round-trip exact on SadeedDiac-25 gold, see --verify).
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(TATWEEL, "")
    units = []
    for ch in text:
        if ARABIC_DIACRITICS.match(ch):
            if units:
                units[-1][1] += ch
        else:
            units.append([ch, ""])
    return [(base, "".join(sorted(marks))) for base, marks in units]


def is_diacritizable(base_char: str) -> bool:
    return bool(ARABIC_LETTER.match(base_char))


def strip_to_bare(diacritized: str) -> str:
    """bare text = base characters only, in order. Must equal the dataset's own `input` column
    (verified 1,200/1,200 exact on SadeedDiac-25 -- see --verify)."""
    return "".join(base for base, _ in segment_chars(diacritized))


class LabelVocab:
    def __init__(self, labels: list):
        assert labels[0] == NONE_LABEL, "id 0 must be the empty/no-diacritic label by convention"
        assert len(set(labels)) == len(labels), "duplicate label in vocab"
        self.labels = list(labels)
        self.label_to_id = {m: i for i, m in enumerate(self.labels)}

    def __len__(self):
        return len(self.labels)

    @classmethod
    def load(cls, path: str) -> "LabelVocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["labels"])

    def save(self, path: str, freq: dict = None):
        payload = {"labels": self.labels}
        if freq is not None:
            payload["frequencies"] = {m: freq.get(m, 0) for m in self.labels}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def encode(self, diacritized: str, strict: bool = True):
        """-> (bare_text, label_ids). label_ids[i] is -100 for non-diacritizable positions
        (space, punctuation, digits) and a vocab id for every diacritizable letter.

        strict=True (training-data prep / --verify): raises on a mark cluster outside the vocab
        — the vocab was built BY scanning this same corpus, so this should never fire; if it
        does, the vocab is stale and must be rebuilt, not silently patched around.

        strict=False (inference on arbitrary text, e.g. the website's decode path): falls back
        to NONE_LABEL (id 0) for an unrecognised cluster rather than raising.
        """
        units = segment_chars(diacritized)
        bare = []
        label_ids = []
        for base, marks in units:
            bare.append(base)
            if not is_diacritizable(base):
                label_ids.append(-100)
                continue
            lid = self.label_to_id.get(marks)
            if lid is None:
                if strict:
                    raise KeyError(
                        f"mark cluster {marks!r} on base {base!r} not in label vocab "
                        f"({len(self.labels)} classes) — rebuild labels.json with --scan"
                    )
                lid = self.label_to_id[NONE_LABEL]
            label_ids.append(lid)
        return "".join(bare), label_ids

    def decode(self, bare: str, label_ids: list) -> str:
        assert len(bare) == len(label_ids), (len(bare), len(label_ids))
        out = []
        for ch, lid in zip(bare, label_ids):
            out.append(ch)
            # A combining mark MUST have a letter to sit on. The head emits a label for every
            # character including spaces, punctuation and digits, and nothing upstream forbids
            # it predicting a mark there -- measured 60,520 such orphan marks over the 1,200
            # SadeedDiac-25 rows (99.9% of rows), against 0 in the gold. They land next to word
            # boundaries, which is exactly where the case ending is scored, so they cost 1.8
            # DER_ce and 7.6 WER_ce while leaving DER_noce untouched (it drops the word-final
            # character anyway). This is a hard constraint knowable a priori, so enforce it at
            # decode rather than hope the model learns it.
            if lid is not None and lid >= 0 and is_diacritizable(ch):
                out.append(self.labels[lid])
        return "".join(out)


def build_vocab_from_texts(texts, min_count: int = 1) -> (LabelVocab, Counter):
    """Scan diacritized texts, collect every OBSERVED mark cluster on a diacritizable base
    character. Sorted by descending frequency so id 0 is always NONE_LABEL and low ids are the
    common classes (fatha/kasra/sukun/damma dominate — see the frequency table --scan prints)."""
    freq = Counter()
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        for base, marks in segment_chars(t):
            if is_diacritizable(base):
                freq[marks] += 1
    freq = Counter({m: n for m, n in freq.items() if n >= min_count})
    if NONE_LABEL not in freq:
        freq[NONE_LABEL] = 0
    ordered = [NONE_LABEL] + [m for m, _ in freq.most_common() if m != NONE_LABEL]
    return LabelVocab(ordered), freq


def verify_roundtrip(vocab: LabelVocab, texts, input_texts=None) -> dict:
    """The single most important check in this package (see the plan). Every row must
    round-trip exactly, and — if input_texts is given — the derived bare text must match the
    dataset's own `input` column exactly, char for char."""
    n = 0
    rt_fail = []
    input_mismatch = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t:
            continue
        n += 1
        bare, label_ids = vocab.encode(t, strict=True)
        recon = vocab.decode(bare, label_ids)
        if unicodedata.normalize("NFC", recon) != unicodedata.normalize("NFC", t):
            rt_fail.append(i)
        if input_texts is not None:
            inp = input_texts[i]
            if isinstance(inp, str) and bare != inp:
                input_mismatch.append(i)
    return {
        "n": n,
        "roundtrip_failures": len(rt_fail),
        "roundtrip_failure_idx_sample": rt_fail[:10],
        "input_mismatches": len(input_mismatch) if input_texts is not None else None,
        "input_mismatch_idx_sample": input_mismatch[:10],
    }


def _main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true", help="derive labels.json from a HF dataset")
    ap.add_argument("--verify", action="store_true", help="round-trip gate against labels.json")
    ap.add_argument("--dataset", default="Misraj/Sadeed_Tashkeela")
    ap.add_argument("--revision", default=os.environ.get("TRAIN_DATASET_REVISION"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--column", default="output", help="diacritized text column")
    ap.add_argument("--input-column", default="input", help="bare text column, for --verify")
    ap.add_argument("--labels-path", default=os.environ.get("LABELS_PATH", "labels.json"))
    ap.add_argument("--limit", type=int, default=None, help="row cap for a fast dev run")
    args = ap.parse_args()

    if not args.scan and not args.verify:
        ap.error("pass --scan and/or --verify")

    # Misraj/Sadeed_Tashkeela is gated:manual -- load_dataset does NOT pick up HF_TOKEN from the
    # environment on its own (measured: 401 DatasetNotFoundError with only the env var set).
    # Same explicit login as Train-Qwen-0.8B-Full-FT/train_full_ft_qwen35_08b.py:745 hf_login().
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split, revision=args.revision,
                       cache_dir=os.environ.get("HF_CACHE_DIR"))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[INFO] loaded {args.dataset}[{args.split}] revision={args.revision} n={len(ds):,}")

    texts = ds[args.column]
    if args.scan:
        vocab, freq = build_vocab_from_texts(texts)
        vocab.save(args.labels_path, freq)
        total = sum(freq.values())
        print(f"[OK] scanned {len(texts):,} rows -> {len(vocab)} label classes -> "
              f"{args.labels_path}")
        for m in vocab.labels:
            names = "+".join(unicodedata.name(ch, "?").replace("ARABIC ", "") for ch in m) or "NONE"
            n = freq.get(m, 0)
            print(f"  {n:>10,}  {100 * n / max(total, 1):6.2f}%  {m!r:16s} {names}")

    if args.verify:
        vocab = LabelVocab.load(args.labels_path)
        input_texts = ds[args.input_column] if args.input_column in ds.column_names else None
        result = verify_roundtrip(vocab, texts, input_texts)
        print(f"[VERIFY] {json.dumps(result, ensure_ascii=False)}")
        ok = result["roundtrip_failures"] == 0 and (
            input_texts is None or result["input_mismatches"] == 0
        )
        if not ok:
            print("[FAIL] round-trip gate did NOT pass — fix labels.json/segment_chars before "
                  "proceeding to Phase 1b. Nothing downstream should run yet.")
            sys.exit(1)
        print(f"[PASS] 100% round-trip on {result['n']:,} rows "
              f"({'input column verified' if input_texts is not None else 'input column not checked'})")


if __name__ == "__main__":
    _main()
