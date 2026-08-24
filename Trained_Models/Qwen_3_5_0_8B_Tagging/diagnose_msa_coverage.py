"""Does the MSA gap come from vocabulary coverage, or from something more data cannot fix?

The question this answers: are MSA errors concentrated on word skeletons the training corpus
never (or rarely) contained? If yes, more MSA data is the right investment. If MSA words the
model saw thousands of times are STILL wrong at a much higher rate than CA words of the same
frequency, the deficit is not coverage and more data will disappoint.

THE CONTROL MATTERS. "MSA errors fall on rare words" is not evidence by itself -- MSA words are
rarer in this corpus by construction, so that would be true even if frequency were the only thing
going on. The diagnostic is the CA-vs-MSA error rate WITHIN each frequency bucket. Equal rates at
matched frequency => pure coverage problem. MSA persistently worse at matched frequency => a
residual domain gap that data alone will not close.

CPU only. Reads the same pinned corpus revision as prepare_data_tagging_qwen.py.
"""

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring  # noqa: E402
from labels import strip_to_bare  # noqa: E402

TRAIN_DATASET = "Misraj/Sadeed_Tashkeela"
TRAIN_DATASET_REVISION = os.environ.get(
    "TRAIN_DATASET_REVISION", "c10bcbb3b50dc96551f62c472389de666a8c1c4e")
INPUT_COLUMN, OUTPUT_COLUMN = "input", "output"
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")

LET = scoring.ARABIC_LETTER
# Buckets over "how many times did training see this exact bare word form".
BUCKETS = [(0, 0), (1, 9), (10, 99), (100, 999), (1000, 9999), (10000, 10**12)]


def bucket_of(n):
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**12 else f"{lo}+")
    return "?"


def build_train_vocab(limit=0, nproc=8):
    from huggingface_hub import login
    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok)
    from datasets import load_dataset

    ds = load_dataset(TRAIN_DATASET, split="train", revision=TRAIN_DATASET_REVISION,
                      cache_dir=CACHE_DIR)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[INFO] {TRAIN_DATASET} n={len(ds):,} revision={TRAIN_DATASET_REVISION}", flush=True)

    # The bare (undiacritized) surface form is the right key: it is exactly what the tagger sees
    # at inference, so "did training contain this token" is asked in the model's own input space.
    def count_shard(shard):
        c = Counter()
        for t in shard[INPUT_COLUMN]:
            if isinstance(t, str) and t:
                c.update(scoring.clean_and_tokenize(t))
        return c

    total = Counter()
    n = len(ds)
    step = max(1, n // 40)
    for start in range(0, n, step):
        total.update(count_shard(ds[start:min(start + step, n)]))
        pct = min(100, int(100 * (start + step) / n))
        print(f"[INFO] scanned {pct}% -- {len(total):,} distinct forms", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import csv
    rows = list(csv.DictReader(open(args.predictions, encoding="utf-8")))
    print(f"[INFO] {len(rows)} benchmark rows", flush=True)

    freq = build_train_vocab(args.limit)
    print(f"[OK] training vocabulary: {len(freq):,} distinct bare word forms, "
          f"{sum(freq.values()):,} tokens", flush=True)

    # Per aligned word: is it wrong, and how often did training see its bare form?
    stats = {}   # (domain, bucket) -> [n_words, n_wrong, n_diac_chars, n_wrong_chars]
    unseen_examples = {"CA": [], "MSA": []}
    for r in rows:
        dom = r["domain"]
        for p, g in scoring.align_words([r["prediction"]], [r["reference"]]):
            if not isinstance(p, str) or not isinstance(g, str):
                continue
            bare = strip_to_bare(g)
            b = bucket_of(freq.get(bare, 0))
            pu, gu = scoring.segment(p), scoring.segment(g)
            if len(pu) != len(gu):
                continue
            nch = sum(1 for gb, _ in gu if LET.match(gb))
            bad = sum(1 for (pb, pm), (gb, gm) in zip(pu, gu) if LET.match(gb) and pm != gm)
            s = stats.setdefault((dom, b), [0, 0, 0, 0])
            s[0] += 1
            s[1] += 1 if bad else 0
            s[2] += nch
            s[3] += bad
            if bad and freq.get(bare, 0) == 0 and len(unseen_examples[dom]) < 15:
                unseen_examples[dom].append({"bare": bare, "gold": g, "pred": p})

    order = [bucket_of(lo) if lo == hi else bucket_of(lo) for lo, hi in BUCKETS]
    print("\n" + "=" * 78)
    print("MSA COVERAGE DIAGNOSTIC -- error rate by training frequency of the word form")
    print("=" * 78)
    print(f"{'train freq':>12} | {'CA words':>9} {'CA err%':>8} {'CA chr%':>8} | "
          f"{'MSA words':>9} {'MSA err%':>8} {'MSA chr%':>8} | {'MSA/CA':>7}")
    print("-" * 78)
    out_rows = []
    for b in order:
        ca, msa = stats.get(("CA", b), [0, 0, 0, 0]), stats.get(("MSA", b), [0, 0, 0, 0])
        cw = 100 * ca[1] / ca[0] if ca[0] else float("nan")
        mw = 100 * msa[1] / msa[0] if msa[0] else float("nan")
        cc = 100 * ca[3] / ca[2] if ca[2] else float("nan")
        mc = 100 * msa[3] / msa[2] if msa[2] else float("nan")
        ratio = (mc / cc) if (cc and cc == cc and mc == mc) else float("nan")
        print(f"{b:>12} | {ca[0]:9,} {cw:7.2f}% {cc:7.2f}% | "
              f"{msa[0]:9,} {mw:7.2f}% {mc:7.2f}% | {ratio:6.2f}x")
        out_rows.append({"bucket": b, "ca_words": ca[0], "ca_word_err_pct": round(cw, 3),
                         "ca_char_err_pct": round(cc, 3), "msa_words": msa[0],
                         "msa_word_err_pct": round(mw, 3), "msa_char_err_pct": round(mc, 3),
                         "msa_over_ca_char": round(ratio, 3)})

    def agg(dom):
        w = sum(v[0] for (d, _), v in stats.items() if d == dom)
        u = sum(v[0] for (d, b), v in stats.items() if d == dom and b == "0")
        ce = sum(v[3] for (d, _), v in stats.items() if d == dom)
        cn = sum(v[2] for (d, _), v in stats.items() if d == dom)
        ue = sum(v[3] for (d, b), v in stats.items() if d == dom and b == "0")
        return w, u, ce, cn, ue

    print("\n--- how much of each domain's error sits on never-seen word forms ---")
    for dom in ("CA", "MSA"):
        w, u, ce, cn, ue = agg(dom)
        print(f"  {dom}: {u:,}/{w:,} words unseen in training ({100*u/w:.2f}%); "
              f"they carry {ue:,}/{ce:,} of its character errors ({100*ue/max(ce,1):.1f}%)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wri.writeheader()
        wri.writerows(out_rows)
    with open(args.out.replace(".csv", "_examples.json"), "w", encoding="utf-8") as f:
        json.dump(unseen_examples, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
