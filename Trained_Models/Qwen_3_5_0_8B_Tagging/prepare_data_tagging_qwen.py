"""
Phase 1b (Qwen arm): build the double-pass, char-labeled, token-aligned dataset.

Each row becomes  bare_text <sep> bare_text  and only the SECOND copy is labeled -- see
model_tagging_qwen.py's docstring for why. `<sep>` reuses the tokenizer's own eos_token
(`<|im_end|>`, a real boundary token the checkpoint already has embeddings and pretrained
semantics for), not a freshly-added special token that would need to learn its role from
scratch in a short run.

Usage:
    python prepare_data_tagging_qwen.py --labels-path labels.json --out $SCRATCH/tagging_data_qwen
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import LabelVocab  # noqa: E402
from model_tagging_qwen import char_id  # noqa: E402

TRAIN_DATASET = "Misraj/Sadeed_Tashkeela"
INPUT_COLUMN = "input"
OUTPUT_COLUMN = "output"

TRAIN_DATASET_REVISION = os.environ.get(
    "TRAIN_DATASET_REVISION", "c10bcbb3b50dc96551f62c472389de666a8c1c4e"
)

# The warm-start source: the ALREADY fine-tuned 2.58-DER checkpoint, not base Qwen -- see
# model_tagging_qwen.py's docstring. Only its tokenizer is needed here.
MODEL_ID = os.environ.get(
    "MODEL_ID", "/scratch/runs/qwen3.5-0.8b-fullft/final_model"
)
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")
RANDOM_SEED = 42
DEV_SET_SIZE = 2000

# Measured on SadeedDiac-25's 1,200 paragraphs: bare-text Qwen-tokenizer length p50=79/p90=127/
# p99=218/max=246. Double-pass (2x + 1 sep) => p50=159/p90=255/p99=437/max=493. 768 leaves ample
# headroom with no truncation for the overwhelming majority of Tashkeela rows; drop (not
# truncate) whatever still clears it -- a truncated stream would desync char_to_token for copy 2.
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "768"))


def _nproc():
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _drop_unusable_rows(ds, name, columns):
    """Same one-null-row trap as every other package here -- see labels.py/prepare_data_tagging.py."""
    def ok(ex):
        return all(isinstance(ex[c], str) and ex[c].strip() for c in columns)

    before = len(ds)
    ds = ds.filter(ok, num_proc=_nproc(), desc=f"dropping unusable {name} rows")
    dropped = before - len(ds)
    if dropped:
        print(f"[WARN] {name}: dropped {dropped:,}/{before:,} rows with a null/blank {columns} "
              f"column")
    else:
        print(f"[OK] {name}: no null/blank rows in {columns}")
    return ds


def build_double_pass_row(vocab: LabelVocab, tokenizer, sep_id: int, ex):
    diac = ex[OUTPUT_COLUMN]
    bare, label_ids = vocab.encode(diac, strict=False)
    if not bare:
        return None

    copy1 = tokenizer(bare, add_special_tokens=False)["input_ids"]
    copy2_enc = tokenizer(bare, add_special_tokens=False, return_offsets_mapping=True)
    copy2 = copy2_enc["input_ids"]

    input_ids = copy1 + [sep_id] + copy2
    if len(input_ids) > MAX_SEQ_LENGTH:
        return None  # dropped, not truncated

    base = len(copy1) + 1  # where copy 2 starts in the full sequence
    char_to_token = [-1] * len(bare)
    char_intra_pos = [0] * len(bare)
    for t_idx, (s, e) in enumerate(copy2_enc["offset_mapping"]):
        for i in range(s, e):
            char_to_token[i] = base + t_idx
            char_intra_pos[i] = i - s
    prev = base
    for i in range(len(char_to_token)):
        if char_to_token[i] == -1:
            char_to_token[i] = prev
        else:
            prev = char_to_token[i]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "char_to_token": char_to_token,
        "char_ids": [char_id(c) for c in bare],
        "char_intra_pos": char_intra_pos,
        "char_labels": label_ids,
        "length": len(input_ids),
        "n_chars": len(bare),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-path", default="labels.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify-input-column", action="store_true")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    vocab = LabelVocab.load(args.labels_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    sep_id = tokenizer.eos_token_id
    print(f"[INFO] {MODEL_ID}: {len(vocab)} label classes, tokenizer vocab={tokenizer.vocab_size}, "
          f"sep_id={sep_id} ({tokenizer.eos_token!r})")

    ds = load_dataset(TRAIN_DATASET, split="train", revision=TRAIN_DATASET_REVISION,
                       cache_dir=CACHE_DIR)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[INFO] {TRAIN_DATASET}[train] revision={TRAIN_DATASET_REVISION} n={len(ds):,}")

    ds = _drop_unusable_rows(ds, "train", (INPUT_COLUMN, OUTPUT_COLUMN))

    if args.verify_input_column:
        mism = 0
        for ex in ds.select(range(min(5000, len(ds)))):
            bare, _ = vocab.encode(ex[OUTPUT_COLUMN], strict=False)
            if bare != ex[INPUT_COLUMN]:
                mism += 1
        print(f"[INFO] derived-bare vs input column mismatch on 5,000-row sample: {mism}")

    _EMPTY_ROW = {"input_ids": None, "attention_mask": None, "char_to_token": None,
                  "char_ids": None, "char_intra_pos": None, "char_labels": None,
                  "length": None, "n_chars": None}

    def _map_fn(ex):
        row = build_double_pass_row(vocab, tokenizer, sep_id, ex)
        return _EMPTY_ROW if row is None else row

    tok_ds = ds.map(_map_fn, remove_columns=ds.column_names, num_proc=_nproc(),
                     desc="double-pass tokenizing + char-aligning")
    before = len(tok_ds)
    tok_ds = tok_ds.filter(lambda ex: ex["input_ids"] is not None, num_proc=_nproc())
    print(f"[INFO] kept {len(tok_ds):,}/{before:,} rows "
          f"(dropped: empty bare text or > {MAX_SEQ_LENGTH} double-pass tokens)")

    has_target = lambda ex: any(l != -100 for l in ex["char_labels"])  # noqa: E731
    before = len(tok_ds)
    tok_ds = tok_ds.filter(has_target, num_proc=_nproc())
    if before != len(tok_ds):
        print(f"[INFO] dropped {before - len(tok_ds):,} rows with no diacritizable character")

    n_total = len(tok_ds)
    assert n_total > DEV_SET_SIZE, f"corpus too small ({n_total}) to hold out a dev set"
    dev_ds = tok_ds.select(range(n_total - DEV_SET_SIZE, n_total))
    train_ds = tok_ds.select(range(n_total - DEV_SET_SIZE))

    lengths = sorted(tok_ds["length"])
    q = lambda p: lengths[min(int(p * len(lengths)), len(lengths) - 1)]
    print(f"[INFO] double-pass token-length p50={q(.5)} p90={q(.9)} p99={q(.99)} max={lengths[-1]}")

    os.makedirs(args.out, exist_ok=True)
    train_ds.save_to_disk(os.path.join(args.out, "train"))
    dev_ds.save_to_disk(os.path.join(args.out, "dev"))
    print(f"[OK] saved train={len(train_ds):,} dev={len(dev_ds):,} rows to {args.out}")


if __name__ == "__main__":
    main()
