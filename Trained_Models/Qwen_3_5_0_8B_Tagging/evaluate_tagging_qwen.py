"""
Phase 4 (Qwen arm): score the double-pass Qwen char-tagger on Misraj/SadeedDiac-25.

Forward pass, not generation. Scored with the SAME corrected metric and CA/MSA convention as the
2.58 generative baseline AND the ../Train-Tagging/ AraBERT control, so all three numbers are
directly comparable.

Usage:
    python evaluate_tagging_qwen.py --model-dir $SCRATCH/runs/qwen-tagging/final_model \
        --out results/metrics_tagging_qwen.csv
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_tagging_qwen import QwenCharTagger, char_id  # noqa: E402
from scoring import compute_der_wer, strip_diacritics, clean_and_tokenize  # noqa: E402

BENCHMARK_DATASET = "Misraj/SadeedDiac-25"
BENCHMARK_REVISION = os.environ.get(
    "BENCHMARK_DATASET_REVISION", "aa311213e44e4cab6cc3f2848daacd753adc1ce1"
)
BENCHMARK_TEXT_COLUMN = "output"
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "768"))


def is_ca(filename: str) -> bool:
    return "fadel" in (filename or "").lower()


def predict_batch(model, tokenizer, vocab, sep_id, bare_texts, device):
    rows = []
    for text in bare_texts:
        copy1 = tokenizer(text, add_special_tokens=False)["input_ids"]
        copy2_enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        copy2 = copy2_enc["input_ids"]
        input_ids = copy1 + [sep_id] + copy2
        truncated = len(input_ids) > MAX_SEQ_LENGTH
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        base = len(copy1) + 1

        char_to_token = [base] * len(text)
        char_intra = [0] * len(text)
        prev = base
        for t_idx, (s, e) in enumerate(copy2_enc["offset_mapping"]):
            tok_pos = base + t_idx
            if tok_pos >= len(input_ids):
                break  # truncated away -- these chars fall back to the last kept token
            for i in range(s, min(e, len(text))):
                char_to_token[i] = tok_pos
                char_intra[i] = i - s
                prev = tok_pos
        for i in range(len(text)):
            if char_to_token[i] >= len(input_ids):
                char_to_token[i] = prev

        rows.append((input_ids, char_to_token, char_intra, truncated))

    max_tok = max(len(r[0]) for r in rows)
    max_chr = max(len(t) for t in bare_texts)

    input_ids = [r[0] + [tokenizer.pad_token_id] * (max_tok - len(r[0])) for r in rows]
    attn = [[1] * len(r[0]) + [0] * (max_tok - len(r[0])) for r in rows]
    c2t = [r[1] + [0] * (max_chr - len(r[1])) for r in rows]
    cpos = [r[2] + [0] * (max_chr - len(r[2])) for r in rows]
    cids = [[char_id(c) for c in t] + [0] * (max_chr - len(t)) for t in bare_texts]

    batch = {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attn, device=device),
        "char_to_token": torch.tensor(c2t, device=device),
        "char_ids": torch.tensor(cids, device=device),
        "char_intra_pos": torch.tensor(cpos, device=device),
    }
    with torch.no_grad():
        logits = model(**batch)["logits"]
    preds = logits.argmax(-1).cpu().tolist()

    out = []
    n_truncated = sum(1 for r in rows if r[3])
    for text, pred in zip(bare_texts, preds):
        out.append(vocab.decode(text, pred[:len(text)]))
    return out, n_truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--base-checkpoint-dir", default=None,
                     help="override the warm-start source recorded in tagger_config.json")
    ap.add_argument("--out", default="results/metrics_tagging_qwen.csv")
    ap.add_argument("--predictions-out", default="results/predictions_tagging_qwen.csv")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    import pandas as pd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, tokenizer = QwenCharTagger.load_bundle(
        args.model_dir, base_checkpoint_dir=args.base_checkpoint_dir, map_location=device)
    model.to(device).eval()
    sep_id = tokenizer.eos_token_id
    print(f"[INFO] loaded {args.model_dir} ({sum(p.numel() for p in model.parameters()):,} params) "
          f"on {device}, {len(vocab)} label classes")

    ds = load_dataset(BENCHMARK_DATASET, split="train", revision=BENCHMARK_REVISION)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    filenames = ds["filename"]
    references = ds[BENCHMARK_TEXT_COLUMN]

    predictions = []
    n_truncated = 0
    for i in range(0, len(references), args.batch_size):
        chunk_refs = references[i:i + args.batch_size]
        bare_chunk = [strip_diacritics(r) for r in chunk_refs]
        preds, nt = predict_batch(model, tokenizer, vocab, sep_id, bare_chunk, device)
        predictions.extend(preds)
        n_truncated += nt
        print(f"[INFO] {min(i + args.batch_size, len(references))}/{len(references)}", flush=True)

    ca_idx = [i for i, fn in enumerate(filenames) if is_ca(fn)]
    msa_idx = [i for i, fn in enumerate(filenames) if not is_ca(fn)]
    assert ca_idx and msa_idx, (
        f"one domain bucket is empty ({len(ca_idx)} CA / {len(msa_idx)} MSA)"
    )
    print(f"[INFO] split: {len(ca_idx)} CA, {len(msa_idx)} MSA (filename-'fadel' convention)")

    def skeleton_corrupted(pred, ref):
        return clean_and_tokenize(strip_diacritics(pred)) != clean_and_tokenize(strip_diacritics(ref))

    rows = []
    buckets = {"CA": ca_idx, "MSA": msa_idx, "pooled": list(range(len(references)))}
    per_row_metrics = {}
    for name, idx in buckets.items():
        preds_b = [predictions[i] for i in idx]
        refs_b = [references[i] for i in idx]
        m = compute_der_wer(preds_b, refs_b)
        corrupted = sum(1 for p, r in zip(preds_b, refs_b) if skeleton_corrupted(p, r))
        rows.append({
            "model": args.model_dir, "method": "tagging_qwen_doublepass", "split": name,
            "n_examples": len(idx), "n_truncated": n_truncated if name == "pooled" else "",
            **m, "skeleton_mismatch_rate": round(100 * corrupted / len(idx), 2),
        })
        per_row_metrics[name] = m

    macro = {k: round((per_row_metrics["CA"][k] + per_row_metrics["MSA"][k]) / 2, 2)
             for k in per_row_metrics["CA"]}
    corrupted_ca = sum(1 for i in ca_idx if skeleton_corrupted(predictions[i], references[i]))
    corrupted_msa = sum(1 for i in msa_idx if skeleton_corrupted(predictions[i], references[i]))
    rows.append({
        "model": args.model_dir, "method": "tagging_qwen_doublepass", "split": "MACRO_MEAN",
        "n_examples": len(ca_idx) + len(msa_idx), "n_truncated": "",
        **macro,
        "skeleton_mismatch_rate": round(100 * (corrupted_ca + corrupted_msa)
                                         / (len(ca_idx) + len(msa_idx)), 2),
    })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame({"filename": filenames, "domain": ["CA" if is_ca(f) else "MSA" for f in filenames],
                  "reference": references, "prediction": predictions}
                 ).to_csv(args.predictions_out, index=False)

    print(f"\n[RESULT] macro DER_noce = {macro['DER_noce']}  (2.58 = the 0.8B generative "
          f"baseline; 10.89 = the AraBERT tagging control; <= 2.58 is a pass)")
    print(f"[RESULT] MSA skeleton corruption = "
          f"{round(100 * corrupted_msa / len(msa_idx), 4)}%")
    for r in rows:
        print(f"  {r['split']:12s} n={r['n_examples']:5d}  DER_noce={r['DER_noce']:6.2f}  "
              f"DER_ce={r['DER_ce']:6.2f}  skel_corrupt={r['skeleton_mismatch_rate']:5.2f}%")
    print(f"[OK] wrote {args.out} and {args.predictions_out}")


if __name__ == "__main__":
    main()
