# DiacriticS fine-tuning runbook

End-to-end sequence for the fine-tuning phase: environment → train → score →
record. Written so someone who wasn't in the conversation can run it.

The experiment matrix comes from the `Fine-Tuning Models` tab of the sheet:

| Owner | Model | Methods | Scored on |
|---|---|---|---|
| Mahdi | `google/gemma-4-E4B-it` | LoRA + QLoRA | Train, Test, SadeedDiac-25 |
| Saad | `Qwen/Qwen3.5-4B` | LoRA + QLoRA | Train, Test, SadeedDiac-25 |

Four runs, three result rows each — twelve rows to fill.

---

## 0. Before anything

**Verify these three, in order.** Each has already cost time once.

1. **Hugging Face token.** The one in commit `b6c05e6` is public and must stay
   revoked. Put a new one on Ibex at `~/.hf_token` with mode `600` — the sbatch
   scripts read it from there and it never enters the repo.
   ```bash
   printf '%s' 'hf_yourNewToken' > ~/.hf_token && chmod 600 ~/.hf_token
   ```
2. **Gemma licence.** `google/gemma-4-E4B-it` is gated. Accept it on the model
   page with the same account the token belongs to, or the download 401s an hour
   into the queue.
3. **KAUST VPN.** `glogin.ibex.kaust.edu.sa` does not resolve off-VPN.

---

## 1. Environment (once)

Never build on the login node — it will be killed for CPU use.

```bash
srun --time=1:00:00 --gpus=1 --cpus-per-gpu=8 --mem=32G --pty bash
cd "/path/to/DiacriticS-Language-Model-for-Arabic-Diacritisation/Train Related/ibex"
bash setup_env.sh
```

This creates `/ibex/scratch/$USER/envs/diacritics` and prints the resolved torch
/ CUDA / GPU versions. Read that output — it is the only confirmation that
bitsandbytes found a working CUDA.

### Assumptions to correct on first run

I could not reach Ibex to verify these. Check them before the first `sbatch`:

```bash
sinfo -o "%20N %10c %10m %25f %10G"     # what your allocation actually offers
module avail cuda                        # is cuda/12.2 real?
```

- `--constraint=a100` in both sbatch files — drop it if a100 nodes queue badly.
  A 4B QLoRA fits on a 32 GB V100 at batch size 1–2.
- `module load cuda/12.2` — falls back to `cuda/11.8`, then to nothing.
- `/ibex/scratch/$USER` as the scratch root.

---

## 2. Train

```bash
cd "/path/to/Train Related/ibex"

# QLoRA arm (4-bit base, the default)
sbatch --job-name=gemma-qlora train.sbatch google/gemma-4-E4B-it

# LoRA arm (bf16 base, same adapter config) — note --out-dir, or the two
# arms overwrite each other's adapter
sbatch --job-name=gemma-lora  train.sbatch google/gemma-4-E4B-it \
    --no-4bit --out-dir /ibex/scratch/$USER/runs/gemma-4-E4B-it__lora
```

Defaults follow `Confg_Info.md`: r=16, alpha=32, dropout 0.05, all seven
projection modules, lr 2e-4, cosine, 6 epochs, effective batch ~32.

`--resume` is passed automatically and is safe on a fresh directory, so a
requeued or time-limited job picks up from the last checkpoint.

Watch it:
```bash
squeue -u $USER
tail -f logs/train_gemma-qlora_*.out
```

Outputs land in `/ibex/scratch/$USER/runs/<slug>/`:
`final_adapter/`, `train_log.csv`, `loss_curve.png`, `run_config.json`.

---

## 3. Score

```bash
sbatch eval.sbatch google/gemma-4-E4B-it adapter gemma-4-E4B-it-QLoRA QLoRA
```

Arguments: `<model> <adapter|base> [label] [method] [splits]`.
Splits default to `train,test,sadeed`.

Chain it behind training so it starts the moment the adapter exists:
```bash
JID=$(sbatch --parsable --job-name=gemma-qlora train.sbatch google/gemma-4-E4B-it)
sbatch --dependency=afterok:$JID eval.sbatch \
    google/gemma-4-E4B-it adapter gemma-4-E4B-it-QLoRA QLoRA
```

Generation checkpoints every 50 paragraphs. Hitting the wall clock costs one
chunk — resubmit and it resumes.

Result: `/ibex/scratch/$USER/sadeed_outputs/<slug>__results.csv`, five rows —
Train, Test, and SadeedDiac-25 split into MSA / CA / Mean.

### Why the Test number is trustworthy

`finetune_qlora.py` and `eval_sadeed.py` both derive the held-out partition from
`HOLDOUT_SEED = 42` and `HOLDOUT_FRACTION = 0.02`, applied to the raw dataset
*before* any filtering. That shared boundary is the only thing making "Test"
genuinely unseen.

**If you change either constant, change it in both files.** They will not warn
you; the Test score will just quietly start measuring memorisation.

---

## 4. Record

```bash
cd "/path/to/Train Related"
python sheets_queue.py add --results-csv .../gemma-4-E4B-it-QLoRA__results.csv
```

Results go to a local queue (`sheets_queue.jsonl`) first, so a job finishing
overnight is never lost. Flush when a browser is available:

```bash
python sheets_queue.py snapshot   # refresh the local copy of the tab
python sheets_queue.py plan       # target cells + paste blocks — ALWAYS read this
python sheets_queue.py done --ids 1,2,3
```

`plan` resolves each row by matching Model / Method / Results against the live
tab, not by hardcoded row numbers, and warns before overwriting a non-empty
cell. Only the benchmark's Mean row goes to the Fine-Tuning tab; MSA and CA
belong on `Results of test models`, which has a Domain Track column.

Local xlsx mirror (the workbook is a stale export — the Sheet is source of truth):
```bash
python record_results.py --results-csv ... --write
```
Defaults to a dry run; `--write` is opt-in and takes a backup first.

---

## 5. Re-scoring the old baselines

Separate from fine-tuning, and it matters for the paper.

The v1 benchmark scripts each carried their own copy of DER/WER and they had
drifted. `aya-expanse-8b` was run twice and scored 96.28 and 86.21 on the same
MSA data — a ten-point spread from the scoring code alone. Numbers from
different implementations cannot be ranked against each other.

Comparable already (scored with the canonical metrics):
`Qwen3.5-4B`, `Qwen3.5-9B`, `gemma-4-E4B-it`, and the `instruct_models_v1.py`
rows for `aya-expanse-8b` / `Moonlight`.

Need re-scoring:
```bash
cd "/path/to/Models to test"
python seq2seq_models_v2.py   # Flan-T5-Tashkeel-Small, Glonor-ByT5-Arabic
python small_models_v2.py     # Fine-Tashkeel, Tashkeel-350M-v2
python medium_models_v2.py    # Qwen3.5-0.8B, gemma-3-1b-pt-10k
python large_models_v2.py     # aya-expanse-8b, Fanar-1-9B, Moonlight-16B
```

These reproduce v1 generation exactly — same per-model prompts, greedy decoding,
per-sample inference — and change only the scoring path, so a moved number is
attributable to the metric fix and nothing else.

Two repo ids were broken in v1 and are corrected here: `Fanar-1-9B-Instruct` and
`gemma-3-1b-pt-10k-diacritization` both lacked their owner prefix (`QCRI/`,
`Bisher/`) and could never have resolved. gemma-3-1b also moves to a `pipeline`
loader, so its delta is "now runs at all" rather than a clean metric-only
comparison.

Budget time: Glonor-ByT5 is byte-level and took 5h15m for 600 paragraphs.
Checkpointing makes it resumable.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `FATAL: no CUDA device visible` | Running on a login node. Use `sbatch`. |
| 401 downloading gemma | Licence not accepted, or `~/.hf_token` missing/unreadable. |
| OOM in training | Lower `--batch-size` to 1 and raise `--grad-accum` to keep the effective batch near 32. |
| `none of the expected projection modules exist` | Architecture uses different names. Print `model.named_modules()` and set `target_modules` by hand. |
| Eval reports many empty predictions | Every blank scores as fully wrong and inflates the rates. Check the eval log for batch failures before trusting the number. |
| `plan` says "no matching row" | The sheet's Model/Method text changed. Refresh with `snapshot`, or pass `--sheet-model`. |
| `plan` says "ambiguous" | Duplicate rows in the tab. Fix the sheet — do not guess. |

## Known gaps

- **ClickUp** is not wired. The connector reports multiple workspaces, returns
  an empty list, and exposes no `workspace_id` parameter on any tool. Fix is to
  disconnect and reconnect the connector, granting the `AI Project` workspace
  explicitly.
- **Sheet writes** need the Claude in Chrome extension. Reads already work
  anonymously via CSV export, which is how `snapshot` and verification work.
- **README results table** is out of sync with the sheet and mixes both metric
  implementations. Reconcile after step 5, not before.
