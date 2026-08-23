---
base_model: google/gemma-4-E4B-it
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:google/gemma-4-E4B-it
- lora
- arabic
- diacritization
- tashkeel
language:
- ar
datasets:
- Misraj/Sadeed_Tashkeela
- Misraj/SadeedDiac-25
---

> **On the Hugging Face Hub:** [`Psycodem/gemma-4-e4b-lora-diacritization`](https://huggingface.co/Psycodem/gemma-4-e4b-lora-diacritization) — this adapter is
> the `10pct` subfolder. Load it directly with
> `PeftModel.from_pretrained(model, "Psycodem/gemma-4-e4b-lora-diacritization", subfolder="10pct")`
> instead of copying these files.

# google/gemma-4-E4B-it + LoRA for Arabic Diacritisation (10% of the training corpus)

A LoRA adapter that restores Arabic diacritics (tashkeel). Part of **DiacriticS**,
a contamination-controlled study of open-weights models on this task
([project site](https://diacritics.vercel.app/) ·
[code](https://github.com/Psycodem/DiacriticS-Language-Model-for-Arabic-Diacritisation)).

## Results

Scored with `Evaluation_Functions_Corrected.py` from the project repository.
All values are percentages; lower is better. CE = case ending (I'rab).

| Split | n | DER (CE) | DER (no CE) | WER (CE) | WER (no CE) |
|---|---:|---:|---:|---:|---:|
| Train sample | 500 | 1.23 | 1.01 | 3.24 | 1.86 |
| Sadeed Tashkeela test | 2,485 | 12.96 | 12.45 | 16.2 | 13.33 |
| **SadeedDiac-25 benchmark** | **1,200** | **3.15** | **2.68** | **7.28** | **4.47** |

The benchmark is the headline number: SadeedDiac-25 is expert-reviewed, balanced
50/50 between Modern Standard and Classical Arabic, and has no overlap with the
training corpus. The Sadeed Tashkeela test split scores worse than the benchmark
because it inherits the residual annotation noise of the source corpus, not
because the model does worse on it.

## Training

Trained on a 10% subset of `Misraj/Sadeed_Tashkeela` (104,270 rows), drawn with
a fixed shuffle seed so the 10% subset is contained in the 30%, and that in the
50%. The corpus has a measured 0.4% overlap with the benchmark.

| | |
|---|---|
| Method | LoRA (base weights frozen, bf16) |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q, k, v, o, gate, up, down |
| Effective batch | 96 |
| Learning rate | 2e-4, cosine |
| Warmup | 5% of total steps (54 of 1,087) |
| Epochs | 1 |
| Max sequence length | 1024 |
| Seed | 42 |
| Run on | Modal (3x A100-80GB) |

Every fraction and both base models share this configuration, so differences
between them reflect data volume and architecture rather than tuning.

Launcher: `modal_config/modal_train.py --model gemma` in the project repository.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE    = "google/gemma-4-E4B-it"
ADAPTER = "<path to this folder>"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.eval()

SYSTEM_PROMPT = (
    "أنت نظام متخصص في التشكيل الآلي للنصوص العربية. "
    "مهمتك إضافة الحركات (التشكيل) الصحيحة إلى النص العربي المُدخل دون تغيير الكلمات أو ترتيبها، "
    "مع مراعاة السياق النحوي والصرفي الكامل للجملة."
)

def diacritize(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print(diacritize("ذهب الطالب إلى المدرسة"))
```

**The prompt must match the one above.** It is what the adapter was trained
against; a different instruction degrades output in ways that look like a bad
model rather than a harness mistake. Decode greedily (`do_sample=False`) — the
numbers above assume it. If you batch, set `tok.padding_side = "left"`.

## Limitations

Sentence-final case endings (I'rab) and Classical Arabic remain the dominant
error sources, as they are for every system in the study. The adapter was trained
for one epoch on a subset of a single corpus and is not expected to transfer to
dialectal Arabic or to Quranic orthography with its additional annotation marks.
