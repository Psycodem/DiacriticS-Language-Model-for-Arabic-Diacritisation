# Published adapters

The three adapters in this folder are on the Hugging Face Hub as one repo, with
each training fraction as a subfolder:

**[Psycodem/gemma-4-e4b-lora-diacritization](https://huggingface.co/Psycodem/gemma-4-e4b-lora-diacritization)**

| Subfolder | Local folder |
|---|---|
| `10pct` | [`10pct/adapter/`](10pct/adapter/) |
| `30pct` | [`30pct/adapter/`](30pct/adapter/) |
| `50pct` | [`50pct/adapter/`](50pct/adapter/) |

```python
from peft import PeftModel
model = PeftModel.from_pretrained(model, "Psycodem/gemma-4-e4b-lora-diacritization", subfolder="50pct")
```

The local copies under each `*/adapter/` folder are byte-identical to what was
uploaded. Prefer the Hub for anything that needs to be reproducible by someone
else — it avoids cloning ~800 MB of weights with the source tree.

Base model: [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it).
