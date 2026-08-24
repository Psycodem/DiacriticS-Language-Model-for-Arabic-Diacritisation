"""
Qwen3.5-0.8B + char-expansion head: the plan's MAIN tagging arm (double-pass design).

Qwen3.5-0.8B is causal (18/24 layers are Gated-DeltaNet linear attention, only 6 are full
softmax attention — config.json `layer_types`, `full_attention_interval=4`). A tagger needs
every position to see the WHOLE sentence, not just what came before it — case endings (i'rab)
depend on syntactic role, which often needs the following words. Un-masking a linear RNN is not
an option (it is causal by construction, not by an attention mask you can drop).

Fix, from the plan: feed the bare text TWICE, separated by a marker, and tag only the second
copy:

    [bare text]  <sep>  [bare text]
                 ^ context pass      ^ tagged positions attend over the full first copy

This needs NO architecture surgery. Standard causal masking already gives every position in
copy 2 full attention over the entirety of copy 1 (which is the whole sentence, unmasked) through
the 6 full-attention layers, plus whatever compressed summary the 18 linear-attention layers'
recurrent state carries forward from copy 1 by the time they reach copy 2. That second part is
the plan's one open risk: a linear RNN's state is a compressed, recency-biased summary, not
per-position full attention, so the "bidirectionality" copy 2 gets through those 18 layers is
weaker than through the 6 real attention layers. If results are poor, that is the first thing to
suspect — the plan's fallback is running the encoder forward AND on reversed text and
concatenating hidden states, BiRNN-style.

Warm-started from the existing 2.58-DER checkpoint (Train-Qwen-0.8B-Full-FT's
step-4070 full fine-tune), not base Qwen — that body already knows how to read this exact bare
Arabic and reason about it; only the head (LM vocab projection -> char-diacritic classification)
is new and randomly initialized.

"Dropping the lm_head" (per the plan) does not free memory: tie_word_embeddings=true means
lm_head and the input embedding table are the SAME tensor. What it drops is the compute: this
model takes hidden states straight from `.model` (the inner decoder) and never runs the
248,320-way vocabulary projection at all.
"""

import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same head design as ../Train-Tagging/model_tagging.py -- copied, not imported, per this repo's
# convention that no package imports another.
MAX_CODEPOINT = 0x800
CHAR_UNK_ID = MAX_CODEPOINT - 1


def char_id(ch: str) -> int:
    cp = ord(ch)
    return cp if cp < MAX_CODEPOINT else CHAR_UNK_ID


class CharTaggingHead(nn.Module):
    def __init__(self, hidden_size: int, char_embed_dim: int, max_intra_pos: int, num_labels: int):
        super().__init__()
        self.char_embed = nn.Embedding(MAX_CODEPOINT, char_embed_dim)
        self.max_intra_pos = max_intra_pos
        self.intra_pos_embed = nn.Embedding(max_intra_pos, char_embed_dim)
        in_dim = hidden_size + 2 * char_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, token_hidden, char_ids, char_intra_pos):
        ce = self.char_embed(char_ids)
        pe = self.intra_pos_embed(char_intra_pos.clamp(max=self.max_intra_pos - 1))
        x = torch.cat([token_hidden, ce, pe], dim=-1)
        return self.mlp(x)


def assert_text_only(model):
    """Copied from Train-Qwen-0.8B-Full-FT/train_full_ft_qwen35_08b.py:1128. Confirms
    AutoModelForCausalLM resolved to the text-only Qwen3_5ForCausalLM (self.model + self.lm_head)
    and not a conditional-generation class carrying a 12-layer vision tower + MTP head that would
    otherwise train, decay and checkpoint ~100M parameters that never see this task's loss."""
    names = {n for n, _ in model.named_modules()}
    bad = sorted(n for n in names if any(part in ("visual", "mtp") for part in n.split(".")))
    if bad:
        raise RuntimeError(
            f"non-text towers were allocated ({bad[:5]}{'...' if len(bad) > 5 else ''}). "
            f"AutoModelForCausalLM resolved to {type(model).__name__} instead of the text-only "
            f"Qwen3_5ForCausalLM."
        )
    print(f"[OK] {type(model).__name__}: text decoder only, no vision/MTP submodules.")


def assert_fast_path():
    """Copied from Train-Qwen-0.8B-Full-FT/train_full_ft_qwen35_08b.py:760. Without
    causal-conv1d + flash-linear-attention, the 18 Gated-DeltaNet layers silently fall back to a
    pure-PyTorch chunked scan -- correct but several times slower, and it only logs
    `warning_once` so a run can look healthy while burning several times the GPU-hours."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    except Exception as e:
        print(f"[WARN] could not import modeling_qwen3_5 to verify is_fast_path_available: {e}")
        return
    flag = getattr(mod, "is_fast_path_available", None)
    if flag is False:
        raise RuntimeError(
            "is_fast_path_available=False -- causal-conv1d / flash-linear-attention are not "
            "usable. The 18 linear-attention layers would fall back to a pure-PyTorch chunked "
            "scan, several times slower. Fix the image before spending real GPU time."
        )
    print(f"[OK] is_fast_path_available={flag}")


def _from_pretrained(cls, model_id, dtype, **kwargs):
    """transformers 5.x renamed torch_dtype -> dtype. Same shim as the generative script."""
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


class QwenCharTagger(nn.Module):
    def __init__(self, checkpoint_dir: str, num_labels: int,
                 char_embed_dim: int = 64, max_intra_pos: int = 48):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.num_labels = num_labels
        self.char_embed_dim = char_embed_dim
        self.max_intra_pos = max_intra_pos

        causal_lm = _from_pretrained(AutoModelForCausalLM, checkpoint_dir, torch.float32)
        assert_text_only(causal_lm)
        self.body = causal_lm.model            # Qwen3_5Model -- no lm_head, no vocab matmul
        self.body.config.use_cache = False      # incompatible with training; wastes memory
        del causal_lm.lm_head                   # tied to embed_tokens -- no memory freed, just
        # drops the reference so nothing accidentally calls through it in this module.

        hidden = self.body.config.hidden_size
        self.head = CharTaggingHead(hidden, char_embed_dim, max_intra_pos, num_labels)

    def forward(self, input_ids, attention_mask, char_to_token, char_ids, char_intra_pos,
                labels=None, **kwargs):
        hidden = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = char_to_token.clamp(min=0)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        token_hidden = torch.gather(hidden, 1, idx_exp)  # [B, C, H]
        logits = self.head(token_hidden, char_ids, char_intra_pos)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-100
            )
        return {"loss": loss, "logits": logits}

    def save_bundle(self, output_dir: str, label_vocab, tokenizer=None):
        import json
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "model_state.pt"))
        with open(os.path.join(output_dir, "tagger_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint_dir": self.checkpoint_dir,
                "num_labels": self.num_labels,
                "char_embed_dim": self.char_embed_dim,
                "max_intra_pos": self.max_intra_pos,
            }, f, indent=2)
        label_vocab.save(os.path.join(output_dir, "labels.json"))
        (tokenizer or AutoTokenizer.from_pretrained(self.checkpoint_dir)).save_pretrained(output_dir)

    @classmethod
    def load_bundle(cls, model_dir: str, base_checkpoint_dir: str = None, map_location=None):
        """base_checkpoint_dir overrides the ORIGINAL warm-start source recorded in
        tagger_config.json -- needed when evaluating a bundle whose original warm-start
        checkpoint path no longer exists (e.g. a different Volume/container), since the encoder
        architecture is re-instantiated from that source before loading this bundle's trained
        state dict on top."""
        import json
        from labels import LabelVocab
        with open(os.path.join(model_dir, "tagger_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        src = base_checkpoint_dir or cfg["checkpoint_dir"]
        model = cls(src, cfg["num_labels"], cfg["char_embed_dim"], cfg["max_intra_pos"])
        state = torch.load(os.path.join(model_dir, "model_state.pt"), map_location=map_location)
        model.load_state_dict(state)
        vocab = LabelVocab.load(os.path.join(model_dir, "labels.json"))
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, vocab, tokenizer


def assert_master_dtype(model: nn.Module, expect=torch.float32):
    """Same guard as Train-Qwen-0.8B-Full-FT's assert_master_dtype and
    ../Train-Tagging/model_tagging.py's -- fp32 master weights, bf16 only under autocast. Loading
    this checkpoint's config.json says "dtype": "bfloat16", so an unspecified dtype resolves to
    bf16 by default and silently rounds away lr-scale updates unless dtype=torch.float32 is
    explicit at load time (see _from_pretrained above)."""
    bad = [n for n, p in model.named_parameters() if p.dtype != expect]
    assert not bad, f"{len(bad)} parameters not in {expect}, e.g. {bad[:5]}"
