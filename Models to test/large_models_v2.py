# -*- coding: utf-8 -*-
"""large_models_v2.py — re-score the large instruct models.

v1 (Large_models_v1.py) scored with the oldest inline metric copy. This is the
tier where the drift is provable: aya-expanse-8b was run twice, once here and
once in instruct_models_v1.py (which had the corrected metrics), and the MSA
track came out 96.28 vs 86.21 for the same model on the same benchmark. Only the
scoring differed. Re-scoring all three puts them on one scale.

    python large_models_v2.py

Fanar's repo id is corrected: v1 listed a bare "Fanar-1-9B-Instruct" with no
owner prefix, which cannot resolve on the Hub. The sheet gives QCRI/Fanar-1-9B-Instruct.

Fanar and Moonlight are both gated or require remote code. Export a token first:
    export HF_TOKEN=hf_...   # a NEW one — the token in commit b6c05e6 is public
"""

from benchmark_runner import run_registry

REGISTRY = {
    "aya-expanse-8b": {
        "repo": "CohereLabs/aya-expanse-8b",
        "kind": "causal_chat",
        "single_line": True,
    },
    "Fanar-1-9B-Instruct": {
        "repo": "QCRI/Fanar-1-9B-Instruct",
        "kind": "causal_chat",
        "single_line": True,
        "trust_remote_code": True,
    },
    "Moonlight-16B-A3B-Instruct": {
        "repo": "moonshotai/Moonlight-16B-A3B-Instruct",
        "kind": "causal_chat",
        "single_line": True,
        "trust_remote_code": True,
    },
}

if __name__ == "__main__":
    run_registry(REGISTRY, "large_models_v2_results.csv")
