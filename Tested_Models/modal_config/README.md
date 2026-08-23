# Modal runners for the zero-shot benchmarks

Every model in `Tested_Models` except `Gemma_Qwen_test.py` was evaluated on Modal
rather than on the Ibex cluster. These are the scripts that produced the numbers
in `Tested Models Results.xlsx` and in the project README.

| Runner | Models |
|---|---|
| `modal_small_models.py` | Fine-Tashkeel, Tashkeel-350M-v2 |
| `modal_large_models.py` | aya-expanse-8b, Moonlight-16B-A3B-Instruct |
| `modal_fanar.py` | Fanar-1-9B-Instruct |
| `modal_flan_byt5.py` | Flan-T5-Tashkeel-Small, Glonor-ByT5-Arabic |
| `modal_gemma3_1b.py` | gemma-3-1b-pt-10k-diacritization |
| `modal_qwen35_08b_plain.py` | Qwen3.5-0.8B (stock, zero-shot baseline) |

All of them import `Evaluation_Functions_Corrected.py` from the repository root
rather than carrying their own copy of the scorer, so the metric cannot drift
between models. Each checkpoints every prediction to a Modal volume as it is
produced, so an interrupted run resumes instead of restarting.

    modal run <runner>.py --probe 5 --wait   # smoke test
    modal run --detach <runner>.py           # full 1,200-paragraph benchmark

Use `--detach` with the default `.spawn()` entrypoint for real runs. A blocking
`.remote()` call is cancelled by Modal if the launching client dies, which is how
two earlier runs were lost.

`Gemma_Qwen_test.py` ran on Ibex; its Slurm script is in `../ibex_config/`.
