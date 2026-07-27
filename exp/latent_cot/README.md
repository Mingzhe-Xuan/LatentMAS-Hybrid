# C0: single-model latent CoT entropy

This directory implements only plan-v3 C0. One `Qwen/Qwen3-4B` model reads a
GSM8K question prompt and performs 50 repository `identical` recurrence
steps. For each true pre-unembedding hidden state, Phase B computes
`softmax(W_out h + b)` entropy in nats. It does not use `W_in` for entropy
readout, cross-model alignment, exact/kernel/linear mappings, or argmax tokens
as CoT text. The
recurrence itself retains the repository's identity feedback plus target
input-embedding mean-norm scaling.

```bash
python exp/latent_cot/run.py \
  --study c0 \
  --model_name Qwen/Qwen3-4B \
  --dataset gsm8k --split test \
  --max_questions 512 --latent_steps 50 --probe_seed 42
```

Full hidden trajectories are cached as flat files under
`exp/cache/trajectories/`. A compatible cache is reused automatically;
`--reuse_trajectory` requires it, and `--force_recollect` replaces it. Cache
filenames contain stable generation parameters and no timestamp or hash.

Each invocation writes a new run under `exp_result/latent_cot/runs/`:

- `metrics/c0_entropy_by_step.parquet`: exactly one row per question and step;
- `summaries/c0_summary.json`: per-step valid counts, mean, median and 95% CI;
- `figures/c0_entropy_vs_step.pdf` and its artifact-context JSON;
- `run_manifest.json`: parameters, versions, cache provenance and failures.

Progress is appended to `exp_state.txt` in the invocation working directory.
