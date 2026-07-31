# C0: single-model latent CoT entropy

This directory implements plan-v3 C0 for GSM8K and MBPP+. One
`Qwen/Qwen3-4B` model performs 50 repository `identical` recurrence steps for
each task. For every true pre-unembedding hidden state, Phase B computes the
entropy of `softmax(W_out h + b)` in nats. It does not use `W_in` for entropy
readout, cross-model alignment, exact/kernel/linear mappings, or argmax tokens
as CoT text. The recurrence retains the repository's identity feedback plus
target input-embedding mean-norm scaling.

By default, one invocation runs both datasets in the fixed order `gsm8k`,
`mbppplus`. The model is loaded once, while trajectory caches remain separate.
The entropy figure contains two side-by-side panels labeled `GSM8K` and
`MBPP+`.

```bash
python exp/latent_cot/run.py \
  --study c0 \
  --model_name Qwen/Qwen3-4B \
  --split test \
  --max_questions 50 --latent_steps 50 --probe_seed 42
```

Submit the same default two-dataset run through PBS without a dataset argument:

```bash
qsub -v "EXP_TARGET=latent_cot" exp.sh
```

`--dataset gsm8k` or `--dataset mbppplus` remains available for a single-dataset
debug run. MBPP+ follows the GSM8K prompt structure, specialized for Python
programming tasks.

Full hidden trajectories are cached as flat files under
`exp/cache/trajectories/`. A compatible per-dataset cache is reused
automatically; `--reuse_trajectory` requires both caches during a default run,
and `--force_recollect` replaces both. Cache filenames contain stable collection
parameters and no timestamp or hash.

Each invocation writes a new run under `exp_result/latent_cot/runs/`:

- `metrics/c0_entropy_by_step.parquet`: one row per dataset, question and step;
- `summaries/c0_summary.json`: separate per-dataset statistics and 95% CIs;
- `figures/c0_entropy_vs_step.pdf`: labeled side-by-side dataset panels;
- `figures/c0_entropy_vs_step.json`: figure provenance for both datasets;
- `run_manifest.json`: parameters, versions, cache provenance and failures.

Progress is appended to `exp_state.txt` in the invocation working directory.
