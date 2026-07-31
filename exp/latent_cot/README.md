# C0: alignment-aware latent CoT entropy

C0 compares three independent same-model latent recurrences on GSM8K and
MBPP+: `identical`, `linear`, and `kernel`. At each latent step, the current
pre-unembedding hidden state is transformed by the selected alignment and fed
back through `inputs_embeds`. Entropy is then computed from the next
pre-unembedding hidden states using `softmax(W_out h + b)`.

The mappings use the repository implementations in `alignment.py`:

- `identical`: identity feedback with target embedding mean-norm scaling;
- `linear`: ridge least-squares mapping from `W_out` to `W_in`, followed by
  target-norm scaling;
- `kernel`: ORF positive-feature approximation using the configured feature
  count, temperature and seed.

By default, one invocation runs both datasets in the fixed order `gsm8k`,
`mbppplus`. The model and alignment states are constructed once, while each
dataset keeps a separate trajectory cache. Each dataset contributes one panel
to the output figure; each panel contains differently colored mean
entropy-versus-step curves for all three alignments with 95% bootstrap bands.

```bash
python exp/latent_cot/run.py \
  --study c0 \
  --model_name Qwen/Qwen3-4B \
  --split test \
  --max_questions 50 --latent_steps 50 --probe_seed 42 \
  --kernel_features 2048 --kernel_temperature 1.0 \
  --kernel_seed 101 --kernel_chunk_size 4096 --align_ridge 1e-5
```

PBS submission needs no dataset or alignment argument:

```bash
qsub -v "EXP_TARGET=latent_cot" exp.sh
```

`--dataset gsm8k` or `--dataset mbppplus` remains available for debugging.
Because the recurrence schema changed from identical-only to a three-alignment
comparison, old C0 trajectory caches are not compatible; the new cache filename
contains the alignment and kernel configuration, so no manual deletion is
required.

Each invocation writes under `exp_result/latent_cot/runs/`:

- `metrics/c0_entropy_by_step.parquet`: one row per dataset, alignment,
  question and step;
- `summaries/c0_summary.json`: per-dataset and per-alignment statistics;
- `figures/c0_entropy_vs_step.pdf`: GSM8K/MBPP+ panels with three colored curves;
- `figures/c0_entropy_vs_step.json`: figure provenance and alignment settings;
- `run_manifest.json`: parameters, cache provenance and failure counts.

Progress is appended to `exp_state.txt` in the invocation working directory.
