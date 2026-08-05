# C0: alignment-aware latent CoT entropy

C0 compares four independent same-model recurrences on GSM8K and MBPP+:
`identical`, `linear`, `kernel`, and `text`. For the three latent recurrences,
the current pre-unembedding hidden state is transformed by the selected
alignment and fed back through `inputs_embeds`. The `text` recurrence performs
greedy argmax decoding followed by ordinary token-embedding feedback, and runs
for the same fixed number of steps as the latent recurrences. Entropy is then
computed from the next pre-unembedding hidden states using
`softmax(W_out h + b)`.

The mappings use the repository implementations in `alignment.py`:

- `identical`: identity feedback with target embedding mean-norm scaling;
- `linear`: ridge least-squares mapping from `W_out` to `W_in`, followed by
  target-norm scaling;
- `kernel`: ORF positive-feature approximation using the configured feature
  count, temperature and seed;
- `text`: greedy argmax decoding followed by ordinary token-embedding feedback.

By default, one invocation runs both datasets in the fixed order `gsm8k`,
`mbppplus`. The model and alignment states are constructed once, while each
dataset keeps a separate trajectory cache. Each dataset contributes one panel
to the output figure; each panel contains differently colored mean
entropy-versus-step curves for all four recurrences with 95% bootstrap bands.
The default trajectory length is 100 steps (indexed 0 through 99).

```bash
python exp/latent_cot/run.py \
  --study c0 \
  --model_name Qwen/Qwen3-4B \
  --split test \
  --max_questions 50 --latent_steps 100 --probe_seed 42 \
  --kernel_features 2048 --kernel_temperature 1.0 \
  --kernel_seed 101 --kernel_chunk_size 4096 --align_ridge 1e-5
```

PBS submission needs no dataset or alignment argument:

```bash
qsub -v "EXP_TARGET=latent_cot" exp.sh
```

`--dataset gsm8k` or `--dataset mbppplus` remains available for debugging.
Because the recurrence schema now includes text feedback alongside the three
latent alignments, old C0 trajectory caches are not compatible; the new cache
filename contains the recurrence and kernel configuration, so no manual
deletion is required.

Each invocation writes under `exp_result/latent_cot/runs/`:

- `metrics/c0_entropy_by_step.parquet`: one row per dataset, alignment,
  question and step;
- `summaries/c0_summary.json`: per-dataset and per-alignment statistics;
- `figures/c0_entropy_vs_step.pdf`: GSM8K/MBPP+ panels with four colored curves;
- `figures/c0_entropy_vs_step.json`: figure provenance and alignment settings;
- `run_manifest.json`: parameters, cache provenance and failure counts.

Progress is appended to `exp_state.txt` in the invocation working directory.
