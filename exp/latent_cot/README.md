# C0: alignment-aware latent CoT entropy

C0 compares five independent same-model recurrences on GSM8K, MBPP+,
ARC-Challenge, and AIME 2025:
`identical`, `linear`, `soft`, `kernel`, and `text`. For the four latent recurrences,
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
- `soft`: full-vocabulary `softmax((W_out h + b) / tau) @ W_in`, with no
  token sampling or argmax;
- `kernel`: ORF positive-feature approximation using the configured feature
  count, temperature and seed; this approximates the `soft` recurrence;
- `text`: greedy argmax decoding followed by ordinary token-embedding feedback.

By default, one invocation runs all four datasets in the fixed order `gsm8k`,
`mbppplus`, `arc_challenge`, `aime2025`. The requested split is used for the
first three datasets; AIME 2025 is resolved to its available `train` split.
The model and alignment states are constructed once, while each
dataset keeps a separate trajectory cache. Each dataset contributes one panel
to a 2x2 output figure; each panel contains differently colored mean
entropy-versus-step curves for all five recurrences with 95% bootstrap bands.
The default trajectory length is 100 steps (indexed 0 through 99).

```bash
python exp/latent_cot/run.py \
  --study c0 \
  --model_name Qwen/Qwen3-4B \
  --split test \
  --max_questions 50 --latent_steps 100 --probe_seed 42 \
  --kernel_features 2048 --kernel_temperature 1.0 \
  --kernel_seed 101 --kernel_chunk_size 4096 --soft_chunk_size 32 --align_ridge 1e-5
```

PBS submission needs no dataset or alignment argument:

```bash
qsub -v "EXP_TARGET=latent_cot" exp.sh
```

`--dataset gsm8k`, `--dataset mbppplus`, `--dataset arc_challenge`, or
`--dataset aime2025` remains available for single-dataset debugging.
Because the recurrence schema includes soft and text feedback alongside the
other latent alignments, old C0 trajectory caches are not compatible; the new cache
filename contains the recurrence and kernel configuration, so no manual
deletion is required.

Each invocation writes under `exp_result/latent_cot/runs/`:

- `metrics/c0_entropy_by_step.parquet`: one row per dataset, alignment,
  question and step;
- `summaries/c0_summary.json`: per-dataset and per-alignment statistics;
- `figures/c0_entropy_vs_step.pdf`: four dataset panels with five colored curves;
- `figures/c0_entropy_vs_step.json`: figure provenance and alignment settings;
- `run_manifest.json`: parameters, cache provenance and failure counts.

Progress is appended to `exp_state.txt` in the invocation working directory.
