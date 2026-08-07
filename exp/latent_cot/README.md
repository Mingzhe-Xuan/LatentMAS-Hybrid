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
The default trajectory length is 150 steps (indexed 0 through 149).

```bash
python exp/latent_cot/run.py \
  --study c0 \
  --model_name Qwen/Qwen3-4B \
  --split test \
  --max_questions 50 --latent_steps 150 --probe_seed 42 \
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

## C1: sequential MAS entropy by agent

C1 reuses the active root `methods/latent_mas.py` sequential organization. A
single Qwen3-8B instance serves as Planner, Critic, Refiner, and Judger. Planner,
Critic, and Refiner each perform the same K latent steps while retaining the
complete accumulated prompt and latent KV cache. The shared C1/C2/C3 collector
records post-feedback hidden-state entropy at every local step and then runs the
Judger, so the same rollout also supplies C2 accuracy. The cumulative index is
Planner `t`, Critic `K+t`, and Refiner `2K+t`.

The MBPP+ subset is the dataset-order prefix of 30 questions (not a shuffled
sample). Colors identify `identical / linear / soft / kernel / text`; line styles
identify the three latent roles. In these sequential MAS studies, `text` is a
fixed-step greedy hard-token control: each latent role projects its current
hidden state through the output head, takes the argmax token, and feeds that
token back through the ordinary input embedding at the next step. It does not
add free-form intermediate text generation; only Judger performs variable-length
text generation.

```bash
python exp/latent_cot/run.py \
  --study c1 \
  --model_name Qwen/Qwen3-8B \
  --dataset mbppplus --split test --max_questions 30 \
  --latent_step_values 20 40 60 80 100 120 140 160 180 \
  --alignments identical linear soft kernel text \
  --device cuda
```

C1 writes `c1_entropy_by_agent_step.parquet`, `c1_summary.json`, and
`c1_entropy_vs_cumulative_step.pdf` in its run-local metrics, summaries, and
figures directories.

## C2: sequential MAS accuracy by per-agent latent steps

C2 uses the same data, role prompts, alignments, K grid, and full sequential KV
retention. After the three latent roles each run K steps, Judger performs greedy
text decoding. MBPP+ correctness uses the repository Markdown Python extraction
and timeout-based test execution.

```bash
python exp/latent_cot/run.py \
  --study c2 \
  --model_name Qwen/Qwen3-8B \
  --dataset mbppplus --split test --max_questions 30 \
  --latent_step_values 20 40 60 80 100 120 140 160 180 \
  --alignments identical linear soft kernel text \
  --max_new_tokens 4096 --device cuda
```

C2 writes `c2_accuracy_by_question.parquet`, `c2_summary.json`, and
`c2_accuracy_vs_steps.pdf`. Each point reports both accuracy and the raw
`correct/30` count with a question-bootstrap 95% interval.

## C3: mean time per question by latent steps

C3 reuses the per-question `wall_seconds` already stored by the shared C1/C2/C3
rollout. For every alignment and K, it plots the mean time across the MBPP+
questions with a question-bootstrap 95% interval. The horizontal axis is K,
the latent steps used by each of Planner, Critic, and Refiner; the corresponding
total latent budget is `3K`. Timing covers the complete
`LatentMASMethod.run_batch` call, including Judger decoding and MBPP+ correctness
evaluation.

```bash
python exp/latent_cot/run.py \
  --study c3 \
  --model_name Qwen/Qwen3-8B \
  --dataset mbppplus --split test --max_questions 30 \
  --latent_step_values 20 40 60 80 100 120 140 160 180 \
  --alignments identical linear soft kernel text \
  --max_new_tokens 4096 --device cuda
```

C3 writes `c3_time_by_question.parquet`, `c3_summary.json`, and
`c3_time_vs_steps.pdf`. If a compatible C1 or C2 shared cache already exists,
C3 only performs aggregation and plotting.

### C1/C2/C3 shared rollout cache

Like C0, C1, C2, and C3 keep validated rollout-derived data in a stable cache
under `exp/cache/latent_cot_mas/`. Each cache contains a C1 entropy table and a
C2/C3 per-question table with accuracy and wall time, produced by one complete
Planner/Critic/Refiner/Judger rollout. Consequently, running any one of C1, C2,
or C3 makes the other two cache hits, provided their rollout arguments match.
A cache miss from any study performs the complete flow, including Judger
decoding.

A normal rerun only regenerates the requested run-local summary and figure; it
does not load the model or perform rollout again. The cache identity covers the
exact MBPP+ question contents, model name, K grid, alignments, generation seed,
`max_new_tokens`, and rollout/alignment settings. It intentionally does not
include `--study`, because C1, C2, and C3 are three views of the same cached
rollout.

Use `--reuse_trajectory` when a cache hit is mandatory (the command fails if no
compatible cache exists), or `--force_recollect` to ignore and replace the
cache. Plot-only settings such as `--bootstrap_replicates` and `--probe_seed`
do not invalidate cached rollout data, so they can be changed when redrawing.
The run manifest records the cache path, integrity hash, and whether it was a
cache hit. Because the alignment list is part of the cache identity, adding the
`text` control creates a new five-alignment cache; an older four-alignment
C1/C2/C3 cache is intentionally not reused.

PBS examples:

```bash
qsub -v "EXP_TARGET=latent_cot,STUDY=c1" exp.sh
qsub -v "EXP_TARGET=latent_cot,STUDY=c2" exp.sh
qsub -v "EXP_TARGET=latent_cot,STUDY=c3" exp.sh
```
