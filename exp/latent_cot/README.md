# C0: alignment-aware latent CoT entropy

C0 compares `linear`, `soft`, and `kernel` recurrence for Qwen3-8B and
Qwen3-14B on AIME 2024, HumanEval+, and MedQA. The current pre-unembedding
hidden state is transformed by the selected alignment and fed back through `inputs_embeds`.
Entropy is computed from the next pre-unembedding hidden states using
`softmax(W_out h + b)`.

The mappings use the repository implementations in `alignment.py`:

- `linear`: ridge least-squares mapping from `W_out` to `W_in`, followed by
  target-norm scaling;
- `soft`: full-vocabulary `softmax((W_out h + b) / tau) @ W_in`, with no
  token sampling or argmax;
- `kernel`: ORF positive-feature approximation using the configured feature
  count, temperature and repeat seed; this approximates the `soft` recurrence.

Every recurrence starts from the sequential LatentMAS Planner prompt, which
asks for a concise step-by-step plan and explicitly forbids producing the final
answer. By default, one invocation runs the three datasets in the fixed order
`aime2024`, `humanevalplus`, `medqa`. The requested split is used for
HumanEval+ and MedQA; AIME 2024 is resolved to its available `train` split.
The complete matrix is `2 models × 3 repeat seeds × 3 datasets = 18`
trajectory files. Each file contains all three alignments. Repeat seeds are
fixed to `42`, `43`, and `44`; each seed
controls both question selection and the kernel random features. Every
model/seed cell runs in an isolated process so GPU memory is released before
loading the next model. Each dataset contributes one panel to the output
figure with mean entropy-versus-step curves for both recurrences and 95%
bootstrap bands.
The default trajectory length is 150 steps (indexed 0 through 149).

```bash
python exp/latent_cot/run.py \
  --study c0 \
  --model_names Qwen/Qwen3-8B Qwen/Qwen3-14B \
  --repeat_seeds 42 43 44 --alignments linear soft kernel \
  --split test \
  --max_questions 50 --latent_steps 150 \
  --kernel_features 2048 --kernel_temperature 0.6 \
  --kernel_chunk_size 4096 --soft_chunk_size 32 --align_ridge 1e-5
```

PBS submission needs no dataset or alignment argument:

```bash
qsub -v "EXP_TARGET=latent_cot" exp.sh
```

For resumable collection, add `--skip_completed_trajectories` to the Python
entry point, `--skip-completed-trajectories` to `exp.py`/`exp.sh`, or set
`SKIP_COMPLETED_TRAJECTORIES=1` for PBS. A whole model/seed cell is skipped
before model loading when all three `.pt` files have compatible complete
manifests and valid SHA256 hashes. In a partially complete cell, complete
datasets are reused and missing, partial, or failed datasets are recollected.

```bash
python exp.py --latent_cot --skip-completed-trajectories
qsub -v "EXP_TARGET=latent_cot,SKIP_COMPLETED_TRAJECTORIES=1" exp.sh
```

`--dataset aime2024`, `--dataset humanevalplus`, or `--dataset medqa` remains
available for single-dataset debugging.
Old C0 trajectory caches are not compatible; the new cache filename contains
the model, repeat seed, `linear/soft/kernel` recurrence list, Planner prompt version, and kernel
configuration, so no manual deletion is required. Trajectory `.pt` files and
their integrity manifests are written under the repository-level `trj/`
directory.

Each invocation writes under `exp_result/latent_cot/runs/`:

- `metrics/c0_entropy_by_step.parquet`: one row per dataset, alignment,
  question and step;
- `summaries/c0_summary.json`: per-dataset and per-alignment statistics;
- `figures/c0_entropy_vs_step.pdf`: three dataset panels with three colored curves;
- `figures/c0_entropy_vs_step.json`: figure provenance and alignment settings;
- `run_manifest.json`: parameters, cache provenance and failure counts.

Progress is appended to `exp_state.txt` in the invocation working directory.

## C6: text-prefix one-step alignment agreement

C6 is a cache-only extension of C0 for AIME 2025 and MBPP+.  For every
question in the longest compatible cached C0 trajectory, it samples up to 100
text-recurrence prefix lengths without replacement.  A position is a tokenizer
token/recurrence step, not a Unicode character.  The cached greedy text-token
path is replayed with the loaded model to recover a mutually consistent hidden
state and KV state.  From the common replayed hidden state at each sampled
prefix, C6 branches one additional transformer step through
`linear`, `kernel`, `soft`, and greedy hard-token (`text`) feedback, then applies
the model output head.  Cached-vs-replayed hidden-state cosine is recorded as a
provenance audit; it does not stop the run unless `--strict_prefix_replay` is
specified (the audit threshold defaults to `0.98`).

For `linear-text`, `kernel-text`, `linear-soft`, and `kernel-soft`, C6 reports
categorical-sample token agreement, top-5 intersection fraction and any-overlap
rate, top-10 intersection size, directional KL in both directions, and
symmetric KL.  Categorical samples use a SHA256-keyed inverse-CDF uniform
variate, so results do not depend on processing order.  Confidence intervals
use question-cluster bootstrap resampling.  The primary KL follows the written
pair order, e.g. `linear|text` means KL(linear || text).

C6 also discovers every cached K that contains all four requested recurrences
and plots separate entropy panels for each K.  Historical caches without
`soft` or `text` are intentionally excluded.

```bash
python exp/latent_cot/c6_prefix_alignment.py \
  --trajectory_dir exp/cache/trajectories \
  --datasets aime2025 mbppplus \
  --comparison_steps 150 \
  --positions_per_trajectory 100 \
  --position_seed 42 --token_sample_seed 42 \
  --device cuda
```

PBS submission uses the same command through `exp.sh`:

```bash
qsub -v "EXP_TARGET=latent_cot_c6" exp.sh
```

The newest `exp_result/latent_cot/runs/c6_prefix_alignment_*` directory contains:

- `metrics/c6_pair_metrics.parquet`;
- `metrics/c6_entropy_by_step.parquet`;
- `summaries/c6_pair_summary.json`;
- `summaries/c6_entropy_summary.json`;
- `figures/c6_top10_overlap_distribution.pdf`;
- `figures/c6_kl_distribution.pdf`;
- one `figures/c6_entropy_by_cached_k_<dataset>.pdf` per dataset;
- `run_manifest.json` with trajectory hashes and exact analysis semantics.

The `prefix_replay_audit` section of `c6_pair_summary.json` reports the minimum
and median cosine, maximum absolute deviation, and the number of sampled
prefixes below the configured audit threshold for each dataset.

For a local input-only smoke test when a copied trajectory has no manifest, use
the explicitly non-formal validation mode.  It loads every record and checks
alignment coverage, hidden shapes, completion flags, and the text token/prompt/
final-hidden fields, but does not load model weights or produce scientific
metrics:

```bash
python exp/latent_cot/c6_prefix_alignment.py \
  --trajectory_dir trajectories --datasets aime2025 \
  --comparison_steps 100 --positions_per_trajectory 2 \
  --allow_missing_manifest --skip_sha256 --validate_only --device cpu
```

Formal runs must not use `--allow_missing_manifest` or `--skip_sha256`.

## C1: sequential MAS entropy by agent

C1 reuses the active root `methods/latent_mas.py` sequential organization. A
single Qwen3-8B instance serves as Planner, Critic, Refiner, and Judger. Planner,
Critic, and Refiner each perform the same K latent steps while retaining the
complete accumulated prompt and latent KV cache. The shared C1/C2/C3 collector
records post-feedback hidden-state entropy at every local step and then runs the
Judger, so the same rollout also supplies C2 accuracy. The cumulative index is
Planner `t`, Critic `K+t`, and Refiner `2K+t`.

By default, each C1/C2/C3 command runs two separate 30-question dataset-order
prefixes: MBPP+ `test` uses K=`20,40,60,80,100,120,140,160,180`, while
AIME2025 `train` (the split exposed by its Hugging Face dataset) uses the reduced
K=`20,60,100,140,180`. Each dataset gets its own run directory, cache, summary,
and figure. Colors identify `identical / linear / soft / kernel / text`; line styles
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
  --dataset all --split test --max_questions 30 \
  --latent_step_values 20 40 60 80 100 120 140 160 180 \
  --aime_latent_step_values 20 60 100 140 180 \
  --alignments identical linear soft kernel text \
  --device cuda
```

C1 writes `c1_entropy_by_agent_step.parquet`, `c1_summary.json`, and
`c1_entropy_vs_cumulative_step.pdf` inside each dataset-specific run directory.
The PDF title identifies MBPP+ or AIME 2025.

## C2: sequential MAS accuracy by per-agent latent steps

C2 uses the same data, role prompts, alignments, K grid, and full sequential KV
retention. After the three latent roles each run K steps, Judger performs greedy
text decoding. MBPP+ correctness uses Markdown Python extraction and
timeout-based test execution; AIME2025 correctness uses the repository's
normalized integer-answer evaluation.

```bash
python exp/latent_cot/run.py \
  --study c2 \
  --model_name Qwen/Qwen3-8B \
  --dataset all --split test --max_questions 30 \
  --latent_step_values 20 40 60 80 100 120 140 160 180 \
  --aime_latent_step_values 20 60 100 140 180 \
  --alignments identical linear soft kernel text \
  --max_new_tokens 4096 --device cuda
```

C2 writes `c2_accuracy_by_question.parquet`, `c2_summary.json`, and
`c2_accuracy_vs_steps.pdf`. Each point reports both accuracy and the raw
`correct/30` count with a question-bootstrap 95% interval. MBPP+ and AIME2025
are plotted and saved separately.

## C3: mean time per question by latent steps

C3 reuses the per-question `wall_seconds` already stored by the shared C1/C2/C3
rollout. For every dataset, alignment, and K, it plots the mean time across
that dataset's questions with a question-bootstrap 95% interval. The horizontal axis is K,
the latent steps used by each of Planner, Critic, and Refiner; the corresponding
total latent budget is `3K`. Timing covers the complete
`LatentMASMethod.run_batch` call, including Judger decoding and the active
dataset's correctness evaluation.

```bash
python exp/latent_cot/run.py \
  --study c3 \
  --model_name Qwen/Qwen3-8B \
  --dataset all --split test --max_questions 30 \
  --latent_step_values 20 40 60 80 100 120 140 160 180 \
  --aime_latent_step_values 20 60 100 140 180 \
  --alignments identical linear soft kernel text \
  --max_new_tokens 4096 --device cuda
```

C3 writes `c3_time_by_question.parquet`, `c3_summary.json`, and
`c3_time_vs_steps.pdf` in each dataset-specific run directory. If compatible
C1 or C2 shared caches already exist, C3 only performs aggregation and plotting.

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
dataset, split, exact question contents, model name, K grid, alignments, generation seed,
`max_new_tokens`, and rollout/alignment settings. It intentionally does not
include `--study`, because C1, C2, and C3 are three views of the same cached
rollout.

The AIME grid can be overridden with `--aime_latent_step_values`; this does not
change `--max_new_tokens`, whose default remains 4096. With 30 questions and
five alignments, AIME2025 now performs `30 * 5 * 5 = 750` single-question
rollouts rather than 1350.

Use `--dataset mbppplus` or `--dataset aime2025` to run only one dataset.
With `--dataset all --reuse_trajectory`, both dataset caches must exist; MBPP+
can continue to reuse a compatible existing shared cache while AIME2025 uses a
separate cache.

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


## C4: hierarchical pre-/post-alignment replacement ablation

C4 is fixed to Qwen3-8B and two datasets: the first 30 AIME2025 train questions
and the first 30 GPQA-Diamond test questions. It uses K=120 for
Planner/Critic/Refiner, `kernel` alignment with 1024 features, and paired
generation seeds 42, 43, 44, and 45. It compares three norm-preserving
conditions independently on each dataset:

- `clean`: normal hierarchical rollout, `e_t=A(h_t)`;
- `pre_replace`: replace each source hidden before alignment with independent
  Gaussian direction rescaled to its source L2 norm,
  `e_t=A(||h_t|| epsilon / ||epsilon||)`;
- `post_replace`: first align normally, then replace the embedding actually
  injected into the model with an independent Gaussian direction rescaled to
  its embedding L2 norm, `e_t~=||A(h_t)|| epsilon / ||epsilon||`.

The post-alignment transform is called only on actual latent inputs, local
steps 0--119. Each condition uses a dedicated noise RNG (`10000 + generation
seed`), so noise sampling does not advance model generation. This produces
`2 datasets * 1 alignment * 3 conditions * 4 seeds * 30 questions = 720`
question-level rows. Results and paired effects are reported separately for
AIME2025 and GPQA-Diamond. Interpret C4 using the paired `pre_replace_minus_clean` and
`post_replace_minus_clean` effects; it does not by itself establish that hidden
content is important.

```bash
python exp/latent_cot/run.py --study c4 --device cuda
```

Each alignment/condition/seed cell is atomically cached under
`exp/cache/latent_cot_c4/`. New C4 cells have an explicit intervention site in
the cache identity. For compatibility, an intact old kernel `clean` cache is
reused as `clean`, and an intact old kernel `random_hidden` cache is read-only
reused in memory as `pre_replace`; the legacy cache files are never changed.
These legacy cells apply only to AIME2025. GPQA-Diamond cells are collected in
new dataset-specific caches. Use `--reuse_trajectory` to require all compatible
cells, or `--force_recollect` to ignore caches.

Run-local artifacts are:

- `metrics/c4_accuracy_cost_by_question.parquet`;
- `metrics/c4_hidden_diagnostics.parquet`;
- `summaries/c4_summary.json`;
- `figures/c4_clean_pre_post_replace.pdf`;
- `run_manifest.json` with per-cell cache-hit provenance.

## C5: Planner-only additive-Gaussian robustness

C5 uses Qwen3-8B with only a latent Planner followed by a text Judger. It
compares `linear` and `kernel` under paired `clean` and `gaussian_005`
conditions on 30 fixed questions each from GSM8K test, MBPP+ test, and
AIME2024 train. The formal per-dataset settings come directly from
`params_dict.json`'s `hierarchical` configuration:

| dataset | Planner latent steps | Judger max tokens | batch size |
| --- | ---: | ---: | ---: |
| GSM8K | 20 | 2048 | 16 |
| MBPP+ | 80 | 4096 | 8 |
| AIME2024 | 10 | 20000 | 2 |

For every causally used Planner source state, the noisy condition applies
`h_t + delta_t` before alignment, where each coordinate of `delta_t` is
Gaussian with standard deviation `0.05 * ||h_t||_2 / sqrt(d)`. A dedicated
RNG keyed by dataset, item, role, step, and repeat gives linear and kernel the
same standard-Gaussian direction without advancing the generation RNG. Eight
paired repeats use generation seeds 42--49 and kernel seeds 101--108.

```bash
python exp/latent_cot/run.py --study c5 --device cuda
```

PBS submission:

```bash
qsub -v "EXP_TARGET=latent_cot,STUDY=c5" \
  exp.sh
```

C5 atomically caches each dataset/alignment/condition/repeat cell under
`exp/cache/latent_cot_c5/` and checkpoints run-local Parquet files after every
cell. Its artifacts are:

- `metrics/c5_accuracy_cost_by_question.parquet`;
- `metrics/c5_perturbation_diagnostics.parquet`;
- `summaries/c5_summary.json`;
- `figures/c5_accuracy_time_tokens.pdf`;
- `figures/c5_paired_accuracy_degradation.pdf`;
- `run_manifest.json`.

The main figure uses color for alignment and solid/hatched bars for clean/noisy.
The primary robustness statistic is the paired accuracy change
`noisy_minus_clean`; `kernel_minus_linear_robustness` is its
difference-in-differences. Alignment-state construction time is recorded in
cell manifests but excluded from the online wall-time bars.
