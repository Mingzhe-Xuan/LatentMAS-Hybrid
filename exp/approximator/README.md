# Two-stage approximator experiments

`run.py` first executes the repository's real sequential
`latent_mas_hybrid` flow and caches sampled observations from its four roles.
It then reloads that `.pt` trajectory and runs S0--S4. S1--S4 always analyse
the semantic edge:

```text
Refiner latent_reply_hidden
  with Refiner W_out -> Judger W_in
```

One model name is copied to all four roles. Four names are assigned in
Planner, Critic, Refiner, Judger order.

```bash
python exp/approximator/run.py \
  --agent_models Qwen/Qwen3-4B \
  --dataset arc_easy --split test --study all
```

Trajectories and their manifests are stored directly under
`exp/cache/trajectories/`; mapping caches and their manifests are stored
directly under `exp/cache/mappings/`. Both directories are flat: filenames
expose stable generation and mapping parameters and contain no timestamps or
hash-derived names. A matching cache is reused by default.
Use `--force_recollect` to replace it, or `--reuse_trajectory` to require a
matching existing cache and prohibit Phase A. Trajectory compatibility uses
only requested question contents, the role-to-model mapping, trajectory
generation settings, collection alignment settings, and the generation seed.
Code hashes, model/tokenizer fingerprints, library versions, and extra cached
questions do not prevent reuse.

S1 and S2 share one full mapping cache. Its flat filename inherits the
trajectory parameters; the manifest additionally verifies the trajectory
manifest, source/target models, probe seed, and mapping implementation. A
valid cache is reused across runs; S1 and S2 summaries and figures are both
derived from those cached rows.

Every invocation creates an independent directory under
`exp_result/approximator/runs/`. The directory name contains the dataset,
split, study, model assignment, generation and kernel parameters, sampling
limits, a `YYYYMMDD_HHMMSS` timestamp, and a short complete-configuration hash.
It contains:

- `run_manifest.json`: full arguments, versions, status, timing, and cache hits;
- `metrics/`: raw Parquet tables;
- `summaries/`: scalar-only JSON summaries for S0--S4;
- `figures/`: plots;
- `manifests/`: trajectory, mapping, and analysis provenance.

Progress from all invocations is appended to `exp_state.txt` in the working
directory from which `run.py` is launched.

Summary JSON files never contain hidden vectors, embeddings, PCA/t-SNE
coordinates, per-state rows, or per-seed arrays. Continuous metrics report
count, mean, median, sample variance, standard deviation, quantiles, and
question-cluster bootstrap confidence intervals. State values are averaged
within each question before cross-question statistics are computed.

## S3 tail-probability analysis

S3 evaluates the absolute mapping error `||F_hat - F||_2` across independent
random-feature seeds. Thresholds are configured with
`--s3_tail_epsilons`; defaults are `0.01 0.02 0.05 0.1 0.2 0.5 1.0`.
For example:

```bash
python exp/approximator/run.py \
  --agent_models Qwen/Qwen3-4B \
  --dataset arc_easy --split test --study s3 --reuse_trajectory \
  --max_questions 50 --max_states_per_question 50 \
  --s3_tail_epsilons 0.01 0.05 0.1 0.2 0.5 1.0
```

Outputs include:

- `metrics/s3_tail_by_question.parquet`: per-question empirical tail
  probability, second moment, and `min(1, MSE / epsilon^2)` Markov bound;
- `metrics/s3_tail_by_seed.parquet`: question-balanced exceedance rate for
  every random-feature seed and threshold;
- `summaries/s3_tail_summary.json`: the `m=2048, tau=1` headline with
  question-cluster bootstrap confidence intervals;
- `summaries/s3_tail_grid_summary.json`: all `(m, tau, epsilon)` cells;
- `figures/s3_tail_probability_epsilon.pdf`: empirical multi-epsilon curves,
  95% question-bootstrap bands, and theoretical upper bounds;
- `figures/s3_tail_probability_temperature.pdf`: temperature ablation;
- `figures/s3_tail_probability_by_seed.pdf`: seed-level exceedance heatmap.

The displayed theoretical curve is a plug-in Markov bound using the empirical
second moment from S3 replicates. It does not assume that the normalized
kernel mapping is unbiased.

## S4 joint alignment visualization

S4 constructs four vectors for every sampled Refiner latent state: the raw
hidden state immediately before the output logits, the exact
probability-weighted target input embedding, and the Linear and Kernel aligned
states. It then creates two independent joint-fit visualizations:

- `figures/s4_linear_joint_reduction.pdf`: hidden, exact embedding, and
  Linear-aligned states;
- `figures/s4_kernel_joint_reduction.pdf`: hidden, exact embedding, and
  Kernel-aligned states.

Each PDF contains joint PCA and joint t-SNE panels by default. Pass
`--no_s4_tsne` to disable t-SNE and generate PCA-only figures. Blue denotes raw hidden states, orange denotes exact
embedding states, and green denotes aligned states. No per-class
standardization is applied, so the plots retain the actual scale and geometry
of the three spaces.

Coordinate artifacts are written to
`metrics/s4_joint_pca_coordinates.parquet` and, when enabled,
`metrics/s4_joint_tsne_coordinates.parquet`. Raw hidden and target embedding
vectors must have the same dimension; S4 fails explicitly when a cross-model
pair does not satisfy that requirement.
