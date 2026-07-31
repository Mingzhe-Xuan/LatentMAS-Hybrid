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
