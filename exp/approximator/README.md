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

Trajectories and their strict configuration manifests are stored under
`exp_result/approximator/cache/trajectories/`. Full S1/S2 mapping caches live
under `exp_result/approximator/cache/mappings/`. A matching cache is reused by default.
Use `--force_recollect` to replace it, or `--reuse_trajectory` to require a
matching existing cache and prohibit Phase A.

S1 and S2 share one full mapping cache. Its filename records
the feature count, kernel temperature, seed, chunk size, and a configuration
digest. The digest also covers the trajectory manifest, source/target models,
probe seed, and mapping implementation. A valid cache is reused across runs;
S1 and S2 summaries and figures are both derived from those cached rows.

Every invocation creates an independent directory under
`exp_result/approximator/runs/`. The directory name contains the dataset,
split, study, model assignment, generation and kernel parameters, sampling
limits, a `YYYYMMDD_HHMMSS` timestamp, and a short complete-configuration hash.
It contains:

- `run_manifest.json`: full arguments, versions, status, timing, and cache hits;
- `exp_state.log`: progress for this invocation only;
- `metrics/`: raw Parquet tables;
- `summaries/`: scalar-only JSON summaries for S0--S4;
- `figures/`: plots;
- `manifests/`: trajectory, mapping, and analysis provenance.

Summary JSON files never contain hidden vectors, embeddings, PCA/t-SNE
coordinates, per-state rows, or per-seed arrays. Continuous metrics report
count, mean, median, sample variance, standard deviation, quantiles, and
question-cluster bootstrap confidence intervals. State values are averaged
within each question before cross-question statistics are computed.
