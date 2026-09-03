# Analysis tasks

Each module is an explicit CLI entry point accepting `--config`, a job selected
with `--job-spec` or `--job-matrix`/`--job-index`, `--cache-only`, and `--force`.
Collection/evaluation tasks may load models; analysis tasks require validated
caches and never fall back to rollout.
