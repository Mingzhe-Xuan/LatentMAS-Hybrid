# Analysis tasks

Each module is an explicit CLI entry point accepting `--config`, a job selected
with `--job-spec` or `--job-matrix`/`--job-index`, `--cache-only`, and `--force`.
Collection/evaluation tasks may load models; analysis tasks require validated
caches and never fall back to rollout.

The bidirectional STT protocol follows the same interface through
`collect_stt_planner_contexts.py`, `evaluate_bidirectional_stt.py`,
`analyze_bidirectional_stt.py`, and `build_bidirectional_stt_report.py`. Its
formal matrices contain 6 planner jobs, 12 four-system evaluation jobs, 3
dataset analyses, and one report job.
