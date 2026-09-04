# Exact soft-token transport artifacts

This directory owns immutable directed cross-vocabulary transport artifacts and
their protocol documentation. `T_algo.md` defines the mathematical transport;
`bidirectional_stt_analysis.md` is the executable experiment specification.

Each `.npz` stores a CSC matrix in `[target, source]` coordinates plus active
token IDs and JSON provenance. Runtime code must load with `allow_pickle=False`
and validate file SHA-256, schema, direction, full sender support, receiver token
bounds, tokenizer fingerprints, model revisions, finite non-negative weights,
and unit source-column mass. It must never infer a missing direction by
transposing another artifact.

The matrix files are experiment inputs, not generated result caches. Planner
contexts belong under `analysis_cache/stt_planner_contexts/`, receiver results
under `analysis_cache/stt_receiver_evaluations/`, and summaries under
`analysis_result/`.
