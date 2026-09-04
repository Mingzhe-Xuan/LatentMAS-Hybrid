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

`complete_reverse_support.py` deterministically upgrades a Bayes-reversed
ordinary-only sender support by encoding each missing source special token's
literal representation with the target tokenizer. The same tool normalizes a
full-support parent without adding columns. It preserves all parent
columns, adds normalized fallback columns, records the exact tokens/weights and
parent hash in provenance, binds both loaded tokenizer mappings and special IDs
to a new runtime fingerprint while retaining the opaque parent fingerprints,
and refuses to overwrite an output artifact.

The matrix files are experiment inputs, not generated result caches. Planner
contexts belong under `analysis_cache/stt_planner_contexts/`, receiver results
under `analysis_cache/stt_receiver_evaluations/`, and summaries under
`analysis_result/`.
