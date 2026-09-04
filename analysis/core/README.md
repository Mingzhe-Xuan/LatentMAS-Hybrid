# Core analysis library

`analysis.core` owns strict configuration and schemas, dataset snapshots,
content-addressed caches, two-agent execution, interventions, evaluation,
statistics, and artifact summaries. Inputs are immutable configuration/job
records plus production model wrappers; outputs are validated dataclasses and
hashed cache artifacts. It never imports experiment implementations from
`exp/`.

`stt.py` adds the versioned bidirectional Exact STT protocol for tokenizer-
incompatible model pairs. It validates directed CSC transport artifacts before
constructing sparse tensors, performs full-source-vocabulary softmax and FP32
transport accumulation, packs aligned sender context before the native judger
prompt, and explicitly drives greedy receiver decoding. STT planner contexts
and receiver evaluations use separate cache namespaces so kernel-analysis-v1
readers cannot accidentally consume different hidden-state semantics.
