# Core analysis library

`analysis.core` owns strict configuration and schemas, dataset snapshots,
content-addressed caches, two-agent execution, interventions, evaluation,
statistics, and artifact summaries. Inputs are immutable configuration/job
records plus production model wrappers; outputs are validated dataclasses and
hashed cache artifacts. It never imports experiment implementations from
`exp/`.
