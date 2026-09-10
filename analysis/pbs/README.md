# PBS orchestration

`build_job_matrix.py` materializes auditable JSONL matrices. `analysis_job.pbs`
validates and runs one row, while `submit_analysis.sh` preserves the formal
dependency chain and supports dry-run, stage, dataset, and smoke filtering.

At repository root, `analysis.sh` is the PBS array submitter. It uses
`build_dataset_run_matrix.py` to group every selected kernel dataset/seed and
deterministic STT dataset run into one auditable bundle. By default it selects
AIME2024, ARC-Challenge, and HumanEval+, producing 12 cells submitted as
`1-12%3`, with one GPU per cell. The same `analysis.sh` executes each bundle
and later runs the dependent cache-only finalizer. Cross-process locks protect
shared immutable Sender caches.

Each submission uses a unique `analysis/jobs/<run-id>/` directory. Array and
finalizer jobs therefore retain the exact matrices they were submitted with,
even if another invocation starts before they finish.

Target, stage, dataset, smoke, and dry-run filters remain available;
`--all-datasets` restores all nine kernel datasets. The array throttle accepts
one, two, or three through `ANALYSIS_MAX_GPUS` and defaults to three. Submit
with `bash analysis.sh` from a PBS login node, not `qsub analysis.sh`, because
the entry point dynamically materializes the manifest before calling `qsub -J`.
