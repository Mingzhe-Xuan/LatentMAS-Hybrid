# PBS orchestration

`build_job_matrix.py` materializes auditable JSONL matrices. `analysis_job.pbs`
validates and runs one row, while `submit_analysis.sh` preserves the formal
dependency chain and supports dry-run, stage, dataset, and smoke filtering.

At repository root, `analysis.sh` is the PBS array submitter. It uses
`build_dataset_run_matrix.py` to group every kernel dataset/seed and every
deterministic STT dataset run into one auditable bundle. The formal compute
array therefore contains 30 cells and is submitted as `1-30%2`, with one GPU
per cell. `analysis_dataset_run.pbs` executes a bundle sequentially and uses
cross-process locks for shared immutable Sender caches. After the complete
array succeeds, `analysis_finalize.pbs` runs the cache-only analyses and both
reports through `run_dataset_bundle.py`.

Each submission uses a unique `analysis/jobs/<run-id>/` directory. Array and
finalizer jobs therefore retain the exact matrices they were submitted with,
even if another invocation starts before they finish.

Target, stage, dataset, smoke, and dry-run filters remain available. The array
throttle accepts only one or two through `ANALYSIS_MAX_GPUS`; it defaults to
two. Submit with `bash analysis.sh` from a PBS login node, not `qsub analysis.sh`,
because the entry point dynamically materializes the manifest before calling
`qsub -J`.
