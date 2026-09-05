# PBS orchestration

`build_job_matrix.py` materializes auditable JSONL matrices. `analysis_job.pbs`
validates and runs one row, while `submit_analysis.sh` preserves the formal
dependency chain and supports dry-run, stage, dataset, and smoke filtering.

At repository root, `analysis.sh` is the single-job PBS entry point matching
the style of `exp.sh`. Its default is a strictly serial kernel-then-STT run in
one GPU allocation; target, stage, dataset, smoke, and dry-run filters are
available through either command-line flags or PBS `-v` variables.
