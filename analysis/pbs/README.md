# PBS orchestration

`build_job_matrix.py` materializes auditable JSONL matrices. `analysis_job.pbs`
validates and runs one row, while `submit_analysis.sh` preserves the formal
dependency chain and supports dry-run, stage, dataset, and smoke filtering.
