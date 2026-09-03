# Slurm orchestration for Guqq

This directory adapts the exact analysis task matrices and dependency graph to
the Slurm-only `Guqq` cluster. It does not replace or modify the formal PBS
interface under `analysis/pbs/`.

`analysis_job.slurm` validates one JSONL row, activates the repository-local
`.venv`, records a locked progress ledger, maps task exit code `10` to
`SKIPPED`, and requests all model computation through Slurm. `submit_analysis.sh`
builds matrices before submission and supports the same stages and filters as
the PBS submitter.

Examples:

```bash
bash analysis/slurm/submit_analysis.sh --dataset aime2024 --smoke --dry-run
bash analysis/slurm/submit_analysis.sh --dataset aime2024 --smoke
bash analysis/slurm/submit_analysis.sh --stage collect
bash analysis/slurm/submit_analysis.sh --stage all
```

The Guqq defaults are partition `compute`, `gpu:1`, 4 CPUs, 64 GiB RAM, one
running array element, and a 72-hour limit. Override them with
`SLURM_PARTITION`, `SLURM_GRES`, `SLURM_CPUS`, `SLURM_MEMORY`,
`SLURM_MAX_RUNNING`, or `SLURM_TIME`.
