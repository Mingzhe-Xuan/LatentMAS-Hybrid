# Kernel alignment analysis

This directory implements the standalone two-agent protocol specified in
[`plan.md`](plan.md). A deterministic Planner/Sender produces one canonical
Kernel trajectory; a task-specific Judger/Receiver always sees the original
question and optionally receives a chronological aligned prefix. Nothing under
`analysis/` imports experimental code from `exp/`.

## Structure

- `configs/kernel_analysis.yaml`: single versioned formal configuration.
- `core/`: schemas, fingerprints, cache integrity, exact Sender/Receiver
  execution, perturbations, evaluation, statistics, and summaries.
- `tasks/`: collection, evaluation, five cache-backed analyses, and report
  entry points.
- `pbs/`: exact JSONL matrix generation, one validated worker, and dependency
  submission.
- `tests/`: protocol, cache, matrix, deterministic-noise, statistics, metric,
  and sentinel-model checks.

Generated caches and results live in `analysis_cache/` and `analysis_result/`
and are intentionally ignored by Git.

## Validate locally

```bash
python -m pytest analysis/tests -q
python analysis/pbs/build_job_matrix.py --dry-run
bash -n analysis/pbs/analysis_job.pbs
bash -n analysis/pbs/submit_analysis.sh
```

The formal dry run reports rows
`18/54/99/144/3/3/3/3/9/1`. A one-question, four-step operational matrix can
be generated without loading a model:

```bash
python analysis/pbs/build_job_matrix.py --dataset aime2024 --smoke --dry-run
```

The authorized perturbation grid is `{0, 0.01, 0.05, 0.10}`. Kernel, soft, and
linear each use the three positive doses, while soft and linear add one clean
control apiece. This yields 11 cells per dataset-seed and 99 formal
perturbation rows.

## Submit

The repository-level PBS submitter builds one dataset/run compute array. Its
formal default has 27 kernel cells (9 datasets x 3 configured seeds) and 3
deterministic STT cells (3 datasets x 1 run). Every cell requests one GPU and
the array throttle is two, so the workflow uses at most two GPUs concurrently.
After the compute array succeeds, one dependent finalize job performs all
cache-only analyses and builds both reports:

```bash
bash analysis.sh
bash analysis.sh --kernel
bash analysis.sh --stt --smoke --dataset aime2024
bash analysis.sh --all --smoke --dataset aime2024 --dry-run
```

Use `--kernel` or `--stt` for one protocol, and `--stage` for one common phase.
Set `ANALYSIS_MAX_GPUS=1` to reduce concurrency; values above two are rejected.
The submitter must run on a PBS login node with `qsub`; `--dry-run` is safe
locally. A bundle treats task exit code `10` as a validated resumable cache hit.
Every submission writes immutable matrices and manifests below a distinct
`analysis/jobs/<run-id>/` directory, so a later submission cannot overwrite the
inputs of an array that is still queued or running.

All model work must use the cluster scheduler:

```bash
bash analysis/pbs/submit_analysis.sh --stage all
bash analysis/pbs/submit_analysis.sh --stage collect
bash analysis/pbs/submit_analysis.sh --stage evaluate
bash analysis/pbs/submit_analysis.sh --stage analyze
bash analysis/pbs/submit_analysis.sh --stage model-pairs
bash analysis/pbs/submit_analysis.sh --dataset aime2024 --smoke
```

Each Python entry accepts `--config` and either a JSON `--job-spec` or a
one-based `--job-matrix`/`--job-index` selection. Exit code `10` means a fully
validated cache hit; the PBS worker records it as `SKIPPED` and returns success
for dependencies. Analysis entry points are strict cache-only consumers.
