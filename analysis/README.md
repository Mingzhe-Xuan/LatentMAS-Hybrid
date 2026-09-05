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
`18/54/126/144/3/3/3/3/9/1`. A one-question, four-step operational matrix can
be generated without loading a model:

```bash
python analysis/pbs/build_job_matrix.py --dataset aime2024 --smoke --dry-run
```

The plan's perturbation prose lists three positive doses but simultaneously
requires four positive doses per method and exactly 126 cells. The versioned
configuration retains the earlier preregistered `0.025` dose, yielding
`{0, 0.01, 0.025, 0.05, 0.10}` and satisfying the explicit 14-cells-per-
dataset-seed and 126-row acceptance invariants.

## Submit

The repository-level PBS entry point runs the complete kernel protocol followed
by the complete bidirectional STT protocol, strictly serially in one allocation:

```bash
qsub analysis.sh
qsub -v ANALYSIS_TARGET=kernel analysis.sh
qsub -v ANALYSIS_TARGET=stt,ANALYSIS_SMOKE=true,DATASET=aime2024 analysis.sh
bash analysis.sh --all --smoke --dataset aime2024 --dry-run
```

Use `--kernel` or `--stt` for one protocol, and `--stage` for one common phase.
The entry point validates exit code `10` as a resumable cache hit. It refuses to
run model work outside a PBS allocation; `--dry-run` is safe locally.

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
