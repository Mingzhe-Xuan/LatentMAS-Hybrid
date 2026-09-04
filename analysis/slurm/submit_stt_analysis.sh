#!/usr/bin/env bash
set -euo pipefail

STAGE=all
DATASET=""
SMOKE=false
DRY_RUN=false
MAX_RUNNING="${SLURM_MAX_RUNNING:-1}"
while (( $# )); do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --smoke) SMOKE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "${STAGE}" in all|collect|evaluate|analyze|report) ;; *) echo "ERROR: invalid stage ${STAGE}" >&2; exit 2 ;; esac
if ! command -v sbatch >/dev/null 2>&1 && [[ "${DRY_RUN}" != true ]]; then
  echo "ERROR: sbatch is unavailable" >&2
  exit 127
fi

PYTHON_BIN="${ANALYSIS_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
  elif [[ -x .venv/bin/python ]]; then PYTHON_BIN=".venv/bin/python"
  else echo "ERROR: no Python interpreter; set ANALYSIS_PYTHON" >&2; exit 127
  fi
fi

WORKER="analysis/slurm/analysis_job.slurm"
CONFIG_PATH="analysis/configs/bidirectional_stt.yaml"
mkdir -p state/analysis
BUILD=("${PYTHON_BIN}" analysis/pbs/build_stt_job_matrix.py --config "${CONFIG_PATH}" --output analysis/jobs)
[[ -n "${DATASET}" ]] && BUILD+=(--dataset "${DATASET}")
[[ "${SMOKE}" == true ]] && BUILD+=(--smoke)
"${BUILD[@]}"

submit_array() {
  local task_name="$1" matrix="$2" cache_mode="$3" dependency="${4:-}"
  local rows
  rows=$(wc -l < "${matrix}")
  local -a command=(sbatch --parsable --job-name "stt_${task_name}"
    --partition "${SLURM_PARTITION:-compute}" --gres "${SLURM_GRES:-gpu:1}"
    --cpus-per-task "${SLURM_CPUS:-4}" --mem "${SLURM_MEMORY:-64G}"
    --time "${SLURM_TIME:-72:00:00}" --array "1-${rows}%${MAX_RUNNING}"
    --output "state/analysis/slurm-stt-%A_%a.out"
    --export "ALL,TASK_NAME=${task_name},JOB_MATRIX=${matrix},CACHE_MODE=${cache_mode},CONFIG_PATH=${CONFIG_PATH}")
  [[ -n "${dependency}" ]] && command+=(--dependency "afterok:${dependency}")
  command+=("${WORKER}")
  if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' "${command[@]}" >&2; printf '\n' >&2
    printf 'dry-%s' "${task_name}"
  else
    "${command[@]}"
  fi
}

if [[ "${STAGE}" == collect ]]; then submit_array collect_stt_planner_contexts analysis/jobs/stt_planner.jsonl reuse; exit; fi
if [[ "${STAGE}" == evaluate ]]; then submit_array evaluate_bidirectional_stt analysis/jobs/stt_evaluation.jsonl reuse; exit; fi
if [[ "${STAGE}" == analyze ]]; then submit_array analyze_bidirectional_stt analysis/jobs/stt_analysis.jsonl cache-only; exit; fi
if [[ "${STAGE}" == report ]]; then submit_array build_bidirectional_stt_report analysis/jobs/stt_report.jsonl cache-only; exit; fi

PLANNER_JOB="$(submit_array collect_stt_planner_contexts analysis/jobs/stt_planner.jsonl reuse)"
EVALUATION_JOB="$(submit_array evaluate_bidirectional_stt analysis/jobs/stt_evaluation.jsonl reuse "${PLANNER_JOB}")"
ANALYSIS_JOB="$(submit_array analyze_bidirectional_stt analysis/jobs/stt_analysis.jsonl cache-only "${EVALUATION_JOB}")"
REPORT_JOB="$(submit_array build_bidirectional_stt_report analysis/jobs/stt_report.jsonl cache-only "${ANALYSIS_JOB}")"
printf 'planner=%s evaluation=%s analysis=%s report=%s\n' "${PLANNER_JOB}" "${EVALUATION_JOB}" "${ANALYSIS_JOB}" "${REPORT_JOB}"
