#!/usr/bin/env bash
#SBATCH --job-name=kernel_analysis
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
set -euo pipefail

TASK_NAME="${TASK_NAME:-}"
JOB_MATRIX="${JOB_MATRIX:-}"
CONFIG_PATH="${CONFIG_PATH:-analysis/configs/kernel_analysis.yaml}"
CACHE_MODE="${CACHE_MODE:-reuse}"
ANALYSIS_CACHE_ROOT="${ANALYSIS_CACHE_ROOT:-analysis_cache}"
ANALYSIS_RESULT_ROOT="${ANALYSIS_RESULT_ROOT:-analysis_result}"
STATE_ROOT="${STATE_ROOT:-state/analysis}"
LEDGER_PATH="${LEDGER_PATH:-${STATE_ROOT}/progress.tsv}"
DEVICE="${DEVICE:-cuda}"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_JOB_ID:-local}"

case "${TASK_NAME}" in
  collect_sender_trajectories|evaluate_kernel_scaling|evaluate_perturbation_stability|evaluate_sender_receiver_performance|analyze_logit_entropy|analyze_kernel_scaling|analyze_aligned_state_variance|analyze_perturbation_stability|analyze_sender_receiver_performance|build_kernel_analysis_report) ;;
  *) echo "ERROR: unknown TASK_NAME=${TASK_NAME}" >&2; exit 2 ;;
esac
case "${CACHE_MODE}" in reuse|cache-only|force) ;; *) echo "ERROR: invalid CACHE_MODE=${CACHE_MODE}" >&2; exit 2 ;; esac
if ! [[ "${ARRAY_INDEX}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be a positive integer" >&2
  exit 2
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}" || exit 1
if [ ! -d analysis ] && [ -d LatentMAS-Hybrid/analysis ]; then cd LatentMAS-Hybrid || exit 1; fi
if [ ! -d analysis ]; then echo "ERROR: analysis directory not found" >&2; exit 1; fi
REPO_ROOT="$(pwd -P)"
VENV_PATH="${ANALYSIS_VENV:-${REPO_ROOT}/.venv}"
if [ ! -x "${VENV_PATH}/bin/python" ]; then
  echo "ERROR: virtual environment not found: ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

MATRIX_ABS="$(realpath -e "${JOB_MATRIX}")" || { echo "ERROR: matrix not found" >&2; exit 2; }
CONFIG_ABS="$(realpath -e "${CONFIG_PATH}")" || { echo "ERROR: config not found" >&2; exit 2; }
case "${MATRIX_ABS}" in "${REPO_ROOT}"/analysis/jobs/*) ;; *) echo "ERROR: JOB_MATRIX must be under analysis/jobs/." >&2; exit 2 ;; esac
case "${CONFIG_ABS}" in "${REPO_ROOT}"/analysis/configs/*) ;; *) echo "ERROR: CONFIG_PATH must be under analysis/configs/." >&2; exit 2 ;; esac
ROW_COUNT="$(python -c 'from pathlib import Path; import sys; lines=Path(sys.argv[1]).read_text(encoding="utf-8").splitlines(); assert all(lines), "blank matrix line"; print(len(lines))' "${MATRIX_ABS}")"
if (( ARRAY_INDEX > ROW_COUNT )); then
  echo "ERROR: array index ${ARRAY_INDEX} exceeds ${ROW_COUNT} rows" >&2
  exit 2
fi

ENTRY="analysis/tasks/${TASK_NAME}.py"
if [ ! -f "${ENTRY}" ]; then echo "ERROR: missing task entry ${ENTRY}" >&2; exit 2; fi
STATE_DIR="${STATE_ROOT}/${TASK_NAME}"
STATE_PATH="${STATE_DIR}/${JOB_ID}_${ARRAY_INDEX}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LEDGER_PATH}")"

append_progress() {
  local status="$1" detail="${2//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  ( flock -x 9
    if [[ ! -s "${LEDGER_PATH}" ]]; then printf 'timestamp\tjob_id\tarray_index\ttask\tstatus\tdetail\n' >&9; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${JOB_ID}" "${ARRAY_INDEX}" "${TASK_NAME}" "${status}" "${detail}" >&9
  ) 9>> "${LEDGER_PATH}"
}

{
  echo "job/index: ${JOB_ID}/${ARRAY_INDEX}"
  echo "task: ${TASK_NAME}"
  echo "host: $(hostname)"
  echo "revision: $(git rev-parse HEAD)"
  echo "started: $(date --iso-8601=seconds)"
  echo "python: $(python --version 2>&1)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi -L
} > "${STATE_PATH}"

ARGS=(--config "${CONFIG_ABS}" --job-matrix "${MATRIX_ABS}" --job-index "${ARRAY_INDEX}" --cache-root "${ANALYSIS_CACHE_ROOT}" --result-root "${ANALYSIS_RESULT_ROOT}" --device "${DEVICE}")
case "${CACHE_MODE}" in cache-only) ARGS+=(--cache-only) ;; force) ARGS+=(--force) ;; esac
append_progress STARTED "state=${STATE_PATH}"
set +e
python "${ENTRY}" "${ARGS[@]}" >> "${STATE_PATH}" 2>&1
STATUS=$?
set -e
if (( STATUS == 0 )); then append_progress COMPLETED "state=${STATE_PATH}"
elif (( STATUS == 10 )); then append_progress SKIPPED "validated cache hit; state=${STATE_PATH}"; STATUS=0
else append_progress FAILED "exit=${STATUS}; state=${STATE_PATH}"; fi
echo "finished: $(date --iso-8601=seconds); exit=${STATUS}" >> "${STATE_PATH}"
exit "${STATUS}"
