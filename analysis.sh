#!/usr/bin/env bash
###############################################################################
# analysis.sh - PBS dataset/run array submitter for analysis/
#
# Default formal submission:
#   27 kernel cells = 9 datasets x 3 configured seeds
#    3 STT cells    = 3 datasets x 1 deterministic run
# At most two one-GPU array cells run concurrently. A dependent one-GPU
# finalize job performs cache-only analyses and builds both reports.
#
# Examples:
#   bash analysis.sh
#   bash analysis.sh --kernel
#   bash analysis.sh --stt --smoke --dataset aime2024
#   bash analysis.sh --all --smoke --dataset aime2024 --dry-run
###############################################################################

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash analysis.sh [--all|--kernel|--stt] [OPTIONS]

Targets (default: all):
  --all                     Submit kernel and STT dataset/run cells
  --kernel                  Submit only kernel dataset/run cells
  --stt                     Submit only bidirectional STT dataset cells

Options:
  --stage NAME              all, collect, evaluate, analyze, or report
  --dataset NAME            Restrict submission to one dataset
  --smoke                   Use isolated smoke matrices
  --max-samples INT         STT smoke sample count (requires --smoke)
  --device DEVICE           Task device (default: cuda)
  --dry-run                 Print matrix/array summary without qsub

Environment overrides:
  ANALYSIS_MAX_GPUS         Array concurrency, 1 or 2 (default: 2)
  ANALYSIS_CACHE_ROOT       Cache root (default: analysis_cache)
  ANALYSIS_RESULT_ROOT      Result root (default: analysis_result)
  ANALYSIS_PYTHON           Python used to build manifests
  ANALYSIS_EXTRA_ARGS       Extra flags passed to every task CLI
  PBS_DEPENDENCY_OPERATOR   afterokarray (default) or afterok
EOF
}

ANALYSIS_TARGET="${ANALYSIS_TARGET:-all}"
ANALYSIS_STAGE="${ANALYSIS_STAGE:-all}"
DATASET="${DATASET:-}"
ANALYSIS_SMOKE="${ANALYSIS_SMOKE:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-false}"
ANALYSIS_MAX_GPUS="${ANALYSIS_MAX_GPUS:-2}"
DEPENDENCY_OPERATOR="${PBS_DEPENDENCY_OPERATOR:-afterokarray}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) ANALYSIS_TARGET="all"; shift ;;
        --kernel) ANALYSIS_TARGET="kernel"; shift ;;
        --stt) ANALYSIS_TARGET="stt"; shift ;;
        --stage) ANALYSIS_STAGE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --smoke) ANALYSIS_SMOKE="true"; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "${ANALYSIS_TARGET}" in all|kernel|stt) ;; *) echo "ERROR: invalid target" >&2; exit 2 ;; esac
case "${ANALYSIS_STAGE}" in all|collect|evaluate|analyze|report) ;; *) echo "ERROR: invalid stage" >&2; exit 2 ;; esac
case "${ANALYSIS_SMOKE}" in true|false) ;; *) echo "ERROR: ANALYSIS_SMOKE must be true or false" >&2; exit 2 ;; esac
case "${DRY_RUN}" in true|false) ;; *) echo "ERROR: DRY_RUN must be true or false" >&2; exit 2 ;; esac
case "${ANALYSIS_MAX_GPUS}" in 1|2) ;; *) echo "ERROR: ANALYSIS_MAX_GPUS must be 1 or 2" >&2; exit 2 ;; esac
case "${DEPENDENCY_OPERATOR}" in afterokarray|afterok) ;; *) echo "ERROR: invalid PBS dependency operator" >&2; exit 2 ;; esac
if [[ -n "${MAX_SAMPLES}" ]]; then
    [[ "${ANALYSIS_SMOKE}" == true ]] || { echo "ERROR: --max-samples requires --smoke" >&2; exit 2; }
    [[ "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --max-samples must be positive" >&2; exit 2; }
fi
if [[ ! -d analysis ]]; then echo "ERROR: run from the repository root" >&2; exit 1; fi

PYTHON_BIN="${ANALYSIS_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
    else echo "ERROR: no Python interpreter; set ANALYSIS_PYTHON" >&2; exit 127
    fi
fi

RUN_ID="${ANALYSIS_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUN_ID="$(printf '%s' "${RUN_ID}" | tr -c 'A-Za-z0-9._-' '_')"
JOB_DIR="analysis/jobs/${RUN_ID}"
BUILD=("${PYTHON_BIN}" analysis/pbs/build_dataset_run_matrix.py
       --target "${ANALYSIS_TARGET}" --stage "${ANALYSIS_STAGE}" --output "${JOB_DIR}")
[[ -n "${DATASET}" ]] && BUILD+=(--dataset "${DATASET}")
[[ "${ANALYSIS_SMOKE}" == true ]] && BUILD+=(--smoke)
[[ -n "${MAX_SAMPLES}" ]] && BUILD+=(--max-samples "${MAX_SAMPLES}")
[[ "${DRY_RUN}" == true ]] && BUILD+=(--dry-run)
"${BUILD[@]}"
[[ "${DRY_RUN}" == true ]] && exit 0

if ! command -v qsub >/dev/null 2>&1; then
    echo "ERROR: qsub is unavailable; run this submitter on a PBS login node" >&2
    exit 127
fi

COMPUTE_MANIFEST="${JOB_DIR}/dataset_runs.jsonl"
FINALIZE_MANIFEST="${JOB_DIR}/analysis_finalize.jsonl"
COMPUTE_ROWS=$(wc -l < "${COMPUTE_MANIFEST}")
FINALIZE_ROWS=$(wc -l < "${FINALIZE_MANIFEST}")
COMMON_EXPORTS="ANALYSIS_CACHE_ROOT=${ANALYSIS_CACHE_ROOT:-analysis_cache},ANALYSIS_RESULT_ROOT=${ANALYSIS_RESULT_ROOT:-analysis_result},DEVICE=${DEVICE}"
if [[ -n "${ANALYSIS_EXTRA_ARGS:-}" ]]; then COMMON_EXPORTS+=",ANALYSIS_EXTRA_ARGS=${ANALYSIS_EXTRA_ARGS}"; fi
if [[ -n "${ANALYSIS_VENV:-}" ]]; then COMMON_EXPORTS+=",ANALYSIS_VENV=${ANALYSIS_VENV}"; fi
if [[ -n "${ANALYSIS_PYTHON_MODULE:-}" ]]; then COMMON_EXPORTS+=",ANALYSIS_PYTHON_MODULE=${ANALYSIS_PYTHON_MODULE}"; fi

COMPUTE_JOB=""
if (( COMPUTE_ROWS > 0 )); then
    COMPUTE_JOB=$(qsub -J "1-${COMPUTE_ROWS}%${ANALYSIS_MAX_GPUS}" \
        -v "${COMMON_EXPORTS},RUN_MANIFEST=${COMPUTE_MANIFEST}" \
        analysis/pbs/analysis_dataset_run.pbs)
fi

FINALIZE_JOB=""
if (( FINALIZE_ROWS > 0 )); then
    FINALIZE_COMMAND=(qsub -v "${COMMON_EXPORTS},FINALIZE_MANIFEST=${FINALIZE_MANIFEST}")
    [[ -n "${COMPUTE_JOB}" ]] && FINALIZE_COMMAND+=(-W "depend=${DEPENDENCY_OPERATOR}:${COMPUTE_JOB}")
    FINALIZE_COMMAND+=(analysis/pbs/analysis_finalize.pbs)
    FINALIZE_JOB=$("${FINALIZE_COMMAND[@]}")
fi

echo "compute_array=${COMPUTE_JOB:-none} rows=${COMPUTE_ROWS} max_concurrent_gpus=${ANALYSIS_MAX_GPUS}"
echo "finalize=${FINALIZE_JOB:-none} rows=${FINALIZE_ROWS} dependency=${COMPUTE_JOB:-none}"
