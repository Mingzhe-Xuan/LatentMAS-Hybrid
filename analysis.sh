#!/usr/bin/env bash
## Job name and per-array-cell resources. As in run_all.sh, the dynamic array
## range is supplied by the submit branch with one qsub -J invocation.
#PBS -N analysis
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

###############################################################################
# analysis.sh - self-submitting single PBS dataset/run array for analysis/
#
# The login-node branch uses only a standard-library Python interpreter to
# build manifests and submit the array. Module loading and virtual-environment
# activation are confined to compute/finalize PBS workers below.
#
# Default formal submission:
#    9 kernel cells = 3 primary datasets x 3 configured seeds
#    3 STT cells    = 3 datasets x 1 deterministic run
# At most three one-GPU array cells run concurrently. The last successful cell
# performs cache-only analyses and builds both reports after a file barrier
# confirms that every compute cell succeeded.
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
  --all-datasets            Run kernel analysis on all nine datasets
  --smoke                   Use isolated smoke matrices
  --max-samples INT         STT smoke sample count (requires --smoke)
  --device DEVICE           Task device (default: cuda)
  --dry-run                 Print matrix/array summary without qsub

Environment overrides:
  ANALYSIS_MAX_GPUS         Array concurrency, 1, 2, or 3 (default: 3)
  ANALYSIS_CACHE_ROOT       Cache root (default: analysis_cache)
  ANALYSIS_RESULT_ROOT      Result root (default: analysis_result)
  ANALYSIS_ALL_DATASETS     true restores all nine kernel datasets
  ANALYSIS_PYTHON           Standard-library Python used to build manifests
  ANALYSIS_EXTRA_ARGS       Extra flags passed to every task CLI
EOF
}

ANALYSIS_TARGET="${ANALYSIS_TARGET:-all}"
ANALYSIS_STAGE="${ANALYSIS_STAGE:-all}"
DATASET="${DATASET:-}"
ALL_DATASETS_MODE="${ANALYSIS_ALL_DATASETS:-false}"
ANALYSIS_SMOKE="${ANALYSIS_SMOKE:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-false}"
ANALYSIS_MAX_GPUS="${ANALYSIS_MAX_GPUS:-3}"
ANALYSIS_EXECUTION_MODE="${ANALYSIS_EXECUTION_MODE:-submit}"
STATE_ROOT="${STATE_ROOT:-state/analysis}"

run_pbs_worker() {
    local manifest array_index state_path job_slug compute_status

    case "${ANALYSIS_EXECUTION_MODE}" in
        compute)
            manifest="${RUN_MANIFEST:-}"
            array_index="${PBS_ARRAY_INDEX:-0}"
            [[ -n "${manifest}" ]] || {
                echo "ERROR: RUN_MANIFEST is required in compute mode" >&2
                return 2
            }
            [[ "${array_index}" =~ ^[1-9][0-9]*$ ]] || {
                echo "ERROR: PBS_ARRAY_INDEX must be a positive integer" >&2
                return 2
            }
            [[ "${ARRAY_SIZE:-}" =~ ^[1-9][0-9]*$ ]] || {
                echo "ERROR: ARRAY_SIZE must be a positive integer" >&2
                return 2
            }
            (( array_index <= ARRAY_SIZE )) || {
                echo "ERROR: PBS_ARRAY_INDEX exceeds ARRAY_SIZE=${ARRAY_SIZE}" >&2
                return 2
            }
            [[ "${FINALIZE_ROWS:-0}" =~ ^[0-9]+$ ]] || {
                echo "ERROR: FINALIZE_ROWS must be a non-negative integer" >&2
                return 2
            }
            [[ "${ANALYSIS_RUN_ID:-}" =~ ^[A-Za-z0-9._-]+$ ]] || {
                echo "ERROR: ANALYSIS_RUN_ID must use only A-Za-z0-9._-" >&2
                return 2
            }
            if (( FINALIZE_ROWS > 0 )); then
                [[ -n "${FINALIZE_MANIFEST:-}" ]] || {
                    echo "ERROR: FINALIZE_MANIFEST is required when FINALIZE_ROWS > 0" >&2
                    return 2
                }
            fi
            ;;
        finalize)
            manifest="${FINALIZE_MANIFEST:-}"
            array_index=1
            [[ -n "${manifest}" ]] || {
                echo "ERROR: FINALIZE_MANIFEST is required in finalize mode" >&2
                return 2
            }
            ;;
        *)
            echo "ERROR: invalid ANALYSIS_EXECUTION_MODE=${ANALYSIS_EXECUTION_MODE}" >&2
            return 2
            ;;
    esac

    module purge
    module load "${ANALYSIS_PYTHON_MODULE:-python/3.12.13}"
    local analysis_venv="${ANALYSIS_VENV:-/home/n2501945g/LatentMAS-Hybrid/.venv}"
    source "${analysis_venv}/bin/activate"
    cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}" || return 1
    [[ -d analysis ]] || { echo "ERROR: analysis directory not found" >&2; return 1; }

    export PYTHONUNBUFFERED=1
    export HF_HOME="${HF_HOME:-/home/n2501945g/.cache/huggingface}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
    if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
        local gpu_count
        gpu_count=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((gpu_count - 1))")
        export CUDA_VISIBLE_DEVICES
    fi

    job_slug="$(printf '%s' "${PBS_JOBID:-local}" | tr -c 'A-Za-z0-9._-' '_')"
    if [[ "${ANALYSIS_EXECUTION_MODE}" == compute ]]; then
        state_path="${STATE_ROOT}/dataset_runs/${job_slug}_${array_index}.log"
    else
        state_path="${STATE_ROOT}/finalize/${job_slug}.log"
    fi
    mkdir -p "$(dirname "${state_path}")"

    compute_status=0
    {
        echo "mode/job/index: ${ANALYSIS_EXECUTION_MODE}/${PBS_JOBID:-local}/${array_index}"
        echo "host: $(hostname)"
        echo "revision: $(git rev-parse HEAD)"
        echo "started: $(date --iso-8601=seconds)"
        echo "python: $(python3 --version 2>&1)"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi -L
        python3 analysis/pbs/run_dataset_bundle.py --manifest "${manifest}" \
            --index "${array_index}" --cache-root "${ANALYSIS_CACHE_ROOT:-analysis_cache}" \
            --result-root "${ANALYSIS_RESULT_ROOT:-analysis_result}" \
            --state-root "${STATE_ROOT}" --device "${DEVICE:-cuda}"
        echo "finished: $(date --iso-8601=seconds)"
    } > "${state_path}" 2>&1 || compute_status=$?

    if [[ "${ANALYSIS_EXECUTION_MODE}" == finalize ]]; then
        return "${compute_status}"
    fi

    local coordination_dir="${STATE_ROOT}/coordination/${ANALYSIS_RUN_ID}"
    local marker_path="${coordination_dir}/cell_${array_index}.status"
    mkdir -p "${coordination_dir}"
    printf '%s\n' "${compute_status}" > "${marker_path}.tmp"
    mv "${marker_path}.tmp" "${marker_path}"
    (( compute_status == 0 )) || return "${compute_status}"
    (( FINALIZE_ROWS > 0 )) || return 0

    claim_finalizer() {
        (
            flock -x 9
            shopt -s nullglob
            local -a markers=("${coordination_dir}"/cell_*.status)
            local marker status
            (( ${#markers[@]} == ARRAY_SIZE )) || exit 1
            [[ ! -e "${coordination_dir}/finalize.started" ]] || exit 1
            for marker in "${markers[@]}"; do
                read -r status < "${marker}"
                if [[ "${status}" != 0 ]]; then
                    : > "${coordination_dir}/finalize.blocked"
                    exit 1
                fi
            done
            : > "${coordination_dir}/finalize.started"
        ) 9>> "${coordination_dir}/barrier.lock"
    }

    if claim_finalizer; then
        local finalize_state="${STATE_ROOT}/finalize/${job_slug}_${array_index}.log"
        local finalize_status=0
        mkdir -p "$(dirname "${finalize_state}")"
        {
            echo "mode/job/index: finalize/${PBS_JOBID:-local}/${array_index}"
            echo "started: $(date --iso-8601=seconds)"
            python3 analysis/pbs/run_dataset_bundle.py --manifest "${FINALIZE_MANIFEST}" \
                --index 1 --cache-root "${ANALYSIS_CACHE_ROOT:-analysis_cache}" \
                --result-root "${ANALYSIS_RESULT_ROOT:-analysis_result}" \
                --state-root "${STATE_ROOT}" --device "${DEVICE:-cuda}"
            echo "finished: $(date --iso-8601=seconds)"
        } > "${finalize_state}" 2>&1 || finalize_status=$?
        printf '%s\n' "${finalize_status}" > "${coordination_dir}/finalize.status"
        if (( finalize_status == 0 )); then
            : > "${coordination_dir}/finalize.done"
        else
            : > "${coordination_dir}/finalize.failed"
        fi
        return "${finalize_status}"
    fi
}

case "${ANALYSIS_EXECUTION_MODE}" in
    compute|finalize)
        [[ $# -eq 0 ]] || {
            echo "ERROR: PBS worker modes do not accept command-line arguments" >&2
            exit 2
        }
        run_pbs_worker
        exit $?
        ;;
    submit) ;;
    *) echo "ERROR: invalid ANALYSIS_EXECUTION_MODE=${ANALYSIS_EXECUTION_MODE}" >&2; exit 2 ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) ANALYSIS_TARGET="all"; shift ;;
        --kernel) ANALYSIS_TARGET="kernel"; shift ;;
        --stt) ANALYSIS_TARGET="stt"; shift ;;
        --stage) ANALYSIS_STAGE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --all-datasets) ALL_DATASETS_MODE="true"; shift ;;
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
case "${ALL_DATASETS_MODE}" in true|false) ;; *) echo "ERROR: ANALYSIS_ALL_DATASETS must be true or false" >&2; exit 2 ;; esac
case "${DRY_RUN}" in true|false) ;; *) echo "ERROR: DRY_RUN must be true or false" >&2; exit 2 ;; esac
case "${ANALYSIS_MAX_GPUS}" in 1|2|3) ;; *) echo "ERROR: ANALYSIS_MAX_GPUS must be 1, 2, or 3" >&2; exit 2 ;; esac
if [[ -n "${MAX_SAMPLES}" ]]; then
    [[ "${ANALYSIS_SMOKE}" == true ]] || { echo "ERROR: --max-samples requires --smoke" >&2; exit 2; }
    [[ "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --max-samples must be positive" >&2; exit 2; }
fi
if [[ "${ALL_DATASETS_MODE}" == true && -n "${DATASET}" ]]; then
    echo "ERROR: --all-datasets conflicts with --dataset" >&2
    exit 2
fi
if [[ "${ALL_DATASETS_MODE}" == true && "${ANALYSIS_TARGET}" == stt ]]; then
    echo "ERROR: --all-datasets is only valid for --all or --kernel" >&2
    exit 2
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
BUILD=("${PYTHON_BIN}" -S analysis/pbs/build_dataset_run_matrix.py
       --target "${ANALYSIS_TARGET}" --stage "${ANALYSIS_STAGE}" --output "${JOB_DIR}")
[[ -n "${DATASET}" ]] && BUILD+=(--dataset "${DATASET}")
[[ "${ALL_DATASETS_MODE}" == true ]] && BUILD+=(--all-datasets)
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
COMMON_EXPORTS="ANALYSIS_CACHE_ROOT=${ANALYSIS_CACHE_ROOT:-analysis_cache},ANALYSIS_RESULT_ROOT=${ANALYSIS_RESULT_ROOT:-analysis_result},STATE_ROOT=${STATE_ROOT},DEVICE=${DEVICE},ANALYSIS_RUN_ID=${RUN_ID}"
if [[ -n "${ANALYSIS_EXTRA_ARGS:-}" ]]; then COMMON_EXPORTS+=",ANALYSIS_EXTRA_ARGS=${ANALYSIS_EXTRA_ARGS}"; fi
if [[ -n "${ANALYSIS_VENV:-}" ]]; then COMMON_EXPORTS+=",ANALYSIS_VENV=${ANALYSIS_VENV}"; fi
if [[ -n "${ANALYSIS_PYTHON_MODULE:-}" ]]; then COMMON_EXPORTS+=",ANALYSIS_PYTHON_MODULE=${ANALYSIS_PYTHON_MODULE}"; fi

ARRAY_JOB=""
if (( COMPUTE_ROWS > 0 )); then
    ARRAY_JOB=$(qsub -J "1-${COMPUTE_ROWS}%${ANALYSIS_MAX_GPUS}" \
        -v "${COMMON_EXPORTS},ANALYSIS_EXECUTION_MODE=compute,RUN_MANIFEST=${COMPUTE_MANIFEST},ARRAY_SIZE=${COMPUTE_ROWS},FINALIZE_ROWS=${FINALIZE_ROWS},FINALIZE_MANIFEST=${FINALIZE_MANIFEST}" \
        "${BASH_SOURCE[0]}")
elif (( FINALIZE_ROWS > 0 )); then
    ARRAY_JOB=$(qsub -J "1-${FINALIZE_ROWS}%${ANALYSIS_MAX_GPUS}" \
        -v "${COMMON_EXPORTS},ANALYSIS_EXECUTION_MODE=finalize,FINALIZE_MANIFEST=${FINALIZE_MANIFEST}" \
        "${BASH_SOURCE[0]}")
else
    echo "No analysis work matched the requested target and stage."
    exit 0
fi

echo "Submitted ${ARRAY_JOB}: ${COMPUTE_ROWS} compute rows, ${FINALIZE_ROWS} in-array finalize bundle, maximum ${ANALYSIS_MAX_GPUS} concurrent GPU jobs."
