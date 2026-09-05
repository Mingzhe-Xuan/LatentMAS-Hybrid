#!/bin/bash
#PBS -N x_missing
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-35%1
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
FORCE_ALL="${FORCE_ALL:-false}"
LIST_ONLY=false
# Run at most one missing configuration (and therefore one GPU) at a time.
MAX_CONCURRENT=1
MAX_SAMPLES="${MAX_SAMPLES:--1}"
GENERATE_BS="${GENERATE_BS:-}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_missing.txt}"

for ARG in "$@"; do
    case "${ARG}" in
        --force_all) FORCE_ALL=true ;;
        --list) LIST_ONLY=true ;;
        *)
            echo "ERROR: unknown argument: ${ARG}"
            echo "Usage: bash run_missing.sh [--force_all] [--list]"
            exit 2
            ;;
    esac
done

# Missing LatentMAS cells in docs/table.tex. Each entry is:
# task|model|prompt|alignment
CONFIGS=(
    # Qwen3-8B, sequential
    "aime2024|Qwen/Qwen3-8B|sequential|identical"
    "aime2025|Qwen/Qwen3-8B|sequential|identical"
    "arc_challenge|Qwen/Qwen3-8B|sequential|soft"
    "arc_easy|Qwen/Qwen3-8B|sequential|soft"
    "gpqa|Qwen/Qwen3-8B|sequential|identical"
    "gpqa|Qwen/Qwen3-8B|sequential|soft"
    "gsm8k|Qwen/Qwen3-8B|sequential|soft"
    "mbppplus|Qwen/Qwen3-8B|sequential|identical"

    # Qwen3-8B, hierarchical
    "aime2024|Qwen/Qwen3-8B|hierarchical|identical"
    "aime2024|Qwen/Qwen3-8B|hierarchical|soft"
    "aime2025|Qwen/Qwen3-8B|hierarchical|identical"
    "aime2025|Qwen/Qwen3-8B|hierarchical|soft"
    "arc_challenge|Qwen/Qwen3-8B|hierarchical|soft"
    "arc_easy|Qwen/Qwen3-8B|hierarchical|soft"
    "gpqa|Qwen/Qwen3-8B|hierarchical|identical"
    "gpqa|Qwen/Qwen3-8B|hierarchical|soft"
    "gsm8k|Qwen/Qwen3-8B|hierarchical|soft"
    "humanevalplus|Qwen/Qwen3-8B|hierarchical|soft"
    "medqa|Qwen/Qwen3-8B|hierarchical|soft"

    # Qwen3-14B, sequential
    "aime2024|Qwen/Qwen3-14B|sequential|identical"
    "aime2024|Qwen/Qwen3-14B|sequential|soft"
    "aime2025|Qwen/Qwen3-14B|sequential|identical"
    "arc_challenge|Qwen/Qwen3-14B|sequential|soft"
    "arc_easy|Qwen/Qwen3-14B|sequential|soft"
    "gpqa|Qwen/Qwen3-14B|sequential|identical"
    "gsm8k|Qwen/Qwen3-14B|sequential|soft"
    "mbppplus|Qwen/Qwen3-14B|sequential|soft"

    # Qwen3-14B, hierarchical
    "aime2024|Qwen/Qwen3-14B|hierarchical|identical"
    "aime2025|Qwen/Qwen3-14B|hierarchical|identical"
    "aime2025|Qwen/Qwen3-14B|hierarchical|soft"
    "arc_challenge|Qwen/Qwen3-14B|hierarchical|soft"
    "arc_easy|Qwen/Qwen3-14B|hierarchical|soft"
    "gpqa|Qwen/Qwen3-14B|hierarchical|identical"
    "gsm8k|Qwen/Qwen3-14B|hierarchical|soft"
    "mbppplus|Qwen/Qwen3-14B|hierarchical|soft"
)

TOTAL_COUNT=${#CONFIGS[@]}

if [[ "${LIST_ONLY}" == "true" ]]; then
    for i in "${!CONFIGS[@]}"; do
        printf '%2d  %s\n' "$((i + 1))" "${CONFIGS[${i}]}"
    done
    echo "Total: ${TOTAL_COUNT} missing configurations."
    exit 0
fi

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT must be a positive integer, got: ${MAX_CONCURRENT}"
    exit 2
fi

# A direct invocation submits the PBS array. Array workers execute one missing
# configuration each, so experiments are never packed onto the same GPU.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH."
        exit 127
    fi
    VARIABLES="FORCE_ALL=${FORCE_ALL},MAX_SAMPLES=${MAX_SAMPLES},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE},EARLY_STOPPING_LENGTH_THRESHOLD=${EARLY_STOPPING_LENGTH_THRESHOLD},EARLY_STOPPING_ENTROPY_THRESHOLD=${EARLY_STOPPING_ENTROPY_THRESHOLD}"
    if [[ -n "${GENERATE_BS}" ]]; then
        VARIABLES+=",GENERATE_BS=${GENERATE_BS}"
    fi
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -J "1-${TOTAL_COUNT}%${MAX_CONCURRENT}" -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${JOB_ID}: ${TOTAL_COUNT} missing configs, one config per GPU, maximum ${MAX_CONCURRENT} concurrent GPU jobs, force_all=${FORCE_ALL}."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[0-9]+$ ]] || \
   (( PBS_ARRAY_INDEX < 1 || PBS_ARRAY_INDEX > TOTAL_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX must be in 1-${TOTAL_COUNT}, got: ${PBS_ARRAY_INDEX}"
    exit 2
fi

IFS='|' read -r TASK MODEL_NAME CONFIG_PROMPT CONFIG_ALIGNMENT \
    <<< "${CONFIGS[$((PBS_ARRAY_INDEX - 1))]}"
CONFIG_METHOD=latent_mas
# Unless GENERATE_BS is explicitly supplied, soft runs use half of the task's
# params_dict.json generation_bs. run.sh floors the division and keeps it >= 1.
if [[ "${CONFIG_ALIGNMENT}" == "soft" ]]; then
    GENERATE_BS_DIVISOR=2
else
    GENERATE_BS_DIVISOR=1
fi
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_DIR="${SUBMIT_DIR}/state"
STATE_PATH="${STATE_DIR}/${TASK}_latent_mas_${CONFIG_ALIGNMENT}_${CONFIG_PROMPT}_${MODEL_SLUG}_state.txt"
mkdir -p "${STATE_DIR}"

append_progress() {
    local status="$1"
    local detail="${2//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    (
        flock -x 9
        if [[ ! -s "${PROGRESS_FILE}" ]]; then
            printf 'timestamp\tjob_id\tarray_index\tdataset\tmethod\tprompt\talignment\tmodel\tstatus\tdetail\n' >&9
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${PBS_JOBID:-local}" "${PBS_ARRAY_INDEX}" \
            "${TASK}" "${CONFIG_METHOD}" "${CONFIG_PROMPT}" "${CONFIG_ALIGNMENT}" \
            "${MODEL_NAME}" "${status}" "${detail}" >&9
    ) 9>> "${PROGRESS_FILE}"
}

state_file_completed() {
    [[ -f "${STATE_PATH}" ]] &&
        [[ "$(tail -n 1 "${STATE_PATH}")" == "Exit status: 0" ]]
}

if [[ "${FORCE_ALL}" != "true" ]] && state_file_completed; then
    append_progress SKIPPED "completed state file: ${STATE_PATH}"
    echo "Skipped completed config: ${STATE_PATH}"
    exit 0
fi

RUN_SCRIPT="${SUBMIT_DIR}/run.sh"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: missing run script: ${RUN_SCRIPT}"
    exit 2
fi

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: ${TASK} ${MODEL_NAME} ${CONFIG_METHOD}/${CONFIG_PROMPT}/${CONFIG_ALIGNMENT}"
cd "${SUBMIT_DIR}" || exit 1
append_progress STARTED "state file: ${STATE_PATH}"

export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT STATE_FILE="${STATE_PATH}"
export MAX_SAMPLES GENERATE_BS GENERATE_BS_DIVISOR
export KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD

if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_PATH}"
else
    STATUS=$?
    append_progress FAILED "exit ${STATUS}; state file: ${STATE_PATH}"
    exit "${STATUS}"
fi
