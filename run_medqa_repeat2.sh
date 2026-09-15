#!/bin/bash
# Re-run the Qwen3-8B MedQA row from docs/table_new.tex.
# Six array jobs cover Single, TextMAS, Linear, Kernel, Kernel-ES, and Soft.
# Every configuration uses the sequential prompt and runs two repetitions.
# The MAS methods use the repository's default four-agent team.
#
# Submit with: bash run_medqa_repeat2.sh
#PBS -N x_medqa_r2
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-6%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
TIMES=2
RUN_TAG="${RUN_TAG:-medqa_rerun2_$(date +%Y%m%d_%H%M%S)}"
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_${RUN_TAG}.txt}"

KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"

# These are the six data columns in the MedQA row of docs/table_new.tex.
CONFIGS=(
    "baseline|identical"                  # Single
    "text_mas|identical"                  # TextMAS
    "latent_mas|linear"                   # Linear
    "latent_mas|kernel"                   # Kernel
    "latent_mas|kernel_early_stopping"    # Kernel-ES
    "latent_mas|soft"                     # Soft
)
TOTAL_COUNT=${#CONFIGS[@]}

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT must be a positive integer, got: ${MAX_CONCURRENT}" >&2
    exit 2
fi

# A direct invocation submits the array. Each array worker owns one GPU and
# runs both repetitions of exactly one table configuration.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH." >&2
        exit 127
    fi

    VARIABLES="RUN_TAG=${RUN_TAG},MAX_CONCURRENT=${MAX_CONCURRENT},MAX_SAMPLES=${MAX_SAMPLES},TIMES=${TIMES},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE},EARLY_STOPPING_LENGTH_THRESHOLD=${EARLY_STOPPING_LENGTH_THRESHOLD},EARLY_STOPPING_ENTROPY_THRESHOLD=${EARLY_STOPPING_ENTROPY_THRESHOLD}"
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -J "1-${TOTAL_COUNT}%${MAX_CONCURRENT}" -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${JOB_ID}: ${TOTAL_COUNT} MedQA sequential configs, two repetitions each, maximum ${MAX_CONCURRENT} concurrent GPU jobs."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[0-9]+$ ]] ||
   (( PBS_ARRAY_INDEX < 1 || PBS_ARRAY_INDEX > TOTAL_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX must be in 1-${TOTAL_COUNT}, got: ${PBS_ARRAY_INDEX}" >&2
    exit 2
fi

IFS='|' read -r CONFIG_METHOD CONFIG_ALIGNMENT \
    <<< "${CONFIGS[$((PBS_ARRAY_INDEX - 1))]}"

TASK=medqa
MODEL_NAME="Qwen/Qwen3-8B"
CONFIG_PROMPT=sequential
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
if [[ "${CONFIG_METHOD}" == "latent_mas" ]]; then
    STATE_METHOD="${CONFIG_METHOD}_${CONFIG_ALIGNMENT}"
else
    STATE_METHOD="${CONFIG_METHOD}"
fi
STATE_DIR="${SUBMIT_DIR}/state/${RUN_TAG}"
STATE_FILE="${STATE_DIR}/${TASK}_${STATE_METHOD}_${CONFIG_PROMPT}_${MODEL_SLUG}_state.txt"
RUN_TIME="${RUN_TAG}"
mkdir -p "${STATE_DIR}"

append_progress() {
    local status="$1"
    local detail="${2//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    (
        flock -x 9
        if [[ ! -s "${PROGRESS_FILE}" ]]; then
            printf 'timestamp\tjob_id\tarray_index\tdataset\tmethod\tprompt\talignment\tmodel\trepetitions\tstatus\tdetail\n' >&9
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${PBS_JOBID:-local}" "${PBS_ARRAY_INDEX}" \
            "${TASK}" "${CONFIG_METHOD}" "${CONFIG_PROMPT}" "${CONFIG_ALIGNMENT}" \
            "${MODEL_NAME}" "${TIMES}" "${status}" "${detail}" >&9
    ) 9>> "${PROGRESS_FILE}"
}

RUN_SCRIPT="${SUBMIT_DIR}/run.sh"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: missing run script: ${RUN_SCRIPT}" >&2
    exit 2
fi

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: ${TASK} ${MODEL_NAME} ${CONFIG_METHOD}/${CONFIG_PROMPT}/${CONFIG_ALIGNMENT}, repetitions=${TIMES}"
cd "${SUBMIT_DIR}" || exit 1
append_progress STARTED "state file: ${STATE_FILE}"

export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT STATE_FILE
export MAX_SAMPLES TIMES RUN_TIME
export KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE
export EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD

if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_FILE}"
else
    STATUS=$?
    append_progress FAILED "exit ${STATUS}; state file: ${STATE_FILE}"
    exit "${STATUS}"
fi
