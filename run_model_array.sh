#!/bin/bash
# Shared one-model experiment array used by run_ds.sh and run_mistral.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"

: "${EXPERIMENT_MODEL:?EXPERIMENT_MODEL must be set by the model wrapper}"
: "${PROGRESS_FILE:?PROGRESS_FILE must be set by the model wrapper}"
: "${STATE_ROOT:?STATE_ROOT must be set by the model wrapper}"
: "${LOG_ROOT:?LOG_ROOT must be set by the model wrapper}"
: "${RESULT_ROOT:?RESULT_ROOT must be set by the model wrapper}"

FORCE_ALL="${FORCE_ALL:-false}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"

DATASETS=(
    aime2024 aime2025 arc_challenge arc_easy gpqa gsm8k
    humanevalplus mbppplus medqa
)
METHODS=(
    baseline baseline text_mas text_mas
    latent_mas latent_mas latent_mas latent_mas latent_mas
    latent_mas latent_mas latent_mas latent_mas latent_mas
)
PROMPTS=(
    sequential hierarchical sequential hierarchical
    sequential sequential sequential sequential sequential
    hierarchical hierarchical hierarchical hierarchical hierarchical
)
ALIGNMENTS=(
    identical identical identical identical
    identical linear kernel kernel_early_stopping soft
    identical linear kernel kernel_early_stopping soft
)

CONFIG_COUNT=${#METHODS[@]}
TOTAL_COUNT=$((${#DATASETS[@]} * CONFIG_COUNT))

if [[ ! "${PBS_ARRAY_INDEX:-}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: submit this file through its model wrapper with qsub." >&2
    exit 2
fi

OFFSET=$((PBS_ARRAY_INDEX - 1))
if (( OFFSET < 0 || OFFSET >= TOTAL_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX} is outside 1-${TOTAL_COUNT}." >&2
    exit 2
fi

DATASET_INDEX=$((OFFSET / CONFIG_COUNT))
CONFIG_INDEX=$((OFFSET % CONFIG_COUNT))
TASK="${DATASETS[${DATASET_INDEX}]}"
CONFIG_METHOD="${METHODS[${CONFIG_INDEX}]}"
CONFIG_PROMPT="${PROMPTS[${CONFIG_INDEX}]}"
CONFIG_ALIGNMENT="${ALIGNMENTS[${CONFIG_INDEX}]}"
MODEL_NAME="${EXPERIMENT_MODEL}"

STATE_METHOD="${CONFIG_METHOD}"
if [[ "${CONFIG_METHOD}" == "latent_mas" ]]; then
    STATE_METHOD="${CONFIG_METHOD}_${CONFIG_ALIGNMENT}"
fi
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_FILE="${STATE_ROOT}/${TASK}_${STATE_METHOD}_${CONFIG_PROMPT}_${MODEL_SLUG}_state.txt"

mkdir -p "${STATE_ROOT}" "$(dirname "${PROGRESS_FILE}")"

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
    [[ -f "${STATE_FILE}" ]] &&
        [[ "$(tail -n 1 "${STATE_FILE}")" == "Exit status: 0" ]]
}

if [[ "${FORCE_ALL}" != "true" ]] && state_file_completed; then
    append_progress SKIPPED "completed state file: ${STATE_FILE}"
    echo "Skipped completed config: ${STATE_FILE}"
    exit 0
fi

RUN_SCRIPT="${SUBMIT_DIR}/run.sh"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: missing run script: ${RUN_SCRIPT}" >&2
    exit 2
fi

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: ${TASK} ${MODEL_NAME} ${CONFIG_METHOD}/${CONFIG_PROMPT}/${CONFIG_ALIGNMENT}"
cd "${SUBMIT_DIR}"

export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT STATE_FILE
export LOG_ROOT RESULT_ROOT MAX_SAMPLES
export KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE
export EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD

append_progress STARTED "state file: ${STATE_FILE}"
if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_FILE}"
else
    STATUS=$?
    append_progress FAILED "exit ${STATUS}; state file: ${STATE_FILE}"
    exit "${STATUS}"
fi
