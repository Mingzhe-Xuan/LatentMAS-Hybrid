#!/bin/bash
#PBS -N x_hetero
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-48%3
#PBS -j oe

# Cross-model TextMAS and latent-alignment matrix used to populate
# docs/paper.tex table 2.
# The only agents are a Sender Planner and a Receiver Judger, matching analysis/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
# Heterogeneous runs rerun every selected configuration by default. Set
# FORCE_ALL=false explicitly to reuse successful state files and skip them.
FORCE_ALL="${FORCE_ALL:-true}"
WORKER_MODE="${WORKER_MODE:-false}"
CONFIG_OFFSET="${CONFIG_OFFSET:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-3}"
RESULT_ROOT="${RESULT_ROOT:-result}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
REPETITION_PENALTY=1.05
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_hetero.txt}"

# Heterogeneous Planner -> Judger experiments use the complete hidden-state
# sequence in this exact order:
#   sender prompt states || sender latent-output states || receiver prompt
# Keep these explicit so run.sh cannot switch this matrix to a truncated
# sequential-info-only or latent-only protocol through inherited defaults.
SEQUENTIAL_INFO_ONLY=false
LATENT_ONLY=false

for arg in "$@"; do
    case "${arg}" in
        --force_all) FORCE_ALL=true ;;
        *)
            echo "ERROR: unknown argument: ${arg}" >&2
            echo "Usage: bash run_hetero.sh [--force_all]" >&2
            exit 2
            ;;
    esac
done

# Matches the six datasets retained by docs/table_new.tex.
DATASETS=(aime2024 aime2025 gpqa humanevalplus mbppplus medqa)
SENDERS=("Qwen/Qwen3-14B" "Qwen/Qwen3-8B")
RECEIVERS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
METHODS=(text_mas latent_mas_hybrid latent_mas_hybrid latent_mas_hybrid)
ALIGNMENTS=(identical linear soft kernel)
PROMPT=sequential

DATASET_COUNT=${#DATASETS[@]}
DIRECTION_COUNT=${#SENDERS[@]}
EXPERIMENT_COUNT=${#METHODS[@]}
TOTAL_COUNT=$((DATASET_COUNT * DIRECTION_COUNT * EXPERIMENT_COUNT))

if ! [[ "${MAX_CONCURRENT_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_CONCURRENT_GPUS must be a positive integer." >&2
    exit 2
fi

# Like run_all.sh, a direct invocation submits one dynamic PBS array; each cell
# owns one GPU and one dataset/direction/alignment configuration.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH." >&2
        exit 127
    fi
    variables="FORCE_ALL=${FORCE_ALL},MAX_SAMPLES=${MAX_SAMPLES},MAX_CONCURRENT_GPUS=${MAX_CONCURRENT_GPUS},RESULT_ROOT=${RESULT_ROOT},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE},SEQUENTIAL_INFO_ONLY=${SEQUENTIAL_INFO_ONLY},LATENT_ONLY=${LATENT_ONLY},REPETITION_PENALTY=${REPETITION_PENALTY}"
    job_id="$(cd "${SCRIPT_DIR}" && qsub -J "1-${TOTAL_COUNT}%${MAX_CONCURRENT_GPUS}" -v "${variables}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${job_id}: ${TOTAL_COUNT} configs, one config per GPU, maximum ${MAX_CONCURRENT_GPUS} concurrent GPU jobs."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[1-9][0-9]*$ ]] || (( PBS_ARRAY_INDEX > TOTAL_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX must be in 1-${TOTAL_COUNT}." >&2
    exit 2
fi

if [[ "${WORKER_MODE}" != true ]]; then
    WORKER_MODE=true CONFIG_OFFSET=$((PBS_ARRAY_INDEX - 1)) bash "${BASH_SOURCE[0]}"
    exit $?
fi
if ! [[ "${CONFIG_OFFSET}" =~ ^[0-9]+$ ]] || (( CONFIG_OFFSET >= TOTAL_COUNT )); then
    echo "ERROR: CONFIG_OFFSET must be in 0-$((TOTAL_COUNT - 1))." >&2
    exit 2
fi

EXPERIMENT_INDEX=$((CONFIG_OFFSET % EXPERIMENT_COUNT))
DATASET_DIRECTION_INDEX=$((CONFIG_OFFSET / EXPERIMENT_COUNT))
DATASET_INDEX=$((DATASET_DIRECTION_INDEX % DATASET_COUNT))
DIRECTION_INDEX=$((DATASET_DIRECTION_INDEX / DATASET_COUNT))

TASK="${DATASETS[${DATASET_INDEX}]}"
SENDER_MODEL="${SENDERS[${DIRECTION_INDEX}]}"
RECEIVER_MODEL="${RECEIVERS[${DIRECTION_INDEX}]}"
CONFIG_METHOD="${METHODS[${EXPERIMENT_INDEX}]}"
CONFIG_ALIGNMENT="${ALIGNMENTS[${EXPERIMENT_INDEX}]}"
CONFIG_PROMPT="${PROMPT}"
MODEL_NAME="${SENDER_MODEL}"
AGENT_MODELS="${SENDER_MODEL} ${RECEIVER_MODEL}"

sender_slug="$(printf '%s' "${SENDER_MODEL}" | tr -c 'A-Za-z0-9._-' '_')"
receiver_slug="$(printf '%s' "${RECEIVER_MODEL}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_DIR="${SUBMIT_DIR}/state/hetero"
if [[ "${CONFIG_METHOD}" = "text_mas" ]]; then
    state_config_slug=textmas
else
    state_config_slug="${CONFIG_ALIGNMENT}"
fi
STATE_PATH="${STATE_DIR}/${TASK}_${state_config_slug}_${sender_slug}_to_${receiver_slug}_state.txt"
mkdir -p "${STATE_DIR}"

append_progress() {
    local status="$1"
    local detail="${2//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    (
        flock -x 9
        if [[ ! -s "${PROGRESS_FILE}" ]]; then
            printf 'timestamp\tjob_id\tarray_index\tdataset\talignment\tsender\treceiver\tstatus\tdetail\n' >&9
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${PBS_JOBID:-local}" "${PBS_ARRAY_INDEX}" \
            "${TASK}" "${CONFIG_ALIGNMENT}" "${SENDER_MODEL}" "${RECEIVER_MODEL}" \
            "${status}" "${detail}" >&9
    ) 9>> "${PROGRESS_FILE}"
}

state_file_completed() {
    [[ -f "${STATE_PATH}" ]] && [[ "$(tail -n 1 "${STATE_PATH}")" == "Exit status: 0" ]]
}

if [[ "${FORCE_ALL}" != true ]] && state_file_completed; then
    append_progress SKIPPED "completed state file: ${STATE_PATH}"
    echo "Skipped completed config: ${STATE_PATH}"
    exit 0
fi

RUN_SCRIPT="${SUBMIT_DIR}/run.sh"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: missing run script: ${RUN_SCRIPT}" >&2
    exit 2
fi

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: ${TASK} ${SENDER_MODEL} -> ${RECEIVER_MODEL} ${CONFIG_METHOD}/${CONFIG_ALIGNMENT}"
cd "${SUBMIT_DIR}" || exit 1
append_progress STARTED "state file: ${STATE_PATH}"
export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME AGENT_MODELS CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT
export STATE_FILE="${STATE_PATH}" MAX_SAMPLES RESULT_ROOT
export KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE
export SEQUENTIAL_INFO_ONLY LATENT_ONLY
export REPETITION_PENALTY
if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_PATH}"
else
    status=$?
    append_progress FAILED "exit ${status}; state file: ${STATE_PATH}"
    exit "${status}"
fi
