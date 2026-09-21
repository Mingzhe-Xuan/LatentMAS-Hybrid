#!/bin/bash
# Qwen3-8B homogeneous four-agent LatentMAS with sequential prompts and
# linear alignment. One PBS array cell per dataset; run.sh performs three
# repeats (seeds 42, 43, and 44) inside each cell.
#
# Submit with:
#   qsub run_8b_seq_linear.sh
#
# Results:
#   result/<dataset>_latent_mas_linear_sequential_Qwen_Qwen3-8B_<timestamp>/
# Per-repeat logs:
#   logging/<dataset>_latent_mas_linear_sequential_Qwen_Qwen3-8B_<timestamp>/
# Array state logs:
#   state_8b_seq_linear/
#
#PBS -N x_8b_linear
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-6%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
RUN_SCRIPT="${SUBMIT_DIR}/run.sh"

MODEL_NAME="Qwen/Qwen3-8B"
CONFIG_METHOD="latent_mas"
CONFIG_PROMPT="sequential"
CONFIG_ALIGNMENT="linear"
TIMES=3

MAX_SAMPLES="${MAX_SAMPLES:--1}"
RESULT_ROOT="${RESULT_ROOT:-result}"
LOG_ROOT="${LOG_ROOT:-logging}"
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_8b_seq_linear.txt}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"

DATASETS=(
    aime2024
    aime2025
    humanevalplus
    mbppplus
    gpqa
    medqa
)
JOB_COUNT=${#DATASETS[@]}

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: missing run script: ${RUN_SCRIPT}" >&2
    exit 2
fi
if [[ ! "${TIMES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TIMES must be a positive integer, got: ${TIMES}" >&2
    exit 2
fi
if [[ ! "${PBS_ARRAY_INDEX:-}" =~ ^[1-9][0-9]*$ ]] ||
   (( PBS_ARRAY_INDEX > JOB_COUNT )); then
    echo "ERROR: submit this script as a PBS array: qsub run_8b_seq_linear.sh" >&2
    echo "       PBS_ARRAY_INDEX must be in 1-${JOB_COUNT}." >&2
    exit 2
fi

TASK="${DATASETS[$((PBS_ARRAY_INDEX - 1))]}"
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_METHOD="${CONFIG_METHOD}_${CONFIG_ALIGNMENT}"
STATE_FILE="${SUBMIT_DIR}/state_8b_seq_linear/${TASK}_${STATE_METHOD}_${CONFIG_PROMPT}_${MODEL_SLUG}_state.txt"

append_progress() {
    local status="$1"
    local detail="${2//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    (
        flock -x 9
        if [[ ! -s "${PROGRESS_FILE}" ]]; then
            printf 'timestamp\tjob_id\tarray_index\tdataset\tmethod\tprompt\talignment\tmodel\trepeats\tstatus\tdetail\n' >&9
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${PBS_JOBID:-local}" "${PBS_ARRAY_INDEX}" \
            "${TASK}" "${CONFIG_METHOD}" "${CONFIG_PROMPT}" "${CONFIG_ALIGNMENT}" \
            "${MODEL_NAME}" "${TIMES}" "${status}" "${detail}" >&9
    ) 9>> "${PROGRESS_FILE}"
}

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: ${TASK} ${MODEL_NAME} ${CONFIG_METHOD}/${CONFIG_PROMPT}/${CONFIG_ALIGNMENT}, repeats=${TIMES}"
cd "${SUBMIT_DIR}" || exit 1

# latent_mas uses the same MODEL_NAME for planner, critic, refiner, and judger.
# AGENT_MODELS is deliberately unset because this is a homogeneous, non-hybrid run.
unset AGENT_MODELS
export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT TIMES
export STATE_FILE RESULT_ROOT LOG_ROOT MAX_SAMPLES
export ALIGN_RIDGE KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE
export EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD

append_progress STARTED "state file: ${STATE_FILE}"
if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_FILE}"
else
    status=$?
    append_progress FAILED "exit ${status}; state file: ${STATE_FILE}"
    exit "${status}"
fi

