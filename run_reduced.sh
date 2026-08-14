#!/bin/bash
# Reduced LatentMAS PBS array: 2 models x 9 datasets x 3 alignments x 2 prompts.
# Submit with: qsub run_reduced.sh
#PBS -N x_reduced
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-108%3
#PBS -j oe

set -euo pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_reduced.txt}"
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
MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
ALIGNMENTS=(kernel kernel_early_stopping soft)
PROMPTS=(sequential hierarchical)

if [[ ! "${PBS_ARRAY_INDEX:-}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: run_reduced.sh must be submitted as a PBS array: qsub run_reduced.sh" >&2
    exit 2
fi

OFFSET=$((PBS_ARRAY_INDEX - 1))
CONFIGS_PER_MODEL_DATASET=$((${#ALIGNMENTS[@]} * ${#PROMPTS[@]}))
TOTAL_COUNT=$((${#MODELS[@]} * ${#DATASETS[@]} * CONFIGS_PER_MODEL_DATASET))
if (( OFFSET < 0 || OFFSET >= TOTAL_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX} is outside 1-${TOTAL_COUNT}" >&2
    exit 2
fi

MODEL_DATASET_INDEX=$((OFFSET / CONFIGS_PER_MODEL_DATASET))
CONFIG_INDEX=$((OFFSET % CONFIGS_PER_MODEL_DATASET))
MODEL_NAME="${MODELS[$((MODEL_DATASET_INDEX / ${#DATASETS[@]}))]}"
TASK="${DATASETS[$((MODEL_DATASET_INDEX % ${#DATASETS[@]}))]}"
CONFIG_ALIGNMENT="${ALIGNMENTS[$((CONFIG_INDEX / ${#PROMPTS[@]}))]}"
CONFIG_PROMPT="${PROMPTS[$((CONFIG_INDEX % ${#PROMPTS[@]}))]}"
STATE_METHOD="latent_mas_${CONFIG_ALIGNMENT}"
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_FILE="${SUBMIT_DIR}/state_reduced/${TASK}_${STATE_METHOD}_${CONFIG_PROMPT}_${MODEL_SLUG}_state.txt"

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
            "${TASK}" "latent_mas" "${CONFIG_PROMPT}" "${CONFIG_ALIGNMENT}" \
            "${MODEL_NAME}" "${status}" "${detail}" >&9
    ) 9>> "${PROGRESS_FILE}"
}

# Deliberately no completed-state check: every array attempt runs its configuration.
echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: ${TASK} ${MODEL_NAME} latent_mas/${CONFIG_PROMPT}/${CONFIG_ALIGNMENT}"
cd "${SUBMIT_DIR}"
export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_PROMPT CONFIG_ALIGNMENT STATE_FILE PROGRESS_FILE
export CONFIG_METHOD=latent_mas MAX_SAMPLES KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD
append_progress STARTED "state file: ${STATE_FILE}"
if bash "${SUBMIT_DIR}/run.sh"; then
    append_progress COMPLETED "state file: ${STATE_FILE}"
else
    STATUS=$?
    append_progress FAILED "exit ${STATUS}; state file: ${STATE_FILE}"
    exit "${STATUS}"
fi
