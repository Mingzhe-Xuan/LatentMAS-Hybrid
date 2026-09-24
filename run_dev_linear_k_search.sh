#!/bin/bash
# Sweep latent steps K for linear alignment on every dataset in data/dev.
#
# Search space: 9 datasets x 2 prompts x 5 K values = 90 one-GPU jobs.
# Each setting runs once with Qwen3-8B. At most four jobs run concurrently.
#
# Submit with either:
#   bash run_dev_linear_k_search.sh
#   qsub run_dev_linear_k_search.sh

#PBS -N x_dev_linear_k
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-90%4
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
RUN_SCRIPT="${SUBMIT_DIR}/run.sh"

DATASETS=(
    aime2024 aime2025 arc_challenge arc_easy gpqa gsm8k
    humanevalplus mbppplus medqa
)
PROMPTS=(sequential hierarchical)
K_VALUES=(0 10 20 40 80)

MODEL_NAME="Qwen/Qwen3-8B"
CONFIG_METHOD="latent_mas"
CONFIG_ALIGNMENT="linear"
TIMES=1
SPLIT=dev
MAX_SAMPLES=-1
MAX_GPU="${MAX_GPU:-4}"
FORCE_ALL="${FORCE_ALL:-false}"

SEARCH_RESULT_ROOT="${SEARCH_RESULT_ROOT:-result/ablation/linear_latent_steps/dev}"
SEARCH_LOG_ROOT="${SEARCH_LOG_ROOT:-logging/ablation/linear_latent_steps/dev}"
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_dev_linear_k_search.tsv}"

# Non-ablated settings match run_all.sh/run.sh.
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"

DATASET_COUNT=${#DATASETS[@]}
PROMPT_COUNT=${#PROMPTS[@]}
K_COUNT=${#K_VALUES[@]}
JOB_COUNT=$((DATASET_COUNT * PROMPT_COUNT * K_COUNT))

if ! [[ "${MAX_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_GPU must be a positive integer, got: ${MAX_GPU}" >&2
    exit 2
fi
if [[ "${FORCE_ALL}" != "true" && "${FORCE_ALL}" != "false" ]]; then
    echo "ERROR: FORCE_ALL must be true or false, got: ${FORCE_ALL}" >&2
    exit 2
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: missing run script: ${RUN_SCRIPT}" >&2
    exit 2
fi

# A normal Bash invocation submits the PBS array. qsub invocation enters below.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH." >&2
        exit 127
    fi
    VARIABLES="MAX_GPU=${MAX_GPU},FORCE_ALL=${FORCE_ALL},SEARCH_RESULT_ROOT=${SEARCH_RESULT_ROOT},SEARCH_LOG_ROOT=${SEARCH_LOG_ROOT},PROGRESS_FILE=${PROGRESS_FILE},ALIGN_RIDGE=${ALIGN_RIDGE},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE}"
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -J "1-${JOB_COUNT}%${MAX_GPU}" -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${JOB_ID}: ${JOB_COUNT} dev-set linear/K settings, one repeat each, maximum ${MAX_GPU} concurrent GPUs."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[1-9][0-9]*$ ]] || ((PBS_ARRAY_INDEX > JOB_COUNT)); then
    echo "ERROR: PBS_ARRAY_INDEX must be in 1-${JOB_COUNT}, got: ${PBS_ARRAY_INDEX}" >&2
    exit 2
fi

OFFSET=$((PBS_ARRAY_INDEX - 1))
K_INDEX=$((OFFSET % K_COUNT))
PROMPT_INDEX=$(((OFFSET / K_COUNT) % PROMPT_COUNT))
DATASET_INDEX=$((OFFSET / (K_COUNT * PROMPT_COUNT)))

TASK="${DATASETS[${DATASET_INDEX}]}"
CONFIG_PROMPT="${PROMPTS[${PROMPT_INDEX}]}"
LATENT_STEPS="${K_VALUES[${K_INDEX}]}"

RESULT_ROOT="${SEARCH_RESULT_ROOT}/${TASK}/${CONFIG_PROMPT}/k_${LATENT_STEPS}"
LOG_ROOT="${SEARCH_LOG_ROOT}/${TASK}/${CONFIG_PROMPT}/k_${LATENT_STEPS}"
STATE_DIR="${SUBMIT_DIR}/state_dev_linear_k_search/${TASK}/${CONFIG_PROMPT}"
STATE_FILE="${STATE_DIR}/k_${LATENT_STEPS}.txt"
mkdir -p "${STATE_DIR}"

append_progress() {
    local status="$1"
    local detail="${2//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    (
        flock -x 9
        if [[ ! -s "${PROGRESS_FILE}" ]]; then
            printf 'timestamp\tjob_id\tarray_index\tdataset\tprompt\tlatent_steps\tmodel\talignment\trepeats\tstatus\tdetail\n' >&9
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${PBS_JOBID:-local}" "${PBS_ARRAY_INDEX}" \
            "${TASK}" "${CONFIG_PROMPT}" "${LATENT_STEPS}" "${MODEL_NAME}" \
            "${CONFIG_ALIGNMENT}" "${TIMES}" "${status}" "${detail}" >&9
    ) 9>>"${PROGRESS_FILE}"
}

state_file_completed() {
    [[ -f "${STATE_FILE}" ]] && [[ "$(tail -n 1 "${STATE_FILE}")" == "Exit status: 0" ]]
}

if [[ "${FORCE_ALL}" != "true" ]] && state_file_completed; then
    append_progress SKIPPED "completed state file: ${STATE_FILE}"
    echo "Skipped completed setting: ${TASK}/${CONFIG_PROMPT}/K=${LATENT_STEPS}"
    exit 0
fi

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: ${TASK}/dev, ${CONFIG_PROMPT}, linear, K=${LATENT_STEPS}, repeat=1"
cd "${SUBMIT_DIR}" || exit 1

# Homogeneous four-agent Qwen3-8B run.
unset AGENT_MODELS
export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT TIMES SPLIT
export LATENT_STEPS MAX_SAMPLES STATE_FILE RESULT_ROOT LOG_ROOT
export ALIGN_RIDGE KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE

append_progress STARTED "state file: ${STATE_FILE}; result root: ${RESULT_ROOT}; log root: ${LOG_ROOT}"
if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_FILE}; result root: ${RESULT_ROOT}; log root: ${LOG_ROOT}"
else
    status=$?
    append_progress FAILED "exit ${status}; state file: ${STATE_FILE}"
    exit "${status}"
fi
