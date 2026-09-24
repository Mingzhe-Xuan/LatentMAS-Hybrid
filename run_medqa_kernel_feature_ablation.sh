#!/bin/bash
# Kernel random-feature dimension ablation on MedQA.
#
# Default configuration:
#   model:      Qwen/Qwen3-8B
#   topology:   hierarchical (MedQA sequential has 0 latent steps)
#   method:     homogeneous four-agent latent_mas
#   alignment:  kernel
#   m:          256, 512, 1024, 2048, 4096
#   repeats:    3 (seeds 42, 43, 44)
#
# Submit from the login node with either:
#   bash run_medqa_kernel_feature_ablation.sh
# or:
#   qsub run_medqa_kernel_feature_ablation.sh
#
# Results and per-repeat logs are separated by feature dimension:
#   result/ablation/kernel_features/medqa/m_<m>/
#   logging/ablation/kernel_features/medqa/m_<m>/

#PBS -N x_medqa_km
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-5%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
RUN_SCRIPT="${SUBMIT_DIR}/run.sh"

FEATURE_DIMS=(256 512 1024 2048 4096)
MAX_GPU="${MAX_GPU:-3}"
FORCE_ALL="${FORCE_ALL:-false}"

TASK="medqa"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
CONFIG_METHOD="latent_mas"
CONFIG_PROMPT="hierarchical"
CONFIG_ALIGNMENT="kernel"
TIMES=3

MAX_SAMPLES="${MAX_SAMPLES:--1}"
ABLATION_RESULT_ROOT="${ABLATION_RESULT_ROOT:-result/ablation/kernel_features/medqa}"
ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT:-logging/ablation/kernel_features/medqa}"
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_medqa_kernel_features.txt}"

# Keep all non-ablated settings aligned with run_all.sh.
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"

JOB_COUNT=${#FEATURE_DIMS[@]}

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

# Like run_all.sh, a normal Bash invocation submits the PBS array once.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH." >&2
        exit 127
    fi
    VARIABLES="MAX_GPU=${MAX_GPU},FORCE_ALL=${FORCE_ALL},MODEL_NAME=${MODEL_NAME},MAX_SAMPLES=${MAX_SAMPLES},ABLATION_RESULT_ROOT=${ABLATION_RESULT_ROOT},ABLATION_LOG_ROOT=${ABLATION_LOG_ROOT},PROGRESS_FILE=${PROGRESS_FILE},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE},EARLY_STOPPING_LENGTH_THRESHOLD=${EARLY_STOPPING_LENGTH_THRESHOLD},EARLY_STOPPING_ENTROPY_THRESHOLD=${EARLY_STOPPING_ENTROPY_THRESHOLD}"
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -J "1-${JOB_COUNT}%${MAX_GPU}" -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${JOB_ID}: ${JOB_COUNT} MedQA Kernel feature-dimension settings, 3 repeats each, maximum ${MAX_GPU} concurrent GPUs."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[1-9][0-9]*$ ]] ||
   (( PBS_ARRAY_INDEX > JOB_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX must be in 1-${JOB_COUNT}, got: ${PBS_ARRAY_INDEX}" >&2
    exit 2
fi

KERNEL_FEATURES="${FEATURE_DIMS[$((PBS_ARRAY_INDEX - 1))]}"
RESULT_ROOT="${ABLATION_RESULT_ROOT}/m_${KERNEL_FEATURES}"
LOG_ROOT="${ABLATION_LOG_ROOT}/m_${KERNEL_FEATURES}"

MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_DIR="${SUBMIT_DIR}/state_medqa_kernel_features/m_${KERNEL_FEATURES}"
STATE_FILE="${STATE_DIR}/${TASK}_${CONFIG_METHOD}_${CONFIG_ALIGNMENT}_${CONFIG_PROMPT}_${MODEL_SLUG}_state.txt"
mkdir -p "${STATE_DIR}"

append_progress() {
    local status="$1"
    local detail="${2//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    (
        flock -x 9
        if [[ ! -s "${PROGRESS_FILE}" ]]; then
            printf 'timestamp\tjob_id\tarray_index\tdataset\tmethod\tprompt\talignment\tmodel\tkernel_features\trepeats\tstatus\tdetail\n' >&9
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${PBS_JOBID:-local}" "${PBS_ARRAY_INDEX}" \
            "${TASK}" "${CONFIG_METHOD}" "${CONFIG_PROMPT}" "${CONFIG_ALIGNMENT}" \
            "${MODEL_NAME}" "${KERNEL_FEATURES}" "${TIMES}" "${status}" "${detail}" >&9
    ) 9>> "${PROGRESS_FILE}"
}

state_file_completed() {
    [[ -f "${STATE_FILE}" ]] &&
        [[ "$(tail -n 1 "${STATE_FILE}")" == "Exit status: 0" ]]
}

if [[ "${FORCE_ALL}" != "true" ]] && state_file_completed; then
    append_progress SKIPPED "completed state file: ${STATE_FILE}"
    echo "Skipped completed m=${KERNEL_FEATURES}: ${STATE_FILE}"
    exit 0
fi

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: MedQA ${MODEL_NAME}, ${CONFIG_PROMPT} Kernel, m=${KERNEL_FEATURES}, repeats=${TIMES}"
cd "${SUBMIT_DIR}" || exit 1

# Homogeneous four-agent run: every role uses MODEL_NAME.
unset AGENT_MODELS
export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT TIMES
export STATE_FILE RESULT_ROOT LOG_ROOT MAX_SAMPLES
export KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE
export EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD

append_progress STARTED "state file: ${STATE_FILE}; result root: ${RESULT_ROOT}; log root: ${LOG_ROOT}"
if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_FILE}; result root: ${RESULT_ROOT}; log root: ${LOG_ROOT}"
else
    status=$?
    append_progress FAILED "exit ${status}; state file: ${STATE_FILE}"
    exit "${status}"
fi
