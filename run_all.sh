#!/bin/bash
#PBS -N x_all
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-240%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
FORCE_ALL="${FORCE_ALL:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-30}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-1.0}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
PROGRESS_FILE="${SUBMIT_DIR}/state.txt"

for ARG in "$@"; do
    case "${ARG}" in
        --force_all) FORCE_ALL=true ;;
        *)
            echo "ERROR: unknown argument: ${ARG}"
            echo "Usage: bash run_all.sh [--force_all]"
            exit 2
            ;;
    esac
done

DATASETS=(
    aime2024 aime2025 arc_challenge arc_easy gpqa gsm8k
    humanevalplus mbppplus medqa
)
MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-4B")
FOUR_B_DATASETS=(
    arc_challenge arc_easy gsm8k humanevalplus mbppplus medqa
)
METHODS=(
    baseline baseline text_mas text_mas
    latent_mas latent_mas latent_mas latent_mas latent_mas latent_mas
)
PROMPTS=(
    sequential hierarchical sequential hierarchical
    sequential sequential sequential hierarchical hierarchical hierarchical
)
ALIGNMENTS=(
    identical identical identical identical
    identical linear kernel identical linear kernel
)

DATASET_COUNT=${#DATASETS[@]}
MODEL_COUNT=${#MODELS[@]}
FOUR_B_DATASET_COUNT=${#FOUR_B_DATASETS[@]}
CONFIG_COUNT=${#METHODS[@]}
MODEL_DATASET_COUNT=$((DATASET_COUNT * 2 + FOUR_B_DATASET_COUNT))
TOTAL_COUNT=$((MODEL_DATASET_COUNT * CONFIG_COUNT))

# `bash run_all.sh` submits the array once. Every array subjob is independently
# queued by PBS; %3 is the global running-job limit.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH."
        exit 127
    fi
    VARIABLES="FORCE_ALL=${FORCE_ALL},MAX_SAMPLES=${MAX_SAMPLES},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE}"
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${JOB_ID}: ${TOTAL_COUNT} independently queued configs, maximum concurrency 3, force_all=${FORCE_ALL}."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX}"
    exit 2
fi
ARRAY_OFFSET=$((PBS_ARRAY_INDEX - 1))
if (( ARRAY_OFFSET < 0 || ARRAY_OFFSET >= TOTAL_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX} is outside 1-${TOTAL_COUNT}."
    exit 2
fi

CONFIG_INDEX=$((ARRAY_OFFSET % CONFIG_COUNT))
MODEL_DATASET_INDEX=$((ARRAY_OFFSET / CONFIG_COUNT))

if (( MODEL_DATASET_INDEX < DATASET_COUNT )); then
    MODEL_NAME="${MODELS[0]}"
    TASK="${DATASETS[${MODEL_DATASET_INDEX}]}"
elif (( MODEL_DATASET_INDEX < DATASET_COUNT * 2 )); then
    MODEL_NAME="${MODELS[1]}"
    TASK="${DATASETS[$((MODEL_DATASET_INDEX - DATASET_COUNT))]}"
else
    MODEL_NAME="${MODELS[2]}"
    TASK="${FOUR_B_DATASETS[$((MODEL_DATASET_INDEX - DATASET_COUNT * 2))]}"
fi
CONFIG_METHOD="${METHODS[${CONFIG_INDEX}]}"
CONFIG_PROMPT="${PROMPTS[${CONFIG_INDEX}]}"
CONFIG_ALIGNMENT="${ALIGNMENTS[${CONFIG_INDEX}]}"

case "${CONFIG_METHOD}" in
    baseline|text_mas)
        if [[ "${CONFIG_ALIGNMENT}" != "identical" ]]; then
            echo "ERROR: ${CONFIG_METHOD} only supports identical alignment."
            exit 2
        fi
        ;;
    latent_mas)
        case "${CONFIG_ALIGNMENT}" in
            identical|linear|kernel) ;;
            *) echo "ERROR: invalid alignment: ${CONFIG_ALIGNMENT}"; exit 2 ;;
        esac
        ;;
    *) echo "ERROR: invalid method: ${CONFIG_METHOD}"; exit 2 ;;
esac
case "${CONFIG_PROMPT}" in
    sequential|hierarchical) ;;
    *) echo "ERROR: invalid prompt: ${CONFIG_PROMPT}"; exit 2 ;;
esac

STATE_METHOD="${CONFIG_METHOD}"
if [[ "${CONFIG_METHOD}" == "latent_mas" ]]; then
    STATE_METHOD="${CONFIG_METHOD}_${CONFIG_ALIGNMENT}"
fi
MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_DIR="${SUBMIT_DIR}/state"
STATE_PATH="${STATE_DIR}/${TASK}_${STATE_METHOD}_${CONFIG_PROMPT}_${MODEL_SLUG}_state.txt"
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

if [[ "${FORCE_ALL}" != "true" && -e "${STATE_PATH}" ]]; then
    append_progress SKIPPED "state file exists: ${STATE_PATH}"
    echo "Skipped existing config: ${STATE_PATH}"
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
export MAX_SAMPLES KERNEL_FEATURES KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
if bash "${RUN_SCRIPT}"; then
    append_progress COMPLETED "state file: ${STATE_PATH}"
else
    STATUS=$?
    append_progress FAILED "exit ${STATUS}; state file: ${STATE_PATH}"
    exit "${STATUS}"
fi
