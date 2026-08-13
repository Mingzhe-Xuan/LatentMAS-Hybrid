#!/bin/bash
#PBS -N x_test
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe
# PBS keeps the outer array-job stdout/stderr as x_test.o* in PBS_O_WORKDIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
# TASKS_PER_GPU="${TASKS_PER_GPU:-3}"  # Disabled: do not pack multiple experiments onto one GPU.
TASKS_PER_GPU=1
WORKER_MODE="${WORKER_MODE:-false}"
CONFIG_OFFSET="${CONFIG_OFFSET:-}"
TEST_LOG="${SUBMIT_DIR}/test_result.txt"
TEST_STATE="${SUBMIT_DIR}/test_state.txt"
MODEL_NAME="Qwen/Qwen3-8B"

KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"

# Capture the PBS array wrapper before validation or child launch can fail.
if [[ -n "${PBS_JOBID:-}" ]]; then
    exec >> "${TEST_STATE}" 2>&1
    printf '\n=== PBS ENTER job=%s array_index=%s host=%s time=%s ===\n' \
        "${PBS_JOBID}" "${PBS_ARRAY_INDEX:-unset}" "$(hostname)" "$(date --iso-8601=seconds)"
fi

DATASETS=(medqa)
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

DATASET_COUNT=${#DATASETS[@]}
CONFIG_COUNT=${#METHODS[@]}
TOTAL_COUNT=$((DATASET_COUNT * CONFIG_COUNT))

if ! [[ "${TASKS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TASKS_PER_GPU must be a positive integer, got: ${TASKS_PER_GPU}" >&2
    exit 2
fi
ARRAY_JOB_COUNT=$(((TOTAL_COUNT + TASKS_PER_GPU - 1) / TASKS_PER_GPU))

# Submission mode: preserve previous debug output and add a boundary for this run.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH." >&2
        exit 127
    fi
    {
        printf '\n%s\n' "================================================================================"
        printf 'Submitting debug matrix at %s: %s configs, model=%s\n' \
            "$(date --iso-8601=seconds)" "${TOTAL_COUNT}" "${MODEL_NAME}"
    } >> "${TEST_STATE}"
    VARIABLES="TASKS_PER_GPU=${TASKS_PER_GPU},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE}"
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -J "1-${ARRAY_JOB_COUNT}%3" -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    printf 'PBS accepted job %s at %s\n' "${JOB_ID}" "$(date --iso-8601=seconds)" >> "${TEST_STATE}"
    echo "Submitted ${JOB_ID}: ${TOTAL_COUNT} debug configs in ${ARRAY_JOB_COUNT} GPU jobs, one config per GPU."
    echo "State log: ${TEST_STATE}"
    echo "Experiment log: ${TEST_LOG}"
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX}" >&2
    exit 2
fi

# One array subjob owns one GPU and launches exactly one configuration.
if [[ "${WORKER_MODE}" != "true" ]]; then
    FIRST_OFFSET=$(((PBS_ARRAY_INDEX - 1) * TASKS_PER_GPU))
    PIDS=()
    OFFSETS=()
    for ((slot = 0; slot < TASKS_PER_GPU; slot++)); do
        CHILD_OFFSET=$((FIRST_OFFSET + slot))
        if (( CHILD_OFFSET >= TOTAL_COUNT )); then
            break
        fi
        echo "Launching config offset ${CHILD_OFFSET} in slot $((slot + 1))/${TASKS_PER_GPU}."
        WORKER_MODE=true CONFIG_OFFSET="${CHILD_OFFSET}" bash "${BASH_SOURCE[0]}" &
        PIDS+=("$!")
        OFFSETS+=("${CHILD_OFFSET}")
    done

    OVERALL_STATUS=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[${i}]}"; then
            echo "Config offset ${OFFSETS[${i}]} finished successfully."
        else
            CHILD_STATUS=$?
            echo "ERROR: config offset ${OFFSETS[${i}]} failed with exit ${CHILD_STATUS}." >&2
            if (( OVERALL_STATUS == 0 )); then
                OVERALL_STATUS=${CHILD_STATUS}
            fi
        fi
    done
    exit "${OVERALL_STATUS}"
fi

if ! [[ "${CONFIG_OFFSET}" =~ ^[0-9]+$ ]] || (( CONFIG_OFFSET >= TOTAL_COUNT )); then
    echo "ERROR: invalid CONFIG_OFFSET=${CONFIG_OFFSET}" >&2
    exit 2
fi

CONFIG_INDEX=$((CONFIG_OFFSET % CONFIG_COUNT))
DATASET_INDEX=$((CONFIG_OFFSET / CONFIG_COUNT))
TASK="${DATASETS[${DATASET_INDEX}]}"
METHOD="${METHODS[${CONFIG_INDEX}]}"
PROMPT="${PROMPTS[${CONFIG_INDEX}]}"
ALIGNMENT="${ALIGNMENTS[${CONFIG_INDEX}]}"

# Record lifecycle in test_state.txt; Python detail output is redirected below.
printf '\n--- START offset=%s dataset=%s method=%s prompt=%s alignment=%s job=%s[%s] ---\n' \
    "${CONFIG_OFFSET}" "${TASK}" "${METHOD}" "${PROMPT}" "${ALIGNMENT}" \
    "${PBS_JOBID:-local}" "${PBS_ARRAY_INDEX}"

# Set up the same cluster environment used by run.sh.
module purge
module load python/3.12.13
source /home/n2501945g/LatentMAS-Hybrid/.venv/bin/activate
cd "${SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export HF_HOME=/home/n2501945g/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
fi

# Every dataset contributes one problem. Soft, TextMAS and entropy-stopped
# kernel retain the task/prompt latent-step setting; every other configuration
# uses 10 latent steps for this debug matrix.
read -r MAX_NEW_TOKENS GENERATE_BS TASK_LATENT_STEPS < <(
    python3 - "${TASK}" "${PROMPT}" <<'PY'
import json
import sys
from pathlib import Path

params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
task = params.get(sys.argv[1], {})
latent = task.get("latent_steps", {})
print(task.get("max_token", 20000), task.get("generation_bs", 10), latent.get(sys.argv[2], 20))
PY
)
if [[ "${METHOD}" == "text_mas" || "${ALIGNMENT}" == "soft" || "${ALIGNMENT}" == "kernel_early_stopping" ]]; then
    LATENT_STEPS="${TASK_LATENT_STEPS}"
else
    LATENT_STEPS=10
fi

COMMAND=(
    python3 run.py
    --method "${METHOD}"
    --prompt "${PROMPT}"
    --align_method "${ALIGNMENT}"
    --model_name "${MODEL_NAME}"
    --task "${TASK}"
    --max_samples 1
    --split test
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --generate_bs "${GENERATE_BS}"
    --latent_steps "${LATENT_STEPS}"
    --temperature 0.6
    --top_p 0.95
    --seed 42
    --align_ridge "${ALIGN_RIDGE}"
    --kernel_features "${KERNEL_FEATURES}"
    --kernel_temperature "${KERNEL_TEMPERATURE}"
    --kernel_chunk_size "${KERNEL_CHUNK_SIZE}"
    --soft_temperature "${SOFT_TEMPERATURE}"
    --soft_chunk_size "${SOFT_CHUNK_SIZE}"
    --trust_remote_code
    --no_write_result
    --log_path "${TEST_LOG}"
)

printf 'Resolved debug parameters: latent_steps=%s max_new_tokens=%s generate_bs=%s\n' \
    "${LATENT_STEPS}" "${MAX_NEW_TOKENS}" "${GENERATE_BS}"

STATUS=0
"${COMMAND[@]}" >> "${TEST_LOG}" 2>&1 || STATUS=$?

printf '%s\n' \
    "--- END offset=${CONFIG_OFFSET} dataset=${TASK} method=${METHOD} prompt=${PROMPT} alignment=${ALIGNMENT} status=${STATUS} ---"
exit "${STATUS}"
