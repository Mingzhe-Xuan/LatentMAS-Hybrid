#!/bin/bash
# Submit only the Kernel cells from run_hetero.sh.
#
# run_hetero.sh stores four experiments per dataset/direction in this order:
# TextMAS, Linear, Soft, Kernel. Therefore PBS indices 4,8,...,48 select all
# 6 datasets x 2 model directions x 1 Kernel configuration.
#
# Submit with: bash run_hetero_kernel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/run_hetero.sh"

MAX_GPU="${MAX_GPU:-1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
FORCE_ALL="${FORCE_ALL:-true}"
RESULT_ROOT="${RESULT_ROOT:-result}"
PROGRESS_FILE="${PROGRESS_FILE:-${SCRIPT_DIR}/state_hetero_kernel.txt}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
REPETITION_PENALTY=1.15
# Keep only the sender's latent output when transferring context to the Judger.
# LATENT_ONLY also implies SEQUENTIAL_INFO_ONLY in the method implementation.
LATENT_ONLY="${LATENT_ONLY:-true}"

# Kernel is the fourth entry in each four-experiment block in run_hetero.sh.
FIRST_KERNEL_INDEX=4
LAST_ARRAY_INDEX=48
EXPERIMENTS_PER_BLOCK=4
KERNEL_JOB_COUNT=12

if ! [[ "${MAX_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_GPU must be a positive integer, got: ${MAX_GPU}" >&2
    exit 2
fi
if [[ "${FORCE_ALL}" != "true" && "${FORCE_ALL}" != "false" ]]; then
    echo "ERROR: FORCE_ALL must be true or false, got: ${FORCE_ALL}" >&2
    exit 2
fi
if [[ "${LATENT_ONLY}" != "true" && "${LATENT_ONLY}" != "false" ]]; then
    echo "ERROR: LATENT_ONLY must be true or false, got: ${LATENT_ONLY}" >&2
    exit 2
fi
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "ERROR: missing worker script: ${TARGET_SCRIPT}" >&2
    exit 2
fi
if ! command -v qsub >/dev/null 2>&1; then
    echo "ERROR: qsub was not found in PATH." >&2
    exit 127
fi

VARIABLES="FORCE_ALL=${FORCE_ALL},MAX_SAMPLES=${MAX_SAMPLES},MAX_CONCURRENT_GPUS=${MAX_GPU},RESULT_ROOT=${RESULT_ROOT},PROGRESS_FILE=${PROGRESS_FILE},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},REPETITION_PENALTY=${REPETITION_PENALTY},LATENT_ONLY=${LATENT_ONLY}"
ARRAY_SPEC="${FIRST_KERNEL_INDEX}-${LAST_ARRAY_INDEX}:${EXPERIMENTS_PER_BLOCK}%${MAX_GPU}"

JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -N x_hetero_k -J "${ARRAY_SPEC}" -v "${VARIABLES}" "${TARGET_SCRIPT}")"
echo "Submitted ${JOB_ID}: ${KERNEL_JOB_COUNT} heterogeneous Kernel jobs, maximum ${MAX_GPU} concurrent GPU jobs, repetition_penalty=${REPETITION_PENALTY}, latent_only=${LATENT_ONLY}."
