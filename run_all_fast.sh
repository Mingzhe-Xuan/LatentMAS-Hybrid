#!/bin/bash
#PBS -N x_fast
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-84%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
FORCE_ALL="${FORCE_ALL:-false}"
TASKS_PER_GPU="${TASKS_PER_GPU:-3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-0.6}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"

for ARG in "$@"; do
    case "${ARG}" in
        --force_all) FORCE_ALL=true ;;
        *)
            echo "ERROR: unknown argument: ${ARG}"
            echo "Usage: bash run_all_fast.sh [--force_all]"
            exit 2
            ;;
    esac
done

if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH."
        exit 127
    fi
    if ! [[ "${TASKS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TASKS_PER_GPU must be a positive integer, got: ${TASKS_PER_GPU}"
        exit 2
    fi
    ARRAY_JOB_COUNT=$(((252 + TASKS_PER_GPU - 1) / TASKS_PER_GPU))
    VARIABLES="FAST_ONLY=true,FORCE_ALL=${FORCE_ALL},TASKS_PER_GPU=${TASKS_PER_GPU},MAX_SAMPLES=${MAX_SAMPLES},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE},EARLY_STOPPING_LENGTH_THRESHOLD=${EARLY_STOPPING_LENGTH_THRESHOLD},EARLY_STOPPING_ENTROPY_THRESHOLD=${EARLY_STOPPING_ENTROPY_THRESHOLD}"
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -J "1-${ARRAY_JOB_COUNT}%3" -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${JOB_ID}: 252 fast configs in ${ARRAY_JOB_COUNT} GPU jobs, ${TASKS_PER_GPU} parallel configs per GPU, maximum 3 concurrent GPU jobs, force_all=${FORCE_ALL}."
    exit 0
fi

export FAST_ONLY=true FORCE_ALL TASKS_PER_GPU MAX_SAMPLES KERNEL_FEATURES
export KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD
RUN_ALL_SCRIPT="${SUBMIT_DIR}/run_all.sh"
if [[ ! -f "${RUN_ALL_SCRIPT}" ]]; then
    echo "ERROR: missing scheduler script: ${RUN_ALL_SCRIPT}"
    exit 2
fi
exec bash "${RUN_ALL_SCRIPT}"
