#!/bin/bash
#PBS -N x_fast
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-180%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
FORCE_ALL="${FORCE_ALL:-false}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
KERNEL_FEATURES="${KERNEL_FEATURES:-1024}"
KERNEL_TEMPERATURE="${KERNEL_TEMPERATURE:-1.0}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"
ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"

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
    VARIABLES="FAST_ONLY=true,FORCE_ALL=${FORCE_ALL},MAX_SAMPLES=${MAX_SAMPLES},KERNEL_FEATURES=${KERNEL_FEATURES},KERNEL_TEMPERATURE=${KERNEL_TEMPERATURE},KERNEL_CHUNK_SIZE=${KERNEL_CHUNK_SIZE},ALIGN_RIDGE=${ALIGN_RIDGE}"
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -v "${VARIABLES}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${JOB_ID}: 180 fast configs, maximum concurrency 3, force_all=${FORCE_ALL}."
    exit 0
fi

export FAST_ONLY=true FORCE_ALL MAX_SAMPLES KERNEL_FEATURES
export KERNEL_TEMPERATURE KERNEL_CHUNK_SIZE ALIGN_RIDGE
RUN_ALL_SCRIPT="${SUBMIT_DIR}/run_all.sh"
if [[ ! -f "${RUN_ALL_SCRIPT}" ]]; then
    echo "ERROR: missing scheduler script: ${RUN_ALL_SCRIPT}"
    exit 2
fi
exec bash "${RUN_ALL_SCRIPT}"
