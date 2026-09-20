#!/bin/bash
#PBS -N x_reduced_fill
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-18%3
#PBS -j oe

# Fill selected results for GSM8K, ARC-Easy, and ARC-Challenge:
# Qwen3-14B Soft plus Qwen3-8B/14B Single, under both prompt structures.
# Every configuration runs one repeat.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
RUN_ALL_SCRIPT="${SUBMIT_DIR}/run_all.sh"

FORCE_ALL="${FORCE_ALL:-false}"
MAX_GPU="${MAX_GPU:-3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
RESULT_ROOT="${RESULT_ROOT:-result}"
PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_14b_soft_reduced.txt}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.6}"
SOFT_CHUNK_SIZE="${SOFT_CHUNK_SIZE:-32}"
EARLY_STOPPING_LENGTH_THRESHOLD="${EARLY_STOPPING_LENGTH_THRESHOLD:-auto}"
EARLY_STOPPING_ENTROPY_THRESHOLD="${EARLY_STOPPING_ENTROPY_THRESHOLD:-auto}"

# Zero-based CONFIG_OFFSET values from run_all.sh's full 336-cell matrix.
RUN_ALL_OFFSETS=(
    # Qwen3-14B Soft: GSM8K, ARC-Easy, ARC-Challenge; Seq/Hier.
    204 209 176 181 162 167
    # Qwen3-8B Single: GSM8K, ARC-Easy, ARC-Challenge; Seq/Hier.
    70 71 42 43 28 29
    # Qwen3-14B Single: GSM8K, ARC-Easy, ARC-Challenge; Seq/Hier.
    196 197 168 169 154 155
)
JOB_COUNT=${#RUN_ALL_OFFSETS[@]}

for arg in "$@"; do
    case "${arg}" in
        --force_all) FORCE_ALL=true ;;
        *)
            echo "ERROR: unknown argument: ${arg}" >&2
            echo "Usage: bash run_14b_soft_reduced.sh [--force_all]" >&2
            exit 2
            ;;
    esac
done

if ! [[ "${MAX_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_GPU must be a positive integer, got: ${MAX_GPU}" >&2
    exit 2
fi
if [[ "${FORCE_ALL}" != true && "${FORCE_ALL}" != false ]]; then
    echo "ERROR: FORCE_ALL must be true or false, got: ${FORCE_ALL}" >&2
    exit 2
fi
if [[ ! -f "${RUN_ALL_SCRIPT}" ]]; then
    echo "ERROR: missing scheduler script: ${RUN_ALL_SCRIPT}" >&2
    exit 2
fi

if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH." >&2
        exit 127
    fi
    variables="FORCE_ALL=${FORCE_ALL},MAX_GPU=${MAX_GPU},MAX_SAMPLES=${MAX_SAMPLES},RESULT_ROOT=${RESULT_ROOT},PROGRESS_FILE=${PROGRESS_FILE},SOFT_TEMPERATURE=${SOFT_TEMPERATURE},SOFT_CHUNK_SIZE=${SOFT_CHUNK_SIZE},EARLY_STOPPING_LENGTH_THRESHOLD=${EARLY_STOPPING_LENGTH_THRESHOLD},EARLY_STOPPING_ENTROPY_THRESHOLD=${EARLY_STOPPING_ENTROPY_THRESHOLD}"
    job_id="$(cd "${SCRIPT_DIR}" && qsub -J "1-${JOB_COUNT}%${MAX_GPU}" -v "${variables}" "${BASH_SOURCE[0]}")"
    echo "Submitted ${job_id}: ${JOB_COUNT} reduced-fill jobs, one repeat each, maximum ${MAX_GPU} concurrent GPU jobs."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[1-9][0-9]*$ ]] || (( PBS_ARRAY_INDEX > JOB_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX must be in 1-${JOB_COUNT}." >&2
    exit 2
fi

offset="${RUN_ALL_OFFSETS[$((PBS_ARRAY_INDEX - 1))]}"

# run_all.sh owns the canonical model/dataset/config mapping and state paths.
# TIMES=1 overrides params_dict.json only for these selected cells.
export FORCE_ALL MAX_SAMPLES RESULT_ROOT PROGRESS_FILE
export SOFT_TEMPERATURE SOFT_CHUNK_SIZE
export EARLY_STOPPING_LENGTH_THRESHOLD EARLY_STOPPING_ENTROPY_THRESHOLD
export TIMES=1 FAST_ONLY=false SLOW_ONLY=false

echo "Array ${PBS_JOBID:-unknown}[${PBS_ARRAY_INDEX}]: run_all.sh CONFIG_OFFSET=${offset}, TIMES=${TIMES}"
cd "${SUBMIT_DIR}" || exit 1
WORKER_MODE=true CONFIG_OFFSET="${offset}" bash "${RUN_ALL_SCRIPT}"
