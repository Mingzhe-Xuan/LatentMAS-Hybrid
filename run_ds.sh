#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B: 9 datasets x 14 method configurations.
# Submit with: qsub run_ds.sh

#PBS -N x_ds
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-126%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"

export EXPERIMENT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
export LOG_ROOT="${LOG_ROOT:-${SUBMIT_DIR}/logging_ds}"
export RESULT_ROOT="${RESULT_ROOT:-${SUBMIT_DIR}/result_ds}"
export PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_ds.txt}"
export STATE_ROOT="${STATE_ROOT:-${SUBMIT_DIR}/state_ds}"

ARRAY_RUNNER="${SUBMIT_DIR}/run_model_array.sh"
if [[ ! -f "${ARRAY_RUNNER}" ]]; then
    echo "ERROR: missing array runner: ${ARRAY_RUNNER}" >&2
    exit 2
fi

exec bash "${ARRAY_RUNNER}" "$@"
