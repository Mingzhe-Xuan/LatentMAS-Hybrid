#!/bin/bash
# Mistral-NeMo-Instruct-2407: 9 datasets x 14 method configurations.
# Submit with: qsub run_mistral.sh

#PBS -N x_mistral
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-126%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"

export EXPERIMENT_MODEL="mistralai/Mistral-Nemo-Instruct-2407"
export LOG_ROOT="${LOG_ROOT:-${SUBMIT_DIR}/logging_mistral}"
export RESULT_ROOT="${RESULT_ROOT:-${SUBMIT_DIR}/result_mistral}"
export PROGRESS_FILE="${PROGRESS_FILE:-${SUBMIT_DIR}/state_mistral.txt}"
export STATE_ROOT="${STATE_ROOT:-${SUBMIT_DIR}/state_mistral}"

exec bash "${SCRIPT_DIR}/run_model_array.sh" "$@"
