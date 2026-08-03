#!/bin/bash
#PBS -N x_all
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-9%3
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
FORCE_ALL="${FORCE_ALL:-false}"

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

TASK_SCRIPTS=(
    run_aime2024.sh
    run_aime2025.sh
    run_arc_challenge.sh
    run_arc_easy.sh
    run_gpqa.sh
    run_gsm8k.sh
    run_humanevalplus.sh
    run_mbppplus.sh
    run_medqa.sh
)

# Preserve the convenient local entry point. Running `bash run_all.sh` submits
# this file once; #PBS -J creates nine subjobs and %3 caps active subjobs.
if [[ -z "${PBS_ARRAY_INDEX:-}" ]]; then
    if ! command -v qsub >/dev/null 2>&1; then
        echo "ERROR: qsub was not found in PATH."
        exit 127
    fi
    JOB_ID="$(cd "${SCRIPT_DIR}" && qsub -v "FORCE_ALL=${FORCE_ALL}" "${BASH_SOURCE[0]}")"
    echo "Submitted dataset array ${JOB_ID}: 9 tasks, maximum concurrency 3, force_all=${FORCE_ALL}."
    exit 0
fi

if ! [[ "${PBS_ARRAY_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX}"
    exit 2
fi

TASK_INDEX=$((PBS_ARRAY_INDEX - 1))
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#TASK_SCRIPTS[@]} )); then
    echo "ERROR: PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX} is outside 1-${#TASK_SCRIPTS[@]}."
    exit 2
fi

SCRIPT_NAME="${TASK_SCRIPTS[${TASK_INDEX}]}"
SCRIPT_PATH="${SUBMIT_DIR}/${SCRIPT_NAME}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "ERROR: missing task script: ${SCRIPT_PATH}"
    exit 2
fi

TASK_NAME="${SCRIPT_NAME#run_}"
TASK_NAME="${TASK_NAME%.sh}"
STATE_PATH="${SUBMIT_DIR}/state_${TASK_NAME}.txt"
if [[ "${FORCE_ALL}" != "true" && -e "${STATE_PATH}" ]]; then
    echo "Skipped ${TASK_NAME}: state file already exists: ${STATE_PATH}"
    exit 0
fi

echo "Array job ${PBS_JOBID:-unknown}, index ${PBS_ARRAY_INDEX}: ${SCRIPT_NAME}"
cd "${SUBMIT_DIR}" || exit 1
exec bash "${SCRIPT_PATH}"
