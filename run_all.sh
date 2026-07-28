#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

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

if ! command -v qsub >/dev/null 2>&1; then
    echo "ERROR: qsub was not found in PATH."
    exit 127
fi

STATUS=0
for SCRIPT_NAME in "${TASK_SCRIPTS[@]}"; do
    SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
    if [ ! -f "${SCRIPT_PATH}" ]; then
        echo "ERROR: Missing PBS script: ${SCRIPT_PATH}"
        STATUS=1
        continue
    fi

    if JOB_ID="$(qsub "${SCRIPT_PATH}")"; then
        echo "Submitted ${SCRIPT_NAME}: ${JOB_ID}"
    else
        echo "ERROR: Failed to submit ${SCRIPT_NAME}"
        STATUS=1
    fi
done

exit "${STATUS}"
