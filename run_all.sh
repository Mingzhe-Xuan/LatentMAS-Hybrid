#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

TASK_SCRIPTS=(
    exp_aime2024.sh
    exp_aime2025.sh
    exp_arc_challenge.sh
    exp_arc_easy.sh
    exp_gpqa.sh
    exp_gsm8k.sh
    exp_humanevalplus.sh
    exp_mbppplus.sh
    exp_medqa.sh
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
