#!/bin/bash
# Measure one-step Text alignment latency on Qwen3-8B / sequential / MedQA.
# Three repeats are run by default. The final report contains the mean and
# population standard deviation in milliseconds per batched alignment step.
#
# Submit with:
#   qsub run_8b_seq_medqa_text_alignment.sh

#PBS -N x_8b_text_align
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${PBS_O_WORKDIR:-${SCRIPT_DIR}}"
RUN_SCRIPT="${SUBMIT_DIR}/run.sh"
SUMMARY_SCRIPT="${SUBMIT_DIR}/summarize_alignment_steps.py"

if [[ ! -f "${RUN_SCRIPT}" || ! -f "${SUMMARY_SCRIPT}" ]]; then
    echo "ERROR: submit from a checkout containing run.sh and summarize_alignment_steps.py" >&2
    exit 2
fi

TASK=medqa
MODEL_NAME="Qwen/Qwen3-8B"
CONFIG_METHOD=latent_mas
CONFIG_PROMPT=sequential
CONFIG_ALIGNMENT=text
TIMES="${TIMES:-3}"
LATENT_STEPS="${LATENT_STEPS:-20}"
GENERATE_BS="${GENERATE_BS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
SEED="${SEED:-42}"

RESULT_ROOT="${RESULT_ROOT:-result/text_alignment_benchmark}"
LOG_ROOT="${LOG_ROOT:-logging/text_alignment_benchmark}"
RUN_TIME="${RUN_TIME:-text_align_${PBS_JOBID:-local}_$(date +%Y%m%d_%H%M%S)}"
STATE_FILE="${STATE_FILE:-state_8b_seq_medqa_text_alignment/${RUN_TIME}.txt}"

MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | tr -c 'A-Za-z0-9._-' '_')"
RESULT_DIR="${RESULT_ROOT}/${TASK}_${CONFIG_METHOD}_${CONFIG_ALIGNMENT}_${CONFIG_PROMPT}_${MODEL_SLUG}_${RUN_TIME}"

cd "${SUBMIT_DIR}" || exit 1
unset AGENT_MODELS
export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK MODEL_NAME CONFIG_METHOD CONFIG_PROMPT CONFIG_ALIGNMENT TIMES
export LATENT_STEPS GENERATE_BS MAX_SAMPLES SEED RUN_TIME
export RESULT_ROOT LOG_ROOT STATE_FILE

bash "${RUN_SCRIPT}"

repeat_files=()
for ((repeat_index = 1; repeat_index <= TIMES; repeat_index++)); do
    repeat_file="${RESULT_DIR}/repeat_${repeat_index}.json"
    if [[ ! -s "${repeat_file}" ]]; then
        echo "ERROR: missing repeat result: ${repeat_file}" >&2
        exit 1
    fi
    repeat_files+=("${repeat_file}")
done

python3 "${SUMMARY_SCRIPT}" \
    --generate-bs "${GENERATE_BS}" \
    --output "${RESULT_DIR}/text_alignment_per_step.json" \
    "${repeat_files[@]}"

echo "Text alignment report: ${RESULT_DIR}/text_alignment_per_step.json"
