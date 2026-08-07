#!/usr/bin/env bash
###############################################################################
# exp.sh - PBS submission script for the experiments under exp/
#
# Examples:
#   qsub exp.sh
#   qsub -v EXP_TARGET=approximator exp.sh
#   qsub -v EXP_TARGET=approximator,STUDY=s3,DATASET=arc_easy,SPLIT=train exp.sh
#   qsub -v EXP_TARGET=latent_cot exp.sh
#   qsub -v EXP_TARGET=latent_comm,STUDY=m2,DATASET=arc_challenge exp.sh
###############################################################################

#PBS -N latent_exp
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  qsub -v EXP_TARGET=approximator[,NAME=value...] exp.sh
  qsub -v EXP_TARGET=latent_cot[,NAME=value...] exp.sh
  qsub -v EXP_TARGET=latent_comm[,NAME=value...] exp.sh

Options override the plan_v2 main-experiment defaults for the selected target:
  --study NAME              S0--S4, C0--C4, or M0--M4 study name
  --model-pair NAME         x1/x2 for operator/communication; c0/c1 for CoT
  --model-name NAME         Single model used by latent_cot C0
  --agent-models "NAMES"    One or four space-separated models for approximator
  --dataset NAME --split NAME
  --method NAME             e.g. identical, linear, kernel, exact, all
  --orf-seed INT --m INT --tau FLOAT
  --probe-seed INT --max-questions INT --latent-steps INT --device DEVICE
  --extra-args "FLAGS"      Extra flags passed verbatim to the Python entry point
EOF
}

# Default to the plan_v2 operator experiment; PBS's -v option overrides it.
# Positional target flags remain useful for local Bash testing only.
EXP_TARGET="${EXP_TARGET:-approximator}"
STUDY="${STUDY:-}"; MODEL_PAIR="${MODEL_PAIR:-}"; DATASET="${DATASET:-}"
SPLIT="${SPLIT:-}"; METHOD="${METHOD:-}"; ORF_SEED="${ORF_SEED:-}"
M="${M:-}"; TAU="${TAU:-}"; PROBE_SEED="${PROBE_SEED:-}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"; LATENT_STEPS="${LATENT_STEPS:-}"
DEVICE="${DEVICE:-}"; EXP_EXTRA_ARGS="${EXP_EXTRA_ARGS:-}"
AGENT_MODELS="${AGENT_MODELS:-}"
MODEL_NAME="${MODEL_NAME:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --approximator) EXP_TARGET="approximator"; shift ;;
        --latent_cot) EXP_TARGET="latent_cot"; shift ;;
        --latent_comm) EXP_TARGET="latent_comm"; shift ;;
        --study) STUDY="$2"; shift 2 ;;
        --model-pair) MODEL_PAIR="$2"; shift 2 ;;
        --agent-models) AGENT_MODELS="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --orf-seed) ORF_SEED="$2"; shift 2 ;;
        --m) M="$2"; shift 2 ;;
        --tau) TAU="$2"; shift 2 ;;
        --probe-seed) PROBE_SEED="$2"; shift 2 ;;
        --max-questions) MAX_QUESTIONS="$2"; shift 2 ;;
        --latent-steps) LATENT_STEPS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --extra-args) EXP_EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1"; usage; exit 2 ;;
    esac
done

if [[ -z "${EXP_TARGET}" ]]; then
    echo "ERROR: choose exactly one target: --approximator, --latent_cot, or --latent_comm."
    usage; exit 2
fi

# Environment setup mirrors run.sh.
module purge
module load python/3.12.13
source /home/n2501945g/LatentMAS-Hybrid/.venv/bin/activate
cd "${PBS_O_WORKDIR}" || exit 1
if [[ ! -d exp ]]; then echo "ERROR: submit from repository root (exp/ was not found)."; exit 1; fi
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/home/n2501945g/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((GPU_COUNT - 1))"); export CUDA_VISIBLE_DEVICES
fi

# plan_v2 fixed numerical settings shared by the three layers.
M="${M:-2048}"; TAU="${TAU:-1.0}"; ORF_SEED="${ORF_SEED:-101}"
PROBE_SEED="${PROBE_SEED:-42}"; DEVICE="${DEVICE:-cuda}"
KERNEL_CHUNK_SIZE="${KERNEL_CHUNK_SIZE:-4096}"; ALIGN_RIDGE="${ALIGN_RIDGE:-1e-5}"
MAX_STATES_PER_QUESTION="${MAX_STATES_PER_QUESTION:-50}"
MAX_REPLY_TOKENS="${MAX_REPLY_TOKENS:-512}"; PROMPT_LIMIT="${PROMPT_LIMIT:-512}"
GENERATION_SEED="${GENERATION_SEED:-77}"

# Main-experiment defaults. The plan_v2 study/dataset matrix requires one PBS
# submission per cell; these select its first primary cell for each layer.
case "${EXP_TARGET}" in
    approximator)
        STUDY="${STUDY:-all}"; MODEL_PAIR="${MODEL_PAIR:-x1}"
        DATASET="${DATASET:-arc_easy}"; SPLIT="${SPLIT:-test}"
        METHOD="${METHOD:-kernel}"; MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
        LATENT_STEPS="${LATENT_STEPS:-50}"
        AGENT_MODELS="${AGENT_MODELS:-Qwen/Qwen3-4B}"
        read -r -a APPROX_MODELS <<< "${AGENT_MODELS}"
        ENTRY="exp/approximator/run.py"
        ARGS=(--study "${STUDY}" --agent_models "${APPROX_MODELS[@]}" --dataset "${DATASET}" --split "${SPLIT}" --kernel_features "${M}" --kernel_temperature "${TAU}" --kernel_seed "${ORF_SEED}" --probe_seed "${PROBE_SEED}" --max_questions "${MAX_QUESTIONS}" --max_states_per_question "${MAX_STATES_PER_QUESTION}" --max_new_tokens "${MAX_REPLY_TOKENS}" --latent_steps "${LATENT_STEPS}" --kernel_chunk_size "${KERNEL_CHUNK_SIZE}" --device "${DEVICE}")
        ;;
    latent_cot)
        STUDY="${STUDY:-c0}"; MODEL_PAIR="${MODEL_PAIR:-c0}"
        SPLIT="${SPLIT:-test}"; LATENT_STEPS="${LATENT_STEPS:-150}"
        ENTRY="exp/latent_cot/run.py"
        if [[ "${STUDY}" == "c1" || "${STUDY}" == "c2" || "${STUDY}" == "c3" ]]; then
            DATASET="${DATASET:-mbppplus}"
            MAX_QUESTIONS="${MAX_QUESTIONS:-30}"
            MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
            LATENT_STEP_VALUES="${LATENT_STEP_VALUES:-20 40 60 80 100 120 140 160 180}"
            ALIGNMENTS="${ALIGNMENTS:-identical linear soft kernel}"
            read -r -a LATENT_STEP_ARRAY <<< "${LATENT_STEP_VALUES}"
            read -r -a ALIGNMENT_ARRAY <<< "${ALIGNMENTS}"
            ARGS=(--study "${STUDY}" --model_name "${MODEL_NAME}" --dataset "${DATASET}" --split "${SPLIT}" --probe_seed "${PROBE_SEED}" --generation_seed "${GENERATION_SEED}" --max_questions "${MAX_QUESTIONS}" --latent_step_values "${LATENT_STEP_ARRAY[@]}" --alignments "${ALIGNMENT_ARRAY[@]}" --max_new_tokens "${LATENT_COT_MAX_NEW_TOKENS:-4096}" --kernel_features "${M}" --kernel_temperature "${TAU}" --kernel_seed "${ORF_SEED}" --kernel_chunk_size "${KERNEL_CHUNK_SIZE}" --align_ridge "${ALIGN_RIDGE}" --device "${DEVICE}")
        else
            DATASET="${DATASET:-all}"
            MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
            MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
            ARGS=(--study "${STUDY}" --model_name "${MODEL_NAME}" --dataset "${DATASET}" --split "${SPLIT}" --probe_seed "${PROBE_SEED}" --max_questions "${MAX_QUESTIONS}" --latent_steps "${LATENT_STEPS}" --kernel_features "${M}" --kernel_temperature "${TAU}" --kernel_seed "${ORF_SEED}" --kernel_chunk_size "${KERNEL_CHUNK_SIZE}" --align_ridge "${ALIGN_RIDGE}" --device "${DEVICE}")
        fi
        ;;
    latent_comm)
        STUDY="${STUDY:-m0}"; MODEL_PAIR="${MODEL_PAIR:-x1}"
        DATASET="${DATASET:-communication_probe}"; SPLIT="${SPLIT:-test}"
        METHOD="${METHOD:-all}"; MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
        LATENT_STEPS="${LATENT_STEPS:-4}"; ENTRY="exp/latent_comm/run.py"
        ARGS=(--study "${STUDY}" --model_pair "${MODEL_PAIR}" --dataset "${DATASET}" --split "${SPLIT}" --method "${METHOD}" --orf_seed "${ORF_SEED}" --m "${M}" --tau "${TAU}" --latent_steps "${LATENT_STEPS}" --generation_seed "${GENERATION_SEED}" --device "${DEVICE}")
        ;;
esac

EXTRA=()
if [[ -n "${EXP_EXTRA_ARGS}" ]]; then read -r -a EXTRA <<< "${EXP_EXTRA_ARGS}"; fi

echo "========================================================================"
echo "PBS job       : ${PBS_JOBID:-interactive}"
echo "Target/study  : ${EXP_TARGET}/${STUDY}"
if [[ "${EXP_TARGET}" == "approximator" ]]; then
    echo "Agent models  : ${AGENT_MODELS}"
else
    echo "Model pair    : ${MODEL_PAIR}"
fi
echo "Dataset/split : ${DATASET}/${SPLIT}"
echo "Method        : ${METHOD}"
echo "ORF (m,tau,seed): ${M}, ${TAU}, ${ORF_SEED}"
echo "Latent steps  : ${LATENT_STEPS}"
echo "Host          : $(hostname)"
nvidia-smi -L
echo "========================================================================"

if [[ ! -f "${ENTRY}" ]]; then
    echo "ERROR: ${ENTRY} does not exist yet; no experiment was launched."
    exit 2
fi
LOG="${EXP_STATE_LOG:-${PBS_O_WORKDIR:-$(pwd)}/exp_state.txt}"
echo "[$(date --iso-8601=seconds)] PBS job ${PBS_JOBID:-interactive}: python3 ${ENTRY} ${ARGS[*]} ${EXTRA[*]}" >> "${LOG}"
python3 "${ENTRY}" "${ARGS[@]}" "${EXTRA[@]}" >> "${LOG}" 2>&1
echo "Completed successfully; progress and output appended to: ${LOG}"
