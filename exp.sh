#!/usr/bin/env bash
###############################################################################
# exp.sh - PBS submission script for the experiments under exp/
#
# Examples:
#   qsub exp.sh
#   qsub -v EXP_TARGET=approximator exp.sh
#   qsub -v EXP_TARGET=approximator,STUDY=s3,DATASET=arc_easy,SPLIT=train exp.sh
#   qsub -v EXP_TARGET=latent_cot,METHOD=kernel,DATASET=gsm8k exp.sh
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --approximator) EXP_TARGET="approximator"; shift ;;
        --latent_cot) EXP_TARGET="latent_cot"; shift ;;
        --latent_comm) EXP_TARGET="latent_comm"; shift ;;
        --study) STUDY="$2"; shift 2 ;;
        --model-pair) MODEL_PAIR="$2"; shift 2 ;;
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
        STUDY="${STUDY:-s1}"; MODEL_PAIR="${MODEL_PAIR:-x1}"
        DATASET="${DATASET:-arc_easy}"; SPLIT="${SPLIT:-test}"
        METHOD="${METHOD:-kernel}"; MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
        LATENT_STEPS="${LATENT_STEPS:-0}"
        ENTRY="exp/approximator/run.py"
        ARGS=(--study "${STUDY}" --model_pair "${MODEL_PAIR}" --dataset "${DATASET}" --split "${SPLIT}" --m "${M}" --tau "${TAU}" --orf_seed "${ORF_SEED}" --probe_seed "${PROBE_SEED}" --max_questions "${MAX_QUESTIONS}" --max_states_per_question "${MAX_STATES_PER_QUESTION}" --max_reply_tokens "${MAX_REPLY_TOKENS}" --prompt_limit "${PROMPT_LIMIT}" --kernel_chunk_size "${KERNEL_CHUNK_SIZE}" --device "${DEVICE}")
        ;;
    latent_cot)
        STUDY="${STUDY:-c1}"; MODEL_PAIR="${MODEL_PAIR:-c0}"
        DATASET="${DATASET:-gsm8k}"; SPLIT="${SPLIT:-test}"
        METHOD="${METHOD:-all}"; MAX_QUESTIONS="${MAX_QUESTIONS:-512}"
        LATENT_STEPS="${LATENT_STEPS:-16}"; ENTRY="exp/latent_cot/run.py"
        ARGS=(--study "${STUDY}" --model_pair "${MODEL_PAIR}" --dataset "${DATASET}" --split "${SPLIT}" --method "${METHOD}" --orf_seed "${ORF_SEED}" --m "${M}" --tau "${TAU}" --latent_steps "${LATENT_STEPS}" --generation_seed "${GENERATION_SEED}" --device "${DEVICE}")
        ;;
    latent_comm)
        STUDY="${STUDY:-m1}"; MODEL_PAIR="${MODEL_PAIR:-x1}"
        DATASET="${DATASET:-communication_probe}"; SPLIT="${SPLIT:-test}"
        METHOD="${METHOD:-all}"; MAX_QUESTIONS="${MAX_QUESTIONS:-512}"
        LATENT_STEPS="${LATENT_STEPS:-4}"; ENTRY="exp/latent_comm/run.py"
        ARGS=(--study "${STUDY}" --model_pair "${MODEL_PAIR}" --dataset "${DATASET}" --split "${SPLIT}" --method "${METHOD}" --orf_seed "${ORF_SEED}" --m "${M}" --tau "${TAU}" --latent_steps "${LATENT_STEPS}" --generation_seed "${GENERATION_SEED}" --device "${DEVICE}")
        ;;
esac

EXTRA=()
if [[ -n "${EXP_EXTRA_ARGS}" ]]; then read -r -a EXTRA <<< "${EXP_EXTRA_ARGS}"; fi

echo "========================================================================"
echo "PBS job       : ${PBS_JOBID:-interactive}"
echo "Target/study  : ${EXP_TARGET}/${STUDY}"
echo "Model pair    : ${MODEL_PAIR}"
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
LOG="exp_${EXP_TARGET}_${PBS_JOBID:-local}.txt"
echo "Command: python3 ${ENTRY} ${ARGS[*]} ${EXTRA[*]}"
python3 "${ENTRY}" "${ARGS[@]}" "${EXTRA[@]}" > "${LOG}" 2>&1
echo "Completed successfully: ${LOG}"
