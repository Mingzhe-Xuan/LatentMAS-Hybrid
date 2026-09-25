#!/bin/bash
#PBS -N kernel_gate
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -J 1-18%3
#PBS -j oe

set -euo pipefail

# This draft is not a controlled scientific comparison. Fail closed.
echo "BLOCKED: Soft and Kernel do not yet share a fixed rollout budget." >&2
echo "Fixed feature IDs came from CPU reconstruction, not a verified generation-time ORF artifact." >&2
echo "Do not submit: match budgets and calibrate with the actual saved operator first." >&2
exit 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PBS_O_WORKDIR:-${SCRIPT_DIR}}"

SEEDS=(42 43 44)
VARIANTS=(exact_soft kernel_soft kernel_argmax kernel_fixed kernel_topk2 kernel_topk8)
# Dominant features selected on cached AIME 2024 C0 trajectories, then held
# fixed for this cross-task MedQA evaluation (matching ORF seed, m=2048,
# tau=0.6).  The same three features also dominate MedQA trajectories, but
# those MedQA routes are not used for feature selection.
declare -A FIXED_FEATURE=(
    [42]=359
    [43]=1477
    [44]=123
)

INDEX="${PBS_ARRAY_INDEX:-1}"
if ! [[ "${INDEX}" =~ ^[0-9]+$ ]] || (( INDEX < 1 || INDEX > 18 )); then
    echo "ERROR: PBS_ARRAY_INDEX must be in 1..18, got ${INDEX}" >&2
    exit 2
fi

OFFSET=$((INDEX - 1))
SEED_INDEX=$((OFFSET / ${#VARIANTS[@]}))
VARIANT_INDEX=$((OFFSET % ${#VARIANTS[@]}))
SEED="${SEEDS[${SEED_INDEX}]}"
VARIANT="${VARIANTS[${VARIANT_INDEX}]}"

CONFIG_ALIGNMENT=kernel
KERNEL_GATE_MODE=soft
KERNEL_FIXED_FEATURE=0
KERNEL_TOPK=8
case "${VARIANT}" in
    exact_soft) CONFIG_ALIGNMENT=soft ;;
    kernel_soft) ;;
    kernel_argmax) KERNEL_GATE_MODE=argmax ;;
    kernel_fixed)
        KERNEL_GATE_MODE=fixed
        KERNEL_FIXED_FEATURE="${FIXED_FEATURE[${SEED}]}"
        ;;
    kernel_topk2)
        KERNEL_GATE_MODE=topk
        KERNEL_TOPK=2
        ;;
    kernel_topk8)
        KERNEL_GATE_MODE=topk
        KERNEL_TOPK=8
        ;;
    *) echo "ERROR: unsupported variant ${VARIANT}" >&2; exit 2 ;;
esac

export FULL_EXP=false TASK_ONLY=true SINGLE_CONFIG=true CAPTURE_ALL_OUTPUT=true
export TASK=medqa MODEL_NAME=Qwen/Qwen3-8B CONFIG_METHOD=latent_mas
export CONFIG_PROMPT=sequential CONFIG_ALIGNMENT
export SEED TIMES=1 MAX_SAMPLES=-1
export KERNEL_FEATURES=2048 KERNEL_TEMPERATURE=0.6 KERNEL_CHUNK_SIZE=4096
export KERNEL_GATE_MODE KERNEL_FIXED_FEATURE KERNEL_TOPK
export RESULT_ROOT=result/kernel_gate_ablation LOG_ROOT=logging/kernel_gate_ablation
export STATE_FILE="state/kernel_gate_ablation/medqa_seed${SEED}_${VARIANT}.txt"

echo "Running MedQA gate ablation: seed=${SEED}, variant=${VARIANT}, fixed=${KERNEL_FIXED_FEATURE}, topk=${KERNEL_TOPK}"
bash run.sh
