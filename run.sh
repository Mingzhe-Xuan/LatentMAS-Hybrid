#!/bin/bash
###############################################################################
# run.sh - PBS job script for the AIME 2025 experiment suite
#
# Submit current config:  qsub run.sh
# Submit full matrix:     qsub -v FULL_EXP=true run.sh
# Local full matrix:      bash run.sh --full_exp
# Monitor: qstat -u $USER
# Runtime log: state.txt by default (overridable with STATE_FILE)
###############################################################################

## Job name
#PBS -N xmz

## Project funding code
#PBS -P ds_ccds_wei.lu

## Queue Name
#PBS -q gpu_ded

## Walltime - HH:MM:SS
#PBS -l walltime=72:00:00

## Resources - select 12 CPUs per GPU selected
#PBS -l select=1:ncpus=12:ngpus=1

## Merge stderr into stdout for cleaner output
#PBS -j oe

## ========================== Environment =====================================
module purge

## The repository imports vllm in methods/latent_mas.py even when --use_vllm is
## not passed, so this module is safer than the plain pytorch module if present.
# module load vllm/0.19.0

## Activate the virtual environment. If you don't have one, create it with:
## python3 -m venv .venv, and then install the requirements with:
## pip install -r requirements.txt
module load python/3.12.13
source /home/n2501945g/LatentMAS-Hybrid/.venv/bin/activate

cd "${PBS_O_WORKDIR}" || exit 1
if [ ! -f run.py ] && [ -f LatentMAS/run.py ]; then
    cd LatentMAS || exit 1
fi
if [ ! -f run.py ]; then
    echo "ERROR: run.py not found. Submit from the LatentMAS directory or its parent."
    exit 1
fi
export PYTHONUNBUFFERED=1

FULL_EXP="${FULL_EXP:-false}"
TASK_ONLY="${TASK_ONLY:-false}"
STATE_FILE="${STATE_FILE:-state.txt}"
for ARG in "$@"; do
    case "${ARG}" in
        --full_exp) FULL_EXP=true ;;
        *)
            echo "ERROR: Unknown argument: ${ARG}"
            echo "Usage: bash run.sh [--full_exp]"
            exit 2
            ;;
    esac
done

## Optional Hugging Face cache location. Uncomment and edit if your cluster
## recommends a project scratch/cache directory.
export HF_HOME=/home/n2501945g/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

## ========================== Debug Info ======================================
echo "PBS_NODEFILE: ${PBS_NODEFILE}"
cat "${PBS_NODEFILE}"
echo ""
echo "hostname: $(hostname)"
echo "working directory: $(pwd)"
echo "date: $(date)"
echo "nvidia-smi check:"
nvidia-smi -L || { echo "ERROR: nvidia-smi failed"; exit 1; }
echo ""

## ========================== Fix CUDA_VISIBLE_DEVICES =========================
## PBS may set CUDA_VISIBLE_DEVICES to GPU UUIDs. PyTorch expects integer IDs.
if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
    echo "Converted CUDA_VISIBLE_DEVICES to indices: ${CUDA_VISIBLE_DEVICES}"
fi

## ========================== Experiment Config ================================
## Keep named values here so the launch command and job summary stay in sync.

## --- Core dataset / model settings ---
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}" # Hugging Face model ID passed to --model_name.
TASK="${TASK:-humanevalplus}"              # Evaluation dataset/task name.
PROMPT_SEQUENTIAL="sequential"      # Sequential multi-agent architecture.
PROMPT_HIERARCHICAL="hierarchical"  # Hierarchical multi-agent architecture.
MAX_SAMPLES=30                # Number of examples; -1 evaluates all examples.
SPLIT="test"                 # Dataset split requested from the task loader.
DEVICE="cuda"                # PyTorch device used by the HF backend.

## --- Generation settings ---
# Empty means use params_dict.json[TASK].max_token; an absent/unknown task
# falls back to 20000. Set a number here to explicitly override every task.
MAX_NEW_TOKENS=""
resolve_max_new_tokens() {
    if [ -n "${MAX_NEW_TOKENS}" ]; then
        printf '%s\n' "${MAX_NEW_TOKENS}"
        return
    fi

    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

fallback = 20000
try:
    params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
    task_params = params.get(sys.argv[1], {})
    value = task_params.get("max_token", fallback) if isinstance(task_params, dict) else fallback
    print(value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback)
except (OSError, json.JSONDecodeError):
    print(fallback)
PY
}
TEMPERATURE=0.6       # Sampling temperature.
TOP_P=0.95            # Nucleus-sampling probability threshold.
# Empty means use params_dict.json[TASK].generation_bs; fallback: 10.
GENERATE_BS="${GENERATE_BS:-}"
resolve_generate_bs() {
    if [ -n "${GENERATE_BS}" ]; then
        printf '%s\n' "${GENERATE_BS}"
        return
    fi

    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

fallback = 10
try:
    params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
    task_params = params.get(sys.argv[1], {})
    value = task_params.get("generation_bs", fallback) if isinstance(task_params, dict) else fallback
    print(value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback)
except (OSError, json.JSONDecodeError):
    print(fallback)
PY
}
SEED=42               # Random seed for reproducibility.

## --- TextMAS / LatentMAS settings ---
TEXT_MAS_CONTEXT_LENGTH=-1  # TextMAS context limit; -1 means unlimited.
# Empty means use params_dict.json[TASK].latent_steps[PROMPT]; fallback: 20.
LATENT_STEPS="${LATENT_STEPS:-}"
resolve_latent_steps() {
    if [ -n "${LATENT_STEPS}" ]; then
        printf '%s\n' "${LATENT_STEPS}"
        return
    fi

    python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

fallback = 20
try:
    params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
    task_params = params.get(sys.argv[1], {})
    latent_steps = task_params.get("latent_steps", {}) if isinstance(task_params, dict) else {}
    value = latent_steps.get(sys.argv[2], fallback) if isinstance(latent_steps, dict) else fallback
    print(value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else fallback)
except (OSError, json.JSONDecodeError):
    print(fallback)
PY
}
TRUST_REMOTE_CODE=true      # Pass --trust_remote_code when the model requires it.
SEQUENTIAL_INFO_ONLY=false   # Retain only each agent's own prompt + latent KV before the next agent.
LATENT_ONLY=false            # Retain only latent KV before the next agent (implies SEQUENTIAL_INFO_ONLY).
THINK=true                  # Insert <think> before latent rollout starts.

## --- Alignment settings ---
ALIGN_RIDGE=1e-5           # Ridge regularization for linear alignment.
KERNEL_FEATURES=1024       # Random-feature count for kernel alignment.
KERNEL_TEMPERATURE=1.0     # Kernel alignment temperature.
KERNEL_CHUNK_SIZE=4096     # Chunk size for kernel alignment computation.

## --- vLLM backend settings ---
USE_VLLM=false              # Whether to enable the optional vLLM backend.
TENSOR_PARALLEL_SIZE=1      # Number of GPUs used for vLLM tensor parallelism.
GPU_MEMORY_UTILIZATION=0.9  # Fraction of each GPU vLLM may reserve.

build_common_args() {
local prompt="$1"
RESOLVED_LATENT_STEPS="$(resolve_latent_steps "${TASK}" "${prompt}")"
COMMON=(
    # Core dataset / model settings
    --model_name "${MODEL_NAME}"             # Required model ID
    --task "${TASK}"                         # run.py default: humanevalplus
    --max_samples "${MAX_SAMPLES}"           # run.py default: -1 (all samples)
    --split "${SPLIT}"                       # run.py default: test; AIME always uses train
    --device "${DEVICE}"                     # run.py default: cuda

    # Generation settings
    --max_new_tokens "${RESOLVED_MAX_NEW_TOKENS}" # Task default from params_dict.json; fallback: 20000
    --temperature "${TEMPERATURE}"           # run.py default: 0.6
    --top_p "${TOP_P}"                       # run.py default: 0.95
    --generate_bs "${RESOLVED_GENERATE_BS}"  # Task default from params_dict.json; fallback: 10
    --seed "${SEED}"                         # run.py default: 42

    # TextMAS / LatentMAS settings
    --text_mas_context_length "${TEXT_MAS_CONTEXT_LENGTH}" # run.py default: -1 (unlimited)
    --latent_steps "${RESOLVED_LATENT_STEPS}" # Task/prompt default from params_dict.json; fallback: 20

    # Alignment settings. --align_method is varied per LatentMAS command below.
    --align_ridge "${ALIGN_RIDGE}"           # run.py default: 1e-5; used by linear
    --kernel_features "${KERNEL_FEATURES}"   # run.py default: 1024; used by kernel
    --kernel_temperature "${KERNEL_TEMPERATURE}" # run.py default: 1.0; used by kernel
    --kernel_chunk_size "${KERNEL_CHUNK_SIZE}" # run.py default: 4096; used by kernel

    # vLLM numeric settings; ignored unless --use_vllm is enabled below.
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" # run.py default: 1
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" # run.py default: 0.9
)

# Qwen3-8B requires custom generation code; run.py default is disabled.
if [ "${TRUST_REMOTE_CODE}" = true ]; then
    COMMON+=(--trust_remote_code)
fi
}

## Boolean / optional arguments for LatentMAS methods.
# LATENT_ONLY implies
# SEQUENTIAL_INFO_ONLY in the Python implementation, so both flags may safely
# be passed when both variables are true.
LATENT_CACHE_ARGS=()
if [ "${THINK}" = true ]; then
    LATENT_CACHE_ARGS+=(--think)
fi
if [ "${SEQUENTIAL_INFO_ONLY}" = true ]; then
    LATENT_CACHE_ARGS+=(--sequential_info_only)
fi
if [ "${LATENT_ONLY}" = true ]; then
    LATENT_CACHE_ARGS+=(--latent_only)
fi
##   --think                  default: true; adds <think> before latent rollout.
##   --kernel_seed SEED       default: None; omitted uses --seed.
##   --use_vllm               default: false; enables vLLM backend.
##   --sequential_info_only   default: false; preserve only the current agent's prompt + latent KV cache.
##   --latent_only            default: false; preserve only latent KV cache; implies --sequential_info_only.
##   --enable_prefix_caching  default: false; only relevant with vLLM.
##   --use_second_HF_model    default: false; only relevant with latent_mas + vLLM.
##   --device2 DEVICE         default: None, then run.py uses --device.
##   --agent_models MODEL...  default: None; only used by latent_mas_hybrid.
##
## --align_method choices/default: identical (default), linear, kernel.
## The current suite runs all three methods explicitly below.

## ========================== Run Experiment Suite =============================
## All experiment output is recorded in state.txt. The commands are chained so
## a failed run stops the suite and causes the PBS job to fail.
run_suite() {
    local suite_model="$1"
    local suite_task="$2"
    MODEL_NAME="${suite_model}"
    TASK="${suite_task}"
    RESOLVED_MAX_NEW_TOKENS="$(resolve_max_new_tokens "${TASK}")"
    RESOLVED_GENERATE_BS="$(resolve_generate_bs "${TASK}")"
    RESOLVED_SEQUENTIAL_LATENT_STEPS="$(resolve_latent_steps "${TASK}" "${PROMPT_SEQUENTIAL}")"
    RESOLVED_HIERARCHICAL_LATENT_STEPS="$(resolve_latent_steps "${TASK}" "${PROMPT_HIERARCHICAL}")"

    echo "========================================================================"
    echo "  Job ID       : ${PBS_JOBID:-local}"
    echo "  Full exp     : ${FULL_EXP}"
    echo "  Model        : ${MODEL_NAME}"
    echo "  Task         : ${TASK}"
    echo "  Latent prompts: ${PROMPT_SEQUENTIAL}, ${PROMPT_HIERARCHICAL}"
    echo "  Generate BS  : ${RESOLVED_GENERATE_BS}"
    echo "  Max samples  : ${MAX_SAMPLES}"
    echo "  Max tokens   : ${RESOLVED_MAX_NEW_TOKENS}"
    echo "  Latent steps : ${PROMPT_SEQUENTIAL}=${RESOLVED_SEQUENTIAL_LATENT_STEPS}, ${PROMPT_HIERARCHICAL}=${RESOLVED_HIERARCHICAL_LATENT_STEPS}"
    echo "  vLLM         : ${USE_VLLM}"
    echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "  Sequential info only: ${SEQUENTIAL_INFO_ONLY}"
    echo "  Latent only  : ${LATENT_ONLY}"
    echo "  Think token  : ${THINK}"
    echo "========================================================================"

    # Baseline and TextMAS use the sequential architecture.
    build_common_args "${PROMPT_SEQUENTIAL}"
    python3 run.py --method baseline --prompt "${PROMPT_SEQUENTIAL}" "${COMMON[@]}" &&
    build_common_args "${PROMPT_HIERARCHICAL}"
    python3 run.py --method baseline --prompt "${PROMPT_HIERARCHICAL}" "${COMMON[@]}" &&
    build_common_args "${PROMPT_SEQUENTIAL}"
    python3 run.py --method text_mas --prompt "${PROMPT_SEQUENTIAL}" "${COMMON[@]}" &&
    build_common_args "${PROMPT_HIERARCHICAL}"
    python3 run.py --method text_mas --prompt "${PROMPT_HIERARCHICAL}" "${COMMON[@]}" &&
    # Run all LatentMAS alignment methods sequentially before hierarchical.
    build_common_args "${PROMPT_SEQUENTIAL}"
    python3 run.py --method latent_mas --prompt "${PROMPT_SEQUENTIAL}" --align_method identical "${COMMON[@]}" "${LATENT_CACHE_ARGS[@]}" && # very weak
    python3 run.py --method latent_mas --prompt "${PROMPT_SEQUENTIAL}" --align_method linear "${COMMON[@]}" "${LATENT_CACHE_ARGS[@]}" && # Easy to explode
    python3 run.py --method latent_mas --prompt "${PROMPT_SEQUENTIAL}" --align_method kernel "${COMMON[@]}" "${LATENT_CACHE_ARGS[@]}" &&
    build_common_args "${PROMPT_HIERARCHICAL}"
    python3 run.py --method latent_mas --prompt "${PROMPT_HIERARCHICAL}" --align_method identical "${COMMON[@]}" "${LATENT_CACHE_ARGS[@]}" &&
    python3 run.py --method latent_mas --prompt "${PROMPT_HIERARCHICAL}" --align_method linear "${COMMON[@]}" "${LATENT_CACHE_ARGS[@]}" &&
    python3 run.py --method latent_mas --prompt "${PROMPT_HIERARCHICAL}" --align_method kernel "${COMMON[@]}" "${LATENT_CACHE_ARGS[@]}"
}

if [ "${FULL_EXP}" = true ]; then
    MODELS=("Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
    if [ "${TASK_ONLY}" = true ]; then
        TASKS=("${TASK}")
    else
        mapfile -t TASKS < <(python3 - <<'PY'
import json
from pathlib import Path

params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
for task in params:
    print(task)
PY
)
    fi
else
    MODELS=("${MODEL_NAME}")
    TASKS=("${TASK}")
fi

{
    STATUS=0
    for CURRENT_MODEL in "${MODELS[@]}"; do
        for CURRENT_TASK in "${TASKS[@]}"; do
            run_suite "${CURRENT_MODEL}" "${CURRENT_TASK}" || {
                STATUS=$?
                echo "ERROR: Suite failed for model=${CURRENT_MODEL}, task=${CURRENT_TASK}"
                break 2
            }
        done
    done

    echo ""
    echo "Finished at: $(date)"
    echo "Exit status: ${STATUS}"
    exit "${STATUS}"
} > "${STATE_FILE}" 2>&1
