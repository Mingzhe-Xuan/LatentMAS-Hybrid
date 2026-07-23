#!/bin/bash
###############################################################################
# run.sh - PBS job script for the AIME 2025 experiment suite
#
# Submit:  qsub run.sh
# Monitor: qstat -u $USER
# Runtime log: state.txt (in the directory where the job is submitted)
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
MODEL_NAME="Qwen/Qwen3-8B"  # Hugging Face model ID passed to --model_name.
TASK="aime2025"             # Evaluation dataset/task name.
PROMPT_SEQUENTIAL="sequential"      # Sequential multi-agent architecture.
PROMPT_HIERARCHICAL="hierarchical"  # Hierarchical multi-agent architecture.
MAX_SAMPLES=-1                # Number of examples; -1 evaluates all examples.
SPLIT="test"                 # Dataset split requested from the task loader.
DEVICE="cuda"                # PyTorch device used by the HF backend.

## --- Generation settings ---
MAX_NEW_TOKENS=20000  # Maximum tokens generated per response.
TEMPERATURE=0.6       # Sampling temperature.
TOP_P=0.95            # Nucleus-sampling probability threshold.
GENERATE_BS=2         # Generation batch size.
SEED=42               # Random seed for reproducibility.

## --- TextMAS / LatentMAS settings ---
TEXT_MAS_CONTEXT_LENGTH=-1  # TextMAS context limit; -1 means unlimited.
LATENT_STEPS=80             # Number of latent reasoning steps.
TRUST_REMOTE_CODE=true      # Pass --trust_remote_code when the model requires it.

## --- Alignment settings ---
ALIGN_RIDGE=1e-5           # Ridge regularization for linear alignment.
KERNEL_FEATURES=1024       # Random-feature count for kernel alignment.
KERNEL_TEMPERATURE=1.0     # Kernel alignment temperature.
KERNEL_CHUNK_SIZE=4096     # Chunk size for kernel alignment computation.

## --- vLLM backend settings ---
USE_VLLM=false              # Whether to enable the optional vLLM backend.
TENSOR_PARALLEL_SIZE=1      # Number of GPUs used for vLLM tensor parallelism.
GPU_MEMORY_UTILIZATION=0.9  # Fraction of each GPU vLLM may reserve.

## All run.py parameters are listed here. Edit the values above as needed.
## --method is required and is set individually in the five commands below.
## --model_name is required; run.py has no default.
COMMON=(
    # Core dataset / model settings
    --model_name "${MODEL_NAME}"             # Required model ID
    --task "${TASK}"                         # run.py default: gsm8k
    --max_samples "${MAX_SAMPLES}"           # run.py default: -1 (all samples)
    --split "${SPLIT}"                       # run.py default: test; AIME always uses train
    --device "${DEVICE}"                     # run.py default: cuda

    # Generation settings
    --max_new_tokens "${MAX_NEW_TOKENS}"     # run.py default: 20000
    --temperature "${TEMPERATURE}"           # run.py default: 0.6
    --top_p "${TOP_P}"                       # run.py default: 0.95
    --generate_bs "${GENERATE_BS}"           # run.py default: 20
    --seed "${SEED}"                         # run.py default: 42

    # TextMAS / LatentMAS settings
    --text_mas_context_length "${TEXT_MAS_CONTEXT_LENGTH}" # run.py default: -1 (unlimited)
    --latent_steps "${LATENT_STEPS}"         # run.py default: 80

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

## Boolean / optional arguments not enabled in this standard HF run:
##   --think                  default: false; add it to COMMON to insert <think>.
##   --kernel_seed SEED       default: None; omitted uses --seed.
##   --use_vllm               default: false; enables vLLM backend.
##   --enable_prefix_caching  default: false; only relevant with vLLM.
##   --use_second_HF_model    default: false; only relevant with latent_mas + vLLM.
##   --device2 DEVICE         default: None, then run.py uses --device.
##   --agent_models MODEL...  default: None; only used by latent_mas_hybrid.
##
## --align_method choices/default: identical (default), linear, kernel.
## The current suite runs all three methods explicitly below.

echo "========================================================================"
echo "  Job ID       : ${PBS_JOBID}"
echo "  Model        : ${MODEL_NAME}"
echo "  Task         : ${TASK}"
echo "  Latent prompts: ${PROMPT_SEQUENTIAL}, ${PROMPT_HIERARCHICAL}"
echo "  Generate BS  : ${GENERATE_BS}"
echo "  Max samples  : ${MAX_SAMPLES} (all)"
echo "  Max tokens   : ${MAX_NEW_TOKENS}"
echo "  Latent steps : ${LATENT_STEPS}"
echo "  vLLM         : ${USE_VLLM}"
echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================================================"
echo ""
## ========================== Run Experiment Suite =============================
## All experiment output is recorded in state.txt. The commands are chained so
## a failed run stops the suite and causes the PBS job to fail.
{
    # Baseline and TextMAS use the sequential architecture.
    python3 run.py --method baseline --prompt "${PROMPT_SEQUENTIAL}" "${COMMON[@]}" &&
    python3 run.py --method text_mas --prompt "${PROMPT_SEQUENTIAL}" "${COMMON[@]}" &&
    # Run all LatentMAS alignment methods sequentially before hierarchical.
    python3 run.py --method latent_mas --prompt "${PROMPT_SEQUENTIAL}" --align_method identical "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt "${PROMPT_SEQUENTIAL}" --align_method linear "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt "${PROMPT_SEQUENTIAL}" --align_method kernel "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt "${PROMPT_HIERARCHICAL}" --align_method identical "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt "${PROMPT_HIERARCHICAL}" --align_method linear "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt "${PROMPT_HIERARCHICAL}" --align_method kernel "${COMMON[@]}"
    STATUS=$?

    echo ""
    echo "Finished at: $(date)"
    echo "Exit status: ${STATUS}"
    exit "${STATUS}"
} > state.txt 2>&1
