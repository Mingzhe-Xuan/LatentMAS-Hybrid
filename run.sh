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
#PBS -l walltime=48:00:00

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
COMMON=(
    --model_name Qwen/Qwen3-8B
    --task aime2025
    --generate_bs 2
    --max_samples -1
    --max_new_tokens 16384
    --trust_remote_code
)

echo "========================================================================"
echo "  Job ID       : ${PBS_JOBID}"
echo "  Model        : Qwen/Qwen3-8B"
echo "  Task         : aime2025"
echo "  Generate BS  : 2"
echo "  Max samples  : -1 (all)"
echo "  Max tokens   : 16384"
echo "  CUDA devices : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================================================"
echo ""

## ========================== Run Experiment Suite =============================
## All experiment output is recorded in state.txt. The commands are chained so
## a failed run stops the suite and causes the PBS job to fail.
{
    python3 run.py --method baseline "${COMMON[@]}" &&
    python3 run.py --method text_mas --prompt sequential "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt sequential --align_method identical "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt sequential --align_method linear "${COMMON[@]}" &&
    python3 run.py --method latent_mas --prompt sequential --align_method kernel "${COMMON[@]}"
    STATUS=$?

    echo ""
    echo "Finished at: $(date)"
    echo "Exit status: ${STATUS}"
    exit "${STATUS}"
} > state.txt 2>&1
