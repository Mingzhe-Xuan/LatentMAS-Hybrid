#!/usr/bin/env bash
###############################################################################
# analysis.sh - serial PBS entry point for the experiments under analysis/
#
# By default one PBS allocation runs the complete kernel analysis followed by
# the complete bidirectional STT analysis. Cached cells are validated and
# skipped by the task CLIs, so re-submitting the entry point is resumable.
#
# Examples:
#   qsub analysis.sh
#   qsub -v ANALYSIS_TARGET=kernel analysis.sh
#   qsub -v ANALYSIS_TARGET=stt,ANALYSIS_SMOKE=true,DATASET=aime2024 analysis.sh
#   bash analysis.sh --all --smoke --dataset aime2024 --dry-run
###############################################################################

#PBS -N latent_analysis
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  qsub analysis.sh
  qsub -v ANALYSIS_TARGET=kernel[,NAME=value...] analysis.sh
  qsub -v ANALYSIS_TARGET=stt[,NAME=value...] analysis.sh

Targets (default: all, executed serially as kernel then stt):
  --all                     Run kernel and then bidirectional STT
  --kernel                  Run only the kernel analysis
  --stt                     Run only the bidirectional STT analysis

Options:
  --stage NAME              all, collect, evaluate, analyze, or report
  --dataset NAME            Restrict both selected analyses to one dataset
  --smoke                   Use the isolated smoke matrices
  --max-samples INT         STT smoke sample count (requires --smoke)
  --device DEVICE           Task device (default: cuda)
  --extra-args "FLAGS"      Extra flags passed to every Python task
  --dry-run                 Validate arguments and print matrix summaries only

PBS -v names are ANALYSIS_TARGET, ANALYSIS_STAGE, DATASET, ANALYSIS_SMOKE,
MAX_SAMPLES, DEVICE, ANALYSIS_EXTRA_ARGS, ANALYSIS_CACHE_ROOT, and
ANALYSIS_RESULT_ROOT.
EOF
}

ANALYSIS_TARGET="${ANALYSIS_TARGET:-all}"
ANALYSIS_STAGE="${ANALYSIS_STAGE:-all}"
DATASET="${DATASET:-}"
ANALYSIS_SMOKE="${ANALYSIS_SMOKE:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DEVICE="${DEVICE:-cuda}"
ANALYSIS_EXTRA_ARGS="${ANALYSIS_EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) ANALYSIS_TARGET="all"; shift ;;
        --kernel) ANALYSIS_TARGET="kernel"; shift ;;
        --stt) ANALYSIS_TARGET="stt"; shift ;;
        --stage) ANALYSIS_STAGE="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --smoke) ANALYSIS_SMOKE="true"; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --extra-args) ANALYSIS_EXTRA_ARGS="$2"; shift 2 ;;
        --dry-run) DRY_RUN="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "${ANALYSIS_TARGET}" in all|kernel|stt) ;; *) echo "ERROR: invalid ANALYSIS_TARGET=${ANALYSIS_TARGET}" >&2; exit 2 ;; esac
case "${ANALYSIS_STAGE}" in all|collect|evaluate|analyze|report) ;; *) echo "ERROR: invalid ANALYSIS_STAGE=${ANALYSIS_STAGE}" >&2; exit 2 ;; esac
case "${ANALYSIS_SMOKE}" in true|false) ;; *) echo "ERROR: ANALYSIS_SMOKE must be true or false" >&2; exit 2 ;; esac
case "${DRY_RUN}" in true|false) ;; *) echo "ERROR: DRY_RUN must be true or false" >&2; exit 2 ;; esac
if [[ -n "${MAX_SAMPLES}" ]]; then
    [[ "${ANALYSIS_SMOKE}" == true ]] || { echo "ERROR: --max-samples requires --smoke" >&2; exit 2; }
    [[ "${MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --max-samples must be positive" >&2; exit 2; }
fi

PYTHON_BIN_EXPLICIT=false
[[ -n "${ANALYSIS_PYTHON:-}" ]] && PYTHON_BIN_EXPLICIT=true
PYTHON_BIN="${ANALYSIS_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
    else echo "ERROR: no Python interpreter; set ANALYSIS_PYTHON" >&2; exit 127
    fi
fi

build_matrices() {
    local target="$1"
    local -a command
    if [[ "${target}" == kernel ]]; then
        command=("${PYTHON_BIN}" analysis/pbs/build_job_matrix.py
                 --config analysis/configs/kernel_analysis.yaml --output analysis/jobs)
    else
        command=("${PYTHON_BIN}" analysis/pbs/build_stt_job_matrix.py
                 --config analysis/configs/bidirectional_stt.yaml --output analysis/jobs)
    fi
    [[ -n "${DATASET}" ]] && command+=(--dataset "${DATASET}")
    [[ "${ANALYSIS_SMOKE}" == true ]] && command+=(--smoke)
    if [[ "${target}" == stt && -n "${MAX_SAMPLES}" ]]; then command+=(--max-samples "${MAX_SAMPLES}"); fi
    [[ "${DRY_RUN}" == true ]] && command+=(--dry-run)
    "${command[@]}"
}

if [[ "${DRY_RUN}" == true ]]; then
    echo "Serial analysis dry run: target=${ANALYSIS_TARGET}, stage=${ANALYSIS_STAGE}, dataset=${DATASET:-all}, smoke=${ANALYSIS_SMOKE}"
    [[ "${ANALYSIS_TARGET}" == all || "${ANALYSIS_TARGET}" == kernel ]] && build_matrices kernel
    [[ "${ANALYSIS_TARGET}" == all || "${ANALYSIS_TARGET}" == stt ]] && build_matrices stt
    exit 0
fi

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "ERROR: model work must run inside a PBS allocation; submit with qsub analysis.sh." >&2
    exit 2
fi

# Environment setup mirrors exp.sh while allowing site-specific overrides.
module purge
module load "${ANALYSIS_PYTHON_MODULE:-python/3.12.13}"
ANALYSIS_VENV="${ANALYSIS_VENV:-/home/n2501945g/LatentMAS-Hybrid/.venv}"
source "${ANALYSIS_VENV}/bin/activate"
if [[ "${PYTHON_BIN_EXPLICIT}" == false ]]; then PYTHON_BIN="$(command -v python3)"; fi
cd "${PBS_O_WORKDIR}" || exit 1
if [[ ! -d analysis ]]; then echo "ERROR: submit from the repository root (analysis/ was not found)." >&2; exit 1; fi
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/home/n2501945g/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 "$((GPU_COUNT - 1))"); export CUDA_VISIBLE_DEVICES
fi

ANALYSIS_CACHE_ROOT="${ANALYSIS_CACHE_ROOT:-analysis_cache}"
ANALYSIS_RESULT_ROOT="${ANALYSIS_RESULT_ROOT:-analysis_result}"
LOG="${ANALYSIS_STATE_LOG:-${PBS_O_WORKDIR}/analysis_state.txt}"
EXTRA=()
if [[ -n "${ANALYSIS_EXTRA_ARGS}" ]]; then read -r -a EXTRA <<< "${ANALYSIS_EXTRA_ARGS}"; fi

echo "========================================================================"
echo "PBS job       : ${PBS_JOBID}"
echo "Target/order  : ${ANALYSIS_TARGET} (all = kernel -> stt)"
echo "Stage         : ${ANALYSIS_STAGE}"
echo "Dataset       : ${DATASET:-all}"
echo "Smoke/samples : ${ANALYSIS_SMOKE}/${MAX_SAMPLES:-default}"
echo "Device        : ${DEVICE}"
echo "Host          : $(hostname)"
nvidia-smi -L
echo "Log           : ${LOG}"
echo "========================================================================"

run_matrix() {
    local config="$1" task="$2" matrix="$3" cache_mode="$4"
    local rows index status
    rows=$(wc -l < "${matrix}")
    for ((index=1; index<=rows; index++)); do
        local -a args=(--config "${config}" --job-matrix "${matrix}" --job-index "${index}"
                       --cache-root "${ANALYSIS_CACHE_ROOT}" --result-root "${ANALYSIS_RESULT_ROOT}"
                       --device "${DEVICE}")
        [[ "${cache_mode}" == cache-only ]] && args+=(--cache-only)
        echo "[$(date --iso-8601=seconds)] START ${task} ${index}/${rows}" | tee -a "${LOG}"
        set +e
        "${PYTHON_BIN}" "analysis/tasks/${task}.py" "${args[@]}" "${EXTRA[@]}" >> "${LOG}" 2>&1
        status=$?
        set -e
        if (( status == 10 )); then
            echo "[$(date --iso-8601=seconds)] SKIP  ${task} ${index}/${rows} (validated cache hit)" | tee -a "${LOG}"
        elif (( status == 0 )); then
            echo "[$(date --iso-8601=seconds)] DONE  ${task} ${index}/${rows}" | tee -a "${LOG}"
        else
            echo "[$(date --iso-8601=seconds)] FAIL  ${task} ${index}/${rows} exit=${status}" | tee -a "${LOG}" >&2
            return "${status}"
        fi
    done
}

run_kernel() {
    local config="analysis/configs/kernel_analysis.yaml"
    build_matrices kernel >> "${LOG}" 2>&1
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == collect ]]; then
        run_matrix "${config}" collect_sender_trajectories analysis/jobs/sender.jsonl reuse
    fi
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == evaluate ]]; then
        run_matrix "${config}" evaluate_kernel_scaling analysis/jobs/kernel_scaling.jsonl reuse
        run_matrix "${config}" evaluate_perturbation_stability analysis/jobs/perturbation.jsonl reuse
        run_matrix "${config}" evaluate_sender_receiver_performance analysis/jobs/model_pairs.jsonl reuse
    fi
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == analyze ]]; then
        run_matrix "${config}" analyze_logit_entropy analysis/jobs/entropy_analysis.jsonl cache-only
        run_matrix "${config}" analyze_kernel_scaling analysis/jobs/scaling_analysis.jsonl cache-only
        run_matrix "${config}" analyze_aligned_state_variance analysis/jobs/variance_analysis.jsonl cache-only
        run_matrix "${config}" analyze_perturbation_stability analysis/jobs/stability_analysis.jsonl cache-only
        run_matrix "${config}" analyze_sender_receiver_performance analysis/jobs/model_pair_analysis.jsonl cache-only
    fi
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == report ]]; then
        run_matrix "${config}" build_kernel_analysis_report analysis/jobs/report.jsonl cache-only
    fi
}

run_stt() {
    local config="analysis/configs/bidirectional_stt.yaml"
    build_matrices stt >> "${LOG}" 2>&1
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == collect ]]; then
        run_matrix "${config}" collect_stt_planner_contexts analysis/jobs/stt_planner.jsonl reuse
    fi
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == evaluate ]]; then
        run_matrix "${config}" evaluate_bidirectional_stt analysis/jobs/stt_evaluation.jsonl reuse
    fi
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == analyze ]]; then
        run_matrix "${config}" analyze_bidirectional_stt analysis/jobs/stt_analysis.jsonl cache-only
    fi
    if [[ "${ANALYSIS_STAGE}" == all || "${ANALYSIS_STAGE}" == report ]]; then
        run_matrix "${config}" build_bidirectional_stt_report analysis/jobs/stt_report.jsonl cache-only
    fi
}

[[ "${ANALYSIS_TARGET}" == all || "${ANALYSIS_TARGET}" == kernel ]] && run_kernel
[[ "${ANALYSIS_TARGET}" == all || "${ANALYSIS_TARGET}" == stt ]] && run_stt
echo "Completed successfully; progress and output appended to: ${LOG}"
