#!/usr/bin/env bash
set -euo pipefail

STAGE=all
DATASET=""
SMOKE=false
DRY_RUN=false
MAX_RUNNING="${SLURM_MAX_RUNNING:-1}"
while (( $# )); do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --smoke) SMOKE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "${STAGE}" in all|collect|evaluate|analyze|model-pairs) ;; *) echo "ERROR: invalid stage ${STAGE}" >&2; exit 2 ;; esac
if ! command -v sbatch >/dev/null 2>&1 && [[ "${DRY_RUN}" != true ]]; then
  echo "ERROR: sbatch is unavailable" >&2
  exit 127
fi

WORKER="analysis/slurm/analysis_job.slurm"
CONFIG_PATH="analysis/configs/kernel_analysis.yaml"
mkdir -p state/analysis
BUILD=(python analysis/pbs/build_job_matrix.py --config "${CONFIG_PATH}" --output analysis/jobs)
[[ -n "${DATASET}" ]] && BUILD+=(--dataset "${DATASET}")
[[ "${SMOKE}" == true ]] && BUILD+=(--smoke)
"${BUILD[@]}"

submit_array() {
  local task_name="$1" matrix="$2" cache_mode="$3" dependency="${4:-}"
  local rows
  rows=$(wc -l < "${matrix}")
  local -a command=(sbatch --parsable --job-name "analysis_${task_name}"
    --partition "${SLURM_PARTITION:-compute}" --gres "${SLURM_GRES:-gpu:1}"
    --cpus-per-task "${SLURM_CPUS:-4}" --mem "${SLURM_MEMORY:-64G}"
    --time "${SLURM_TIME:-72:00:00}" --array "1-${rows}%${MAX_RUNNING}"
    --output "state/analysis/slurm-%A_%a.out"
    --export "ALL,TASK_NAME=${task_name},JOB_MATRIX=${matrix},CACHE_MODE=${cache_mode},CONFIG_PATH=${CONFIG_PATH}")
  [[ -n "${dependency}" ]] && command+=(--dependency "afterok:${dependency}")
  command+=("${WORKER}")
  if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' "${command[@]}" >&2; printf '\n' >&2
    printf 'dry-%s' "${task_name}"
  else
    "${command[@]}"
  fi
}

if [[ "${STAGE}" == collect ]]; then submit_array collect_sender_trajectories analysis/jobs/sender.jsonl reuse; exit; fi
if [[ "${STAGE}" == evaluate ]]; then
  submit_array evaluate_kernel_scaling analysis/jobs/kernel_scaling.jsonl reuse
  submit_array evaluate_perturbation_stability analysis/jobs/perturbation.jsonl reuse
  submit_array evaluate_sender_receiver_performance analysis/jobs/model_pairs.jsonl reuse
  exit
fi
if [[ "${STAGE}" == analyze ]]; then
  submit_array analyze_logit_entropy analysis/jobs/entropy_analysis.jsonl cache-only
  submit_array analyze_kernel_scaling analysis/jobs/scaling_analysis.jsonl cache-only
  submit_array analyze_aligned_state_variance analysis/jobs/variance_analysis.jsonl cache-only
  submit_array analyze_perturbation_stability analysis/jobs/stability_analysis.jsonl cache-only
  submit_array analyze_sender_receiver_performance analysis/jobs/model_pair_analysis.jsonl cache-only
  exit
fi
if [[ "${STAGE}" == model-pairs ]]; then submit_array evaluate_sender_receiver_performance analysis/jobs/model_pairs.jsonl reuse; exit; fi

SENDER_JOB="$(submit_array collect_sender_trajectories analysis/jobs/sender.jsonl reuse)"
SCALING_JOB="$(submit_array evaluate_kernel_scaling analysis/jobs/kernel_scaling.jsonl reuse "${SENDER_JOB}")"
PERTURB_JOB="$(submit_array evaluate_perturbation_stability analysis/jobs/perturbation.jsonl reuse "${SENDER_JOB}")"
PAIR_JOB="$(submit_array evaluate_sender_receiver_performance analysis/jobs/model_pairs.jsonl reuse "${SENDER_JOB}")"
ENTROPY_JOB="$(submit_array analyze_logit_entropy analysis/jobs/entropy_analysis.jsonl cache-only "${SCALING_JOB}")"
SCALING_ANALYSIS_JOB="$(submit_array analyze_kernel_scaling analysis/jobs/scaling_analysis.jsonl cache-only "${SCALING_JOB}")"
VARIANCE_JOB="$(submit_array analyze_aligned_state_variance analysis/jobs/variance_analysis.jsonl cache-only "${SCALING_JOB}")"
STABILITY_JOB="$(submit_array analyze_perturbation_stability analysis/jobs/stability_analysis.jsonl cache-only "${PERTURB_JOB}:${SCALING_JOB}")"
PAIR_ANALYSIS_JOB="$(submit_array analyze_sender_receiver_performance analysis/jobs/model_pair_analysis.jsonl cache-only "${PAIR_JOB}:${SCALING_JOB}")"
REPORT_DEPENDENCY="${ENTROPY_JOB}:${SCALING_ANALYSIS_JOB}:${VARIANCE_JOB}:${STABILITY_JOB}:${PAIR_ANALYSIS_JOB}"
REPORT_JOB="$(submit_array build_kernel_analysis_report analysis/jobs/report.jsonl cache-only "${REPORT_DEPENDENCY}")"
printf 'sender=%s scaling=%s perturbation=%s model_pairs=%s report=%s\n' "${SENDER_JOB}" "${SCALING_JOB}" "${PERTURB_JOB}" "${PAIR_JOB}" "${REPORT_JOB}"
