#!/usr/bin/env bash
set -euo pipefail

STAGE=all
DATASET=""
SMOKE=false
DRY_RUN=false
MAX_RUNNING="${MAX_RUNNING:-3}"
DEPENDENCY_OPERATOR="${DEPENDENCY_OPERATOR:-afterok}"
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
case "${DEPENDENCY_OPERATOR}" in afterok|afterokarray) ;; *) echo "ERROR: invalid dependency operator" >&2; exit 2 ;; esac

PBS_TEMPLATE="analysis/pbs/analysis_job.pbs"
CONFIG_PATH="analysis/configs/kernel_analysis.yaml"
BUILD=(python3 analysis/pbs/build_job_matrix.py --config "${CONFIG_PATH}" --output analysis/jobs)
[[ -n "${DATASET}" ]] && BUILD+=(--dataset "${DATASET}")
[[ "${SMOKE}" == true ]] && BUILD+=(--smoke)
[[ "${DRY_RUN}" == true ]] && BUILD+=(--dry-run)
"${BUILD[@]}"
[[ "${DRY_RUN}" == true ]] && exit 0

submit_array() {
  local task_name="$1" matrix="$2" cache_mode="$3" dependency="${4:-}"
  local rows
  rows=$(wc -l < "${matrix}")
  local -a command=(qsub -J "1-${rows}%${MAX_RUNNING}" -v "TASK_NAME=${task_name},JOB_MATRIX=${matrix},CACHE_MODE=${cache_mode},CONFIG_PATH=${CONFIG_PATH}" )
  [[ -n "${dependency}" ]] && command+=(-W "depend=${DEPENDENCY_OPERATOR}:${dependency}")
  command+=("${PBS_TEMPLATE}")
  "${command[@]}"
}

if [[ "${STAGE}" == collect ]]; then submit_array collect_sender_trajectories analysis/jobs/sender.jsonl reuse; exit; fi
if [[ "${STAGE}" == evaluate ]]; then
  submit_array evaluate_kernel_scaling analysis/jobs/kernel_scaling.jsonl reuse
  submit_array evaluate_perturbation_stability analysis/jobs/perturbation.jsonl reuse
  submit_array evaluate_sender_receiver_performance analysis/jobs/model_pairs.jsonl reuse
  exit
fi
if [[ "${STAGE}" == model-pairs ]]; then
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
