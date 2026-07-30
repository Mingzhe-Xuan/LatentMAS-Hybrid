#!/bin/bash
###############################################################################
# Scan LatentMAS latent_steps on the first 30 samples of every benchmark.
#
# Submit: qsub scan_step.sh
# Override alignment: qsub -v ALIGN_METHOD=identical scan_step.sh
# Runtime log: step_state.txt
# Incremental machine-readable summary: step_summary.tsv
###############################################################################

#PBS -N x_scan_step
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

module purge
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

STATE_FILE="${STATE_FILE:-step_state.txt}"
SUMMARY_FILE="${SUMMARY_FILE:-step_summary.tsv}"
exec > "${STATE_FILE}" 2>&1
export PYTHONUNBUFFERED=1

export HF_HOME=/home/n2501945g/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
    GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
fi

MODEL_NAME="Qwen/Qwen3-8B"
PROMPT="hierarchical"
ALIGN_METHOD="${ALIGN_METHOD:-kernel}"
LATENT_STEP_VALUES=(0 10 20 40 80)

MAX_SAMPLES=30
SPLIT="test"
DEVICE="cuda"
TEMPERATURE=0.6
TOP_P=0.95
GENERATE_BS=10
SEED=42
TEXT_MAS_CONTEXT_LENGTH=-1
ALIGN_RIDGE=1e-5
KERNEL_FEATURES=1024
KERNEL_TEMPERATURE=1.0
KERNEL_CHUNK_SIZE=4096

mapfile -t TASK_VALUES < <(python3 - <<'PY'
import json
from pathlib import Path

params = json.loads(Path("params_dict.json").read_text(encoding="utf-8"))
for task in params:
    print(task)
PY
)

resolve_max_new_tokens() {
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

print_summary() {
    python3 - "${SUMMARY_FILE}" "${1:-}" <<'PY'
import csv
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
task_filter = sys.argv[2]
with summary_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if task_filter:
    rows = [row for row in rows if row["task"] == task_filter]
if not rows:
    print("No completed runs to summarize.")
    raise SystemExit

headers = list(rows[0])
widths = {
    header: max(len(header), *(len(row[header]) for row in rows))
    for header in headers
}
print("  ".join(header.rjust(widths[header]) for header in headers))
print("  ".join("-" * widths[header] for header in headers))
for row in rows:
    print("  ".join(row[header].rjust(widths[header]) for header in headers))
PY
}

printf '%s\n' \
    $'task\tstep\tstatus\taccuracy\tcorrect\tsamples\ttime_s\ts_per_sample\ttext_in\tlatent_in\ttext_out\tlatent_out\ttokens_total\tresult_json' \
    > "${SUMMARY_FILE}"

echo "========================================================================"
echo "Latent-step scan started at: $(date)"
echo "Model        : ${MODEL_NAME}"
echo "Tasks        : ${TASK_VALUES[*]}"
echo "Method       : latent_mas"
echo "Prompt       : ${PROMPT}"
echo "Alignment    : ${ALIGN_METHOD}"
echo "Latent steps : ${LATENT_STEP_VALUES[*]}"
echo "Max samples  : ${MAX_SAMPLES}"
echo "Summary file : ${SUMMARY_FILE}"
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================================================"

STATUS=0

for TASK in "${TASK_VALUES[@]}"; do
    MAX_NEW_TOKENS="$(resolve_max_new_tokens "${TASK}")"
    RESULT_PATTERN="${TASK}_latent_mas_prompt_${PROMPT}_model_Qwen_Qwen3-8B_align_${ALIGN_METHOD}_*.json"

    echo ""
    echo "========================================================================"
    echo "DATASET: ${TASK} (first ${MAX_SAMPLES} samples, max tokens ${MAX_NEW_TOKENS})"
    echo "========================================================================"

    for LATENT_STEPS in "${LATENT_STEP_VALUES[@]}"; do
        echo ""
        echo "------------------------------------------------------------------------"
        echo "Running task=${TASK}, latent_steps=${LATENT_STEPS} at $(date)"
        echo "------------------------------------------------------------------------"

        MARKER_FILE="$(mktemp)"
        python3 run.py \
            --method latent_mas \
            --model_name "${MODEL_NAME}" \
            --task "${TASK}" \
            --prompt "${PROMPT}" \
            --align_method "${ALIGN_METHOD}" \
            --max_samples "${MAX_SAMPLES}" \
            --split "${SPLIT}" \
            --device "${DEVICE}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --generate_bs "${GENERATE_BS}" \
            --seed "${SEED}" \
            --text_mas_context_length "${TEXT_MAS_CONTEXT_LENGTH}" \
            --latent_steps "${LATENT_STEPS}" \
            --align_ridge "${ALIGN_RIDGE}" \
            --kernel_features "${KERNEL_FEATURES}" \
            --kernel_temperature "${KERNEL_TEMPERATURE}" \
            --kernel_chunk_size "${KERNEL_CHUNK_SIZE}" \
            --tensor_parallel_size 1 \
            --gpu_memory_utilization 0.9 \
            --trust_remote_code \
            --think
        RUN_STATUS=$?

        if [ "${RUN_STATUS}" -ne 0 ]; then
            echo "ERROR: task=${TASK}, latent_steps=${LATENT_STEPS} failed with status ${RUN_STATUS}."
            printf '%s\t%s\tfailed(%s)\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n' \
                "${TASK}" "${LATENT_STEPS}" "${RUN_STATUS}" >> "${SUMMARY_FILE}"
            STATUS=1
            continue
        fi

        RESULT_FILE="$(
            find result -type f -name "${RESULT_PATTERN}" -newer "${MARKER_FILE}" -print |
                sort |
                tail -n 1
        )"
        if [ -z "${RESULT_FILE}" ]; then
            echo "ERROR: Could not locate the result JSON for task=${TASK}, latent_steps=${LATENT_STEPS}."
            printf '%s\t%s\tmissing_json\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n' \
                "${TASK}" "${LATENT_STEPS}" >> "${SUMMARY_FILE}"
            STATUS=1
            continue
        fi

        python3 - "${SUMMARY_FILE}" "${TASK}" "${LATENT_STEPS}" "${RESULT_FILE}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_file, task, step, result_file = sys.argv[1:]
summary = json.loads(Path(result_file).read_text(encoding="utf-8"))
results = summary["results"]
timing = summary["timing"]
tokens = results.get("tokens", {})

def token_total(name):
    value = tokens.get(name, {})
    return int(value.get("total", 0)) if isinstance(value, dict) else 0

text_in = token_total("text_input")
latent_in = token_total("latent_input")
text_out = token_total("text_output")
latent_out = token_total("latent_output")
row = [
    task,
    step,
    "ok",
    f'{results["accuracy"]:.6f}',
    results["correct"],
    results["processed"],
    f'{timing["total_seconds"]:.4f}',
    f'{timing["seconds_per_sample"]:.4f}',
    text_in,
    latent_in,
    text_out,
    latent_out,
    text_in + latent_in + text_out + latent_out,
    result_file,
]
with Path(summary_file).open("a", encoding="utf-8", newline="") as handle:
    csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(row)
PY
        echo "Result JSON: ${RESULT_FILE}"
    done

    echo ""
    echo "SUMMARY FOR ${TASK}"
    print_summary "${TASK}"
done

echo ""
echo "========================================================================"
echo "FULL LATENT-STEP SUMMARY"
echo "========================================================================"
print_summary
echo ""
echo "tokens_total = text_in + latent_in + text_out + latent_out"
echo "Note: input token totals are summed across agent roles and may include reused context."

echo ""
echo "Finished at: $(date)"
echo "Exit status: ${STATUS}"
exit "${STATUS}"
