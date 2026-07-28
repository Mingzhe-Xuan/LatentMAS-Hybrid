#!/bin/bash
###############################################################################
# Scan LatentMAS latent_steps on the default Qwen3-8B HumanEval+ experiment.
#
# Submit: qsub scan_step.sh
# Override alignment: qsub -v ALIGN_METHOD=identical scan_step.sh
# Runtime log and final summary: step_state.txt
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
TASK="humanevalplus"
PROMPT="sequential"
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

MAX_NEW_TOKENS="$(python3 - "${TASK}" <<'PY'
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
)"

echo "========================================================================"
echo "Latent-step scan started at: $(date)"
echo "Model        : ${MODEL_NAME}"
echo "Task         : ${TASK}"
echo "Method       : latent_mas"
echo "Prompt       : ${PROMPT}"
echo "Alignment    : ${ALIGN_METHOD}"
echo "Latent steps : ${LATENT_STEP_VALUES[*]}"
echo "Max samples  : ${MAX_SAMPLES}"
echo "Max tokens   : ${MAX_NEW_TOKENS}"
echo "CUDA devices : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================================================"

SUMMARY_ARGS=()
STATUS=0
RESULT_PATTERN="${TASK}_latent_mas_prompt_${PROMPT}_model_Qwen_Qwen3-8B_align_${ALIGN_METHOD}_*.json"

for LATENT_STEPS in "${LATENT_STEP_VALUES[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Running latent_steps=${LATENT_STEPS} at $(date)"
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
        echo "ERROR: latent_steps=${LATENT_STEPS} failed with status ${RUN_STATUS}."
        STATUS="${RUN_STATUS}"
        break
    fi

    RESULT_FILE="$(
        find result -type f -name "${RESULT_PATTERN}" -newer "${MARKER_FILE}" -print |
            sort |
            tail -n 1
    )"
    if [ -z "${RESULT_FILE}" ]; then
        echo "ERROR: Could not locate the result JSON for latent_steps=${LATENT_STEPS}."
        STATUS=1
        break
    fi

    SUMMARY_ARGS+=("${LATENT_STEPS}" "${RESULT_FILE}")
    echo "Result JSON: ${RESULT_FILE}"
done

echo ""
echo "========================================================================"
echo "LATENT-STEP SUMMARY"
echo "========================================================================"

if [ "${#SUMMARY_ARGS[@]}" -gt 0 ]; then
    python3 - "${SUMMARY_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
headers = [
    "step",
    "accuracy",
    "correct",
    "samples",
    "time_s",
    "s/sample",
    "text_in",
    "latent_in",
    "text_out",
    "latent_out",
    "tokens_total",
]
rows = []

for step, result_file in pairs:
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
    rows.append([
        step,
        f'{results["accuracy"]:.6f}',
        str(results["correct"]),
        str(results["processed"]),
        f'{timing["total_seconds"]:.4f}',
        f'{timing["seconds_per_sample"]:.4f}',
        str(text_in),
        str(latent_in),
        str(text_out),
        str(latent_out),
        str(text_in + latent_in + text_out + latent_out),
    ])

widths = [
    max(len(headers[index]), *(len(row[index]) for row in rows))
    for index in range(len(headers))
]
print("  ".join(value.rjust(widths[index]) for index, value in enumerate(headers)))
print("  ".join("-" * width for width in widths))
for row in rows:
    print("  ".join(value.rjust(widths[index]) for index, value in enumerate(row)))

print()
print("tokens_total = text_in + latent_in + text_out + latent_out")
print("Note: input token totals are summed across agent roles and may include reused context.")
PY
else
    echo "No successful runs to summarize."
fi

echo ""
echo "Finished at: $(date)"
echo "Exit status: ${STATUS}"
exit "${STATUS}"
