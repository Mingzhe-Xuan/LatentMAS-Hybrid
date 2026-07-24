#!/usr/bin/env bash
# Download every model and remote dataset required by docs/plan_v2.md.
# Run this on a login node with network access before submitting offline jobs.
set -euo pipefail

module load python/3.12.13
source /home/n2501945g/LatentMAS-Hybrid/.venv/bin/activate

# This location must be visible from both login and compute nodes.
export HF_HOME=/home/n2501945g/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# C0/C1: same-model Latent CoT. X1/X2: operator and communication experiments.
MODELS=(
    Qwen/Qwen2.5-1.5B
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-7B-Instruct
)

for model in "${MODELS[@]}"; do
    echo "Downloading ${model}..."
    hf download "${model}"
done

python - <<'PY'
from datasets import load_dataset

# All remote datasets and splits explicitly required by docs/plan_v2.md.
for dataset, subset, split in (
    # S0--S4 operator calibration, test, and OOD.
    ("allenai/ai2_arc", "ARC-Easy", "train"),
    ("allenai/ai2_arc", "ARC-Easy", "test"),
    ("allenai/ai2_arc", "ARC-Challenge", "test"),
    ("gsm8k", "main", "train"),
    ("gsm8k", "main", "test"),
    ("evalplus/mbppplus", None, "test"),
    ("fingertap/GPQA-Diamond", None, "test"),
    # C1/C2 Latent CoT.
    ("HuggingFaceH4/aime_2024", None, "train"),
    ("yentinglin/aime_2025", None, "train"),
):
    print(f"Downloading {dataset} / {subset or '-'} ({split})...")
    load_dataset(dataset, subset, split=split)
PY

# MedQA is intentionally local in this repository: data.py reads it with the
# JSON loader instead of downloading a Hub dataset.
if [[ ! -f data/medqa.json ]]; then
    echo "ERROR: required local MedQA file is missing: data/medqa.json"
    exit 1
fi

# # M1/M3 require a fixed, repository-owned probe rather than a Hub dataset.
# # It must contain the plan_v2 1,024 examples and fixed 256/256/512 split.
# if [[ ! -f data/communication_probe.jsonl ]]; then
#     echo "WARNING: data/communication_probe.jsonl is not present."
#     echo "         It is required for M1/M3 and must be generated/versioned separately."
# fi

echo "Download complete. The cache is ready for offline PBS jobs."
