#!/usr/bin/env bash
# Run this on a login node with network access before submitting offline jobs.
set -euo pipefail

module load python/3.12.13
source /home/n2501945g/LatentMAS-Hybrid/.venv/bin/activate

# This location must be visible from both login and compute nodes.
export HF_HOME=/home/n2501945g/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Qwen3 has official 4B, 8B, and 14B dense checkpoints. There is no official
# Qwen/Qwen-16B repository, so Qwen3-14B is used as the available large model.
MODELS=(
    Qwen/Qwen3-4B
    Qwen/Qwen3-8B
    Qwen/Qwen3-14B
)

for model in "${MODELS[@]}"; do
    echo "Downloading ${model}..."
    hf download "${model}"
done

python - <<'PY'
from datasets import load_dataset

for dataset, split in (
    ("HuggingFaceH4/aime_2024", "train"),
    ("yentinglin/aime_2025", "train"),
):
    print(f"Downloading {dataset} ({split})...")
    load_dataset(dataset, split=split)
PY

echo "Download complete. The cache is ready for offline PBS jobs."