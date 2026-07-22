module load python/3.12.13
source /home/n2501945g/LatentMAS-Hybrid/.venv/bin/activate

export HF_HOME=/home/n2501945g/.cache/huggingface
huggingface-cli download Qwen/Qwen3-8B
python -c "from datasets import load_dataset; load_dataset('yentinglin/aime_2025', split='train')"