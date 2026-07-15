python run.py \
  --method baseline \
  --model_name Qwen/Qwen3-4B \
#   --agent_models Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-4B Qwen/Qwen3-8B \
  --task gsm8k \
  --prompt sequential \
  --max_samples 160 \
  --generate_bs 4 \
#   --latent_steps 4 \
  --max_new_tokens 512 \
  --seed 42

python run.py \
  --method baseline \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --max_samples 160 \
  --generate_bs 4 \
  --max_new_tokens 4096 \
  --seed 42