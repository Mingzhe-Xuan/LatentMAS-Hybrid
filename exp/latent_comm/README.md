# M0: heterogeneous model communication

M0 measures whether a Planner prefill from one model improves a visible-question
Judger in another model. It evaluates Qwen3-4B and Qwen3-8B in all ordered
sender-to-receiver pairs: 4B->4B, 4B->8B, 8B->4B, and 8B->8B.

For every dataset and architecture, M0 reports:

- each receiver's `text` accuracy, which is the 4B/8B single-model baseline;
- each ordered pair's `soft`, `linear`, and `kernel` communication accuracy;
- question-bootstrap 95% confidence intervals and the number of transmitted
  Planner hidden states.

The receiver always sees the original question. The latent methods prepend the
sender's complete single-prefill Planner hidden-state sequence, aligned to the
receiver input-embedding space. `text` bypasses Agent A and gives the original
question directly to the receiver, so it isolates the receiver's own task
accuracy rather than claiming a source-dependent text transfer.

Supported datasets are `gpqa_diamond`, `gsm8k`, and `arc_challenge` (test
split), plus `aime2024`, `aime2025`, and `medqa` (train split). Both repository
prompt organizations are supported:
`sequential` and `hierarchical`. The sender uses the corresponding Planner prompt
and the receiver uses the corresponding Judger prompt.

```bash
python exp/latent_comm/run.py \
  --study m0 --dataset gpqa_diamond --split test \
  --prompt sequential --max_questions 30 --device cuda

python exp/latent_comm/run.py \
  --study m0 --dataset aime2025 --split train \
  --prompt hierarchical --max_questions 30 --device cuda

python exp/latent_comm/run.py \
  --study m0 --dataset gsm8k --split test \
  --prompt sequential --max_questions 30 --device cuda

python exp/latent_comm/run.py \
  --study m0 --dataset medqa --split train \
  --prompt sequential --max_questions 30 --device cuda

python exp/latent_comm/run.py \
  --study m0 --dataset arc_challenge --split test \
  --prompt sequential --max_questions 30 --device cuda
```

Stable sender and answer caches are stored in `exp/cache/latent_comm_m0_v2/`.
Each run writes `metrics/m0_answers.parquet`, `summaries/m0_summary.json`,
`figures/m0_accuracy.pdf`, and `run_manifest.json` under
`exp_result/latent_comm/runs/`.