# M0: question-blind latent communication

M0 measures whether a receiver Agent B can answer a private ARC-Easy question
seen only by sender Agent A. It samples a fixed 100-question test subset. A
uses the repository sequential Planner prompt, prefills the original question,
and produces a fixed 10-state latent trajectory with identical same-model
recurrence. That sender trajectory is cached once per source model.

The 10 states are transported from A to B with `linear`, `kernel`, `soft`, or
`text`. The first three use the shared cross-model vocabulary alignment. The
`text` control greedily maps each A state to a token ID and embeds that token
with B. B receives the transported message followed by a constant sequential
receiver prompt. Neither B's messages nor rendered prompt contains the source
question or choices; this is checked at runtime and recorded in every row.

The ordered model pairs are:

- Qwen3-4B -> Qwen3-4B
- Qwen3-4B -> Qwen3-8B
- Qwen3-8B -> Qwen3-4B
- Qwen3-8B -> Qwen3-8B

Run locally:

```bash
python exp/latent_comm/run.py \
  --study m0 --dataset arc_easy --split test \
  --max_questions 100 --latent_steps 10 --prompt sequential \
  --model_pair all --method all \
  --alignments linear kernel soft text \
  --device cuda
```

PBS:

```bash
qsub -v "EXP_TARGET=latent_comm,STUDY=m0" exp.sh
```

Stable caches live in `exp/cache/latent_comm_m0/`. Sender `.pt` trajectories
are shared across target models and alignment methods. Receiver answer caches
are keyed by ordered model pair and alignment. A normal rerun reuses compatible
caches; `--reuse_cache` requires every cache to exist, while
`--force_recollect` rebuilds them.

Each run writes:

- `metrics/m0_answers.parquet`: 1600 per-question answer rows;
- `summaries/m0_summary.json`: accuracy and bootstrap intervals;
- `figures/m0_accuracy.pdf`: one panel per ordered model pair;
- `run_manifest.json`: sample identity, cache provenance, and leak audit.
