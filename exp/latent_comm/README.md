# M0: question-blind prefill communication

M0 measures whether a receiver Agent B can answer a private ARC-Easy question
seen only by sender Agent A. It samples a fixed 100-question test subset.

Agent A is an encoder only. Its chat contains the original question as the sole
user message, with no Planner instruction and no generated reasoning. A performs
one forward prefill and caches the complete last-layer hidden-state sequence?one
state per source prompt token. There is no fixed latent-step recurrence, so the
communication length is the actual A prompt length and can differ by question.

The complete prefill sequence is transported from A to B with `linear`,
`kernel`, `soft`, or `text`. The first three use the shared cross-model
vocabulary alignment. The `text` control greedily maps each A prefill state to a
token ID and embeds that token with B. The aligned sequence is prepended to B's
constant question-blind receiver prompt. B then prefills the combined embedding
sequence and generates one boxed choice. Neither B's messages nor rendered
prompt contains the source question or choices; this is checked at runtime and
recorded in every row.

The ordered model pairs are:

- Qwen3-4B -> Qwen3-4B
- Qwen3-4B -> Qwen3-8B
- Qwen3-8B -> Qwen3-4B
- Qwen3-8B -> Qwen3-8B

Run locally:

```bash
python exp/latent_comm/run.py \
  --study m0 --dataset arc_easy --split test \
  --max_questions 100 --prompt sequential \
  --model_pair all --method all \
  --alignments linear kernel soft text \
  --device cuda
```

PBS:

```bash
qsub -v "EXP_TARGET=latent_comm,STUDY=m0" exp.sh
```

Stable caches live in `exp/cache/latent_comm_m0/`. Sender `.pt` prefill
sequences are shared across target models and alignment methods. Receiver answer
caches are keyed by ordered model pair and alignment. Cache schema v2 prevents
reuse of the former 10-step recurrent trajectories. A normal rerun reuses
compatible caches; `--reuse_cache` requires every cache to exist, while
`--force_recollect` rebuilds them.

Each run writes:

- `metrics/m0_answers.parquet`: 1600 per-question answer rows;
- `summaries/m0_summary.json`: accuracy, bootstrap intervals, and communication-length statistics;
- `figures/m0_accuracy.pdf`: one panel per ordered model pair;
- `run_manifest.json`: sample identity, cache provenance, and leak audit.
