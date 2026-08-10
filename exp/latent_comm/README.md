# M0: prefill communication with receiver-visibility controls

M0 samples a fixed 100-question ARC-Easy test subset and evaluates four ordered
Qwen3-4B/Qwen3-8B sender-receiver pairs.

Agent A is an encoder only. Its chat contains the original question as the sole
user message, with no Planner instruction and no generated reasoning. A performs
one forward prefill and caches the complete last-layer hidden-state sequence,
one state per source prompt token.

For each of `linear`, `kernel`, and `soft`, M0 runs two B conditions:

- `blind`: B receives the aligned A prefill states followed by a fixed prompt
  that does not contain the original question;
- `visible`: B receives the same aligned states followed by a prompt that also
  contains the complete original question and choices.

`text` remains a direct-text B-only baseline. It bypasses Agent A and inserts the
original question directly into B's normal prompt. Therefore each ordered model
pair has seven cells: three alignments times two visibility conditions, plus one
direct-text baseline. With 100 questions and four pairs, a complete run contains
2800 rows.

## Figure encoding

Each ordered model pair gets one panel. The x-axis is `linear`, `kernel`, `soft`,
and `text only`.

- solid colored bars: aligned latent states, B question-blind;
- hatched bars of the same alignment color: aligned latent states plus the
  original question in B's prompt;
- single gray `text only` bar: direct original question, no Agent A.

This keeps alignment method on the x-axis and uses fill/hatching exclusively for
question visibility, so the two experimental dimensions are not conflated.

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
sequences are shared by both visibility conditions and all latent alignments.
Existing compatible blind and direct-text answer caches remain reusable. Only
the new `visible` cells require fresh answer rollout. `--reuse_cache` requires
every needed cache to exist; `--force_recollect` rebuilds them.

Each run writes:

- `metrics/m0_answers.parquet`: 2800 per-question condition rows;
- `summaries/m0_summary.json`: accuracy, bootstrap intervals, visibility audits,
  and communication-length statistics;
- `figures/m0_accuracy.pdf`: grouped visibility comparison in four panels;
- `run_manifest.json`: sample identity, cache provenance, and visibility audit.
