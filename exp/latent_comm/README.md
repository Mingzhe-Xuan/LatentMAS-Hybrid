# M0: prefill communication with a direct-text baseline

M0 samples a fixed 100-question ARC-Easy test subset and evaluates four ordered
Qwen3-4B/Qwen3-8B sender-receiver pairs.

For `linear`, `kernel`, and `soft`, Agent A is an encoder only. Its chat contains
the original question as the sole user message, with no Planner instruction and
no generated reasoning. A performs one forward prefill and caches the complete
last-layer hidden-state sequence, one state per source prompt token. The sequence
is aligned to B and prepended to B's constant question-blind prompt. B never sees
the original question in these three conditions.

`text` is a direct-text upper baseline. It bypasses Agent A entirely: the
original question and choices are inserted directly into B's normal user prompt,
then B tokenizes, prefills, and answers. It does not apply A's LM head, argmax a
predicted token, or consume A hidden states. Consequently B is expected to see
the original question only for `text`; runtime and cache audits enforce this
per-alignment visibility contract.

The ordered model-pair labels are:

- Qwen3-4B -> Qwen3-4B
- Qwen3-4B -> Qwen3-8B
- Qwen3-8B -> Qwen3-4B
- Qwen3-8B -> Qwen3-8B

For `text`, A is bypassed, so rows with the same B but different A labels should
produce the same greedy answers. The duplicated pair labels are retained so the
four-panel comparison has the same shape as the latent conditions.

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
sequences remain reusable by `linear`, `kernel`, and `soft`. Existing compatible
answer caches for those alignments remain reusable as well. The former argmax
`text` answer cache has a different protocol identity and is not reused; the
first run collects the new direct-text results. `--reuse_cache` requires every
needed cache to exist, while `--force_recollect` rebuilds them.

Each run writes:

- `metrics/m0_answers.parquet`: 1600 per-question answer rows;
- `summaries/m0_summary.json`: accuracy, bootstrap intervals, visibility audits, and communication lengths;
- `figures/m0_accuracy.pdf`: one panel per ordered model pair;
- `run_manifest.json`: sample identity, cache provenance, and visibility audit.
