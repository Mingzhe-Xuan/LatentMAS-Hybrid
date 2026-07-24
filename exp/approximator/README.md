# S0--S4 approximator experiments

`exp/approximator/run.py` implements the operator layer in `docs/plan_v2.md`.
It verifies full tokenizer-ID compatibility before loading an experiment; a
mismatch stops the run. Exact `F` always scans the full vocabulary.

All artefacts are outside the source tree, under `exp_result/approximator/`:

- `metrics/`: raw Parquet tables (mapping, single-kernel, calibration,
  variance, embeddings and performance), plus cluster-bootstrap summaries;
- `figures/`: stratified S0/S1/S2 distributions and S4 shared-PCA plots;
- `manifests/`: arguments, models, compatibility, versions and git commit.

Examples:

```bash
python exp/approximator/run.py --study s0 --model_pair x1 --dataset arc_easy --split test
python exp/approximator/run.py --study s2 --model_pair x1 --dataset arc_easy --split train --run_s2_calibration
python exp/approximator/run.py --study s3 --model_pair x1 --dataset arc_easy --split train
python exp/approximator/run.py --study s4 --model_pair x1 --dataset arc_easy --split test
```

S3 uses the fixed 32 ORF seeds and retains each question's last prompt state
plus up to 16 evenly spaced reply states. S4 fits a single PCA over equal-sized
prompt/reply state samples; Qwen3-4B/8B have different hidden dimensions, so
there is intentionally no `identical` baseline for X1/X2.

Progress is appended, never overwritten, to
`exp_result/approximator/exp_state.txt`. PBS submissions use the same fixed
filename; follow it with `tail -f exp_result/approximator/exp_state.txt`.