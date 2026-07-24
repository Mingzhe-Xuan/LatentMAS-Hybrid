# S0--S4 approximator experiments

This package implements the operator layer of `docs/plan_v2.md`.  It is
deliberately independent from the Latent CoT and communication experiments.

Run from the repository root, for example:

```powershell
python exp/approximator/run.py --study s1 --model_pair x1 --dataset arc_easy --split test
python exp/approximator/run.py --study s3 --model_pair x1 --dataset arc_easy --split train
```

The entry point first verifies exact tokenizer compatibility and writes the
result to `result/manifests/compatibility.json`; any mismatch stops the run.
`F` always uses the full vocabulary.  Outputs are limited to this package's
`result/` directory: manifests, raw Parquet metrics, and figures.

`s4` writes every valid B-space mapping as raw rows; `identical` is omitted when source and target hidden dimensions differ.  This keeps fitting the
single shared PCA (and optional t-SNE) an explicit, inspectable downstream
plotting step rather than silently fitting one projection per method.
