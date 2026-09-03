# GPU / remote access log

- 2026-09-03: Slurm array 254 was submitted for the AIME 2024 one-question Sender smoke. Array task 1 started on node221/RTX 5090 after the existing user job released the sole GPU. Guqq compute nodes cannot resolve Hugging Face, so the worker now defaults HF Hub, datasets, and Transformers to offline-cache mode while permitting explicit overrides.
- 2026-09-03: Sender array 254 completed the Qwen3-8B cell and failed the Qwen3-14B cell because Guqq has no 14B snapshot (available Qwen3 caches: 0.6B, 4B, 8B). Job 256 validated the 8B cache-hit path. AIME smoke scaling array 257 completed 9/9 cells; perturbation array 258 (42 cells, concurrency 1) is running. Cache-only jobs 270--273 depend on successful completion of 258.
- 2026-09-03: Perturbation array 258 completed 42/42 cells, giving exactly 51 Receiver Parquet caches with the 9 scaling cells. Scaling (271) and stability (273) analyses completed. Entropy (270) exposed a scalar reshape defect and variance (272) exhausted GPU memory while retaining the full fp32 model; both are fixed and locally verified in `986db77`. Guqq remains at `237534d` because repeated bounded `git pull --ff-only` attempts cannot reach GitHub, so corrected retries cannot yet be submitted. Qwen3-14B remains absent and the Slurm queue is empty.

- 2026-09-03：计划连接 `Guqq`，用途是只读确认远程仓库位置、当前提交及可用调度器（`qsub`/`sbatch`），为 `analysis/plan.md` 的 one-question/Kmax=4 smoke 做准备。连接后首先在 `/home/n2501945g/LatentMAS-Hybrid` 执行 `git pull --ff-only`；不在登录节点运行模型、评测或其他明显计算任务，也不直接修改受 Git 管理源码。
- 2026-09-03：远程预检未执行。安全审批要求用户明确授权连接该远程并执行 `git pull` 后方可继续；未尝试绕过审批。
- 2026-09-03：用户已明确授权提交 GPU smoke 和 Git 操作。计划将本地验收版本推送到独立验证分支，连接后先 `git pull --ff-only`，再只做调度器检查与 scheduler submission。
