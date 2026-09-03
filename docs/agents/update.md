# Progress updates

- 2026-09-03: Authorized Slurm smoke array 254 reached node221 and loaded the cached Qwen3-8B checkpoint. Added default offline-cache environment settings to eliminate repeated Hugging Face DNS retries on Guqq; local analysis suite remains green (21 passed).
- 2026-09-03: Real AIME smoke produced and hash-validated the 8B Sender cache, proved a subsequent cache reuse exits as SKIPPED, and completed all 9 kernel-scaling Receiver cells. Perturbation array 258 and dependent cache-only analyses 270--273 were submitted. The 8B/14B matrix remains externally blocked by the absent Qwen3-14B snapshot on the offline cluster.
- 2026-09-03: AIME evaluation smoke finished all 51 expected Receiver cells without evaluation failures. Real cache-only analysis then found and drove fixes for one-dimensional entropy output and variance peak GPU memory; commit `986db77` passes 31 targeted tests. Runtime re-verification is pending because Guqq cannot currently pull from GitHub, and 14B coverage is still impossible without a shared Qwen3-14B snapshot.

- 2026-09-03：审阅 `analysis/plan.md`、生产模型/对齐/数据接口与 `analysis/AGENTS.md`，确认分析目录尚无实现，开始建立完整框架。
- 2026-09-03：完成独立分析框架：严格配置与 schema、数据快照、safetensor 分片缓存、精确 Sender 隐藏态捕获、批量 Receiver 前缀注入、任务评测、扰动与统计、五类分析、报告、PBS 矩阵/worker/提交链及本地测试。正式矩阵为 18 Sender + 324 去重 Receiver cells。
- 2026-09-03：本地验收 20 tests passed；全仓遗留测试另有 13 个与本改动无关的既存失败，真实 GPU smoke 等待 Git 同步授权。
- 2026-09-03：用户授权后推送验证提交 `af831ae`。远端确认仅部署 Slurm，新增 `analysis/slurm/` array worker 与依赖提交器，复用同一组严格 JSONL 矩阵和任务入口，并将 Guqq 默认并发设为单 GPU 一次一个 array cell。
