# Agent state

## 2026-09-04 双向 Exact STT goal

- 当前状态：本地 cross-vocab STT 配置/schema/cache/runtime、四个 task、6/12/3/1 matrix、Slurm 链与 cache-only 统计报告均已实现；干净 `origin/main` 基线上 `analysis/tests` 35 项全部通过。两个 runtime-v3 artifacts 均已按 `ModelWrapper` tokenizer 归一化策略通过 strict gate，正反向 source support 都完整。
- 当前计划：推送 runtime-v3 配置更新；通过 SFTP 将两个 Git-ignored 大 artifact 放到 Guqq，在远端 pull 成功后通过 Slurm 依次运行双向 smoke、三数据集 integration smoke、12 个正式单元和最终报告。
- 边界：保留 kernel-analysis-v1 的配置校验、矩阵计数、cache identity 与现有结果；STT 不导入 `exp/`，并沿用 analysis 的 dataset、prompt、CLI、cache/result 和 scheduler contracts。

## 变更记录

- 2026-09-04：将 `analysis/transport/bidirectional_stt_analysis.md` 作为 active goal 启动；确认 full-context STT 与 recurrent `SenderTrajectoryStore/evaluate_receiver_batch()` 语义不同，采用公共基础设施复用加独立版本化 STT store/runtime 的实现路径。
- 2026-09-04：完成本地 STT 主体与回归验证（33 passed）；正式矩阵为 6 planner + 12 evaluation + 3 analysis + 1 report。artifact gate 证明正向满足静态协议，反向缺少 source IDs `0,1,2`，下一阶段需先修复该输入 artifact。
- 2026-09-04：检测到用户并发更新的反向 artifact 已独立重建为 full-source-support。使用锁定 Qwen/Mistral revision 生成 runtime-v2 派生 artifacts，以 analysis 规范重新绑定完整 token→ID mapping 与 special IDs；正向 `[131069,151669]`、反向 `[151643,131072]` 均通过 SHA、revision、fingerprint、support 和列质量 gate。
- 2026-09-04：发现 `ModelWrapper` 会把 Mistral 的缺失 pad token 绑定为 EOS，因而 runtime-v2 的原始-tokenizer指纹与真实执行状态不一致。生成 runtime-v3，并使用同一归一化函数和 strict loader 重新验证；Qwen vocab 151669、Mistral vocab 131072/pad 2，双向 gate 均通过。

## 2026-09-04 perturbation 网格调整

- 当前状态：已按用户授权移除 `alpha=0.025`，配置、矩阵、测试和计划计数已同步并通过回归验证。
- 当前计划：本次调整已完成；不改动 transport 等用户文件。
- 变更记录：perturbation 正式矩阵由 126 行降为 99 行，三个 Receiver 数组由 324 个唯一单元降为 297 个。

## 当前状态

`analysis/plan.md` 的本地实现与静态验收已完成：核心协议、缓存、任务、正式/冒烟矩阵和 PBS 链均已建立。真实 Qwen GPU smoke 尚未提交，因为当前修改还未被用户授权提交并推送到远程仓库；服务器规范禁止直接修改或传送未由 Git 管理的源码。

## 当前计划

1. 创建并推送独立验证分支。
2. 为远端实际存在的 Slurm 增加独立提交路径，同时保留正式 PBS 接口并完成本地回归测试。
3. 推送增量提交，在远端 `git pull --ff-only` 后提交真实 one-question/Kmax=4 smoke，持续检查直到完成。

## 变更记录

- 2026-09-03：开始实现 `analysis/plan.md`；先完成可本地验证的框架与 fake-model 测试，再将真实 GPU smoke 作为 Slurm 作业入口。
- 2026-09-03：完成本地四阶段实现；20 项 analysis 测试、10 个 CLI、Python 编译、shell 静态检查和正式/冒烟矩阵检查通过。下一步需要通过 Git 同步后在集群运行真实模型 smoke。
- 2026-09-03：远程只读调度器预检被安全审批拒绝，原因是缺少用户对连接目标及远程 `git pull` 的明确授权；本地工作不受影响，远程 smoke 暂停。
- 2026-09-03：用户明确授权 GPU smoke 与 Git 操作；恢复远程验证流程，下一步推送独立验证分支。
- 2026-09-03：远端仓库实际路径为 `/home/xmz/LatentMAS-Hybrid`，仅有 Slurm `sbatch`（compute partition, `gpu:1`），没有 PBS `qsub`；远端原有 README、已跟踪 pyc 与 state.txt 修改均予以保留。开始增加不改变正式 PBS 契约的 Slurm 适配目录。
- 2026-09-03：Slurm worker/submitter 已实现并通过本地语法、静态契约与 AIME2024 smoke dry-run；下一步提交增量验证 commit 并在 Guqq 运行轻量环境检查。
- 2026-09-03：首次 Sender smoke 提交在创建作业前失败：Guqq 非交互登录环境没有全局 `python` 命令。调整 submitter，使其自动回退到仓库 `.venv/bin/python` 后重试。
