# Analysis test record

## 2026-09-04 双向 Exact STT 实现计划

- 配置/schema：STT 配置严格固定两模型、三 primary datasets、greedy、`tau=0.6`、sender budget 1024、双向 artifact hash；既有 `load_config()` 行为不变。
- artifact：使用 `allow_pickle=False`，验证 CSC schema、shape、token IDs、fingerprint、revision、非负有限列质量、SHA-256 和方向；任一不匹配 fail closed。
- 数值核心：小词表 dense oracle 对照稀疏 exact transport；验证完整 source softmax、全部边、FP32 累加、chunk invariance、无 causal shift 和 prefix 顺序。
- cache：planner full-context 分片与 receiver 结果均使用内容身份、锁、原子写入、文件哈希和完整 manifest；损坏/身份冲突不得复用。
- 矩阵/任务：正式矩阵行数精确为 6/12/3/1，无重复 ID；所有 task 复用统一 CLI，cache hit 返回 10，cache-only 缺失时不 rollout。
- 回归/调度：现有 `analysis/tests`、kernel 正式矩阵计数和 PBS/Slurm contracts 不变；STT submitter dry-run 依赖正确。GPU 计算仅通过 Slurm 提交。
- 实验验收：双方向单题 GPU smoke、每数据集至少四题 integration smoke、12 个正式单元及统一统计报告全部完成后，goal 才可标记 complete。

### 当前实际结果

- `python -m pytest analysis/tests -q`：33 passed；新增 STT 测试覆盖配置隔离、CSC gate、dense oracle、chunk invariance、prefix/position IDs、planner cache、fake causal-LM planner→judger、McNemar、6/12/3/1 矩阵和 cache-only report。
- `python -m compileall -q analysis`：通过；四个新增 task 的统一 CLI `--help` 均通过。
- `bash -n analysis/slurm/analysis_job.slurm` 与 `bash -n analysis/slurm/submit_stt_analysis.sh`：通过。
- AIME first-1 submitter dry-run：正确生成 2 planner、4 evaluation、1 analysis、1 report，并形成 `afterok` 依赖链。
- 正式 STT matrix dry-run：精确生成 6/12/3/1 行，无重复 effective cache ID。
- 真实正向 artifact 静态 gate：通过；shape `[131069,151669]`，最大 source-column mass error `1.2299050666797484e-12`。
- 真实反向 artifact 静态 gate：按预期 fail closed；source IDs 为 `3..131071`，缺少 Mistral IDs `0,1,2`，不满足完整 sender vocabulary。修复 artifact 前不能完成反向 GPU smoke 或 12 个正式单元。
- Guqq 只读 tokenizer 预检：连接后的首个 `git pull --ff-only` 再次受 GitHub 网络阻塞，命令被中止，未绕过 pull 执行后续检查。
- 干净 `origin/main` worktree 回归：`36 passed`；早先全局 `%TEMP%` 权限导致 fixture setup error，改用 workspace 内显式 `--basetemp` 后全部通过。
- 锁定 revision 的真实 tokenizer 已下载并离线加载：Mistral vocab 131072，Qwen vocab 151669。旧 builder fingerprints 为 opaque scheme，因此生成保留父 provenance 的 runtime-v2 artifacts，并改用 analysis 可重算的 token mapping + special IDs 指纹。
- 正向 runtime-v2 strict gate：SHA `04b8c1e2a553eb61233fb71dfcea692a471fab14ac204359fab484a0c42d6944`，shape `[131069,151669]`，source IDs 完整，最大列质量误差 `1.2299050666797484e-12`。
- 反向 runtime-v2 strict gate：SHA `9880bb4885dc792e3d61786b8dad5531cea70c796c578191c9fd0724e8b92b2c`，shape `[151643,131072]`，source IDs `0..131071` 完整，最大列质量误差 `7.178257988016412e-12`。
- 运行时 tokenizer 策略回归：新增测试确认 artifact 工具复现 `ModelWrapper` 的 `pad=eos` 与 left-padding 行为。
- 正向 runtime-v3 strict gate：SHA `b7ce13823f3a09750aa944dbe7f6a419df2d9c7987883a20be854d32be857e17`，shape `[131069,151669]`，Qwen source vocab 151669 完整，最大列质量误差 `1.2299050666797484e-12`。
- 反向 runtime-v3 strict gate：SHA `257c46a67c2e68c7888cca5ae32e6f2d89afcd68c0faa8eeae386225bb30cd32`，shape `[151643,131072]`，Mistral source vocab 131072 完整且运行时 pad ID 为 2，最大列质量误差 `7.178257988016412e-12`。
- Exact STT dense oracle：未分块输出与 position-size=1、target-size=1 的双重分块输出均在 `atol=1e-6` 内等于显式 dense 公式；正式 chunk sizes 已进入 receiver cache identity。
- Cache-only 依赖隔离：matrix 显式绑定每个 dataset 的四个 receiver cache IDs 及 report 所需的三个 analysis IDs；测试注入同 dataset/system 的 stale cache 后，analysis 只读取 job 声明的四个已验证 cache。
- 结果报告：每个 dataset/system 明确输出 score、无法解析率、HumanEvalPlus 代码执行失败率和总错误率；两个预注册方向效应在 Markdown 与 JSON 中都报告 10,000 次 paired-bootstrap 95% CI、exact McNemar p 值与 discordant 数。
- 模型版本一致性：Qwen/Mistral 的 40 位 commit hash 在正式配置中逐模型锁定，并写入全部 planner/baseline/cross job 与 cache identity；wrapper 将 revision 传给 tokenizer/model loader，任务在加载后再次 fail-closed 比较解析 revision。

## 2026-09-04 perturbation 强度调整

- 计划：验证正式配置只允许 `alpha={0,0.01,0.05,0.10}`；每个 dataset-seed 包含 11 个 perturbation 单元；正式 perturbation 矩阵共 99 行，三个 Receiver 数组共 297 个唯一 cache ID。
- 预期：`analysis/tests/test_contracts.py` 与 `analysis/tests/test_job_matrix.py` 全部通过，矩阵 dry-run 输出更新后的计数。
- 实际：相关测试 `6 passed`；正式矩阵 dry-run 生成 99 个 perturbation 单元（每个 primary dataset 33 个），所有正式计数校验通过；`git diff --check` 通过。

## 2026-09-03 计划

- 配置/schema：严格校验正式协议、稳定哈希和 clean 条件规范化。
- 数据与缓存：内容指纹、原子分片、哈希损坏检测、断点续作、完整缓存零 forward。
- Sender/Receiver：fake model 验证只捕获最终层最后位置的 readout 前隐藏态、前缀长度、prompt 分支、计时和 unembedding 计数。
- 干预/统计：相同 noise key 跨方法产生相同方向；paired/nested bootstrap 保持问题聚类。
- 作业矩阵/PBS：精确行数 18/54/126/144/3/3/3/3/9/1、无重复有效缓存 ID、复用 canonical clean 条件、shell 静态校验。
- 架构：AST 检查 `analysis/` 不导入 `exp`；cache-only 分析不加载或调用模型。
- 集成：一个问题、`Kmax=4` 的 fake-model Sender→cache→Receiver→summary 流程。

预期：所有本地测试通过；真实 Qwen smoke 只能通过 Slurm 提交，不能在本地或登录节点直接运行。

## 2026-09-03 实际结果

- `python -m pytest analysis/tests -q`：20 passed；仅有本机 pandas 对可选 numexpr/bottleneck 版本的两条 warning。
- `python -m compileall -q analysis`：通过。
- 十个任务入口逐一执行 `--help`：全部通过。
- `python analysis/pbs/build_job_matrix.py --dry-run`：通过；正式行数精确为 `18/54/126/144/3/3/3/3/9/1`，324 个 Receiver ID 无重复。
- `python analysis/pbs/build_job_matrix.py --dataset aime2024 --smoke --dry-run`：通过；使用独立的 first-1/Kmax=4 缓存身份。
- `bash -n analysis/pbs/analysis_job.pbs` 与 `bash -n analysis/pbs/submit_analysis.sh`：由测试调用并通过。
- `git diff --check`：通过。
- 全仓 `python -m pytest -q`：收集阶段因本机未安装可选 `vllm` 而失败；该包也未在当前 `requirements.txt` 启用。
- 排除环境探针 `tests/test_libs.py` 后：114 passed、13 failed。失败全部位于任务开始前已有且本次未修改的 `exp/`/`run_all.sh` 行为与其历史测试不一致（S3 1项、latent_comm M0 7项、latent_cot C4 4项、run_all 1项）；analysis 的 20 项测试仍全部通过。
- 真实 Qwen/PBS smoke：未运行。按 `analysis/AGENTS.md`，源码必须先经本地 Git 提交并由服务器 `git pull` 同步；本轮没有擅自 push 到远程 main。
- 远程预检：审批拒绝，未连接或修改服务器。需要用户明确授权 `ssh Guqq` 与远程 `git pull --ff-only`。

## 2026-09-03 Slurm 适配测试计划

- 对 Slurm worker/submitter 执行 `bash -n`，并静态验证 array index、路径白名单、exit-10/SKIPPED、FAILED、flock ledger 和 afterok 依赖。
- 本地 dry-run 必须生成独立 smoke cache IDs，输出完整 `sbatch` 命令且不实际提交。
- 推送后先在 Guqq 虚拟环境执行轻量 import/config/matrix 检查，再通过 `sbatch` 提交 AIME2024 first-1/Kmax=4 smoke；不得在登录节点运行模型。

实际：21 项 analysis tests passed；PBS/Slurm 两套 shell 均通过 `bash -n`；Slurm AIME2024 smoke dry-run 正确生成 2/9/42/12 个 Sender/scaling/perturbation/model-pair array rows 和完整 afterok 链，未提交作业。

- Guqq 首次 `--stage collect`：提交前失败（exit 1），原因是 login shell 无全局 `python`；未创建 Slurm job。新增 `.venv/bin/python` fallback 回归测试后重试。
