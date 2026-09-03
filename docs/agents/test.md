# Analysis test record

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
