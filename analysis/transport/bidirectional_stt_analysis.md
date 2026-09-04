# 双向 Exact STT Planner–Judger 分析实验设计

## 1. 实验目标

本实验研究不同 tokenizer、不同 hidden size 的语言模型之间，能否通过 exact soft-token transport（STT）传递 planner 的完整上下文，并提高 judger 的最终任务表现。`planner` 和 `judger` 严格采用现有 `analysis` 的角色定义，不另外创建 `thinker` 角色。

实验只复用 `analysis/` 和仓库根目录中的公共实现，不导入 `exp/`。主协议遵循 `analysis/transport/T_algo.md`：

- sender 先基于题目生成符合 `planner` 角色的求解计划；
- 对 `sender prompt + sender plan` 的全部有效位置重新 forward；
- 每个位置使用完整 sender vocabulary softmax；
- 使用全部有效稀疏 transport 边，不使用 top-k 或离散 source token；
- aligned sender context 放在 receiver native judger prompt 之前；
- receiver native judger prompt 再次显式包含同一道题；
- sender 和 receiver 都使用 greedy decoding；
- 主协议不做 causal shift。

核心问题是：双向 STT 是否携带了对当前题目有用的 planner 信息，而不是仅仅给 judger 增加一个连续前缀。

### 1.1 与现有 `analysis` 的结构兼容契约

本实验是现有 `analysis` 框架中的一个新协议，不建立第二套实验框架。实现时必须保持以下公共约定：

- Python 包仍以 `analysis.core` 作为可复用逻辑层，以 `analysis.tasks` 作为薄 CLI 层；task 文件只做参数解析、依赖装配和调用 core API；
- 配置放在 `analysis/configs/`，作业矩阵放在 `analysis/jobs/`，缓存根目录仍为 `analysis_cache/`，分析结果仍写入 `analysis_result/<task>/<effective_cache_id>/`；
- 所有 task 继续接受 `_common.parser()` 定义的 `--config`、`--job-spec` 或 `--job-matrix/--job-index`、`--cache-root`、`--result-root`、`--device`、`--cache-only` 和 `--force`；
- job matrix 每行仍是一个 JSON object，至少包含 `task` 与内容寻址的 `effective_cache_id`；数组下标仍为从 1 开始，禁止空行和重复 cache ID；
- 完整且身份一致的不可变 cache 命中时返回退出码 `10`，让现有 Slurm worker 记录为 `SKIPPED`；缺失、不完整或身份不一致的 cache 必须 fail closed；
- dataset、prompt、evaluation、hash、原子写入、锁和统计能力优先复用 `analysis.core`，不得在 STT task 中复制实现；
- cache-only task 只读经 manifest 与文件 SHA-256 验证的 cache，不得隐式加载模型或重新 rollout；
- 不导入 `exp/`，也不改变现有 kernel-analysis-v1 的正式配置、矩阵计数、cache identity 或结果。

建议的代码映射如下；文件名也是实现验收时的默认边界：

| 责任 | 位置 | 与现有结构的关系 |
|---|---|---|
| STT 配置及严格校验 | `analysis/configs/bidirectional_stt.yaml`、`analysis/core/config.py` 中的 `load_stt_config()` | 与 `load_config()` 并列；不得放宽或改写 kernel 配置校验 |
| STT schema/identity | `analysis/core/schemas.py` | 沿用 frozen dataclass、`stable_hash()` 和显式 `schema_version` |
| artifact loader 与 exact transport | `analysis/core/stt.py` | 新增窄模块；只负责 artifact gate、稀疏输运、prefix 拼接和 STT receiver 执行 |
| dataset/prompt/evaluator/statistics | `analysis/core/datasets.py`、`schemas.py`、`evaluation.py`、`statistics.py` | 直接复用或作向后兼容扩展 |
| planner/receiver cache | `analysis/core/cache.py` | 复用 `DatasetSnapshotStore`、`CacheHandle`、`CacheLock`、原子写入与 manifest 规范；为 full-context 语义增加独立 store/schema |
| 模型任务入口 | `analysis/tasks/collect_stt_planner_contexts.py`、`evaluate_bidirectional_stt.py` | 继续使用 `_common.parser()`、`load_job()`、`load_wrapper()` 与 `model_revision()` |
| 纯缓存分析与报告 | `analysis/tasks/analyze_bidirectional_stt.py`、`build_bidirectional_stt_report.py` | 遵守现有 result manifest、provenance 和 cache-only 行为 |
| 作业矩阵 | `analysis/pbs/build_stt_job_matrix.py` | 复用现有 JSONL contract，但与 kernel job matrix 分开校验和计数 |
| Slurm 编排 | `analysis/slurm/submit_stt_analysis.sh`、扩展 `analysis_job.slurm` task allowlist | 复用同一个 worker、状态日志和 progress ledger |
| 测试 | `analysis/tests/test_stt_*.py` | 与现有 core/task/job-matrix 测试放在同一测试包 |

`SenderTrajectoryStore` 当前保存的是 recurrent analysis 的 `[steps, hidden_dim]` 状态，其 hidden semantics 与 STT 的逐 token full-context `[sequence_length, hidden_dim]` 不同；`evaluate_receiver_batch()` 当前也要求 tokenizer 相容并采用现有 recurrent-prefix 拼接。因此只能复用它们的公共基础设施和结果约定，不能伪装成同一种 cache 或直接调用。STT 应使用独立、版本化的 planner-context identity/store，并继续产出兼容 `ReceiverItemResult` 与 `ReceiverEvaluationStore` 的逐题结果。任何对公共 dataclass 的扩展都必须提供默认值，保证既有 kernel cache 与测试仍可读取。

建议的矩阵固定为：

| Matrix | Task | 正式行数 |
|---|---|---:|
| `stt_planner.jsonl` | `collect_stt_planner_contexts` | 6（3 datasets × 2 planners） |
| `stt_evaluation.jsonl` | `evaluate_bidirectional_stt` | 12（3 datasets × 4 systems） |
| `stt_analysis.jsonl` | `analyze_bidirectional_stt` | 3（每个 dataset 一行） |
| `stt_report.jsonl` | `build_bidirectional_stt_report` | 1 |

这里的 Stage A/B/C 只表示同一主实验内部的执行与缓存依赖，不扩展实验范围；正式实验始终只有第 2 节定义的四个系统和 12 个主单元。

## 2. 四个主实验系统

正式实验只包含以下四个系统。

| 系统 ID | Planner | Judger/回答模型 | Judger 输入 |
|---|---|---|---|
| `qwen_only` | 无 | Qwen3-8B | Qwen native judger prompt |
| `mistral_only` | 无 | Mistral-Nemo | Mistral native judger prompt |
| `qwen_to_mistral` | Qwen3-8B | Mistral-Nemo | aligned Qwen full context + Mistral native judger prompt |
| `mistral_to_qwen` | Mistral-Nemo | Qwen3-8B | aligned Mistral full context + Qwen native judger prompt |

其中：

- `qwen_to_mistral - mistral_only` 测量 Qwen planner 对 Mistral judger 的增益；
- `mistral_to_qwen - qwen_only` 测量 Mistral planner 对 Qwen judger 的增益。

`qwen_to_mistral` 与 `mistral_to_qwen` 的绝对分数可以并列报告，但不能把二者差异全部解释成 transport 方向差异，因为两个条件的 judger 本身不同。

为保证公平，单模型 baseline 必须使用其作为 judger 时完全相同的 native judger prompt、generation budget、greedy decoding 和评测器。通信条件相对对应的单模型 baseline 只增加 transported planner context。

## 3. 数据集与现有 analysis 对齐

主实验严格使用现有 `analysis` 的三个 primary datasets：

| Dataset | Split | 类型 | 主指标 |
|---|---|---|---|
| `aime2024` | `train` | 数学推理 | accuracy |
| `arc_challenge` | `test` | 多项选择 | accuracy |
| `humanevalplus` | `test` | 代码生成 | pass@1 |

所有系统必须复用：

- `analysis.core.datasets.load_analysis_items()`；
- 相同的 loader、split、题目格式和 gold answer；
- 相同的 `DatasetSnapshot`、`selection_policy` 和 dataset fingerprint；
- 相同的 `item_id + question_hash`；
- `analysis.core.evaluation.evaluate_answer()` 的评分逻辑。

同一个 snapshot 中的同一道题分别进入四个系统，从而进行逐题配对比较。不得为不同模型重新抽样或改变题目顺序。

本实验的数据集范围固定为上述三个 primary datasets，不包含其他扩展数据集。

OpenHermes-2.5 是 transport matrix 的构建语料，不是下游评测集，不得混入评测 `DatasetSnapshot`。

## 4. 双向 Transport Artifacts

当前提供两个 artifact：

| 方向 | 文件 | Shape `[target, source]` | NNZ | SHA-256 |
|---|---|---:|---:|---|
| Qwen3-8B → Mistral-Nemo | `analysis/transport/qwen3_8b_to_mistral_nemo_openhermes_500k_runtime_v3.npz` | `[131069, 151669]` | 2,733,518 | `b7ce13823f3a09750aa944dbe7f6a419df2d9c7987883a20be854d32be857e17` |
| Mistral-Nemo → Qwen3-8B | `analysis/transport/mistral_nemo_to_qwen3_8b_openhermes_500k_full_vocab_runtime_v3.npz` | `[151643, 131072]` | 2,693,524 | `257c46a67c2e68c7888cca5ae32e6f2d89afcd68c0faa8eeae386225bb30cd32` |

已完成的静态检查：

- 两个矩阵均为 CSC，`indptr` 长度分别等于 source support size 加一；
- 所有 transport 权重均有限且非负；
- 正向矩阵最大列质量误差约为 `1.23e-12`；
- 反向矩阵最大列质量误差约为 `7.18e-12`；
- 两个正式 runtime-v3 artifacts 的 source support 都覆盖对应 sender tokenizer 的完整词表；target support 是对应 receiver 的 ordinary-token 子集；
- 正反向矩阵分别独立构建，反向不是正向条件矩阵的转置；
- runtime-v3 派生过程不改变已有 transport 数值边，只使用锁定 revision 的真实 tokenizer，并复现 `ModelWrapper` 的 pad-token/left-padding 归一化，再将运行时 token→ID 映射与 special IDs 重新指纹化；parent SHA-256、原始 fingerprints 和构建 provenance 均被保留。

正向 artifact 记录的 revision 是：

- Qwen source revision：`b968826d9c46dd6066d109eabc6255188de91218`
- Mistral target revision：`04d8a90549d23fc6bd7f642064003592df51e9b3`

### 4.1 正式运行前的 artifact gate

loader 必须使用 `allow_pickle=False`，并在加载模型后逐项检查：

1. schema、坐标系和稀疏结构完整；
2. `T.shape == [len(target_token_ids), len(source_token_ids)]`；
3. source token IDs 覆盖协议要求的完整 sender tokenizer vocabulary；
4. target token IDs 全部落在 receiver tokenizer vocabulary 内；
5. source/target token IDs 唯一且顺序与矩阵坐标一致；
6. source/target fingerprint 与实际 tokenizer 一致；
7. 模型 revision 与 artifact 或受版本控制的 sidecar manifest 一致；
8. 权重有限、非负，各 source 列质量在容差内等于 1；
9. artifact 文件 SHA-256 与配置完全一致；
10. 配置方向与 artifact 方向完全一致。

任何检查失败都必须立即停止，不能自动转置矩阵、删除 fingerprint 检查或退化成 hard mapping。

正式 runtime-v3 artifacts 已在上述锁定 revision 的实际 Mistral/Qwen tokenizer 上完成 fingerprint normalization 与 full-source-support 检查，其中 Mistral 按模型运行时策略将缺失的 pad token 绑定为 EOS（ID 2）。父 artifact 的 opaque builder fingerprint 仍保留在 `runtime_tokenizer_validation.parent_*_fingerprint`；运行时 gate 比较的是可由当前 `analysis` 独立重算的 `analysis-tokenizer-mapping-plus-special-ids-sha256-v1` 指纹。大文件不进入普通 Git，必须通过受控数据传输放入配置声明的路径，并在运行前再次校验配置 SHA-256。

## 5. Prompt 与生成协议

不得为 STT 实验硬编码新的 system/user 文本。每道题必须直接调用现有 `analysis` prompt API：

```python
planner_messages = build_role_messages(
    role="planner",
    question=item.question,
    task=dataset,
    model_name=sender_model_id,
)
judger_messages = build_role_messages(
    role="judger",
    question=item.question,
    task=dataset,
    model_name=receiver_model_id,
)

sender_prompt_text = render_role_prompt(
    sender, planner_messages, sender_model_id
)
receiver_prompt_text = render_role_prompt(
    receiver, judger_messages, receiver_model_id
)
```

这会复用 `prompts.build_agent_message_sequential_latent_mas()` 的现有角色语义：

- `planner` 根据题目产生简洁、分步骤的求解计划，并被要求不直接给出最终答案；
- `judger` 显式接收同一道 `Target Question`，把 latent 信息作为可忽略的参考，并按照数据集规定输出最终答案格式；
- AIME、ARC 和 HumanEvalPlus 分别沿用现有数学、多选和代码任务提示；
- Qwen/Mistral 的 system identity 和 reasoning cue 由现有 `render_role_prompt()` 与 `reasoning_models.py` 处理。

不得新增 `thinker` role，不得用自定义文本替换 `planner`/`judger`，也不得为 Mistral 人工插入 Qwen 专属的 `<think>` 标记。最终 messages、渲染文本、token IDs、attention mask、message fingerprint 和 rendered prompt hash 都必须进入 cache provenance。

`qwen_only` 和 `mistral_only` 同样调用 `build_role_messages(role="judger", ...)`。即使 K=0 时没有 latent prefix，也不删除 judger prompt 中关于 latent reference 的描述，从而与现有 `analysis` 的 receiver-only 条件以及对应通信条件保持完全相同的 prompt。

建议第一版固定：

- `sender_budget = 1024`；
- `tau = 0.6`；
- sender `do_sample = false`；
- receiver `do_sample = false`；
- receiver budget 沿用现有 dataset-specific 配置；
- `causal_shift = false`；
- batch 内先去除每个样本的 padding，再拼接并统一右 padding。

正式 prefix 顺序只能是：

```text
aligned sender prompt
+ aligned sender planner output
+ receiver native judger prompt
```

receiver 不得接收 sender token IDs。

## 6. 三阶段执行与缓存

### 6.1 Stage A：Planner context collection

分别使用 Qwen 和 Mistral：

1. 渲染 planner prompt；
2. greedy 生成 planner output（cache 字段建议记为 `sender_plan_ids`）；
3. 验证生成序列以 sender prompt 开头；
4. 对 `sender_prompt + sender_plan` 重新执行完整 forward；
5. 保存全部有效位置的 final-layer hidden states。

Planner cache 至少保存：

- dataset/split/selection fingerprint；
- `item_id` 和 `question_hash`；
- sender model ID、revision、tokenizer fingerprint；
- rendered prompt、prompt hash 和 prompt token count；
- sender plan IDs、文本和 token count；
- full context IDs、mask 和 hidden states；
- hidden capture semantics；
- greedy 参数和 sender budget；
- generation 与 full-forward 耗时；
- schema version 和代码 revision。

该 cache 与 transport direction、receiver、`tau` 无关，可以被多个 downstream condition 复用。

### 6.2 Stage B：Exact STT 与 Judger evaluation

对于 sender hidden chunk `H [B,L,d_A]`：

```text
logits_A = LMHead_A(H)
p_A_full = softmax(logits_A / tau, dim=-1)
p_A = gather(p_A_full, source_token_ids)
p_B^T = sparse_mm(T, p_A^T)
Z_B = p_B @ E_B[target_token_ids]
```

实现必须：

- 使用完整 source vocabulary softmax；
- 使用全部稀疏边；
- 不 dense 化完整 `T`；
- 只按 sender position 或 target vocabulary 分块；
- 按原位置顺序还原 `Z_B`；
- transport 累加使用 FP32，最后才转换成 receiver embedding dtype；
- 用小型 dense oracle 验证稀疏和分块实现。

拼接 aligned sender context 和 receiver prompt 后，显式计算 attention mask 与 position IDs，先 prefill，再使用 receiver 自己生成的离散 token 和 KV cache 进行 greedy decode。

Receiver cache identity 必须包含：

- sender manifest hash；
- transport artifact SHA-256、方向和 schema；
- source/target fingerprints；
- sender/receiver model revisions；
- `tau`、sender budget、receiver budget；
- full-context、no-shift、greedy 等协议标志；
- receiver prompt hash；
- dataset fingerprint、selection policy 和 evaluator version；
- numeric dtype 与代码 revision。

### 6.3 Stage C：Cache-only analysis

分析任务只能读取已经验证的 cache，不得在 cache 缺失时隐式重新 rollout。输出包括逐题表、dataset summary、paired statistics、图和统一报告。

## 7. 正式计算环境与作业调度

完整实验将在另一张具备足够显存的 GPU 上运行，设计中不设置显存约束。正式实现允许 sender、receiver、sender LM head、receiver input embeddings 和稀疏 transport matrix 同时驻留，无需卸载完整 sender、选择性读取 LM head 或把一次样本执行拆成跨作业的模型切换。

推荐的单个双模型实验单元按协议直接执行：

1. 同时加载并验证 sender、receiver 和对应方向的 transport artifact；
2. sender 生成 planner output，并对 `sender prompt + sender plan` 完整 forward；
3. 使用仍在显存中的 sender LM head 立即执行 exact STT；
4. 将 aligned full context 前置到 receiver native judger prompt；
5. receiver 完成 prefill、greedy decode 和任务评测；
6. 原子写入 planner context、receiver result、计时和 provenance cache。

正式配置使用 32-position chunk 和 8192-target-vocabulary chunk；二者都只用于等价的 FP32 分块累加，并已纳入 receiver cache identity。不同 chunk size 必须通过 dense oracle 的 chunk-invariance 验证。不得因为显存充足而 dense 化完整 `T`，也不得改变完整 vocabulary softmax、全部稀疏边或 prefix 顺序。

Stage A 的 planner cache 仍然保留，用于确保确定性 trajectory 可复查、可复现；Stage A/Stage B 的逻辑边界是缓存与 provenance 边界，不代表必须卸载模型或使用不同 GPU 作业。每个 dataset analysis job 必须显式列出其四个 receiver cache IDs，最终 report job 必须显式列出每个 dataset 的 analysis result ID；cache-only 阶段不得通过目录扫描猜测依赖，以免旧提交或旧 smoke cache 污染当前结果。

若运行环境使用 Slurm，所有计算任务仍按 scheduler 规则提交；若正式高显存 GPU 使用其他调度环境，则保持相同任务入口、job matrix、cache identity 和依赖关系，不把调度器差异写入实验条件。

Slurm 依赖关系建议为：

```text
collect_qwen_planner ─┐
                     ├─ qwen_to_mistral ─┐
collect_mistral_planner ─ mistral_to_qwen ├─ cache_only_analysis ─ report
mistral_only ─────────────────────────────┤
qwen_only ────────────────────────────────┘
```

四个系统在每个 dataset 上形成一个主单元，共 `3 datasets × 4 systems = 12` 个正式主实验单元。

## 8. 指标与统计

### 8.1 任务指标

- AIME/ARC：accuracy；
- HumanEvalPlus：pass@1；
- 无法解析率；
- 代码执行失败率；
- 每个 dataset 单独结果；
- 三个 dataset 的 macro-average，不直接混合不同任务的题目。

### 8.2 配对比较

预先指定两个主要效应：

```text
Delta_Q_to_M = score(qwen_to_mistral) - score(mistral_only)
Delta_M_to_Q = score(mistral_to_qwen) - score(qwen_only)
```

统计方法：

- 同题 paired difference；
- 10,000 次 question-level paired bootstrap，报告 95% CI；
- 二元正确率补充 exact McNemar test；
- 分 dataset 报告后再计算 macro-average；
- 不把 greedy decoding 的不同 seed 当成独立实验重复。

### 8.3 协议与效率诊断

每个样本同时记录：

- sender prompt/plan/full-context token counts；
- aligned prefix length 与 transferred-length ratio；
- `abs(sum(p_A)-1)` 和 `abs(sum(p_B)-1)`；
- `H(p_A)`、`H(p_B)` 与 effective support size；
- aligned embedding norm 和非有限值计数；
- planner generation、full forward、STT、receiver prefill、decode、evaluation 耗时；
- peak GPU memory；
- sender/receiver output token counts；
- error 类型和 cache/provenance IDs。

## 9. 测试与验收顺序

1. 先运行既有 `analysis/tests/` 回归测试，确认 kernel-analysis-v1 的配置、cache 和结果 contracts 未改变；
2. 验证既有正式 kernel job matrix 的文件集合与预期计数完全不变；
3. 验证 STT 四个矩阵的正式行数分别严格为 6、12、3、1，且无空行、重复 cache ID 或非法 task；
4. 用小词表和小型稀疏矩阵验证 dense oracle；
5. 验证不同 position/target chunk size 给出相同结果；
6. 验证双方向 shape、fingerprint、revision 和 artifact hash；
7. 验证任一方向不匹配时 fail closed；
8. 验证 source vocabulary 完整覆盖和 target IDs 合法；
9. 验证 sender full context 严格等于 planner prompt + planner output；
10. 验证 no causal shift；
11. 验证 batch 去 padding、prefix 顺序和 position IDs；
12. 验证没有 sender token IDs 进入 receiver；
13. 验证 cache identity 对方向、artifact、tau、模型 revision 和 prompt 敏感；
14. 验证 STT task 可由现有 `analysis_job.slurm` worker 调度，并通过 submitter dry-run 的依赖图检查；
15. Slurm 单题、双方向 GPU smoke；
16. 每个 primary dataset 至少四题的 integration smoke；
17. 运行 12 个正式主实验单元；
18. cache-only 统计与最终报告。

只有两个方向都通过同一套 strict artifact gate、协议测试和 GPU smoke 后，才能把双向结果作为对称实验进行比较。
