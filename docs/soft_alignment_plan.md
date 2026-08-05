# Soft alignment 实施计划

## 1. 目标

在不增加新的方法层级的前提下，将 `soft` 直接加入现有的
`align_method`：

```text
identical / linear / kernel / soft
```

`soft` 固定表示 full-vocabulary exact softmax。它不采用 top-k，不使用
kernel 近似，也不对最终的期望 embedding 做额外范数归一化。

给定当前步最后一层 hidden state：

\[
h_t \in \mathbb{R}^{d_A},
\]

源模型输出层给出的 logits 和概率为：

\[
z_t = W_{\mathrm{out},A}h_t+b_A,
\]

\[
p_t = \operatorname{softmax}(z_t/\tau_{\mathrm{soft}}).
\]

送入目标模型的 soft embedding 为：

\[
e_t = p_t^\top E_{\mathrm{in},B}
    = \sum_{i=1}^{V}p_{t,i}E_{\mathrm{in},B}[i].
\]

同模型递推时，模型 A 与模型 B 是同一个模型；Hybrid 跨模型通信时，概率
由源模型 A 产生，embedding 由目标模型 B 提供。

## 2. `alignment.py`

### 2.1 类型和状态

将 `AlignMethod` 扩展为：

```python
Literal["identical", "linear", "kernel", "soft"]
```

为 `AlignmentState` 增加 exact soft 所需的信息：

- source output weight；
- source output bias；
- target input embedding weight；
- soft temperature；
- soft query chunk size。

增加 `build_soft_state()`，由源模型 output head 和目标模型 input embedding
构造 soft alignment 状态。

### 2.2 精确映射

在 `apply_alignment()` 中增加 `soft` 分支：

```python
logits = F.linear(hidden, output_weight, output_bias)
probabilities = torch.softmax(logits / soft_temperature, dim=-1)
aligned = probabilities @ input_weight
```

数值计算约定：

- hidden、logits 和 softmax 使用 FP32；
- 加权聚合使用 FP32；
- 最终结果转换回调用方的原始 dtype；
- 不应用 `identical/linear` 分支中的平均 embedding 范数缩放。

## 3. 显存控制

Hybrid 可能一次迁移形状为 `[batch, latent_steps, hidden_dim]` 的全部状态。
直接展开计算会构造：

```text
[batch * latent_steps, vocabulary_size]
```

大小的 logits 和概率张量，可能产生较高的峰值显存。

因此增加：

```text
--soft_chunk_size 32
```

该参数按 hidden query 数量分块：

```text
hidden chunk
→ full-vocabulary logits
→ full softmax
→ expected embedding
```

每个 query 仍访问完整词表，因此分块不会把 exact soft 变成 top-k 或近似方法。

## 4. `models.py`

在 `ModelWrapper` 中完成以下接入：

1. `_build_alignment_state()` 接受 `align_method=soft`；
2. 从 source model 取得 output weight/bias；
3. 从 target model 取得 input embedding weight；
4. `_apply_latent_realignment()` 继续统一调用 `apply_alignment()`；
5. `align_hidden_to()` 使用相同的 soft 实现完成跨模型映射。

完成后，下列现有路径无需各自复制公式：

- HF `generate_latent_batch()`；
- vLLM 辅助 HF `generate_latent_batch_hidden_state()`；
- Hybrid `align_hidden_to()`。

## 5. 同模型 soft thinking

每个 latent step 的数据流为：

```text
last_hidden
→ source lm_head
→ full-vocabulary logits
→ temperature softmax
→ probability-weighted input embedding
→ inputs_embeds
→ Transformer
→ next hidden
```

每个 soft embedding 仍占据一个序列位置，并正常产生该位置的 K/V。现有
Planner、Critic、Refiner、Judger 之间通过 KV cache 传递上下文的结构保持
不变。

## 6. Hybrid soft communication

模型从 A 切换到 B 时执行：

\[
p_A = \operatorname{softmax}
\left(
\frac{W_{\mathrm{out},A}h_A+b_A}{\tau_{\mathrm{soft}}}
\right),
\]

\[
e_B = p_A^\top E_{\mathrm{in},B}.
\]

这表示：

- token 概率语义来自源模型 A；
- 输入向量位于目标模型 B 的 embedding space；
- 不直接复用 A 的 KV cache；
- B 使用迁移后的 soft embeddings 重建自己的 KV cache。

第一版继续要求 A、B 的 token-to-ID vocabulary 完全一致。若词表大小、token
字符串或 token ID 对应关系不同，应明确报错，不进行未经定义的跨 tokenizer
概率映射。

## 7. 命令行参数

在 `run.py` 中：

- `--align_method` 增加 `soft`；
- 增加 `--soft_temperature`，默认值为 `1.0`；
- 增加 `--soft_chunk_size`，默认值为 `32`。

三个温度参数的语义必须独立：

| 参数 | 用途 |
| --- | --- |
| `--temperature` | Judger 最终文本生成和采样 |
| `--kernel_temperature` | kernel alignment 的核近似目标 |
| `--soft_temperature` | exact soft logits 分布 |

## 8. 运行脚本

更新实际负责运行和提交实验的脚本，包括：

- `run.sh`；
- `run_all.sh`；
- `run_all_fast.sh`；
- 其他包含 alignment 白名单或参数透传的 run 脚本。

修改内容包括：

1. alignment 合法值加入 `soft`；
2. 透传 `soft_temperature` 和 `soft_chunk_size`；
3. 需要运行全方法组合时，把 `soft` 加入 alignment 数组；
4. 保持用户显式指定 alignment 时的过滤行为；
5. 不改变现有重复实验、数据量默认值和结果汇总行为。

按当前命名规则，soft 实验的输出目录应自然包含：

```text
<dataset>_latent_mas_soft_<prompt>_<model>_<time>
```

其中每次重复和汇总继续采用：

```text
result/<config>/repeat_<n>.json
result/<config>/summary.json
logging/<config>/repeat_<n>.txt
```

## 9. 测试计划

### 9.1 Alignment 单元测试

使用小型人工权重验证 `soft` 与直接参考实现一致：

```python
expected = torch.softmax(
    F.linear(hidden.float(), output_weight.float(), output_bias.float())
    / soft_temperature,
    dim=-1,
) @ input_weight.float()
```

覆盖：

- output bias 正确生效；
- `soft_temperature` 正确生效；
- batch hidden 输入；
- 三维 `[batch, steps, hidden_dim]` 输入；
- chunked 和 non-chunked 结果一致；
- 不发生额外范数缩放；
- 输出形状和 dtype 正确；
- 非有限输入或非法 temperature 得到明确错误。

### 9.2 同模型和 Hybrid 测试

验证：

- 同模型 soft recurrence 能连续运行多个 latent steps；
- KV cache 每一步正常增长；
- 跨模型时使用 A 的 output head 和 B 的 input embedding；
- 同词表、不同 hidden size 的模型能在维度允许时正确映射；
- token-to-ID vocabulary 不一致时明确拒绝。

### 9.3 回归和静态验证

执行：

- Python 编译检查；
- Shell 脚本语法检查；
- 现有 alignment 和 metrics 测试；
- `identical`、`linear`、`kernel` 回归测试；
- 使用可用的小模型或构造模型完成一次 soft latent smoke test。

## 10. Logging、result 与文档

每次运行的配置和最终 `summary.json` 应记录：

```json
{
  "align_method": "soft",
  "soft_temperature": 1.0,
  "soft_chunk_size": 32
}
```

logging 可以额外记录 soft distribution 的诊断统计，例如平均 entropy、最大
概率和 top-k token/probability，但这些信息只用于观察，不参与实际递推，也不能
被描述成 latent state 的唯一文本解码。

相关算法文档应明确区分：

| Alignment | 定义 |
| --- | --- |
| `identical` | hidden state 的单位映射及范数缩放 |
| `linear` | hidden state 到输入 embedding 的线性映射及范数缩放 |
| `kernel` | soft expected embedding 的随机特征近似 |
| `soft` | full-softmax exact expected embedding |

## 11. 验收标准

实现完成需同时满足：

1. `--align_method soft` 可以从直接命令和所有调度脚本运行；
2. 每个 latent step 使用完整词表概率的期望 embedding；
3. 实现不进行采样、argmax、top-k 截断或 kernel 近似；
4. soft 输出不做额外范数缩放；
5. 同模型、辅助 HF 和 Hybrid 路径复用同一公式；
6. 跨模型仅在 token-to-ID vocabulary 完全一致时执行；
7. 原有 alignment 方法和结果目录结构保持不变；
8. 新增测试与现有回归测试全部通过。
