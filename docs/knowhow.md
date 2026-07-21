# LatentMAS 推理机制速读

本文整理仓库中模型推理、隐状态通信、`past_key_values`、`attention_mask` 与 batch 的工作方式。这里的重点是**代码实际如何调用模型**，而非泛泛的多智能体概念。

## 1. 从哪里开始运行

实验入口是 [`run.py`](../run.py)。它完成以下流程：

```text
命令行参数
  -> 加载数据集（data.py）
  -> 创建 ModelWrapper（models.py）
  -> 创建一种方法（methods/）
  -> 将题目按 batch 送入 method.run_batch()
  -> 解析答案、计算正确率、写入日志和结果 JSON
```

可选择的四种方法如下：

| 方法 | 中间通信方式 | 最终输出 |
| --- | --- | --- |
| `baseline` | 无 | 单模型直接生成文本 |
| `text_mas` | 前一 Agent 的可见文本 | Judger 生成文本 |
| `latent_mas` | 共享 Transformer KV cache 中的隐状态 | Judger 生成文本 |
| `latent_mas_hybrid` | 跨模型对齐后的隐状态 | Judger 生成文本 |

四个固定 Agent 的顺序为：

```text
Planner -> Critic -> Refiner -> Judger
```

前三者形成中间推理，`Judger` 负责输出最终可读答案。

## 2. 一个模型内部的推理在哪里实现

本仓库**没有手写 Transformer 的 Attention、MLP 或采样器**。它通过 Hugging Face Transformers（或可选 vLLM）加载由 `--model_name` 指定的模型：

```python
self.model = AutoModelForCausalLM.from_pretrained(model_name, ...)
```

见 [`models.py`](../models.py)。因此，具体的模型内部实现由模型架构决定；例如 Qwen 模型会在 Transformers 对应的 `modeling_qwen*.py` 中实现。

一个典型 Decoder-only LLM 的内部数据流是：

```text
token IDs
  -> token embedding
  -> 多个 Decoder Layer
       -> RMSNorm -> Self-Attention -> residual
       -> RMSNorm -> MLP / SwiGLU -> residual
  -> final RMSNorm
  -> lm_head（映射到词表 logits）
  -> 按采样策略选下一个 token，并重复
```

本仓库进入模型的两种主要调用是：

```python
# 标准文本自回归生成
self.model.generate(...)

# 单次 Transformer 前向传播
self.model(
    input_ids=...,
    attention_mask=...,
    past_key_values=...,
    output_hidden_states=True,
)
```

分别封装在 `ModelWrapper.generate_text_batch()` 与 latent 生成函数中。

## 3. `output_hidden_states=True` 得到什么

开启 `output_hidden_states=True` 后，模型返回 `outputs.hidden_states`。对于仓库常用的 Qwen 类 Decoder 架构，通常可按以下方式理解：

```text
hidden_states[0]       = 输入 token 的 embedding
hidden_states[1..L-1]  = 各 Transformer layer 的残差输出
hidden_states[-1]      = 最后一层输出再经过 final RMSNorm
```

所以 [`models.py`](../models.py) 中：

```python
e_t = outputs.hidden_states[0][:, -1, :]
last_hidden = outputs.hidden_states[-1][:, -1, :]
```

`e_t` 是输入 embedding；`last_hidden` 是最后一个位置的最终层表示，通常已经过模型末尾的 RMSNorm。

注意，中间 layer 的输出并不是每一项都额外经过最终 RMSNorm。虽然每个 layer 内部会在 Attention 与 MLP 前做 RMSNorm，但残差相加后的 layer 输出仍是残差流状态。

## 4. 普通文本推理

Baseline 的路径最直接：

```text
问题
  -> prompts.py 构造消息
  -> tokenizer 编码为 input_ids 和 attention_mask
  -> model.generate()
  -> tokenizer 解码为文本
  -> utils.py 提取答案或运行代码测试
```

TextMAS 的差别只在于四个 Agent 依次生成文本，并把前三个 Agent 的输出拼接到后续 Agent 的 prompt：

```text
Planner 文本
  -> Critic 的 context
  -> Refiner 的 context
  -> Judger 的 context
  -> 最终答案
```

其实现见 [`methods/text_mas.py`](../methods/text_mas.py)。

## 5. LatentMAS 如何进行“隐状态推理”

LatentMAS 的前三个 Agent 不生成可见推理文本。每个 Agent 先处理自身 prompt，再从最后层取出一个隐状态，将其作为连续 embedding 反复输入模型：

```text
Agent prompt
  -> Transformer 前向
  -> 取最后 token 的 last_hidden
  -> 转成可作为输入的 latent embedding
  -> 用 inputs_embeds 再前向一次
  -> 重复 latent_steps 次
  -> 把更新的 KV cache 交给下一个 Agent
```

核心循环位于 [`models.py`](../models.py) 的 `generate_latent_batch()`：

```python
latent_vec = self._apply_latent_realignment(last_hidden, source_model)
latent_embed = latent_vec.unsqueeze(1)
outputs = self.model(
    inputs_embeds=latent_embed,
    attention_mask=latent_mask,
    past_key_values=past,
    use_cache=True,
    output_hidden_states=True,
    return_dict=True,
)
```

`_apply_latent_realignment()` 依据 `--align_method` 将最终 hidden state 调整为可作为输入的 embedding。`identical` 使用单位映射并缩放到目标 embedding 的平均范数；`linear` 使用输入/输出 embedding 构造岭回归线性映射后缩放；`kernel` 按 `docs/algo_detail.md` 预聚合 ORF 核近似统计量，在在线阶段直接输出目标 embedding 空间向量而不扫描全词表。

LatentMAS 的 Agent 链是：

```text
Planner 的 prompt + latent steps -> past_kv_1
Critic 的 prompt + past_kv_1 + latent steps -> past_kv_2
Refiner 的 prompt + past_kv_2 + latent steps -> past_kv_3
Judger 的 prompt + past_kv_3 -> 生成最终文本
```

实现见 [`methods/latent_mas.py`](../methods/latent_mas.py)。因此，Planner、Critic、Refiner 对后续 Agent 的影响不以文本出现，而是留存在 cache 中。

## 6. `past_key_values` 是如何维护的

`past_key_values`（代码中也称 `past_kv`）是每个 Transformer layer 的 Attention Key/Value cache。概念上的单层形状为：

```text
K, V: [batch, kv_heads, cached_sequence_length, head_dim]
```

### 6.1 首次前向

首次调用时 `past_key_values=None`：

```text
prompt tokens
  -> 模型为每层计算 K/V
  -> outputs.past_key_values 保存 prompt 的 K/V
```

### 6.2 后续前向

后续调用把上次 cache 传回模型：

```python
outputs = self.model(
    ...,
    past_key_values=past,
    use_cache=True,
)
past = outputs.past_key_values
```

模型内部会保留旧 K/V，只为本轮新输入计算 K/V，再将新位置追加到 cache。一次 latent step 会在每层 cache 中新增一个没有离散 token ID 的连续向量位置：

```text
[prompt K/V]
  -> [prompt K/V, latent-1 K/V]
  -> [prompt K/V, latent-1 K/V, latent-2 K/V]
```

仓库负责传入旧 cache 和接收新 cache；具体的 K/V 追加由 Transformers 的模型实现完成。

### 6.3 裁剪 cache

`LatentMASMethod._truncate_past()` 会保留每层 K/V 在序列维度的最后 `N` 个位置：

```python
layer_cache[..., -tokens_to_keep:, :]
```

该逻辑只在 `sequential_info_only` 或 `latent_only` 为真时启用。当前 `run.py` 没有暴露这两个命令行参数，因此通常默认不裁剪，cache 会随 Agent prompt 和 latent step 累积。

## 7. `attention_mask` 的含义与维护

`attention_mask` 是 0/1 张量，含义是“输入位置是否有效”：

```text
1 = 真实 token / 有效 latent embedding，可以被 attention 使用
0 = 为 batch 对齐加入的 padding，应被 attention 忽略
```

例如两题长度不同：

```text
input_ids:
A: [token1, token2, token3, token4]
B: [token1, token2, PAD,    PAD]

attention_mask:
A: [1, 1, 1, 1]
B: [1, 1, 0, 0]
```

它由 tokenizer 在 `padding=True` 时生成，见 [`models.py`](../models.py) 中的 `prepare_chat_batch()`。模型会将它和因果 mask 结合：一个位置既不能注意到未来 token，也不能注意到 padding。

### 7.1 有历史 KV cache 时

仓库没有单独持久化历史 mask，而是根据 cache 长度重新构造：

```python
past_mask = torch.ones((batch, past_len))
attention_mask = torch.cat([past_mask, current_attention_mask], dim=-1)
```

语义是：

```text
[历史 cache 的位置全部视为有效] + [当前 prompt 的真实 token 为 1、padding 为 0]
```

当追加一个 latent embedding 时，代码构造长度为 `past_len + 1` 的全 1 mask，因为新 latent 位置本身一定有效。

### 7.2 一个 batch 相关的注意点

在 `generate_bs > 1` 且不同样本长度不同时，较短样本的早期 prompt padding 也会占据 cache 位置。下一轮该仓库将历史 cache 全部设置为 1，因此可能把原先的历史 padding 当成有效位置。

更严格的实现应维护每个样本自己的 `cached_attention_mask`，并在追加或裁剪 KV cache 时同步追加或裁剪该 mask。`generate_bs=1` 时不存在这个 batch padding 问题。

## 8. Batch 维度：能否同时推理多个问题

可以。`--generate_bs` 控制每次送入模型的题目数量。`run.py` 收集到一批题目后调用 `method.run_batch(batch)`。

典型张量形状：

```text
input_ids:      [B, prompt_length]
attention_mask: [B, prompt_length]
hidden_states:  [B, sequence_length, hidden_dim]
K/V cache:      [B, kv_heads, cached_length, head_dim]
```

其中 `B = generate_bs`。尽管 Python 层面只有一个 `past_kv` 变量，它的第 0 维含有 B 道题各自独立的 cache。它们共享一次 GPU 的矩阵计算，但每一行只关注自身的上下文，不会在逻辑上互相通信。

增大 `generate_bs` 能提高吞吐，但会增加显存占用。对于 LatentMAS，显存还会随 prompt 长度、Agent 数量和 `latent_steps` 增加。

## 9. Hybrid：模型切换时如何处理 cache

KV cache 不能跨模型直接复用：模型 A 与模型 B 的层数、hidden size、注意力头或权重可能不同。

Hybrid 模式维护两类状态：

```text
cumulative_prompts          = 所有前序 Agent 的 prompt 文本
cumulative_latent_hiddens   = 前序 Agent 产生的原始 latent hidden states
```

切换模型时，[`methods/latent_mas_hybrid.py`](../methods/latent_mas_hybrid.py) 会：

```text
1. 用新模型重新编码此前 prompt 文本；
2. 将旧模型的 raw latent hidden states 映射为新模型的输入 embeddings；
3. 拼接「新模型 prompt embeddings + 迁移后的 latent embeddings」；
4. 在新模型上做一次前向，重新建立新模型专属的 KV cache；
5. 用该 cache 继续新 Agent 的 latent steps 或 Judger 解码。
```

跨模型迁移的不是 `past_kv` 本身，而是 hidden state。使用的映射为：

```text
W_cross = (W_out,A^T W_out,A + lambda I)^-1 W_out,A^T W_in,B
embedding_B = hidden_A @ W_cross
```

对应函数为 `transfer_via_realignment()`。

## 10. 推荐的源码阅读路径

若目标是彻底读懂一次运行，建议按此顺序：

1. [`run.py`](../run.py)：运行、批处理、方法选择；
2. [`methods/text_mas.py`](../methods/text_mas.py)：最直观的多 Agent 文本上下文传递；
3. [`models.py`](../models.py)：tokenization、文本生成、latent 前向、KV cache；
4. [`methods/latent_mas.py`](../methods/latent_mas.py)：共享 cache 的四 Agent 流程；
5. [`methods/latent_mas_hybrid.py`](../methods/latent_mas_hybrid.py)：切模型时如何重建 cache；
6. 实际 `--model_name` 对应的 Transformers `modeling_*.py`：Attention、RMSNorm、MLP、KV cache 的底层实现。

## 11. vLLM：本仓库的高吞吐生成后端

vLLM 是面向高吞吐 LLM 推理的引擎。相较逐条调用 Hugging Face 的 `model.generate()`，它会调度多个请求，并高效管理自己的 KV cache。仓库通过 `--use_vllm` 启用它；固定依赖版本是 [`requirements.txt`](../requirements.txt) 中的 `vllm==0.17.0`。

### 11.1 模型下载、缓存与普通推理

模型名来自 `--model_name`。仓库在 [`run.py`](../run.py) 中将 `HF_ENDPOINT` 设为 `https://hf-mirror.com`，所以 Hugging Face 风格的模型和 tokenizer 会从该镜像下载。首次初始化 `LLM(model=model_name)` 时下载或读取权重、配置与 tokenizer；之后复用本地 Hugging Face 缓存。建议设置：

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

启用 vLLM 后，[`models.py`](../models.py) 创建引擎：

```python
self.vllm_engine = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    gpu_memory_utilization=gpu_util,
)
```

普通推理路径：

```text
messages -> tokenizer 的 chat template 渲染为文本 prompt
         -> vllm_engine.generate([prompt_1, prompt_2, ...], SamplingParams)
         -> 每个请求的生成文本
```

仓库先自行套用 chat template；`SamplingParams` 中，`temperature` 控制随机性，`top_p` 控制 nucleus sampling，`max_tokens` 限制新增输出 token 数。

### 11.2 仓库 batch 与 vLLM 连续批处理

`--generate_bs` 是一次传给 `method.run_batch()` 的题目数。vLLM 在 `generate()` 内部还会进行连续批处理（continuous batching），动态混排各请求的 prompt 处理（prefill）和逐 token 解码（decode）。前者是仓库实验批大小，后者是 vLLM 引擎调度机制。

### 11.3 Prefix caching

若多个请求拥有完全相同的长前缀，vLLM 可以复用该前缀对应的引擎 KV cache。`latent_mas + --use_vllm` 时，`run.py` 自动启用 `enable_prefix_caching=True`。它不同于、也不能直接替代 Hugging Face 的 `past_key_values`。

### 11.4 LatentMAS 的双后端结构

LatentMAS 需要读取 hidden states、构造连续 embedding、维护 Hugging Face 格式的 `past_key_values`；这些由辅助 HF 模型完成。vLLM 则用于 Judger 的最终文本生成：

```text
HF 辅助模型（device2）
  Planner / Critic / Refiner
  -> raw latent hidden states + HF past_key_values
                 |
                 | 收集 latent embeddings
                 v
vLLM Engine（device）
  Judger prompt + latent embeddings（prompt_embeds）
  -> 最终答案文本
```

HF 的 `past_key_values` 不会交给 vLLM。代码将前三个 Agent 的 embedding 插入 Judger prompt，然后调用：

```python
vllm_engine.generate([{"prompt_embeds": embeds}, ...], sampling_params)
```

因此该路径以 `enable_prompt_embeds=True` 初始化；当前代码还限制 latent-vLLM 路径仅支持 Qwen。

### 11.5 多 GPU 与显存

`--tensor_parallel_size N` 表示将同一个模型的矩阵计算在 N 张 GPU 上分片并通信合并，而不是将 N 道题各放到一张卡。例如 Baseline/TextMAS 可用：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --method baseline \
  --model_name Qwen/Qwen3-14B --use_vllm --tensor_parallel_size 2
```

`--gpu_memory_utilization` 是 vLLM 单个引擎的目标显存使用比例；它影响可留给运行时和 vLLM KV cache 的空间。增大可能提高长请求并发能力，但也更容易 OOM。

对于 `latent_mas + vLLM`，通常应分两张卡且 vLLM 使用单卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B --task gsm8k --use_vllm \
  --device2 cuda:1 --tensor_parallel_size 1
```

```text
GPU 0: vLLM Engine（Judger）
GPU 1: HF 辅助模型（latent 推理）
```

此双模型设置不宜再设 `--tensor_parallel_size 2`：vLLM 会占用两张可见卡，而辅助 HF 模型也在 `cuda:1`，从而发生显存竞争；仓库不会主动阻止该冲突。

### 11.6 当前支持范围

| 方法 | vLLM 支持情况 |
| --- | --- |
| `baseline` | 支持，vLLM 直接生成文本 |
| `text_mas` | 支持，每个 Agent 由 vLLM 生成文本 |
| `latent_mas` | 支持，HF 负责 latent、vLLM 负责 Judger 最终生成 |
| `latent_mas_hybrid` | 主运行流程未接入 vLLM；跨模型路径使用 HF 模型 |

## 12. `run.py` 命令行参数速查

入口是 `python run.py`；以下以当前代码实际行为为准。`--method` 与 `--model_name` 是仅有的两个必填参数。

### 12.1 实验与数据集

| 参数 | 默认值 / 可选值 | 含义与适用范围 |
| --- | --- | --- |
| `--method` | 必填：`baseline`、`text_mas`、`latent_mas`、`latent_mas_hybrid` | 选择单模型、文本通信、同模型 latent 通信或异构 latent 通信的方法实现。 |
| `--model_name` | 必填 | Hugging Face 模型名/本地路径，亦为未指定 `--agent_models` 时各 Agent 的模型。当前 TextMAS/LatentMAS 提示词主路径要求 Qwen。 |
| `--task` | `gsm8k` | 数据集、提示词和评测规则；可选 `gsm8k`、`aime2024`、`aime2025`、`gpqa`、`arc_easy`、`arc_challenge`、`mbppplus`、`humanevalplus`、`medqa`。 |
| `--split` | `test` | 仅 GSM8K 实际使用；AIME 固定 `train`，其余当前入口固定 `test`。 |
| `--max_samples` | `-1` | 评测题数；`-1` 表示任务全部样本。 |
| `--seed` | `42` | 随机种子。HF/vLLM、硬件和采样实现仍可能造成细小差异。 |

### 12.2 Agent 结构与文本生成

| 参数 | 默认值 / 可选值 | 含义与适用范围 |
| --- | --- | --- |
| `--prompt` | `sequential`；或 `hierarchical` | 多 Agent 的提示词结构；影响 TextMAS 的文本上下文组织，以及 LatentMAS 的角色提示词。`baseline` 不使用。 |
| `--max_new_tokens` | `4096` | 生成上限：Baseline 为直接输出，TextMAS 为每个 Agent，LatentMAS/Hybrid 为 Judger 最终输出。 |
| `--temperature` | `0.6` | 采样温度；越高越随机。 |
| `--top_p` | `0.95` | nucleus sampling 阈值。 |
| `--generate_bs` | `20` | 外层 `run_batch()` 的题目数量，也作为生成 batch size；增大吞吐但提高显存。对 LatentMAS，变长样本批处理还需留意第 7.2 节的 mask/cache 限制。 |
| `--text_mas_context_length` | `-1` | TextMAS 传给后续 Agent 的已有文本长度，代码为 `context[:value]`。默认 `-1` 会删掉最后一个字符，**不是**无限长度；请设足够大的正数以保留全文，`0` 为不传上下文。 |

### 12.3 LatentMAS 参数

| 参数 | 默认值 / 可选值 | 含义与适用范围 |
| --- | --- | --- |
| `--latent_steps` | `50` | Planner、Critic、Refiner 各自的连续 latent embedding rollout 步数；会增长耗时、KV cache 和显存。仅 LatentMAS/Hybrid。 |
| `--think` | 关闭；指定即开启 | 为 latent Agent 手动追加 think token，改变 latent rollout 起点。仅 LatentMAS/Hybrid。 |
| `--align_method` | `identical`；`identical` / `linear` / `kernel` | latent hidden→目标输入 embedding 的对齐策略。`identical` 为单位映射加范数缩放；`linear` 为岭回归线性映射加范数缩放；`kernel` 使用 ORF 正随机特征的预聚合核近似。跨模型时当前要求 token 到 ID 的词表映射完全一致。非 `identical` 的 vLLM 路径需要 HF 辅助模型。 |
| `--align_ridge` | `1e-5` | `linear` 对齐的岭回归正则系数。 |
| `--kernel_features` | `1024` | `kernel` 的随机特征数 $\mathit{m}$；越大通常近似越好，但预计算、显存与在线计算开销也越大。 |
| `--kernel_temperature` | `1.0` | `kernel` 中 softmax 映射的温度 $\tau$，独立于文本生成的 `--temperature`。 |
| `--kernel_seed` | 继承 `--seed` | ORF 随机方向的种子，保证离线统计量与实验可复现。 |
| `--kernel_chunk_size` | `4096` | 构建 kernel 统计量时每次处理的词表行数，用于控制预计算峰值显存。 |
| `--agent_models` | `None` | 仅 Hybrid。按 `Planner Critic Refiner Judger` 顺序给出四个模型名；未给则全部用 `--model_name`，数量不是 4 会断言失败。应选 tokenizer 兼容的同族模型。 |

### 12.4 设备与 vLLM

| 参数 | 默认值 / 可选值 | 含义与适用范围 |
| --- | --- | --- |
| `--device` | `cuda` | 主 HF 模型或 vLLM 引擎的设备（如 `cuda:0`、`cpu`）。 |
| `--use_vllm` | 关闭；指定即开启 | 请求 vLLM；未安装时 `ModelWrapper` 回退 HF。Hybrid 的附加模型仍强制为 HF，因而不等同完整 Hybrid-vLLM 支持。 |
| `--tensor_parallel_size` | `1` | vLLM 切分同一个模型的 GPU 数，不是每卡一题。 |
| `--gpu_memory_utilization` | `0.9` | vLLM 单卡目标显存比例；调高可扩大其 cache 空间，也更易 OOM。 |
| `--device2` | 缺省时同 `--device` | HF 辅助模型的设备；建议 latent+vLLM 时与 vLLM 分置两卡。 |
| `--use_second_HF_model` | 关闭；指定即开启 | vLLM 同时使用时加载 HF 辅助模型，完成 latent hidden state、embedding 与 KV cache 计算。 |
| `--enable_prefix_caching` | 关闭；指定即开启 | 启用 vLLM prefix cache；当前只在 `latent_mas` 时传给 vLLM，不能控制 HF `past_key_values`。 |
