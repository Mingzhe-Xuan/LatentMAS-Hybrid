# `run.sh` 中的 Token 与时间统计口径

## 结论

`run.sh` **不直接计算实际消耗的 token 或推理耗时**。它做三件事：

1. 为每个任务确定生成上限 `max_new_tokens` 与重复次数 `times`；
2. 逐次调用 `run.py`，每次将结果写到 `result/<config>/repeat_N.json`；
3. 调用 `aggregate_results.py`，对这些 JSON 的数值字段逐层取算术平均，写为 `result/<config>/summary.json`。

真正的 token 计数和时间测量发生在 `run.py`、`models.py`、`methods/*.py` 中。

## `run.sh` 传入的 token 上限（不是实际 token 数）

`MAX_NEW_TOKENS` 默认是空字符串。`resolve_max_new_tokens()` 按以下优先级给 `RESOLVED_MAX_NEW_TOKENS` 赋值：

1. 若环境/脚本变量 `MAX_NEW_TOKENS` 非空，直接使用它；
2. 否则读取 `params_dict.json[TASK].max_token`；
3. 文件不可读、JSON 无效、任务不存在或值不是正整数时，回退为 `20000`。

该值通过 `--max_new_tokens "${RESOLVED_MAX_NEW_TOKENS}"` 传给 `run.py`，是**单次文本生成的最大新 token 上限**，不是最终统计值。各方法的使用方式为：

| 方法 | 此上限应用位置 |
| --- | --- |
| `baseline` | 单个 Agent 的文本生成 |
| `text_mas` | 每个文本 Agent（包括 Judger）的文本生成 |
| `latent_mas` | 最终 Judger 的文本生成；latent Agent 的步数由 `--latent_steps` 控制 |

`latent_steps` 同样由 `params_dict.json[TASK].latent_steps[prompt]`（或显式变量）解析，但它是 latent rollout 的步数，不是文本 token 上限。

## 实际 token 如何计算

每题的每个 Agent 都由 `build_agent_metrics()` 写入四类整数；`run.py` 再按角色及全体角色求和。

| 字段 | 计算口径 |
| --- | --- |
| `text_input` | 当前角色可见的文字 prompt 的非 padding token 数。LatentMAS 会包含仍被保留的历史文字 prompt。 |
| `latent_input` | 当前 prompt 前已有 KV cache 的长度（`_past_length(past_key_values)`）。其中可能混有历史文字和 latent 位置，且可能包含 batch cache 的物理 padding。 |
| `text_output` | 文本生成时为模型返回的 completion token ID 数（包含第一个 EOS、排除之后的 padding）；latent Agent 中则是其执行的全词表 `W_out` 投影次数。 |
| `latent_output` | 实际执行的 latent rollout 步数；每一步对应一个连续 embedding。早停时可小于配置的 `latent_steps`。 |

顶层 `results.tokens.<field>.total` 是该字段在全部题目、全部角色上的和；`average_per_problem = total / processed`。角色下的平均数则是角色总数除以该角色的 `samples`。

`results.output_tokens` 正常情况下等于所有角色 `text_output` 的总和。仅当旧格式结果没有 role metrics 时，才把每个可见 `agent.output` 用 tokenizer 重新编码求和；此回退路径不计 latent state。

注意：`text_output` 不能一概理解为“可见文本”。对 latent Agent，`soft` 每个 latent step 都会执行一次完整的 `W_out`→softmax→`W_in`，因此其 `text_output = latent_output`；`kernel_early_stopping` 仅每 10 个 latent steps 为熵检查执行一次完整 `W_out` 投影，因此 `text_output = floor(latent_output / 10)`。`identical`、`linear`、`kernel` 为 0。`latent_output` 始终表示实际 latent rollout 步数。

## 实际时间如何计算

### 总时间

`run.py` 在 `ModelWrapper` 构造完成后记录 `start_time = time.time()`，在所有 batch 推理、答案解析/判分和日志写入完成后计算：

```text
total_seconds = time.time() - start_time
seconds_per_sample = total_seconds / processed
```

因此 `timing.total_seconds` 不含模型/tokenizer 加载和初始化阶段建立的 alignment state；但包含数据迭代、tokenize、prompt 构造、模型调用、解析/判分、进度条和日志 I/O。`seconds_per_sample` 是整批运行的均摊吞吐成本，不是单题真实延迟。

### 模型阶段时间

模型阶段写在 `timing.model_phases`，每个 batch 先用 `perf_counter()`（GPU 调用前后会 `torch.cuda.synchronize()`）测墙钟时间，再由 `build_agent_metrics()` 除以 batch size 分摊到 batch 内每题，最后 `run.py` 汇总。字段含义如下：

| phase | 测量边界 |
| --- | --- |
| `prefill_seconds` | Latent 路径中为 prompt 的一次前向；HF 文本生成中为 `generate()` 开始到首 token logits；vLLM 中按请求 TTFT/decode 比例由 batch 墙钟时间估计。 |
| `latent_decode_seconds` | 全部 latent steps 的墙钟时间，包含 alignment、模型前向、Python 循环和张量操作。 |
| `alignment_seconds` | latent loop 内 hidden state 到输入 embedding 的对齐时间；CUDA 用 event，CPU 用 `perf_counter()`。 |
| `text_decode_seconds` | HF 中从首 token logits 到 `generate()` 返回；vLLM 中按请求指标比例估计的 decode 部分。 |

`model_phases.<phase>.total` 是上述“按 batch 均摊后”再加总的值，`average_per_problem = total / processed`。这些 phase **不能相加后与 `total_seconds` 对账**：`alignment_seconds` 是 `latent_decode_seconds` 的子集；而 `total_seconds` 还包含许多 phase 未覆盖的工作。vLLM 的 phase 切分也是按请求指标比例估计，并非严格的独占时间。

## 重复运行与平均

`run_repeated()` 以 `SEED, SEED + 1, ...` 运行 `RESOLVED_TIMES` 次；`times` 的来源也遵循“显式 `TIMES` → `params_dict.json[TASK].times` → 1”的优先级。每次重复都保留独立 JSON。

随后 `aggregate_results.py` 对所有共同存在的数值叶子节点取算术平均（保留 6 位小数），所以 `summary.json` 中：

```text
average.timing.total_seconds
    = mean(repeat_N.timing.total_seconds)
average.results.tokens.text_output.total
    = mean(repeat_N.results.tokens.text_output.total)
```

也就是说，`summary.json` 的 `total` 仍是**单次重复实验的平均总量**，并不是把所有重复的总量累加。聚合元数据 `aggregation.repetitions`、`aggregation.seeds` 和 `aggregation.source_files` 可用于追溯来源。

## 与 PBS 时间限制及日志的关系

- `#PBS -l walltime=72:00:00` 是调度器允许的最长作业时间，不是统计出的实验耗时，也不会写入 JSON 指标。
- `STATE_FILE`（默认 `state/run_state.txt`）记录 shell 输出；它是运行日志，不参与 token/time 数值计算。
- 结果指标在每次的 `repeat_N.json` 和聚合后的 `summary.json` 中，而非 state log。

## 代码入口

- `run.sh`：参数解析、重复运行和聚合。
- `run.py`：总耗时、四类 token 与角色/全局汇总，生成单次结果 JSON。
- `utils.py`：将 batch 级 phase 时间均摊给单题/单角色。
- `models.py`：HF、vLLM、latent rollout 与 alignment 的底层计时和输出 token ID 计数。
- `aggregate_results.py`：跨重复的递归数值平均。

## Linear method: cross-dataset ratios of tokens/steps to decode time

Each calculation uses the `average` totals in `result/*linear*/summary.json` (the mean over repetitions for the same configuration). Text throughput is `text_output.total / text_decode_seconds.total`; latent throughput is `latent_output.total / latent_decode_seconds.total`; alignment cost per step is `alignment_seconds.total / latent_output.total`. The table pairs only counts and time from the same phase: `alignment_seconds` is not added to `latent_decode_seconds`, because it is a subset of it.

| Dataset | Model | Topology | text tokens/s | text ms/token | latent steps/s | latent us/step | alignment us/step | alignment / latent decode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Challenge | Qwen3-14B | hierarchical | 119.49 | 8.369 | 430.69 | 2321.84 | 8.20 | 0.353% |
| ARC-Challenge | Qwen3-8B | hierarchical | 159.38 | 6.274 | 461.14 | 2168.54 | 6.49 | 0.299% |
| ARC-Challenge | Qwen3-14B | sequential | 98.17 | 10.187 | 471.50 | 2120.90 | 7.56 | 0.357% |
| ARC-Challenge | Qwen3-4B | sequential | 117.34 | 8.522 | 514.37 | 1944.14 | 4.69 | 0.241% |
| ARC-Challenge | Qwen3-8B | sequential | 151.94 | 6.582 | 491.13 | 2036.11 | 5.96 | 0.293% |
| ARC-Easy | Qwen3-14B | hierarchical | 140.67 | 7.109 | 500.88 | 1996.47 | 8.05 | 0.403% |
| ARC-Easy | Qwen3-8B | hierarchical | 177.39 | 5.637 | 514.75 | 1942.69 | 6.54 | 0.337% |
| ARC-Easy | Qwen3-14B | sequential | 141.80 | 7.052 | 498.58 | 2005.69 | 8.09 | 0.403% |
| ARC-Easy | Qwen3-8B | sequential | 189.29 | 5.283 | 535.22 | 1868.41 | 6.40 | 0.343% |
| GSM8K | Qwen3-14B | hierarchical | 102.71 | 9.736 | 494.54 | 2022.09 | 7.52 | 0.372% |
| GSM8K | Qwen3-8B | hierarchical | 171.31 | 5.837 | 534.94 | 1869.36 | 5.84 | 0.312% |
| GSM8K | Qwen3-14B | sequential | 100.56 | 9.944 | 487.08 | 2053.03 | 7.65 | 0.372% |
| GSM8K | Qwen3-8B | sequential | 162.73 | 6.145 | 520.54 | 1921.10 | 5.71 | 0.297% |
| HumanEval+ | Qwen3-14B | hierarchical | 89.59 | 11.163 | 180.50 | 5540.27 | 12.34 | 0.223% |
| HumanEval+ | Qwen3-8B | hierarchical | 81.53 | 12.266 | 193.22 | 5175.43 | 10.30 | 0.199% |
| HumanEval+ | Qwen3-14B | sequential | 74.47 | 13.428 | 162.80 | 6142.36 | 13.36 | 0.217% |
| HumanEval+ | Qwen3-8B | sequential | 79.61 | 12.562 | 175.17 | 5708.87 | 11.50 | 0.201% |
| MBPP+ | Qwen3-14B | hierarchical | 113.98 | 8.774 | 232.74 | 4296.72 | 11.71 | 0.273% |
| MBPP+ | Qwen3-8B | hierarchical | 60.74 | 16.464 | 251.68 | 3973.38 | 9.50 | 0.239% |
| MBPP+ | Qwen3-14B | sequential | 65.22 | 15.334 | 162.33 | 6160.18 | 14.30 | 0.232% |
| MBPP+ | Qwen3-8B | sequential | 70.93 | 14.097 | 173.33 | 5769.41 | 12.50 | 0.217% |
| MedQA | Qwen3-14B | hierarchical | 55.30 | 18.083 | 164.05 | 6095.56 | 14.97 | 0.246% |
| MedQA | Qwen3-8B | hierarchical | 83.55 | 11.968 | 176.03 | 5680.86 | 12.46 | 0.219% |
| MedQA | Qwen3-14B | sequential | 82.75 | 12.084 | N/A | N/A | N/A | N/A |
| MedQA | Qwen3-8B | sequential | 114.78 | 8.712 | N/A | N/A | N/A | N/A |

Conclusion: there is no fixed cross-dataset ratio. Text throughput ranges from 55.30 to 189.29 tokens/s across all configurations (3.42x); even within Qwen3-8B + sequential, the six datasets range from 70.93 to 189.29 tokens/s (2.67x). For the runs with usable latent counts, latent throughput ranges from 162.33 to 535.22 steps/s (3.30x); within Qwen3-8B + sequential and excluding MedQA's missing counts, it still ranges from 173.33 to 535.22 steps/s (3.09x). Alignment costs 4.69--14.97 us/step and accounts for 0.199%--0.403% of latent decode time, so it is not a fixed proportion either.

MedQA sequential has `latent_output.total = 0` and near-zero `latent_decode_seconds` in both aggregate results, indicating legacy timing/counting data are unavailable. Its latent columns are therefore marked N/A instead of incorrectly suggesting zero cost per step. These ratios are useful empirical throughput values for a particular model, topology, dataset, and runtime environment; input/cache length, output length, batch shape, and GPU scheduling can all change them.