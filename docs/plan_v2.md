# Latent CoT 与 Latent Communication 的分层分析实验计划

## 1. 结论边界

暂设 token-to-ID 词表完全一致：

$$
T_{A\to B}=I.
$$

本计划严格区分三层。

| 层级 | 状态关系 | 核心问题 | 不能推出的结论 |
| --- | --- | --- | --- |
| 算子层 | $q\mapsto F(q)$ 或 $\hat F(q)$ | ORF 是否准确、稳定、快速 | 不说明 CoT 或通信成功 |
| Latent CoT | 同模型跨时间步 $h_t\to e_{t+1}$ | 注入状态后能否持续推理 | 不说明跨模型语义保留 |
| Latent Communication | $h^A\to e^B$ | B 是否理解和利用 A 的消息 | 不说明同模型 rollout 稳定 |

跨模型的 exact 映射为

$$
F_{A\to B}(q)=W_{\mathrm{in}}^B\operatorname{softmax}
\left(W_{\mathrm{out}}^Aq/\tau+b^A\right).
$$

记 $w_i^\top$ 为 $W_{\mathrm{out}}^A$ 的第 $i$ 行，$c_i=(W_{\mathrm{in}}^B)_{:,i}$，并令

$$
\phi_{\Omega}(x)=
\frac{1}{\sqrt m}
\left[
\exp\left(\omega_r^\top x-\frac{\|x\|_2^2}{2}\right)
\right]_{r=1}^{m},
$$

其中 $\Omega=[\omega_1^\top;\ldots;\omega_m^\top]$ 由 block ORF 构造。离线预聚合

$$
S=\sum_{i=1}^{V_A}c_i e^{b_i^A}\phi_{\Omega}(w_i)^\top,
\qquad
z=\sum_{i=1}^{V_A}e^{b_i^A}\phi_{\Omega}(w_i).
$$

于是 kernel 映射的在线形式为

$$
\hat F_{A\to B}(q)=
\frac{S\phi_{\Omega}(q/\tau)}
{z^\top\phi_{\Omega}(q/\tau)}.
$$

exact $F$ 在线需要完整词表扫描；$\hat F$ 在线只需计算 $m$ 个特征和一次矩阵向量乘法，复杂度为 $O(m(d_A+d_B))$。所有实验中，$F$ 仅作为离线 reference 或通信中的数值 oracle，$\hat F$ 才是可部署的核近似。

$F$ 是 kernel $\hat F$ 的数值 oracle，不是语义真值。对同模型 CoT，$F_{M\to M}$ 只是一个待检验的状态投影策略；其效果必须由 rollout 和任务结果判断。

## 2. 固定模型、数值设置和数据

### 2.1 模型矩阵

| ID | A | B | 用途 |
| --- | --- | --- | --- |
| C0 | `Qwen/Qwen2.5-1.5B-Instruct` | 同一模型 | Latent CoT 主实验 |
| C1 | `Qwen/Qwen2.5-7B-Instruct` | 同一模型 | CoT 尺度复现 |
| X1 | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | 通信主实验，含 identical/linear/kernel |
| X2 | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | 跨尺度通信，含 linear/kernel |

每个 X 组合必须先逐项检查 tokenizer 的 vocab size、每个 ID 的字符串、special token ID 一致；任何一项不一致即停止该组合。保存模型 revision、hidden dimension、weight tying 信息和运行库版本到 `artifacts/plan_v2/manifests/compatibility.json`。

### 2.2 固定参数

- 模型：bfloat16、`eval()`、`torch.inference_mode()`；
- 算子和统计量：float32；每组合额外 256 个 state 以 float64 exact reference 审计，若 relative-$L_2$ 差异 p99 大于 $10^{-4}$ 则停止；
- 主 kernel：`m=2048`、$\tau=1.0$、`kernel_chunk_size=4096`、ORF seed 101；
- linear：`align_ridge=1e-5`；生成：`temperature=0`、`top_p=1.0`、seed 77；
- 主 ORF seed 集：$\{101,202,303,404,505\}$；方差专用 seed：$1001,\ldots,1032$。

### 2.3 数据和统计单位

算子分析的状态池分为两类，且所有图表必须分层报告，不能只给混合后的总数：

1. **问题状态（prompt states）**：只输入 question、query 或代码 prompt，不输入参考答案或人工 CoT。prompt 上限 512 token，超长保留前 480 与后 32；若 non-special token 不超过 50 则全部保留；否则等距抽取 50 个位置，并强制包含末位置。
2. **回复状态（reply states）**：对同一条输入，以 `latent_mas` 的实际 agent 配置（模型组合、role prompt、`prompt=sequential`、greedy 解码、生成 seed 77）生成回复；从每个 agent 的模型生成 token 对应的末层 pre-unembedding hidden state 中抽取至多 50 个位置（不含 special token）；不足 50 个则全取，超过时等距抽取并强制包含回复末位置。回复最多保留 512 token；若产生 EOS 则在 EOS 前截断。这里的回复是模型在线生成的内容，而非数据集 gold answer 或 teacher-forced CoT，故不向算子注入标注答案。

对单 agent 设置，采样该 agent 的回复；对 X1/X2 通信设置，S0--S3 以发送方 A 的回复 state 为主，并在附录以接收方 B 的原生回复 state 复现。若一次 `latent_mas` 运行有多个 agent turn，则每个 turn 独立标记 `turn_id`、`agent_id`、`prompt/reply`；主分析仅使用第一轮 A 回复，避免同一题后续通信造成的依赖。`probe_seed=42` 确定随机题目和保留的 turn。

| 用途 | 数据 | 随机题目数 | state 采样 |
| --- | --- | ---: | ---: |
| 算子 calibration | ARC-Easy train | 至多 50 | question/reply 各至多 50 state |
| 通用 test | ARC-Easy test | 至多 50 | question/reply 各至多 50 state |
| 难度 test | ARC-Challenge test | 至多 50 | question/reply 各至多 50 state |
| 数学 OOD | GSM8K test | 至多 50 | question/reply 各至多 50 state |
| 专业 OOD | MedQA | 至多 50 | question/reply 各至多 50 state |
| 代码 OOD | MBPP+ test | 至多 50 | question/reply 各至多 50 state |
| 小样本 OOD | GPQA Diamond test | 至多 50 | question/reply 各至多 50 state |

ARC-Easy train 由分析脚本直接读取，因为当前 loader 固定 test。每个算子数据集以 `probe_seed=42` 不放回随机抽取至多 50 题；每题的 prompt 与一条 `latent_mas` 回复各贡献至多 50 个 non-special hidden states。算子实验以题目为 cluster：先在每个题目内分别平均 prompt state 与 reply state，再在同一数据集、同一状态来源内汇总；不得把长回复因 token 多而赋予更高权重。除分层曲线/表外，可给 prompt/reply 等权的汇总值，但必须明确标注。所有主指标报告 mean、median、p90/p95/p99 和按题目 cluster bootstrap 1,000 次 95% CI。

CoT 数据为 GSM8K train 128（calibration）、GSM8K test 512、ARC-Challenge test 512、AIME 2024/2025 全部；预算分别为 1024、1024、512、4096 token。CoT 的统计单位是一条完整题目 rollout。通信的统计单位是一条 message 或协作题目。

## 3. 第一层：算子近似

### S0. 范数和困难区域

**运行**：X1、X2，所有静态数据的 prompt 与 A-reply state；X1 的 B-reply state 置附录。统计 B input embedding 每个 token 的 $\|(W_{\mathrm{in}}^B)_{:,i}\|_2$、A output key 的 $\|(W_{\mathrm{out}}^A)_{i,:}\|_2$，以及每一来源状态的 $\|h_L^A\|_2$、$\|q\|_2$、$\|w_i+q/\tau\|_2$。

输出按 `prompt/reply` 分面的 histogram、ECDF、p1/p5/p25/p50/p75/p95/p99；回复额外按 reply token index 的相对位置四分位分面。误差热图按 entropy、置信度、输入 prompt length、reply length 和状态来源分桶。

**意义**：解释低温、高范数、尖锐分布为何更难近似；不将此图解释为推理质量。

### S1. exact-$F$ 静态保真和性能

**运行**：X1 在全部 test/OOD；X2 在 ARC-Easy 与 GSM8K。每个数据集均对 prompt 与 A-reply state 运行；X1 的 B-reply state 在 ARC-Easy/GSM8K 附录复现。ARC 两集跑 5 个主 ORF seed，其他集跑前 3 个。

完整 vocab 计算 $F(q)$，报告

$$
\frac{\|\hat F(q)-F(q)\|_2}{\|F(q)\|_2+10^{-8}},
\qquad \cos(\hat F(q),F(q)),
$$

同时直接评估单核近似。对每个 state 令 $x=q/\tau$，以 exact softmax rank 分层抽取 key：rank 1、rank 2--10 中稳定 hash 选 3 个、rank 11--100/101--1000/大于 1000 中各均匀抽 3 个；固定抽样由 state ID 与 `probe_seed=42` 决定。对每个 $(w_i,x)$ 计算

$$
k_i(x)=\exp(w_i^\top x),\qquad
\hat k_i(x)=\phi(w_i)^\top\phi(x),
$$

并报告

$$
|\hat k_i-k_i|,\qquad
\frac{|\hat k_i-k_i|}{k_i+10^{-8}},\qquad
|\log(\hat k_i+10^{-8})-\log(k_i+10^{-8})|.
$$

单核的 mean、median、p90/p95、rank 分层误差和 $\hat k_i/k_i$ 校准散点图，与 $\hat F$ 的相对 $L_2$ 误差及 cosine 一起按 `prompt/reply` 分面；另报告每题的 reply--prompt 误差差值及配对题目 bootstrap CI。按状态来源分别报告分母异常率、NaN/Inf 率。性能仅测在线映射：warm-up 200 次，CUDA 同步后测 1,000 次；从两类状态各均匀抽取至多 500 个 state，保证性能分布不被单一来源主导。
**意义**：证明 kernel 是否以可接受误差替代完整词表扫描；只是方法必要条件。

### S2. softmax 误差传递与 ORF/iid 消融

**运行**：X1，ARC-Easy/ARC-Challenge/GSM8K 的 prompt 与 A-reply state。显式计算 $p$ 与 $\hat p$，按来源报告 KL、JS、TV、top-1/top-10/top-100 overlap、exact top-k mass，并报告 `reply - prompt` 的分层效应量。画单核误差、$\|p-\hat p\|_1$、$\|F-\hat F\|_2$ 的散点图，点形区分状态来源。

仅 ARC-Easy train calibration 运行：block ORF、iid Gaussian RF、可选重复方向负对照，

$$
m\in\{256,512,1024,2048,4096\},
\quad \tau\in\{0.7,1.0,1.3\},
\quad \mathrm{seed}\in\{101,202,303,404,505\}.
$$

**意义**：验证误差是否按“核 $\to$ softmax $\to$ embedding”传播，并检验 ORF 是否比 iid 更稳定。calibration 只作附表，不能修改主设置。

### S3. 固定文本的 variance--$\tau$ 与 std--$\tau$

**运行**：X1、ARC-Easy train。固定随机 50 条题目；每题取 prompt 最后 state，以及从同次 `latent_mas` A 回复中等距抽取 16 个 state（回复不足 16 个则全取）。每个 state 固定取 exact rank 1、稳定 hash 选出的 rank 2--10、rank 100--1000 key。prompt/reply 各自汇总，不能将两类 state 混为同一独立样本。GSM8K test 随机 50 条在 $m=2048$ 复现。

对 $m\in\{512,1024,2048\}$、$\tau\in\{0.5,0.6,\ldots,2.0\}$，仅改变 $R=32$ 个 ORF 矩阵。计算

$$
\hat k_{ij}^{(r)}=\phi_{\Omega_r}(w_i)^\top\phi_{\Omega_r}(q_j/\tau),
$$

$$
s^2_{k,ij}=\frac{1}{R-1}\sum_r(\hat k_{ij}^{(r)}-\overline k_{ij})^2,
\qquad
s^2_{F,j}=\frac{1}{d_B(R-1)}\sum_r\|\hat F_j^{(r)}-\overline F_j\|_2^2.
$$

画 kernel 与 $\hat F$ 的原始、相对 variance/std--$\tau$ 曲线，原始方差用 log y 轴；按 rank 与 `prompt/reply` 双分面，阴影为题目 cluster bootstrap 1,000 次 CI。另画每题 reply/prompt 方差比的森林图，补充

$$
\operatorname{MSE}_j=
\frac{\|\overline F_j-F(q_j)\|_2^2}{d_B}+s^2_{F,j}.
$$

**意义**：严格度量“固定文本，只换 ORF”造成的随机不确定性，区分低方差和低偏差。

### S4. 通信空间 PCA

**运行**：仅 X1；ARC-Easy test 随机 50 题的全部 prompt 与 A-reply states，主设置和 seed 101；GSM8K 图置附录。为避免某一来源主导主成分，对两类状态各分层随机抽取相同数量（最多 2,000）后，将 $Y_F,Y_{\mathrm{identical}},Y_{\mathrm{linear}},Y_{\mathrm{kernel}}$ 拼接并只拟合一次全局中心化 PCA，四面板共享坐标。颜色为 exact entropy 四分位、点形区分 `prompt/reply`；另画灰色 exact 点与三方法叠加图、每种来源各 100 条配对箭头。

可选 t-SNE 只对四组拼接后的分层 2,000 点拟合一次，参数固定为 `init=pca`、`perplexity=50`、`learning_rate=auto`、`n_iter=1500`、seed 101。

**意义**：展示映射在 B 输入空间相对 exact 的几何偏移；不得解释为 CoT trajectory。

## 4. 第二层：同模型 Latent CoT

### C0. 接口、基线和实验范围

**运行**：C0 为主；C1 在 GSM8K test 的固定 128 题复现。使用仓库 `latent_mas` 同模型路径，固定 `prompt=sequential`、相同 agent role prompt 和 judge；主 `latent_steps=16`，扫描 $K\in\{0,4,8,16,32\}$。

比较：纯文本 CoT、`identical`、`linear`、exact $F_{M\to M}$、kernel $\hat F_{M\to M}$。文本 CoT 是可读行为参考，不是 latent mapping oracle；exact $F_{M\to M}$ 是 kernel 的数值 oracle，不是正确 latent thought 的语义 oracle。

**意义**：明确区分“近似算得准”与“同模型状态接口适合推理”。

### C1. rollout 稳定性和最终任务结果

**运行**：C0 于 GSM8K test 512、ARC-Challenge test 512、AIME 全部；C1 于 GSM8K test 128。每种方法使用相同题目、生成 seed 和 $K$。

每条 rollout 保存最终答案、格式成功、每 step hidden norm、相邻 step cosine、分母异常、NaN/Inf、文本长度、重复率。报告相对 `identical` 和文本 CoT 的 paired accuracy difference、bootstrap CI、accuracy--$K$ 与异常率--$K$ 曲线。

**意义**：直接检验同模型的跨时间状态注入能否支撑推理；即使 kernel 接近 exact $F$，若两者均表现差，也不能将 $F$ 宣称为好的 CoT 接口。

### C2. 文本 CoT 锚点

**运行**：C0；GSM8K train 128 和 GSM8K test 256。先为每题生成一条固定 greedy 文本 CoT，保存每个 token 前的 $h_t^{\mathrm{text}}$、预测分布和真实 token embedding $W_{\mathrm{in}}[:,x_t]$。仅在 $\max_i p_i\ge0.8$ 的位置，比较四种接口输出到真实下一 token embedding 的 cosine、rank、最近邻 token；并比较文本/latent trajectory 的 readout entropy 与 top-k stability。

**意义**：提供可读文本推理过程的弱参考，而非逐步监督标签；低置信和多种等价思路不应被当作错误。

### C3. language-lens 轨迹

对第 $k$ 个同模型 latent state 读出

$$
p_k^M=\operatorname{softmax}\left(
W_{\mathrm{out}}^M\operatorname{Norm}_M(h_k^M)+b^M
\right).
$$

**运行**：C1 所有题目；GSM8K/ARC-Challenge 各固定 20 题输出逐步案例。保存 top-10、probability、entropy、top-10 mass、special-token mass；画随 $k$ 的轨迹，并人工标注重复、空白/special token 主导、主题突变。

**意义**：使 latent CoT 的动态退化可见。它只是 language-lens，不把单个 top-1 token 当作完整 thought 的语义标签；主要结论仍以 C1 任务结果为准。

### C4. CoT 中的 ORF seed 敏感性

**运行**：C0；GSM8K test 128、ARC-Challenge test 128；$m=2048,\tau=1.0,K=16$，seed $\{101,202,303,404,505\}$。以题目为 cluster 记录最终答案、每 step entropy、top-k、轨迹长度、异常率，画 seed 雨云图、answer agreement 和 step-wise 置信带。

**意义**：静态方差小不保证反馈系统稳定；该实验测随机特征误差是否在同模型 rollout 中放大。

## 5. 第三层：跨模型 Latent Communication

### M0. 单次消息的接收端保真

**运行**：X1 于 ARC-Challenge、GSM8K、MBPP+ 的各 4,096 static states；X2 于 ARC-Challenge、GSM8K 复现。source state 映射成 B input embedding，在相同 B prefix、attention mask、position ID 下插入一次；完成 B 前向后，读取该位置 B 末层的 next-token distribution。

报告 kernel/linear/identical 相对 exact $F$ 的 B-logits KL/JS、top-1/top-10 agreement 和 greedy first-token agreement。

**意义**：验证 B-space 中的小近似误差是否保留 B 的局部行为；这仍只是接收端数值 fidelity，不等于复杂语义理解。

### M1. 私有信息恢复

**运行**：X1 主实验、X2 复现。新增固定 `data/communication_probe.jsonl` 共 1,024 条：每条有 entity、attribute、16 类 value；A 只看完整事实，B 只看 entity/attribute 查询。split 固定 train/calibration/test 为 256/256/512，value/entity/template 分层均衡；仅作评测，禁止训练。A 做 4 个 latent step，最终 state 一次映射给 B。

比较无消息 B、随机匹配消息、错配消息、exact $F$、kernel、linear，以及 X1 的 identical。报告 top-1/top-3 accuracy、NLL、ECE、混淆矩阵和 kernel 相对 exact 的 paired difference。

**意义**：B 无法从自身查询猜到 value，因此该实验能直接检验消息是否传递了 A 的私有信息；随机/错配对照排除先验和额外计算效应。

### M2. 共享问题协作

**运行**：X1 于 ARC-Challenge test 512、GSM8K test 512；X2 各 128 题复现。A、B 都看题；A 先做 16 个 latent step，B 接收最终 message 后独立 greedy 作答，最大生成长度分别为 512/1024。

比较 B-alone、A-alone、B+随机消息、B+错配消息、B+exact $F$、B+kernel、B+linear，以及 X1 的 B+identical。报告 task accuracy、B 相对 B-alone 的 paired improvement、kernel 相对 exact 的 paired degradation、总延迟与传递 embedding 数。

**意义**：测消息能否改善真实推理，而非把 B 自身能力、额外 prompt 或额外计算误判为通信成功。

### M3. 多跳通信链

**运行**：X1；私有信息 test 固定 256 条和 GSM8K test 固定 128 条。比较 A$\to$B 一跳与 A$\to$B$\to$A 两跳，每跳采用同一映射并保存 receiver readout。比较 exact $F$、kernel、linear，X1 另加 identical。

报告私有信息 accuracy、相对 exact chain 的 readout KL、top-k overlap、消息范数、异常率和 hop 退化曲线。

**意义**：测跨模型消息重复重编码的误差累积；不得称为 Latent CoT。

### M4. 通信 language-lens 案例

对 X1 私有信息与共享问题各固定 20 条，保存 A 原生读出和 B+exact/kernel/linear/identical 的 top-5。B 读出只能来自注入 message 后完成 B 前向的末层 state：

$$
p^{B,M}=\operatorname{softmax}\left(
W_{\mathrm{out}}^B\operatorname{Norm}_B(h_L^{B,M})+b^B
\right).
$$

禁止把 A$\to$B 的 input embedding 直接乘 B unembedding。

**意义**：解释成功、错译、退化 special token 等消息案例；它是定性辅助，结论由 M1--M3 的量化指标支持。

## 6. 图表、统计、产物和停止条件

主报告必须有：算子 error ECDF/分位数表、latency--error Pareto、ORF/iid variance--$\tau$ 与 bias--variance--MSE、通信 PCA、CoT accuracy--$K$ 与 language-lens、私有信息 accuracy/ECE、共享任务 paired improvement、多跳退化、paired forest plot 和 seed 雨云图。

所有方法差异以 prompt/message/rollout 为成对单位做 bootstrap 95% CI；对多个数据集、$m$、$\tau$、方法声明显著性时报告 Benjamini--Hochberg 校正后的 q value，同时报告效应量和失败率。

### 6.1 实现实验目录

所有实验实现必须置于 `exp` 下，并按结论层级分目录；不得把三类实验混在同一入口中。

| 实验范围 | 实现目录 | 覆盖内容 |
| --- | --- | --- |
| S0--S4 算子近似 | `exp\approximator` | 状态采样、exact $F$、单核误差、ORF/iid、variance--$\tau$、PCA/t-SNE 与性能测量 |
| C0--C4 Latent CoT | `exp\latent_cot` | 同模型 rollout、接口对照、答案评测、language-lens、动态 seed 稳定性 |
| M0--M4 Latent Communication | `exp\latent_comm` | 跨模型消息注入、接收端保真、私有信息、协作、多跳与通信 readout |

每个目录应有独立的可执行入口、配置文件、README 和 `result/` 子目录；运行产物只写入各自的 `result/`，并在 manifest 中记录对应的实验目录、配置路径和 git commit。 `exp` 中的分析实验可以读取或调用工作目录下的其他文件和模块，但绝对不得修改 `exp` 外的任何内容；所有新建、覆盖或删除操作只能作用于该实验目录自身及其 `result/` 子目录。建议入口分别为 `exp\approximator\run.py`、`exp\latent_cot\run.py`、`exp\latent_comm\run.py`；三者使用一致的 `--model_pair`、`--dataset`、`--method`、`--orf_seed` 参数命名，输出：

- `exp/<实验目录>/result/manifests/*.json`：revision、样本、token position、参数、seed；
- `exp/<实验目录>/result/metrics/*.parquet`：prompt/message/trajectory 原始指标；
- `exp/<实验目录>/result/readouts/*.jsonl`：top-k 与案例；
- `exp/<实验目录>/result/figures/*.pdf`：预注册图。

停止规则：

1. tokenizer 不一致、float64 审计失败、分母非有限/非正时停止该组合；
2. 显存不足时减少 state batch，不减少 vocab、不用 top-k 代替 exact $F$；
3. 私有信息数据或其 split 不可复现时，不报告语义通信结论；
4. 同模型 rollout 接口无法与仓库实现严格对应时，只报告算子和通信实验，不伪造 CoT 结论。

## 7. 展望：训练型方法

当前比较均为零额外训练。未来可加入 ThoughtComm、C2C 等 training-required 方法，但必须统一模型配对、tokenizer、任务 split、消息预算和生成预算；单列训练数据、监督信息、参数量、训练步数、GPU 时间、峰值显存、在线延迟和额外存储。

应分别绘制零训练与训练后方法的 Pareto 图，比较训练成本、推理成本、算子保真、信息恢复、CoT 稳定性和最终任务收益。若方法使用额外配对数据、教师输出或标签，必须将其列为额外信息预算，而不能将收益单独归因于对齐算法。


## 8. 各实验的意义速览

本节是对前述设计的解释性索引。每个实验只能支持其对应层级的结论；尤其不能将“算子误差小”直接表述为“Latent CoT 有效”或“跨模型通信成功”。

| 实验 | 它检验的假设 | 若结果良好，意味着什么 | 若结果不佳，优先怀疑什么 | 不能单独证明什么 |
| --- | --- | --- | --- | --- |
| S0 范数与尺度 | ORF 误差受 $\|w_i\|$、$\|q\|$、$\tau$ 和分布尖锐度影响 | 已定位方法适用的数值区域，可解释误差异质性 | 高范数、低温或数值稳定化不足 | CoT 或通信的语义有效性 |
| S1 exact-$F$ 保真 | $\hat F$ 能近似完整词表 $F$ 且更快 | kernel 是 exact soft-token 算子的有效压缩 | 特征数不足、温度不合适、ORF 实现/数值错误 | $F$ 本身是好的 CoT 或通信接口 |
| S2 分布误差链 | kernel 误差会以可解释方式传到 $\hat p$ 和 B-space 输出 | 数值误差来源清晰，不是 embedding 聚合偶然抵消 | 某些 key 或尖锐分布被随机特征系统性扭曲 | B 已理解消息语义 |
| S2 ORF/iid 消融 | 正交采样比 iid 随机方向更稳定 | ORF 的额外结构确有方差或 seed 稳定性收益 | ORF 不适合当前维度/温度，或收益不足以抵偿复杂度 | 任意输入上 ORF 都严格更优 |
| S3 variance--$\tau$ | 固定文本时，随机 $\Omega$ 的条件方差随 $\tau,m$ 可控 | 单次离线随机投影的 seed 风险已被量化；可据此选择 $m$ | 固定 seed 误差大，需提高 $m$、改变温度或做 ensemble | 闭环 CoT 一定稳定；动态系统可能放大小方差 |
| S4 通信 PCA | 各映射在同一 B-space 中的几何偏移可见 | kernel 的整体几何形状接近 exact $F$ | 某方法发生尺度、方向或簇结构偏移 | latent thought 的完整语义或 CoT 轨迹质量 |
| C0 CoT 接口对照 | 同模型时间接口与跨模型映射是不同设计问题 | 已建立文本 CoT、identical、linear、exact、kernel 的公平比较框架 | 接口或实验管线未对齐 | 任一映射天然优于其他映射 |
| C1 CoT rollout 与答案 | 状态注入后同一模型能否继续完成推理 | 某接口在动态反馈与最终答案上真正有用 | 反馈累积误差、接口不兼容、任务本身过难 | 跨模型通信也会成功 |
| C2 文本 CoT 锚点 | latent 状态接口与可读文本推理存在弱对应 | 接口输出在高置信位置与正常 token 轨迹相容 | 映射偏离正常 token 输入尺度/方向 | 每一步 latent thought 有唯一文本翻译 |
| C3 CoT language-lens | hidden readout 可暴露重复、塌缩和主题漂移 | 可以定位 rollout 在何时、以何种方式退化 | state 动力学异常、special-token 吸引子、过长 rollout | top-1 token 就是完整 thought 含义 |
| C4 CoT seed 敏感性 | 固定 ORF 随机性不会在反馈中被不可接受地放大 | 静态近似误差在动态 rollout 中仍可控 | 小静态误差被闭环放大，应提高 $m$ 或改接口 | 单一 seed 的好结果可泛化 |
| M0 单跳 B 行为 | kernel 相对 exact $F$ 保留 B 的局部 logits 行为 | 数值近似进入 B 后未明显改变即时行为 | B-space 小误差被 B 非线性放大 | B 能恢复复杂私有信息或完成协作 |
| M1 私有信息恢复 | A 的消息向 B 传递了 B 自身不可获得的信息 | 通信具有可量化的语义内容；kernel 损失可相对 exact 测量 | 映射失真、A 未编码事实、B 未能读取消息 | 共享任务上的协作一定提高 |
| M2 共享问题协作 | A 的消息能给 B 的真实推理带来增量 | 通信在真实任务上有实用价值 | 消息无信息、B 忽略消息，或收益来自额外计算/提示结构 | 任意任务、任意模型组合都会提升 |
| M3 多跳通信 | 单跳误差经过重复重编码不会迅速失控 | 方法可支持更接近多 agent 的消息链 | hop 间误差累积或接收方重编码损失 | 同模型 Latent CoT 的稳定性 |
| M4 通信 language-lens | 成功/失败消息可被接收端读出解释 | 可定位错译、special-token 退化或合理重述的机制 | 消息注入位置或 receiver dynamics 有问题 | 定性案例可替代 M1--M3 的统计证据 |

因此，推荐的论证顺序是：先以 S0--S4 证明 kernel 算子可控；再以 C1--C4 单独论证同模型 Latent CoT；最后以 M0--M3 论证跨模型 Latent Communication。三类结论应在论文或报告中分节书写，避免相互替代。
