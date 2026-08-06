# Paper v0：ICLR 行文逻辑与写作蓝图

## 1. 论文的唯一中心主张

论文不应写成三个并列项目：approximator、latent CoT 和 latent communication。三类实验应构成同一个方法主张的逐层证据。

推荐的一句话中心论点：

> We introduce a vocabulary-mediated reasoning-and-communication protocol whose soft latent interface is efficiently approximated by random features, preserving full-softmax recurrent and communicative behavior at substantially lower online cost.

对应中文：

> 我们定义一个基于共享词表的 reasoning-and-communication protocol，其核心 `soft` latent interface 由随机特征高效近似；kernel 不仅在单步数值上逼近 full-softmax soft mapping，而且在闭环推理和跨 agent 通信中保留其行为，同时降低在线成本。

整篇论文的论证链固定为：

```text
多步推理与多 agent 协作首先需要选择 intermediate-state protocol
→ text、vocabulary distribution 和 hidden state 各有不同的信息—成本—兼容性权衡
→ Latent CoT 与 Latent Communication 共享 representation-to-input alignment 问题
→ shared vocabulary 提供 source-to-target 中介
→ soft decoding 可重写为 key/value 解耦的 vocabulary attention
→ full-softmax soft mapping 定义自然但在线成本高
→ kernel preaggregation 降低在线复杂度
→ 验证单步数值保真
→ 验证闭环误差不会快速放大
→ 验证任务行为与信息通信
→ 验证端到端效率和跨设置泛化
```

不要把“观察 entropy”或“PCA 中形状相近”写成中心贡献；它们只能解释主要行为结果。

### 1.1 先区分两个场景，再统一接口

Latent CoT 和 Latent Communication 不应混用。建议全文固定以下术语：

| 场景 | source | target | 核心问题 | 主要证据 |
| --- | --- | --- | --- | --- |
| **Latent CoT / latent recurrence** | agent $A$ 在 step $t$ 的状态 | 同一 agent $A$ 在 step $t+1$ 的输入空间 | 跨时间反馈是否落在模型可消费的 input manifold，并能否稳定迭代 | accuracy--$K$、failure--$K$、entropy 与闭环 divergence |
| **Latent Communication** | sender agent/model $A$ 的状态 | receiver agent/model $B$ 的输入空间 | 跨 agent、可能跨模型的 message 是否对齐且携带可利用信息 | private-information recovery、receiver gain、端到端协作准确率 |

二者共享的上位问题是 **representation-to-input alignment**，但不能笼统写成“hidden-space alignment”。即使 Latent CoT 中 $A=B$，最后层 hidden state、LM-head vocabulary geometry 与原生 input embeddings 也不是自动可互换的；Latent Communication 在此基础上还增加了模型维度、几何结构和 vocabulary compatibility 的差异。

统一记号使用 target-indexed interface

$$
F_{A\rightarrow B}: \mathcal H_A\rightarrow\mathcal X_B.
$$

Latent CoT 是 $F_{A\rightarrow A}$ 在时间上的反复应用；Latent Communication 是 $F_{A\rightarrow B}$ 在 agent edge 上的应用，通常 $A\neq B$。这个统一只发生在方法层，problem statement、实验假设和结论仍分别陈述。

## 2. Introduction 的五段结构

### 第 1 段：从 intermediate-state protocol 引出共同问题

不要从“LLM agents 近年来受到广泛关注”开始。第一段直接指出：任何多步语言模型系统都必须决定，如何让一次模型调用的中间计算被下一次调用消费。把 **protocol** 定义为中间状态的 encoding、transport 和 target-side injection 规则，而不仅是一种 tensor format。

推荐开头：

> Multi-step language-model systems require a protocol for making intermediate computation consumable by a subsequent model invocation. Such a protocol specifies how a source state is encoded, transported, and injected into a target input.

> We study two distinct uses of the same representation interface. Latent CoT feeds a model-derived state back into the same model across reasoning steps, whereas latent communication transfers a sender state into a receiver model across an agent edge.

> Both require mapping a source representation into a valid target input space, but they pose different behavioral questions: recurrent stability for Latent CoT and usable information transfer for Latent Communication.

随后比较三类 protocol。正文最好使用 **vocabulary-distribution (logit-mediated)**，而不是简称 `logits`，因为本文 `soft` 传递/消费的是 logits 诱导的完整词表分布及其 expected embedding，并非未经处理的 raw-logit vector。

| Protocol | Intermediate representation | 优点 | 核心局限 |
| --- | --- | --- | --- |
| **Text** | sampled discrete tokens | 可解释；直接落在模型原生输入分布；跨模型兼容性强 | 自回归 decoding 昂贵；sampling 丢失完整分布信息；消息通常较长 |
| **Vocabulary-distribution / logit-mediated** | 完整 next-token distribution 或其 expected target embedding | 保留不确定性；以共享词表为语义坐标；不需要 sampling | full-softmax 计算/显存随 $V$ 增长；要求 vocabulary 对齐 |
| **Hidden-state** | continuous activation/KV representation | 紧凑且信息丰富；绕过离散 decoding | source/target geometry、维度、尺度和 input manifold 不自动兼容；通常需要 adapter |

第一段最后收束为 protocol-selection problem：

> Text protocols are interoperable but discretize and serialize intermediate computation; hidden-state protocols retain continuous information but lack a shared coordinate system; vocabulary-distribution protocols offer a principled bridge between them, but incur vocabulary-sized online computation.

`identical` 和 `linear` 不作为第四、第五种上位 protocol，而应视为 hidden-state protocol 下的 alignment implementations；`soft` 和 `kernel` 则属于 vocabulary-mediated protocol 的 reference 与 efficient implementation。

### 第 2 段：共享词表把 soft decoding 化为跨模型 vocabulary attention

关键观察应表述为：source 和 target 即使 hidden dimension 或 representation geometry 不同，只要 tokenizer vocabulary 和 token ID 对齐，就共享一个离散坐标系。

因此可以定义：

$$
h_A
\xrightarrow{W_{\mathrm{out}}^A}
p_A
\xrightarrow{W_{\mathrm{in}}^B}
e_B,
$$

其中

$$
p_A=\operatorname{softmax}\!\left(
\frac{W_{\mathrm{out}}^Ah_A+b_A}{\tau}
\right),
\qquad
e_B=p_A^\top W_{\mathrm{in}}^B.
$$

这就是 `soft` vocabulary-mediated latent interface。忽略 output bias 时，令

$$
q=\frac{h_A}{\sqrt{\tau}},\qquad
k_i=\frac{W_{\mathrm{out},i}^A}{\sqrt{\tau}},\qquad
v_i=W_{\mathrm{in},i}^B,
$$

则

$$
e_B
=
\frac{\sum_{i=1}^{V}\exp(q^\top k_i)v_i}
{\sum_{i=1}^{V}\exp(q^\top k_i)}
=\operatorname{Attn}(q,K_A,V_B).
$$

因此，soft latent decoding 可以解释为对 vocabulary-sized memory 的 single-query attention：sender 的 output representations 是 keys，receiver 的 input embeddings 是 values。若存在 output bias，则其对应每个 vocabulary item 的固定乘性权重 $\exp(b_i/\tau)$，必须在 soft 与 kernel 两种实现中保持一致。

这个代数对应本身不能宣称为本文首创。LAVA NAT 已将 $\operatorname{softmax}(zW^\top)W$ 称为 vocabulary attention；已有工作也把 next-token prediction 描述为 context query 与 vocabulary keys 的 softmax 点积。Introduction 只需简洁承认这一背景，然后立刻指出本文的推进：

> We operationalize this correspondence in a different regime: the sender supplies vocabulary keys, the receiver supplies values, and a Performer-style factorization makes the resulting cross-model latent interface efficient.

必须同时声明：

- 它不 sampling、不 argmax；
- 它不是 latent thought 的唯一语义解释；
- 它允许 source/target hidden dimension 不同；
- 第一版要求 token-to-ID vocabulary 完全一致。

### 第 3 段：soft 的在线瓶颈

Soft 每个 query 需要完整词表 projection 和 expected-embedding aggregation：

$$
O(Vd_A+Vd_B).
$$

在多 agent、多 edge、长 latent recurrence 中，成本随

$$
\text{agents}\times\text{edges}\times\text{latent steps}\times V
$$

增长。由此自然提出问题：能否将 vocabulary-dependent computation 移到离线阶段，同时保留 soft interface 的行为？

### 第 4 段：方法

概述 ORF kernel approximation：将 exponential dot-product kernel 写成正随机特征，把 target vocabulary embeddings 预聚合为固定的 $S,z$，在线仅计算

$$
\hat F(q)=\frac{S\phi(q/\tau)}
{z^\top\phi(q/\tau)}.
$$

在线成本由词表大小 $V$ 转为 feature count $m$。此处只给 intuition，不在 Introduction 展开完整推导。

### 第 5 段：明确静态保真不足以支持行为 claim

主动指出论文面对的核心科学风险：

> A small one-step approximation error does not guarantee stable recurrent computation, useful latent reasoning, or successful communication after a nonlinear receiver processes the message.

因此论文依次验证：

1. broad end-to-end task effectiveness；
2. static fidelity and efficiency；
3. closed-loop behavioral fidelity；
4. task behavior of latent recurrence；
5. controlled information transfer and real-task collaboration；
6. generalization across model directions and agent edges。

这一段让后续实验看起来是为排除替代解释而设计，而不是事后堆叠分析。

## 3. Contributions：限制为三点

### Contribution 1：Cross-model interface and algorithm

> We generalize vocabulary attention into a key--value-decoupled cross-model interface, where sender output representations serve as keys and receiver input embeddings serve as values, and derive an ORF-based approximation that preaggregates receiver-side values and removes vocabulary-sized tensors from online communication.

### Contribution 2：Closed-loop evaluation

> We connect static mapping fidelity to recurrent behavior through step-wise soft--kernel divergence, accuracy--$K$, failure analysis, and seed sensitivity.

### Contribution 3：Broad empirical, communication, and systems evidence

> Across a broad end-to-end benchmark spanning model scales, task families, agent topologies, and alignment methods, we evaluate task effectiveness, recoverable private information, downstream collaboration, and the latency/memory--quality trade-off.

如果 M2 没有显著正结果，第三点改为“controlled information transfer and systems characterization”，不要宣称改善真实协作。

## 4. 推荐正文结构

### 4.1 Introduction

约 1--1.25 页。包含五段逻辑、方法总览图和三点贡献。

### 4.2 Background and Problem Formulation

约 0.75 页。这里是正式拆分 Latent CoT 与 Latent Communication 的最佳位置，建议使用三个连续小节。

#### 4.2.1 Latent CoT as intra-model recurrence

定义单个模型内的时间递推：

$$
x_{t+1}^A=F_{A\rightarrow A}(h_t^A).
$$

强调它研究的是 recurrent reasoning，而不是 agent 间通信。即使 source 和 target 属于同一模型，$h_t^A$ 也未必是模型训练时见过的 token embedding；因此问题是 output-state-to-input compatibility、闭环稳定性以及任务效用。对应实验是 C0--C4，正文主要由 RQ2 回答。

#### 4.2.2 Latent Communication as inter-agent transfer

定义 agent edge 上的消息：

$$
m_{A\rightarrow B}=F_{A\rightarrow B}(h_A),\qquad
x_B'=\operatorname{Inject}(x_B,m_{A\rightarrow B}).
$$

强调它研究的是 sender 信息能否被 receiver 恢复和利用。$A$ 与 $B$ 可以是同架构的不同 agent，也可以是不同模型；后者额外要求维度、geometry 和 vocabulary compatibility。对应实验是 M0--M3，正文主要由 RQ3 回答。

#### 4.2.3 Unified alignment interface

最后再统一二者：同一个 $F_{A\rightarrow B}$ 解决 source representation 到 target input space 的转换，Latent CoT 取 $B=A$，Latent Communication 允许 $B\neq A$。`identical`、`linear`、`soft` 和 `kernel` 是 interface choices，不是两个场景的名称。

在这里正式定义 protocol 与 interface 的层级，防止全文交替混用：

$$
\Pi_{A\rightarrow B}
=
\bigl(
\operatorname{Encode}_A,\,
\operatorname{Transport}_{A\rightarrow B},\,
\operatorname{Inject}_B
\bigr),
$$

其中 $F_{A\rightarrow B}$ 是 Encode/alignment 的核心映射，Transport 规定传递 text、distribution 或 continuous vector 的方式，Inject 规定结果如何进入 target prompt、embedding sequence 或 KV state。Latent CoT 中 transport 通常是同一进程内的时间反馈；Latent Communication 中 transport 对应显式 agent edge。本文的算法创新集中在 vocabulary-mediated $F_{A\rightarrow B}$，而端到端实验评价完整 protocol。

这里集中定义：

- same-model latent recurrence；
- cross-agent message injection；
- source/target vocabulary compatibility；
- `soft` mapping；
- 本文 claim 的边界。

Related Work 可紧随其后，控制在约 0.75--1 页，按问题组织：

1. latent reasoning and continuous thoughts；
2. language-model multi-agent communication；
3. soft tokens / expected embeddings；
4. random features and efficient softmax attention。

Related Work 的关键任务是解释：随机特征本身并不新，本文的新意来自 vocabulary-mediated cross-model mapping、target-value preaggregation，以及对静态保真、闭环行为和通信效用的完整连接。

建议明确组织为“已有组成部分—本文组合”的边界：

- [LAVA NAT](https://arxiv.org/abs/2002.03084) 已给出 $\operatorname{softmax}(zW^\top)W$ 形式的 vocabulary attention，但其 keys/values 来自同一个词表矩阵，目标是改进 non-autoregressive translation；
- [Correlation and Navigation in the Vocabulary Key Representation Space](https://arxiv.org/abs/2410.02284) 明确把 next-token prediction 表述为 context query 对 fixed vocabulary keys 的 softmax dot product，但不构造 receiver-side values；
- [Towards Decoding as Continuous Optimisation](https://aclanthology.org/D17-1014/) 等工作早已使用 probability-weighted expected embeddings，近期 [Soft Thinking](https://arxiv.org/abs/2505.15778) 将其用于连续推理；
- [Performer](https://arxiv.org/abs/2009.14794) 提供 softmax attention 的 FAVOR+ 随机特征近似，但不研究 vocabulary-mediated cross-model communication。

因此不要写“we first discover that decoding is attention”。更准确的 novelty statement 是：

> Our contribution is not the algebraic correspondence alone, but its use as a scalable cross-model communication interface: we decouple vocabulary keys and values across sender and receiver models, preaggregate receiver-side values with positive orthogonal random features, and evaluate whether the approximation remains faithful under closed-loop reasoning and communication.

### 4.3 Kernelized Soft Latent Interface

约 2 页，严格按照以下顺序：

1. 用统一的 $F_{A\rightarrow B}$ 定义 full-softmax `soft` reference；
2. decoding-as-vocabulary-attention correspondence；
3. sender-key/receiver-value decoupling；
4. numerator/denominator reformulation；
5. positive random features 与 block-ORF；
6. receiver-side value preaggregation；
7. online algorithm；
8. numerical stabilization；
9. 分别说明 $F_{A\rightarrow A}$ 的 Latent CoT 用法与 $F_{A\rightarrow B}$ 的 communication 用法；
10. complexity comparison。

方法章节可以共享公式和算法，但不要共享场景描述。推荐统一使用：

- **latent state / latent step / recurrence** 描述 Latent CoT；
- **latent message / sender / receiver / edge** 描述 Latent Communication；
- **interface / alignment method** 仅描述 `identical`、`linear`、`soft`、`kernel`。

建议给出复杂度表：

| 方法 | 离线成本 | 每 query 在线成本 | 在线 vocabulary-sized tensor |
| --- | ---: | ---: | --- |
| soft | 无 | $O(V(d_A+d_B))$ | 是 |
| linear | 拟合矩阵 | $O(d_Ad_B)$ | 否 |
| kernel | 一次词表预聚合 | $O(m(d_A+d_B))$ | 否 |

不要直接声称归一化后的 ratio estimator 无偏。单个 exponential-kernel estimator 的性质与最终 ratio mapping 的性质应严格区分。

如果理论工作量允许，增加一个简洁的误差分解：在 denominator 有下界时，$\|\hat F-F\|$ 如何由 numerator error 和 denominator error 控制。还可以用

$$
\|h_{t+1}^{K}-h_{t+1}^{S}\|
\le L_t\|h_t^{K}-h_t^{S}\|+\epsilon_t
$$

解释为什么静态 S1 不能替代闭环 C2；不必在无法验证假设时把它包装成强收敛定理。

### 4.4 Experimental Setup

约 0.75 页。不要按内部代码的 S/C/M 编号介绍，而应提出四个 research questions，直接对应 Results 的四个主体部分：

- **RQ0:** Does LatentMAS work end to end across model scales, task families, topologies, and alignment methods?
- **RQ1:** Does kernel approximate soft accurately and efficiently, both statically and under closed-loop use?
- **RQ2:** When does Latent CoT improve reasoning, and when does recurrent feedback become unstable?
- **RQ3:** Do latent messages transfer usable information and improve receiver behavior?

集中说明模型、数据集、split、seed、paired evaluation、bootstrap、失败计数、硬件和 temperature 约定。`run_all.sh` 提供 RQ0 的端到端主表：Qwen3-8B/14B 覆盖 9 个数据集，Qwen3-4B 覆盖其中 6 个数据集；每个 model--dataset 组合比较 12 个配置，即 baseline/TextMAS 的 sequential/hierarchical，以及 LatentMAS 的 identical/linear/kernel/soft × sequential/hierarchical，共 288 个数组任务。内部的 S0/C2/M1 编号可以放括号或附录索引，不应支配正文叙事。

## 5. Results 的叙事顺序

结果部分必须按照 claim 排列，而不是按照实验脚本编号排列。正文使用四个主体 block：**end-to-end effectiveness → kernel approximation → Latent CoT → Latent Communication**。Generalization 作为最后的 cross-cutting robustness section，不另起一条中心故事。

### 5.1 Overall end-to-end benchmark: does LatentMAS work?

首先给出由 `run_all.sh` 产生的 **Table 1**。这张表是论文的 headline result，回答 RQ0：在真实端到端设置中，不同 latent interface 相对 single-agent baseline 与 TextMAS 是否具有一致的任务价值。

主表覆盖：

- Qwen3-8B 与 Qwen3-14B：AIME 2024/2025、ARC-Challenge/Easy、GPQA、GSM8K、HumanEvalPlus、MBPPPlus、MedQA；
- Qwen3-4B：ARC-Challenge/Easy、GSM8K、HumanEvalPlus、MBPPPlus、MedQA；
- sequential 与 hierarchical 两种 agent 拓扑；
- baseline、TextMAS、identical、linear、kernel 与 soft，其中四种 latent alignment 均在两种拓扑下评价。

建议把同一张 Table 1 分成按模型堆叠的三个 panel，行保持为 12 个配置，列为各数据集，并在末尾增加 average rank 和 win/tie/loss。4B 未运行的 AIME/GPQA 单元格写 N/A，不能按 0 参与平均。由于不同任务的基准难度和指标尺度不同，原始 accuracy/pass@1 的宏平均不能作为唯一汇总指标；正文必须保留逐任务结果。

每个单元格报告跨 repetitions/seeds 的均值和不确定性。若当前产物只有 run-level aggregate，就报告 mean ± standard deviation/CI，并明确统计单位；只有保存了逐样本结果后才使用 paired bootstrap 或 paired significance。加粗和下划线只在同一 model--dataset、同等实验预算内标记最好与次好结果。

Table 1 证明的是 **broad end-to-end effectiveness**，不单独证明 kernel 忠实逼近 soft，也不单独证明 latent message 导致了协作收益。因此后续小节分别解释：结果是否来自近似保真、闭环是否稳定、消息是否真正携带可用信息，以及成本是否更低。output tokens 与 seconds/sample 更适合放在紧邻主表的 compact companion table 或 Pareto 图中，不要把 accuracy、latency、memory 和 tokens 全塞入 Table 1。

### 5.2 Kernel approximation: is it faithful and efficient?

这一部分只回答 kernel 是否是 soft 的可靠且更便宜的近似，不在这里讨论 Latent CoT 或 communication 本身是否优于 baseline。证据分为 static fidelity 和 closed-loop fidelity 两层。

#### 5.2.1 Static cost--fidelity trade-off

对应 S1、S2、E0，以及 S3 的核心结果。

正文回答：

- kernel 的 embedding error 多大；
- error 是否集中在高 entropy/低 confidence state；
- denominator/NaN failure 是否可忽略；
- 不同 $m$ 下 latency、显存和 error 如何变化；
- 包含离线预计算后，break-even query 数是多少。

主图建议：

1. relative-$L_2$ ECDF；
2. latency--error 或 latency--accuracy Pareto；
3. peak memory/throughput 紧凑表。

S3 的完整 seed--temperature grid 放附录，正文只保留一句最重要的 bias--variance 结论。

#### 5.2.2 Does static fidelity persist under closed-loop use?

对应 C2、C4。

从相同 prompt/KV 启动 soft 与 kernel，报告：

- feedback embedding divergence；
- hidden cosine；
- readout JS/KL；
- top-k agreement；
- 首次分叉 step；
- 最终答案 agreement；
- ORF seed sensitivity。

主图是 soft--kernel divergence vs. step，辅图是 answer agreement vs. $m$。这是整篇论文最重要的桥梁：它把静态 approximation method 与实际 recurrent computation 连接起来。

### 5.3 Latent CoT: when does recurrent latent reasoning help?

这一部分把 $F_{A\rightarrow A}$ 作为推理机制单独评价。核心不是证明 latent trajectory “看起来连续”，而是确定它在哪些任务、step budget 和 interface 下提高 accuracy，以及何时发生累积退化。

#### 5.3.1 Task effectiveness across latent-step budgets

对应 C1，C0/C3 作为解释。

主结果：

$$
K\in\{0,4,8,16,32,50,100\}
$$

下五种方法的 accuracy、parse success 和 failure rate。必须报告：

- accuracy--$K$；
- failure--$K$；
- kernel 相对 soft 的 paired degradation；
- soft/kernel 相对 text 的 paired difference；
- compute/position-matched text baseline。

#### 5.3.2 Where and why does Latent CoT help or fail?

Entropy、hidden norm、相邻 cosine、special-token mass 用来解释“为什么”，不能代替 accuracy。将这些诊断与 accuracy--$K$ 和 failure--$K$ 对齐，重点区分有效区间、收益饱和区间和退化区间。四数据集 entropy 图通常放附录；正文只在其揭示关键退化模式时选一个紧凑面板。

### 5.4 Latent Communication: do messages help receivers?

这一部分把 $F_{A\rightarrow B}$ 作为通信机制单独评价，证据按“message 是否携带信息”到“receiver 是否在真实任务中利用信息”递进。不能只凭主表中的协作 accuracy 推断因果通信收益。

#### 5.4.1 Do latent messages carry recoverable information?

对应 M1，是最干净的通信证据。

A 看到 `entity / attribute / value`，B 只看到 entity/attribute query。比较：

- no message；
- random/mismatched message；
- matched soft；
- matched kernel；
- matched linear；
- 可选 text reference。

正文报告 private-information accuracy、kernel vs. soft paired difference、NLL 和 ECE。Confusion matrix 可放附录。只有 matched message 显著优于 no-message/random/mismatch，才支持 message 携带信息。

#### 5.4.2 Do latent messages improve real-task collaboration?

对应 M2。A、B 都看到真实任务问题，评价 message 是否给 B 带来增量收益。

必须比较 B-alone、compute-matched B-alone、random/mismatched message、soft、kernel、linear 和 TextMAS。报告 paired accuracy improvement、总 latency、传递 embedding/token 数和失败率。

如果 M1 成功但 M2 不成功，应把结论写成：

> Latent messages carry recoverable information, but current receivers do not reliably exploit it for complex collaborative reasoning.

这是可信的边界结论，不应被隐藏在附录。

### 5.5 Generalization and boundary conditions

对应 G0，用一个紧凑 forest plot 或表格复现：

- 4B→4B 与 8B→8B；
- 4B→8B 与 8B→4B；
- Refiner→Judger 与 Planner→Critic；
- ARC-Challenge 与 GSM8K。

不需要重复所有诊断图；复用 RQ1 的 mapping error、RQ2 的 Latent CoT task metric、RQ3 的 private-information accuracy 即可。

## 6. 主文与附录取舍

### 6.1 主文必须保留

- 方法总览图；
- soft/kernel 公式和复杂度表；
- `run_all.sh` 的 Table 1 端到端主表；
- E0 latency/memory--quality Pareto；
- C1 accuracy--$K$；
- C2 divergence--step；
- M1 private-information recovery；
- M2 核心协作结果；
- 一个 G0 泛化表或 forest plot。

### 6.2 更适合附录

- S0 norm histograms；
- S2 全部分布指标；
- S3 完整 $m\times\tau\times$ seed grid；
- S4 PCA/t-SNE；
- C0 四数据集完整 entropy 图；
- C3 全部 language-lens 案例；
- M0 无题目压力测试的完整输出；
- M3 多跳扩展；
- prompts、Parquet schema、额外模型方向和超参数。

S4/PCA 不应占据正文主结果位置，否则容易给 reviewer 留下“以可视化替代行为证据”的印象。

## 7. Discussion、Limitations 与 Conclusion

### 7.1 Discussion 应回答的机制问题

- soft interface 为什么可能比 hidden identity/linear 更自然；
- kernel error 在什么 state 上被闭环放大；
- entropy 与 accuracy 是否相关，何时不相关；
- 信息可恢复为什么不必然带来复杂协作收益；
- $m$、$K$ 和 message bandwidth 之间的折中。

### 7.2 Limitations 必须主动写明

- 第一版要求 vocabulary 和 token IDs 对齐；
- soft 是 full-softmax reference，不是 latent semantics oracle；
- kernel 只能逼近 soft，不能修复 soft interface 自身的问题；
- 单个 expected embedding 的通信容量有限；
- 多步 recurrence 会放大小误差；
- 主要实验集中于一个模型族；
- AIME 2025 样本量小；
- 合成私有信息只证明可控信息传输，不等价于开放域协作。

### 7.3 Conclusion

控制在半页以内，按 RQ0--RQ3 逐句回答，不重复摘要和实验清单。

## 8. 结果出来后的叙事分支

### 情况 A：完整正向结果

```text
kernel ≈ soft
soft 有任务价值
kernel 保留任务价值
kernel 显著更高效
M1/M2 显示通信有效
```

此时可写成完整 method paper，中心是 efficient latent communication。

### 情况 B：Kernel 很准，但 soft/kernel 都不如 text

不要硬写成成功方法论文。更合适的中心结论是：

> Numerical fidelity is not sufficient for latent reasoning.

论文转为诊断性 empirical study：kernel 忠实复现 soft，但 soft expected embedding 本身不是可靠 CoT interface；静态误差、entropy 或 PCA 都不能预测任务表现。

### 情况 C：M1 成功、M2 不成功

限制主张为“latent message 携带可恢复的私有信息”，不要写成“改善 multi-agent reasoning”。正文应分析 receiver 为什么不能利用该信息。

### 情况 D：Kernel 没有速度优势或闭环迅速分叉

当前 method story 不成立。应优先修复 estimator、temperature/bias 定义或实现，而不是用更多 entropy/PCA 图补救。

## 9. 标题候选

### 正向 method paper

- **Kernelized Soft Latent Communication for Efficient Multi-Agent Reasoning**
- **Vocabulary-Mediated Latent Communication via Kernelized Soft Tokens**
- **Efficient Soft Latent Interfaces for Language-Model Agents**

第一项最直接，但只有在 E0、C1、C2、M1 至少成立时使用。

### 诊断性论文

- **When Numerical Fidelity Is Not Enough: Evaluating Soft Latent Interfaces for Language Models**
- **From Static Approximation to Closed-Loop Behavior in Latent Language-Model Communication**

## 10. 摘要模板

摘要严格用五个逻辑句群：

1. **Problem.** 不同语言模型的 hidden spaces 不兼容，文本通信又带来离散生成与长上下文成本。
2. **Principled interface.** 将 soft decoding 写成 key/value 解耦的 vocabulary attention：sender 提供 keys，receiver 提供 values。
3. **Method.** 用 ORF kernel 和 receiver-value preaggregation 消除在线 vocabulary-sized aggregation。
4. **Evidence.** 在 288 个端到端配置的主 benchmark 上评价 task performance，并进一步评价 static fidelity、closed-loop divergence、private-information recovery 和端到端效率。
5. **Finding.** 用具体数字写保留多少行为、获得多少 speed/memory improvement、在哪些条件下失败。

不要在摘要中罗列 entropy、PCA、top-k overlap、所有 seed 和所有数据集；这些是支持证据，不是贡献。

## 11. ICLR 页面与写作规划

ICLR reviewer 会明确判断：问题是否具体、方法是否有充分动机、证据是否支持 claim、工作是否提供足够的新知识与价值。写作和实验应直接对应这些问题：

- [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)

ICLR 2026 的 submission/camera-ready page limit 是 10 页；投稿目标年份仍须以当届官方指南为准：

- [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide)

建议按 9.5--10 页规划：

| 部分 | 建议页数 |
| --- | ---: |
| Introduction | 1.25 |
| Problem + Related Work | 1.25 |
| Method | 2.0 |
| Setup | 0.75 |
| Results | 3.5--4.0 |
| Discussion + Limitations + Conclusion | 0.75--1.0 |

准备简洁的 Reproducibility Statement，指向匿名代码、配置、数据处理、seed、失败记录和附录。官方 Author Guide 也明确鼓励说明复现材料：

- [ICLR Author Guide: Reproducibility](https://iclr.cc/Conferences/2025/AuthorGuide)

## 12. 写作检查清单

提交前逐项确认：

- 标题和摘要只有一个中心 claim；
- Introduction 在第一页明确 representation mismatch；
- Introduction 第一段定义 intermediate-state protocol，并比较 text、vocabulary-distribution 与 hidden-state 三类选择；
- `logits` 仅作为 logit-mediated 的便捷称呼，正式表述使用 vocabulary distribution/expected embedding；
- 明确 protocol 包含 encode、transport、inject，而 $F_{A\rightarrow B}$ 是其中的 alignment interface；
- Introduction 和 Problem Formulation 分别定义 Latent CoT 与 Latent Communication；
- Latent CoT 写成 $F_{A\rightarrow A}$ 的时间递推，Latent Communication 写成 $F_{A\rightarrow B}$ 的 agent-edge transfer；
- 不用“latent communication”指代模型内部 recurrence，也不用“Latent CoT”指代跨 agent message；
- 将 decoding--attention correspondence 写成方法动机而非首创 claim；
- 明确区分已有同模型 vocabulary attention 与本文 sender-key/receiver-value 的跨模型接口；
- `soft` 始终指 full-softmax expected embedding；
- 不再把 alignment 方法称为 `exact`；
- kernel 和 soft 的 temperature、bias、normalization 定义一致且可审计；
- 不声称 ratio estimator 无偏，除非有严格证明；
- 每个 Results 小节明确回答一个 RQ；
- Table 1 明确列出 3 个模型、数据集覆盖、12 个配置、重复次数和汇总规则，缺失项记为 N/A；
- 不用跨异质任务的 raw macro-average 代替逐数据集结果，至少补充 average rank 或 win/tie/loss；
- accuracy、通信和效率是主要证据；
- entropy/PCA/language-lens 只用于诊断；
- 所有方法差异有 paired statistics 和失败率；
- M1 排除数据泄漏，M2 有 compute-matched baseline；
- 跨模型实验明确 vocabulary compatibility 条件；
- 负结果在正文说明，不隐藏到附录；
- Reproducibility Statement 能定位代码、配置、seed 和原始指标。

最重要的写作原则：

> 不要把论文写成“我们做了很多分析”，而要写成“我们提出一个清晰的方法，并逐层排除了从数值近似到实际通信之间的替代解释”。
