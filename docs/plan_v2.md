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

静态算子/通信 probe 只输入 question、query 或代码 prompt，不输入答案或 CoT。prompt 上限 512 token，超长保留前 480 与后 32；每题等距取最多 8 个非-special position，含最后一个位置。`probe_seed=20260723` 确定样本和位置。

| 用途 | 数据 | 题目数 | 最大 state 数 |
| --- | --- | ---: | ---: |
| 算子 calibration | ARC-Easy train | 512 | 4,096 |
| 通用 test | ARC-Easy test | 512 | 4,096 |
| 难度 test | ARC-Challenge test | 512 | 4,096 |
| 数学 OOD | GSM8K test | 512 | 4,096 |
| 专业 OOD | MedQA | 512 | 4,096 |
| 代码 OOD | MBPP+ test | 至多 512 | 4,096 |
| 小样本 OOD | GPQA Diamond test | 全部 | 至多 1,584 |

ARC-Easy train 由分析脚本直接读取，因为当前 loader 固定 test。静态实验以 prompt 为统计单位，先平均其 state 再汇总。所有主指标报告 mean、median、p90/p95/p99 和 cluster bootstrap 1,000 次 95% CI。

CoT 数据为 GSM8K train 128（calibration）、GSM8K test 512、ARC-Challenge test 512、AIME 2024/2025 全部；预算分别为 1024、1024、512、4096 token。CoT 的统计单位是一条完整题目 rollout。通信的统计单位是一条 message 或协作题目。

## 3. 第一层：算子近似

### S0. 范数和困难区域

**运行**：X1、X2，所有静态数据。统计 B input embedding 每个 token 的 $\|(W_{\mathrm{in}}^B)_{:,i}\|_2$、A output key 的 $\|(W_{\mathrm{out}}^A)_{i,:}\|_2$，以及状态的 $\|h_L^A\|_2$、$\|q\|_2$、$\|w_i+q/\tau\|_2$。

输出 histogram、ECDF、p1/p5/p25/p50/p75/p95/p99 和按 entropy、置信度、prompt length 分桶的误差热图。

**意义**：解释低温、高范数、尖锐分布为何更难近似；不将此图解释为推理质量。

### S1. exact-$F$ 静态保真和性能

**运行**：X1 在全部 test/OOD；X2 在 ARC-Easy 与 GSM8K。ARC 两集跑 5 个主 ORF seed，其他集跑前 3 个。

完整 vocab 计算 $F(q)$，报告

$$
\frac{\|\hat F(q)-F(q)\|_2}{\|F(q)\|_2+10^{-8}},
\qquad \cos(\hat F(q),F(q)),
$$

分母异常率、NaN/Inf 率，以及 batch size 1/32 的 p50/p95 latency、tokens/s、峰值显存。性能仅测在线映射：warm-up 200 次，CUDA 同步后测 1,000 次。

**意义**：证明 kernel 是否以可接受误差替代完整词表扫描；只是方法必要条件。

### S2. softmax 误差传递与 ORF/iid 消融

**运行**：X1，ARC-Easy/ARC-Challenge/GSM8K。显式计算 $p$ 与 $\hat p$，报告 KL、JS、TV、top-1/top-10/top-100 overlap、exact top-k mass。画单核误差、$\|p-\hat p\|_1$、$\|F-\hat F\|_2$ 的散点图。

仅 ARC-Easy train calibration 运行：block ORF、iid Gaussian RF、可选重复方向负对照，

$$
m\in\{256,512,1024,2048,4096\},
\quad \tau\in\{0.7,1.0,1.3\},
\quad \mathrm{seed}\in\{101,202,303,404,505\}.
$$

**意义**：验证误差是否按“核 $\to$ softmax $\to$ embedding”传播，并检验 ORF 是否比 iid 更稳定。calibration 只作附表，不能修改主设置。

### S3. 固定文本的 variance--$\tau$ 与 std--$\tau$

**运行**：X1、ARC-Easy train。固定 128 条 prompt 的最后 state；每个 state 固定取 exact rank 1、稳定 hash 选出的 rank 2--10、rank 100--1000 key，共 384 个 $(w_i,q_j)$。GSM8K test 128 条在 $m=2048$ 复现。

对 $m\in\{512,1024,2048\}$、$\tau\in\{0.5,0.6,\ldots,2.0\}$，仅改变 $R=32$ 个 ORF 矩阵。计算

$$
\hat k_{ij}^{(r)}=\phi_{\Omega_r}(w_i)^\top\phi_{\Omega_r}(q_j/\tau),
$$

$$
s^2_{k,ij}=\frac{1}{R-1}\sum_r(\hat k_{ij}^{(r)}-\overline k_{ij})^2,
\qquad
s^2_{F,j}=\frac{1}{d_B(R-1)}\sum_r\|\hat F_j^{(r)}-\overline F_j\|_2^2.
$$

画 kernel 与 $\hat F$ 的原始、相对 variance/std--$\tau$ 曲线，原始方差用 log y 轴；按 rank 分面，阴影为 pair/state bootstrap 1,000 次 CI。补充

$$
\operatorname{MSE}_j=
\frac{\|\overline F_j-F(q_j)\|_2^2}{d_B}+s^2_{F,j}.
$$

**意义**：严格度量“固定文本，只换 ORF”造成的随机不确定性，区分低方差和低偏差。

### S4. 通信空间 PCA

**运行**：仅 X1；ARC-Easy test 的 4,096 states，主设置和 seed 101；GSM8K 图置附录。将 $Y_F,Y_{\mathrm{identical}},Y_{\mathrm{linear}},Y_{\mathrm{kernel}}$ 拼接后只拟合一次全局中心化 PCA，四面板共享坐标。颜色为 exact entropy 四分位；另画灰色 exact 点与三方法叠加图、200 条配对箭头。

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

建议新增只读入口 `analysis/run_plan_v2.py`，参数为 `--layer {operator,cot,communication}`、`--model_pair`、`--dataset`、`--method`、`--orf_seed`，输出：

- `artifacts/plan_v2/manifests/*.json`：revision、样本、token position、参数、seed；
- `artifacts/plan_v2/metrics/*.parquet`：prompt/message/trajectory 原始指标；
- `artifacts/plan_v2/readouts/*.jsonl`：top-k 与案例；
- `artifacts/plan_v2/figures/*.pdf`：预注册图。

停止规则：

1. tokenizer 不一致、float64 审计失败、分母非有限/非正时停止该组合；
2. 显存不足时减少 state batch，不减少 vocab、不用 top-k 代替 exact $F$；
3. 私有信息数据或其 split 不可复现时，不报告语义通信结论；
4. 同模型 rollout 接口无法与仓库实现严格对应时，只报告算子和通信实验，不伪造 CoT 结论。

## 7. 展望：训练型方法

当前比较均为零额外训练。未来可加入 ThoughtComm、C2C 等 training-required 方法，但必须统一模型配对、tokenizer、任务 split、消息预算和生成预算；单列训练数据、监督信息、参数量、训练步数、GPU 时间、峰值显存、在线延迟和额外存储。

应分别绘制零训练与训练后方法的 Pareto 图，比较训练成本、推理成本、算子保真、信息恢复、CoT 稳定性和最终任务收益。若方法使用额外配对数据、教师输出或标签，必须将其列为额外信息预算，而不能将收益单独归因于对齐算法。

