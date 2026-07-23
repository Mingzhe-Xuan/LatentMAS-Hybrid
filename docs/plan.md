# 核对齐的分析性实验计划（暂设 $T_{A\to B}=I$）

## 1. 目标与问题拆分

本计划暂不研究跨词表迁移。A、B 使用完全相同的 token-to-ID 词表，因而

$$
T_{A\to B}=I,
\qquad
c_i=(W_{\mathrm{in}}^B)_{:,i}.
$$

需要区分两个不同问题：

1. **soft-token 对齐目标是否合理**：
   $$
   F(q)=W_{\mathrm{in}}^B\operatorname{softmax}
   (W_{\mathrm{out}}^Aq/\tau+b^A)
   $$
   是否能把 A 的 hidden state 表达为 B 可接收的连续输入 embedding；
2. **核近似是否准确且值得**：ORF 预聚合得到的
   $$
   \hat F(q)=\frac{S\phi_{\mathrm{orth}}(q/\tau)}
   {z^\top\phi_{\mathrm{orth}}(q/\tau)}
   $$
   是否足够接近 exact $F(q)$，并以可接受的代价换取在线加速。

最终 reasoning 正确率不足以区分这两类误差。因此分析实验必须保留 exact $F$ 作为 oracle：若 exact $F$ 的下游行为已差，问题在 soft-token 对齐假设；若 exact $F$ 好而 $\hat F$ 差，问题在 ORF 特征数、温度或数值实现。

## 2. Probe 语料及数据划分

分析实验只使用题目文本（`question`、`query` 或代码 prompt），**不输入** `solution`、gold answer 或模型生成的 CoT。将每段文本送入 A，并在有效 token 位置收集

$$
q=\operatorname{Norm}_A(h_L^A).
$$

这不需要额外标签；语料的作用是覆盖模型在自然输入上实际遇到的 $q$ 分布。

| 划分 | 数据集 | 用途 |
| --- | --- | --- |
| calibration | ARC-Easy `train` 的 512 条固定子集 | 只用于选择 $\tau$ 和特征数 $m$；见第 3.2 节。 |
| in-domain test | ARC-Easy `test` 及 ARC-Challenge `test` 的各 512 条固定子集 | 报告通用自然语言下的主结果；见第 3.2 节。 |
| OOD: 数学 | `gsm8k` | 数字、运算符和步骤性表达；若 GSM8K 是主任务结果，绝不用于调参。 |
| OOD: 专业问答 | `gpqa`、`medqa` | 测试科学/医学术语与知识密集输入。 |
| OOD: 代码 | `mbppplus`、`humanevalplus` | 测试代码、标点和 API token 密集的分布。 |
| 极端数学 probe | `aime2024`、`aime2025` | 样本少，仅作补充压力测试，不作为主统计集。 |

固定 `probe_seed=20260723`；保存每个 split 的样本 ID、token 位置及模型/分词器版本。完整的预算和抽样方式见第 3.2 节。各 test 集从不参与 $m$、$\tau$、seed 或其他阈值的选择。

## 3. 共同设置

- A、B 选同一 tokenizer 的 checkpoint。先做 A=B 的自一致性实验，再做同族、不同规模或不同 checkpoint 的 A $\to$ B 实验。
- exact oracle 对每个采样位置显式扫描 A 的完整词表，计算 logits、$p=\operatorname{softmax}(\cdot)$ 及 $F(q)$。这一步仅用于离线诊断，不计入线上推理成本。
- kernel 端使用仓库实现的 block ORF，固定离线和在线相同的 `omega`/seed；报告多个独立 ORF seed 的均值与标准差。
- 所有核统计量均包含输出 bias；online feature 使用共同 log-scale 稳定化，exact 与 kernel 均以 float32/float64 的一致设置计算。
- 主报告使用 token 位置为统计单位，同时采用“每样本先平均、再跨样本平均”的汇总，避免超长 prompt 主导结果。

### 3.1 固定模型矩阵

主分析统一采用 Qwen2.5 系列，并以 Hugging Face `eval()` 模式加载：`Qwen/Qwen2.5-1.5B`、`Qwen/Qwen2.5-1.5B-Instruct`、`Qwen/Qwen2.5-7B-Instruct`。这些组合的 tokenizer 必须通过逐 token ID 和 special-token 配置的一致性检查后才能运行。

| ID | A（source） | B（target） | 用途 | `identical` |
| --- | --- | --- | --- | --- |
| M0 | `Qwen/Qwen2.5-1.5B-Instruct` | 同一 checkpoint | 自一致性和实现 sanity check | 运行 |
| M1 | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | 主跨 checkpoint 对齐；维度相同，四组可视化齐全 | 运行 |
| M2 | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | 跨尺度压力测试 | 不运行，因 $d_A\ne d_B$ |

M1 是论文主图、PCA/t-SNE 图和 `identical`/`linear`/`kernel` 三方法比较的唯一主模型配对。M0 用于确认实现不会在同一模型上引入不必要偏差；M2 只报告 exact $F$、`linear`、`kernel`，用于检验结论是否能迁移到不同 hidden dimension。任何 tokenizer 检查失败的组合直接排除，不能以 $T=I$ 继续实验。

### 3.2 固定语料、prompt 数与 probe state 预算

所有题目只取 loader 输出中的 `question`/`query`/代码 prompt；不拼接答案、选择正确项、参考解答或系统生成内容。ARC-Easy `train` 必须由分析脚本直接调用 `load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")` 读取，因为当前 `data.py` 的 ARC loader 固定使用 `test`；同时记录数据集 revision。每条 prompt 按模型 tokenizer 截断到 512 token；若超过上限，保留前 480 与后 32 token。对每条 prompt，在有效、非 special 的 token 位置等距抽取最多 8 个 state，并始终包含最后一个有效位置。使用固定 `probe_seed=20260723` 打乱样本 ID 后截取。

| 阶段 | 数据 | 题目数 | 最大 probe states | 目的 |
| --- | --- | ---: | ---: | --- |
| calibration | ARC-Easy `train` | 512 | 4,096 | 只选择主设置，不报告 test 结论。 |
| 主 test | ARC-Easy `test` | 512 | 4,096 | 通用自然语言主结果。 |
| 难度 test | ARC-Challenge `test` | 512 | 4,096 | 同格式、更难问题。 |
| OOD 数学 | GSM8K `test` | 512 | 4,096 | 数字与符号密集输入。 |
| OOD 专业 | MedQA | 512 | 4,096 | 医学问答。 |
| OOD 代码 | MBPP+ `test` | 全部，至多 512 | 4,096 | Python/标点/tokenizer 压力。 |
| 小样本补充 | GPQA Diamond `test` | 全部 | 至多 1,584 | 高难科学问题。 |
| 极端补充 | AIME 2024 + 2025 | 全部 | 至多 480 | 仅报告，不用于选择设置。 |

同一个数据集、模型配对和 probe seed 下，exact $F$、`identical`、`linear`、每个 ORF seed 必须使用完全相同的 probe state 索引。主表对每条 prompt 先在其被抽到的 states 上平均，再对 prompt 平均，并以 bootstrap $1{,}000$ 次给出 95% CI。

### 3.3 固定超参数、消融范围与选择规则

主设置固定为：`align_method=kernel`、`kernel_features=2048`、`kernel_temperature=1.0`、`kernel_chunk_size=4096`、`kernel_seed=101`。对齐统计量与 ORF 均在 float32 构建；模型前向使用 bfloat16；exact reference 使用 float32 的稳定 log-sum-exp。每个模型组合额外随机抽取 256 个 states，用 float64 exact reference 复算；若 float32/float64 的 relative-$L_2$ 差异 p99 超过 $10^{-4}$，该组合停止并先修正数值实现。

唯一允许的 calibration 消融为：

$$
m\in\{256,512,1024,2048,4096\},
\qquad
\tau\in\{0.7,1.0,1.3\},
\qquad
\mathrm{ORF\ seed}\in\{101,202,303,404,505\}.
$$

主结果仍使用预先固定的 $m=2048,\tau=1.0$，不因 test/OOD 结果再调参。消融结果仅在 ARC-Easy `train` 的 4,096 个 calibration states 上运行，选择规则是：在分母失败率为零的候选中，选取使 median relative-$L_2$ 最小的配置；如与最优值相差不超过 1%，取更小的 $m$。该规则确定一个“calibrated setting”，可在附表报告，但不能替代预注册的主设置。

`linear` 固定使用 `align_ridge=1e-5`；另只在 calibration 上做 $\lambda\in\{10^{-6},10^{-5},10^{-4},10^{-3}\}$ 敏感性表，不以其结果修改主基线。`identical`、`linear` 与 `kernel` 都沿用仓库当前的目标 embedding 平均范数缩放行为；exact $F$ 不额外作范数缩放。

### 3.4 精确参考、B 端行为与性能测量协议

对每个 $q$，exact reference 必须按完整词表计算：先以稳定 log-sum-exp 得到 $p$，再计算 $F(q)=W_{\mathrm{in}}^Bp$。不得用 top-k softmax、采样 softmax 或截断词表替代 exact oracle。

B 端局部行为使用同一 token prefix：若 source state 来自第 $t$ 个位置，则将 B 的前 $t$ 个相同 token 作为 context，在其后追加一个方法输出的连续 embedding，并比较该位置后 B 的 next-token logits。exact $F$ 是 kernel、linear、identical 的共同行为参照；每个方法均在相同 B context、attention mask 和 position ID 下测量。

性能只测在线 `apply_alignment`，不把构建 $S,z$ 的离线开销混入。对每个模型配对在 CUDA 上分别以 batch size 1 和 32 测量 200 次 warm-up 后的 1,000 次调用；每次前后 `torch.cuda.synchronize()`，报告 p50/p95 延迟、tokens/s、`torch.cuda.max_memory_allocated()`。exact $F$ 作为离线诊断基线同样测量，但单列呈现，不声称其可线上部署。

### 3.5 各实验的固定执行矩阵

| 实验 | 模型配对 | 数据与 probe states | 固定设置 |
| --- | --- | --- | --- |
| 实验 0（范数） | M0、M1、M2 | 第 3.2 节所有 calibration/test/OOD 集 | 仅前向；记录全部预定 states。 |
| PCA 主图 | M1 | ARC-Easy `test` 的 4,096 states | $m=2048,\tau=1.0$，kernel seed 101；四面板和重叠图均由此产生。 |
| PCA 复现图 | M1 | GSM8K `test` 的 4,096 states | 同主图设置；作为 OOD 图置附录。 |
| 实验 A/B | M1 | ARC-Easy、ARC-Challenge、GSM8K、MedQA、MBPP+、GPQA 的全部第 3.2 节 states | 主设置；ARC 两集 5 seed，OOD 3 seed。 |
| 实验 C（ORF） | M1 | ARC-Easy `train` 的 4,096 calibration states；其中固定 128 条文本用于方差曲线 | 网格和 iid RF 对照；variance--$\tau$/std--$\tau$ 使用 32 个 ORF seed。 |
| 实验 D（B logits） | M1 | ARC-Challenge、GSM8K、MBPP+ 各 4,096 states | 主设置，比较 exact、identical、linear、kernel。 |
| 实验 E（闭环） | M1 | ARC-Challenge 与 GSM8K 各 128 条固定 prompt | 主设置、greedy decoding、$K\in\{1,2,4,8,16\}$；保存 B 末层 unembedding 读出。 |
| 实验 F（模型变化） | M0、M1、M2 | ARC-Easy `test` 与 GSM8K `test` 各 4,096 states | 主设置；M2 不含 `identical`。 |
| 性能 | M1、M2 | 合成的 32 个真实 ARC-Easy states 循环调用 | batch size 1、32；exact、linear、kernel 分开测量。 |

该表之外不扩展组合或数据集，除非先修改计划并重新声明为探索性实验。

## 4. 实验 0：向量范数分布与尺度诊断

该实验为后续所有保真结果提供尺度背景。特别地，`algo_detail.md` 的误差分析依赖 $\|w_i+q/\tau\|_2$：高范数 key、hidden state 或低温度均可能增大随机特征误差。因此，不应只报告平均范数。

### 4.1 embedding / 输出头的逐 token 范数

对全部词表 token 计算并保存：

$$
r^{\mathrm{in},B}_i=\|(W_{\mathrm{in}}^B)_{:,i}\|_2,
\qquad
r^{\mathrm{out},A}_i=\|(W_{\mathrm{out}}^A)_{i,:}\|_2.
$$

其中 $r^{\mathrm{in},B}$ 是用户关心的 B 输入 embedding layer 中每个 token vector 的范数；$r^{\mathrm{out},A}$ 是核函数的 key 范数，虽不等同于 embedding layer，但直接参与近似难度，必须一并报告。对 A=B 时仍分别报告 input embedding 与 output head（它们未必 weight tying）。

报告并绘制：

- min、max、mean、std、median、p1/p5/p25/p75/p95/p99；
- 直方图与 ECDF（同一图中显示 A output head 和 B input embedding）；
- token frequency 分桶下的范数分布，以及高范数 token 的字符串/类别抽样（special token、数字、代码片段、普通文本）；
- 若比较多个 checkpoint，使用相同横轴范围绘制各模型的分布。

### 4.2 语料 hidden-state 范数

在第 2 节规定的每个 calibration/test/OOD split 上，对每一个有效输入位置记录原始末层状态及其核输入：

$$
r_h=\|h_L^A\|_2,
\qquad
r_q=\|q\|_2,
\quad q=\operatorname{Norm}_A(h_L^A).
$$

两者都需要保存：实际实现进入核函数的是 $q/\tau$，而 $r_h$ 可揭示模型末层表示本身的尺度与 LayerNorm 的影响。对每个数据域报告与逐 token embedding 相同的分位数统计、直方图和 ECDF，并按 token 位置、prompt 长度、exact softmax entropy、$\max_i p_i$ 分桶。

额外报告 $r_q/\tau$ 的分布，并对每个 probe 的 exact top-k key 记录

$$
\left\|w_i+q/\tau\right\|_2.
$$

将该量与实验 A 的 relative-$L_2$、实验 B 的 TV/KL 作散点图或按分位数分桶作图。这将实证检验范数、温度与核近似误差之间的关系，而不是把范数分布仅作为描述性统计。
## 5. 表示空间可视化：exact $F$ 与三种对齐方法

在同一批从 probe 语料抽取的 hidden states $\{h_j\}_{j=1}^N$ 上，构造四组处于 B 输入 embedding 空间的向量：

$$
Y_F=\{F(q_j)\}_{j=1}^N,
\quad
Y_{\mathrm{identical}}=\{g_{\mathrm{identical}}(h_j)\}_{j=1}^N,
\quad
Y_{\mathrm{linear}}=\{g_{\mathrm{linear}}(h_j)\}_{j=1}^N,
\quad
Y_{\mathrm{kernel}}=\{\hat F(q_j)\}_{j=1}^N.
$$

其中 $q_j=\operatorname{Norm}_A(h_j)$，$F(q_j)$ 是全词表 exact oracle，且四组的第 $j$ 个点必须来自同一个 source hidden state。`identical` 仅适用于 $d_A=d_B$；若维度不同，则只在 A=B 或维度相同的模型组合中绘制它，并在图注明确说明。

### 5.1 主图：联合 PCA 的四面板对比

主图固定使用 M1、ARC-Easy `test` 的 4,096 个 states、$m=2048$、$\tau=1.0$、kernel seed 101；图中不混入 calibration 或其他语料。首选 PCA，而非独立运行 t-SNE。将四组原始 B-space 向量按行拼接后，只做一次全局中心化并仅拟合一次二维 PCA；不按维度标准化，以保留对齐方法造成的真实尺度差异：

$$
P=\operatorname{PCA}_2\left(
[\,Y_F;Y_{\mathrm{identical}};Y_{\mathrm{linear}};Y_{\mathrm{kernel}}\,]
\right).
$$

将同一 $P$ 投影分别画入一个 $2\times2$ 面板：`exact F`、`identical`、`linear`、`kernel`。四个面板必须共享完全相同的 x/y 坐标范围、相同的 4,096 个点和相同的颜色语义。颜色固定为 exact softmax entropy 的四分位桶，以便观察高置信度与低置信度状态在不同方法中的整体形状和簇结构是否保持。

另输出一张重叠图：exact $F$ 用半透明灰色圆点作参考，三种方法分别用蓝/橙/绿的半透明点叠加在同一 PCA 坐标系。该图应使用随机分层抽取的 $N$ 个点，避免点过密；主图可取 $N=2{,}000$--$10{,}000$，并固定抽样 seed。

### 5.2 配对偏移图与定量补充

仅看投影散点可能掩盖高维误差。对每个方法 $M$，在 PCA 平面叠加从 $P(F(q_j))$ 指向 $P(g_M(h_j))$ 的箭头；为避免遮挡，只随机显示 100--300 条箭头。同步报告高维量：

$$
\frac{1}{N}\sum_j
\frac{\|g_M(h_j)-F(q_j)\|_2}{\|F(q_j)\|_2+10^{-8}},
\qquad
\frac{1}{N}\sum_j\cos\left(g_M(h_j),F(q_j)\right).
$$

因此图中的近邻/簇结构和数值保真指标可相互校验。预期 `kernel` 的点云和配对箭头最接近 exact $F$；`linear`、`identical` 是否接近则是需要由实验检验的基线结果，不能预设。

### 5.3 可选 t-SNE 图

PCA 是唯一主图。如果需要补充 t-SNE，仅在 M1 的 ARC-Easy `test` 图上运行：从 4,096 点按 entropy 四分位分层抽取 2,000 点，将四组拼接矩阵一次性送入同一个 t-SNE，固定 `init=pca`、`perplexity=50`、`learning_rate=auto`、`n_iter=1500`、`random_state=101`；再按四组切回四个共享坐标轴面板。严禁分别拟合。t-SNE 仅用于局部邻域的辅助可视化，不应用于比较全局距离、尺度或作定量结论。
## 6. 实验 A：exact--kernel 映射保真度（核心）

对每个 probe $q$，计算 $F(q)$ 与 $\hat F(q)$。仅在 ARC-Easy `train` calibration states 上执行第 3.3 节规定的 $m$、$\tau$、ORF-seed 网格；所有 test/OOD 主结果固定为 $m=2048,\tau=1.0$。ARC-Easy `test` 与 ARC-Challenge `test` 以五个预定 ORF seed 重复，OOD 集以 seed 101 为主，并额外以 seed 202、303 复核。

报告以下指标的均值、中位数、p90、p99 和最坏 1%：

- 相对 embedding 误差 $\|\hat F-F\|_2/(\|F\|_2+10^{-8})$；
- embedding cosine $\cos(\hat F,F)$；
- 分母 $z^\top\phi(q/\tau)$ 的最小值、非有限值比例及异常率；
- 运行一次 exact 和一次 kernel 映射的延迟、峰值显存。

在 calibration 集绘制“误差--$m$”和“误差--$\tau$”曲线；在 ARC-Easy/ARC-Challenge test 绘制五个 seed 的误差分布；在所有 test/OOD 集绘制固定主设置的“延迟--误差”Pareto 图。理想现象是：随着 $m$ 增大，误差下降且 seed 方差收敛；当 $m\ll V$ 时，kernel 在线成本显著低于 exact。

## 7. 实验 B：softmax 分布保真及理论链

诊断时显式构造

$$
\hat p_i(q)=
\frac{e^{b_i^A}\,\phi(w_i)^\top\phi(q/\tau)}
{z^\top\phi(q/\tau)}.
$$

对 $p$ 与 $\hat p$ 报告：

- KL $(p\|\hat p)$、JS divergence 和 total variation $\tfrac12\|p-\hat p\|_1$；
- top-1 一致率、top-10/top-100 overlap；
- 概率质量在 exact top-k token 上的保留比例。

在同一批点上记录单核相对误差、$\|p-\hat p\|_1$、$\|F-\hat F\|_2$，绘制散点图及相关系数。该实验检验 `algo_detail.md` 第 12 节中的误差传递链，而不是仅观察到 embedding 上的偶然抵消。

## 8. 实验 C：ORF 的必要性和超参数敏感性

固定 $m,\tau$ 后，比较：

1. block ORF（当前方法）；
2. iid Gaussian positive random features；
3. 若可行，重复方向或随机置换方向的负对照。

每种方法运行不少于 5 个随机 seed，并报告实验 A/B 全部误差的均值和标准差。重点检验 ORF 是否在相同 $m$ 下获得更低误差或更小 seed 方差；避免作“对所有输入严格更优”的过强结论。

进一步按下列变量分桶，报告每桶误差与分母异常率：

- $\|q\|_2$ 分位数；
- exact softmax entropy 分位数；
- $\max_i p_i$ 分位数；
- $\|w_i+q/\tau\|_2$（可对 exact top-k token 统计）分位数；
- token 位置和 prompt 长度。

这将直接检验文档的预期：低 $\tau$、高范数及尖锐分布是随机特征近似的困难区域。

### 8.1 固定文本下的 ORF 条件方差：variance--$\tau$ 与 std--$\tau$

本节的随机性**仅**来自 ORF 矩阵 $\Omega$。文本、模型权重、token 位置、$q$、key $w_i$、bias、$m$ 和 $\tau$ 在每个条件内均固定；不要将不同 prompt 或不同 token 的差异混入“seed 方差”。

主实验固定为 M1，在 ARC-Easy `train` calibration 集中按 `probe_seed=20260723` 选取前 128 条 prompt，并只取每条 prompt 的最后一个有效 token state $q_j$。对每个 $q_j$，按 exact softmax rank 固定选择三个 key：rank 1、由稳定 hash 选出的 rank 2--10、由稳定 hash 选出的 rank 100--1000。因此共有 384 个固定 $(w_i,q_j)$ 核对；hash 规则和实际 token ID 必须保存。OOD 复现使用 GSM8K `test` 的 128 条 prompt、相同的选 key 规则。

对每个 $m\in\{512,1024,2048\}$、$\tau\in\{0.5,0.6,\ldots,2.0\}$，生成 $R=32$ 个独立 block-ORF 矩阵，seed 固定为 $1001,1002,\ldots,1032$。主 variance--$\tau$ 曲线使用 ARC-Easy；GSM8K 仅对 $m=2048$ 完整复现。iid Gaussian RF 使用同一 $R$、相同 $m$、相同文本和相同 $\tau$，作为方差基线。

对固定核对 $(i,j)$，第 $r$ 个 ORF 的核估计记为

$$
\hat k_{ij}^{(r)}(\tau)=
\phi_{\Omega_r}(w_i)^\top
\phi_{\Omega_r}(q_j/\tau).
$$

在 $R$ 个 seed 上计算无偏样本方差和标准差：

$$
s^2_{k,ij}(\tau)=
\frac{1}{R-1}\sum_{r=1}^{R}
\left(\hat k_{ij}^{(r)}(\tau)-\overline{k}_{ij}(\tau)\right)^2,
\qquad
s_{k,ij}(\tau)=\sqrt{s^2_{k,ij}(\tau)}.
$$

核图必须输出两组曲线：

1. 原始 $\operatorname{median}_{i,j}(s_k^2)$ 和 $\operatorname{median}_{i,j}(s_k)$ 对 $\tau$ 的曲线，y 轴为 log scale；
2. 相对方差和相对标准差曲线，即将上式分别除以 $k_{ij}(\tau)^2$ 与 $k_{ij}(\tau)$，其中 $k_{ij}(\tau)=\exp(w_i^\top q_j/\tau)$。

每条线以 384 个固定核对为统计单位，阴影为对这些核对 bootstrap $1{,}000$ 次的 95% CI；同时按 rank-1、rank-2--10、rank-100--1000 分三面板，防止 top token 主导结论。

对固定文本 state $q_j$，第 $r$ 个 ORF 输出为 $\hat F_j^{(r)}(\tau)$。将向量方差定义为协方差矩阵的 trace 除以目标维度：

$$
s^2_{F,j}(\tau)=
\frac{1}{d_B(R-1)}\sum_{r=1}^{R}
\left\|\hat F_j^{(r)}(\tau)-\overline F_j(\tau)\right\|_2^2,
\qquad
s_{F,j}(\tau)=\sqrt{s^2_{F,j}(\tau)}.
$$

输出对应的 $\operatorname{median}_j(s_F^2)$--$\tau$ 与 $\operatorname{median}_j(s_F)$--$\tau$ 原始/相对曲线；相对版本分别除以 $\|F(q_j)\|_2^2/d_B$ 和 $\|F(q_j)\|_2/\sqrt{d_B}$。阴影为 128 个固定文本 state 的 bootstrap 95% CI。为解释“低方差是否也准确”，同图或相邻面板加入以下 bias--variance 分解：

$$
\operatorname{MSE}_j(\tau)=
\frac{\|\overline F_j(\tau)-F(q_j)\|_2^2}{d_B}
+s^2_{F,j}(\tau).
$$

主图中 $m=512,1024,2048$ 各为一条线；图注必须写明 $R=32$、固定文本/keys、seed 范围、方差定义及 y 轴是否为 log scale。不得将跨文本的总方差标为 ORF 方差。

### 8.2 统计图表与检验的补充建议

除上述 variance--$\tau$、std--$\tau$ 曲线外，主结果建议额外提供：

- **误差 ECDF 与分位数表**：对 prompt-level relative-$L_2$、TV、B-logits KL 画 ECDF，并报告 median、p90、p95、p99；比单一均值更能显示长尾失效。
- **配对效应 forest plot**：以每条 prompt 的方法差值 $\Delta_M=\mathrm{error}(M)-\mathrm{error}(\mathrm{kernel})$ 为单位，给出 `identical`、`linear`、iid RF 相对 kernel 的 mean difference 和 bootstrap 95% CI。不同 token state 不应被当作独立样本。
- **bias--variance--MSE 堆叠图**：沿 $\tau$ 或 $m$ 展示第 8.1 节的 squared bias、ORF variance、MSE，避免只因方差下降而误判近似更优。
- **误差条件热图**：横轴为 exact entropy 四分位，纵轴为 $\|q\|_2$ 或 $\|w_i+q/\tau\|_2$ 四分位，单元格为 median relative-$L_2$；同时标注样本数，揭示困难区域。
- **seed 稳定性雨云图**：在固定 $m=2048,\tau=1.0$ 下，画每个 ORF seed 的 prompt-level error 分布及 seed 均值，区分“少数文本难”与“某个随机特征实例异常”。
- **exact $F$ 覆盖率图**：对每个 $q$ 的 32 个 ORF 输出形成经验 2.5%--97.5% 区间，报告 exact $F(q)$ 的每维覆盖率和向量投影覆盖率；这是对近似中心与随机波动的校准检查，不假设正态性。
- **多重比较控制**：若在多个数据集、多个 $m$、多个 $\tau$、多个方法间声明显著差异，对预先声明的 prompt-level 检验报告效应量、bootstrap CI 和 Benjamini--Hochberg 校正后的 $q$ value；探索性分桶仅报告 CI，不作强显著性结论。

## 9. 实验 D：目标空间语义与 B 端局部行为

### D.1 高置信 token probe

从真实状态中选取 $\max_i p_i$ 高的点，按置信度分桶。对每个点令 $i^*=\arg\max p_i$，比较 $F(q)$、$\hat F(q)$ 与 $(W_{\mathrm{in}}^B)_{:,i^*}$：

- cosine 和欧氏距离；
- 在 B 输入 embedding 表中的最近邻 token 是否为 $i^*$；
- 相对 exact $F$ 的最近邻排名变化。

注意这不是要求所有 soft distribution 都还原为单一 token；只有高置信点才预期接近 $i^*$ 的 embedding。

### D.2 B 端 logits 保真

在相同 B 上下文中，分别把 exact $F(q)$ 与 $\hat F(q)$ 作为连续输入 embedding，比较 B 随后的 next-token logits：

- logits KL/JS、top-k overlap、top-1 一致率；
- greedy 首 token 一致率；
- 依 $m$、$\tau$、语料域和 A$\to$B 组合分层报告。

它回答“embedding 差异是否会改变 B 的实际局部行为”。同时加上 `identical` 与 `linear` 两个基线；exact $F$ 是 kernel 的直接 oracle，但不是这两个基线的 oracle。

## 10. 实验 E：多步闭环稳定性

选择固定、无标签的 prompts；相同模型、采样参数和初始上下文下，分别以 exact $F$ 与 $\hat F$ 进行 $K\in\{1,2,4,8,16\}$ 次 latent rollout。每步记录：

- 对齐 embedding 的相对误差和 cosine；
- B 端 next-token logits KL 与 top-1 一致；
- 轨迹中分母异常和 NaN/Inf 比例；
- 若使用 greedy decoding，最终文本的完全一致率及 token-level edit distance。

若单步保真高但随 $K$ 急剧发散，应如实报告：这表示近似误差在闭环中累积，不能仅以单步误差宣称方法有效。

### 10.1 latent CoT 的 unembedding 语义读出

多步稳定性不应只依据向量距离判断；还应检查 latent CoT state 是否仍能被读成连贯、相近的 token-level 语义。这里的 unembedding 是**诊断探针**，不等同于将连续 input embedding 直接当作语言模型末层状态。

对第 $k$ 个 latent step 的 source state $h_k^A$，先用 A 自身的末层规范化和 output head 计算原生读出：

$$
p_k^A=operatorname{softmax}
\left(W_{\mathrm{out}}^A\operatorname{Norm}_A(h_k^A)+b^A\right).
$$

对每种映射 $M\in\{F,\hat F,\mathrm{linear},\mathrm{identical}\}$，将 $M(h_k^A)$ 作为 B 的连续输入 embedding 插入相同的 B prefix，并完成一次 B 前向，取得该插入位置的**B 末层** hidden state $h_k^{B,M}$。仅在此之后才用 B 的 output head 读出：

$$
p_k^{B,M}=operatorname{softmax}
\left(W_{\mathrm{out}}^B\operatorname{Norm}_B(h_k^{B,M})+b^B\right).
$$

禁止直接对 $M(h_k^A)$ 使用 $W_{\mathrm{out}}^B$：它属于 B 的输入 embedding 空间，而非 B 的末层 residual space，直接 unembedding 没有架构上的可解释性。

在第 3.5 节规定的实验 E 的 256 条 prompt、每个 $k\in\{1,2,4,8,16\}$ 上保存：

- $p_k^A$、$p_k^{B,F}$、$p_k^{B,\hat F}$、$p_k^{B,\mathrm{linear}}$、$p_k^{B,\mathrm{identical}}$ 的 top-10 token、概率和 rank；
- 每个方法相对 exact 读出 $p_k^{B,F}$ 的 KL/JS、top-1 一致率、top-10 overlap 和 top-10 概率质量保留率；
- `kernel` 相对 exact 的这些量随 step $k$ 的轨迹，以及 source $p_k^A$ 与 exact-B $p_k^{B,F}$ 的对应读出，供端到端语义传递诊断；
- 对 ARC-Challenge 和 GSM8K 各固定 20 条 prompt，导出逐步表格：source top-5、exact top-5、kernel top-5、linear top-5、identical top-5。special token 用可读名称显示，且不据此作定量语义判断。

主统计结论仍以 kernel 相对 exact $F$ 的 B 端读出保真为准；A 到 B top token 不完全一致不自动表示失败，因为 B 的一次前向可合法地改写 token 分布。人工查看的逐步读出表用于发现语义漂移、重复、退化为空白/special token 或在某一步突然切换主题等现象，必须与 KL、top-k 和闭环误差曲线一起解释。

## 11. 实验 F：跨 checkpoint / 模型尺度

在 tokenizer 相同的前提下，构造：A=B、自小到大、自大到小、相邻 checkpoint 等组合。对每组执行实验 A--E，并与 `identical`、`linear` 比较。

该实验不涉及跨词表；它检验的只是同 ID token 条件下，A 的输出头与 B 的输入 embedding 之间的无训练对齐是否随模型差异保持有效。报告时应明确区分：

- exact $F$ 对 B 行为的有效性：soft-token 对齐假设；
- $\hat F$ 相对 exact $F$ 的退化：核近似误差；
- `linear`/`identical` 的表现：替代映射基线。

## 12. 结果表与判定标准

每个 A $\to$ B、数据域、$m$、$\tau$、随机 seed 组合保存一行机器可读结果；主表至少包含：relative-$L_2$、cosine、TV、KL、top-k overlap、B-logits KL、denominator failure rate、在线延迟、峰值显存。

主结论按以下顺序作出：

1. exact $F$ 在 B 端是否明显优于 `identical`/`linear`，或至少能稳定保持局部行为；
2. $\hat F$ 是否在所有预先指定的 test/OOD 集上接近 exact $F$，且无数值失效；
3. ORF 是否相对 iid 在相同特征数下更准或更稳定；
4. 所需的 $m$ 下是否仍有明确在线加速和可接受存储；
5. 哪些低温、高范数、尖锐分布或长 rollout 条件构成明确边界。

不将单一 reasoning benchmark 的最终正确率作为对齐有效性的唯一证据；它仅作为上述机制验证完成后的补充下游结果。

## 13. 展望：与训练型通信/对齐方法的公平比较

本计划当前聚焦于无需额外训练的 $T_{A\to B}=I$ 对齐：`identical`、`linear` 与 ORF kernel 均直接由现有模型权重构造。因此，当前结论只应表述为“在零训练成本下的映射保真、稳定性与在线效率”，不应与通过训练获得的通信接口直接作强弱比较。

后续可加入 ThoughtComm、C2C 等需要训练的 latent communication / alignment 方法作为对照。比较必须在同一模型配对、相同 tokenizer、相同 probe 语料与相同下游任务上进行，并完整报告：

- 训练语料来源、样本量、是否与本计划的 calibration/test 集重叠；
- 训练参数量、训练步数、batch size、学习率、随机 seed、GPU 时间及峰值显存；
- 推理阶段的在线延迟、额外参数/缓存存储，以及是否需要访问完整词表；
- 与本计划一致的 exact-$F$ 保真、B-logits、闭环稳定性、variance--$\tau$ 和最终任务指标；
- 以“零训练 ORF”与“训练后方法”分别作 Pareto 图，明确比较的是训练成本、推理成本、保真度和下游收益之间的权衡。

若训练型方法使用了额外的配对数据、教师模型输出或任务监督，应单列为额外信息预算，而不能归因于对齐算法本身。这样才能回答：训练能带来多少额外收益，以及该收益是否足以抵偿训练成本与部署复杂度。




