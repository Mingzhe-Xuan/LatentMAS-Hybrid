# plan v4 preview：从算子近似到有效 LatentMAS

> 状态：预览稿。本文以 `plan_v3.md` 为基础，保留 S0--S4、C0、M0，补入闭环任务效果、受控通信、端到端效率和泛化实验。实现前仍需按算力预算冻结样本数；一旦主实验开始，不再根据结果改动主要指标、对照或停止规则。

## 1. Motivation、主张与论证顺序

本项目的核心 motivation 不是单独证明随机特征可以近似 softmax，也不是仅观察 latent state 的 entropy，而是验证以下完整链条：

```text
kernel 能以更低在线成本近似 full-softmax soft mapping
→ 单步近似误差在闭环 recurrence 中不会快速放大
→ soft/kernel latent recurrence 能维持或改善任务行为
→ 跨 agent latent message 携带接收方原本不知道的信息
→ 在真实协作任务中，通信收益足以覆盖额外计算
→ 结论能跨数据集、模型方向和 agent edge 复现
```

各层主张、必要证据和禁止外推如下：

| 层级 | 可支持的主张 | 必要证据 | 不能由该层单独推出 |
| --- | --- | --- | --- |
| S：算子层 | kernel 数值上接近 soft reference | embedding/distribution error、异常率、latency | latent recurrence 有用、消息有语义 |
| C：同模型闭环 | 某接口能稳定递推并产生任务行为 | accuracy--K、轨迹分叉、失败率、动态诊断 | 跨模型通信成立 |
| M：通信层 | message 向 B 传递信息并改善行为 | 私有信息恢复、真实任务 paired improvement | 任意模型或任意 edge 都有效 |
| E：系统层 | kernel 形成有意义的效率--质量折中 | 端到端 latency、显存、吞吐、break-even | 数值更快必然带来任务收益 |
| G：泛化层 | 结论不依赖单模型/单 edge/单数据集 | 预注册复现矩阵 | 对未测试模型族普遍成立 |

## 2. 统一术语与映射定义

### 2.1 `soft` 是 full-softmax reference

后续方法名统一为 `soft`，不再把 alignment 方法命名为 `exact`。若需要描述数值精度，可写“full-softmax soft reference”或“float64 audit”，但图例、配置和结果表中的方法名固定为：

```text
identical / linear / soft / kernel / text
```

令 source hidden 为行向量 $q\in\mathbb R^{d_A}$，source output weight 为 $W_{\mathrm{out}}^A\in\mathbb R^{V\times d_A}$，target input embedding 为 $W_{\mathrm{in}}^B\in\mathbb R^{V\times d_B}$，则

$$
p(q)=\operatorname{softmax}\!\left(\frac{W_{\mathrm{out}}^Aq+b^A}{\tau}\right),
\qquad
F_{\mathrm{soft}}(q)=p(q)^\top W_{\mathrm{in}}^B.
$$

`soft` 不 sampling、不 argmax、不 top-k 截断，也不做 identical/linear 的额外 norm rescaling。

### 2.2 Kernel reference

设 block-ORF feature 为

$$
\phi_\Omega(x)=\frac{1}{\sqrt m}
\left[\exp\left(\omega_r^\top x-\frac{\|x\|_2^2}{2}\right)\right]_{r=1}^m,
$$

并预聚合 target values 后得到 $S,z$，在线估计为

$$
\hat F_{\mathrm{kernel}}(q)=
\frac{S\phi_\Omega(q/\tau)}
{z^\top\phi_\Omega(q/\tau)}.
$$

主设置暂沿用 $m=2048,\tau=1.0$、block-ORF seed 101。所有 soft/kernel 比较必须确认二者的 temperature、bias、normalization 和词表约定一致；若定义不一致，不得把差异解释为近似误差。

## 3. 统一实验口径

### 3.1 数据集与 split

- GSM8K：`test`；
- MBPP+：`test`；
- ARC-Challenge：`test`；
- AIME 2025：数据源提供的 `train` split，作为评测集使用；
- 受控通信数据：固定 train/calibration/test 模板和随机种子，仅用于评测，不训练模型。

AIME 2025 样本量小，报告逐题结果和 paired effect，不把其置信区间与大数据集作同等强度解释。

### 3.2 配对、随机性与统计

1. 方法比较以同一题目、同一初始 prompt/KV、同一生成配置配对；
2. 数据选择 seed、文本生成 seed、ORF seed 分开记录；
3. 连续指标报告 mean、median、分位数、题目 cluster bootstrap 95% CI；
4. accuracy 报告原值、paired difference、95% CI 和解析失败率；
5. 多数据集/多方法显著性声明使用 Benjamini--Hochberg q value，并同时给效应量；
6. NaN/Inf、非正 kernel denominator、OOM、解析失败和提前终止保留为原始行，不静默删除；
7. 主结果以预注册 seed 为准，其他 seed 进入稳定性实验，不选择最好 seed。

### 3.3 产物规范

每次运行统一写入：

```text
exp_result/<experiment>/runs/<run>/
  run_manifest.json
  metrics/*.parquet
  summaries/*.json
  figures/*.pdf
  figures/*.json
```

Parquet 保存可重新聚合的逐题/逐 state/逐 step 数据；summary 不保存大 embedding；每张 PDF 旁的 JSON 记录输入指标、模型、数据集、seed、方法顺序和绘图参数。

## 4. S：算子层实验

S0--S4 保留 `plan_v3` 的研究边界：它们只评价映射算子的数值性质，不证明 latent thought 或通信语义。

### S0. 状态与 embedding 数值尺度（保留）

对真实 Hybrid 轨迹中的 source hidden、source $W_{\mathrm{out}}$ 和 target $W_{\mathrm{in}}$ 报告 norm 分布，按 role、state kind、step 区间分层。保留：

- `s0_hidden_states.parquet`；
- `s0_embedding_norm_hist.pdf`；
- `s0_hidden_norm_hist.pdf`。

### S1. Kernel 对 soft 的单步保真与在线性能（保留）

在固定 source state 上计算 $F_{\mathrm{soft}}(q)$ 与 $\hat F_{\mathrm{kernel}}(q)$，报告 relative-$L_2$、cosine、denominator、NaN/Inf，并对 soft distribution 报告 entropy/confidence 分桶。可选 float64 audit 用于验证实现，不作为独立方法。

性能测试必须分别记录 warm-up 后 latency、peak allocated memory 和输入 batch/query 数；不得只报告单次未经同步的 CUDA wall time。

### S2. Distribution error 与误差传播（保留）

比较 soft distribution 与 kernel-induced approximation 的 KL、JS、TV、$L_1$、top-1 agreement、top-10/top-100 overlap，并分析它们与 embedding error、entropy、confidence 的关系。

校准扫描保留：

$$
m\in\{256,512,1024,2048,4096\},\quad
\tau\in\{0.7,1.0,1.3\},
$$

并比较 block-ORF 与 iid Gaussian RF；校准结果不能替代闭环 C2。

### S3. 固定 state 的 ORF bias--variance（保留）

固定真实 source state，只替换 ORF seed，分解 variance、bias$^2$ 和 MSE。该实验隔离随机特征不确定性，但不得外推为 recurrence 稳定性。

### S4. Receiver embedding geometry（保留但降为辅助）

继续比较 hidden、soft、linear、kernel 与真实 TextMAS receiver-side token embedding 的几何关系。PCA 是描述性辅助，不作为 kernel 有效、latent thought 可读或通信成功的主要证据；不再为增加定量结论而扩展更多 t-SNE 图。

### S5. State-distribution 与 edge 泛化（P1，新增）

为避免 S1--S3 只说明 ARC-Easy 的 `Refiner → Judger` state，增加固定复现池：

- 数据集：ARC-Challenge、GSM8K；
- step：early/middle/late 三段；
- confidence：soft entropy 四分位；
- edge：至少 `Planner → Critic`、`Refiner → Judger`；
- model setting：同模型一组、跨模型一组。

主指标仍为 S1 的 relative-$L_2$/cosine/异常率；报告分层 forest plot。S5 只回答算子是否随 state distribution 改变，不回答任务收益。

## 5. C：同模型 Latent CoT 闭环实验

C 系列必须把“分布看起来稳定”和“任务行为有效”分开。所有方法从同一 prompt state 独立展开，不共享后续 KV；`text` 使用 greedy token feedback，`soft` 使用 full-softmax expected embedding，`kernel` 近似 soft。

### C0. Entropy trajectory（已实现，保留）

**数据。** GSM8K、MBPP+、ARC-Challenge、AIME 2025；每数据集按固定 seed 抽取至多 512 题。

**方法。** `identical / linear / soft / kernel / text`，$K=100$。

**指标。** 对每个 dataset × method × question × step 保存 pre-unembedding readout entropy：

$$
p_t=\operatorname{softmax}(W_{\mathrm{out}}h_t+b),
\qquad
H_t=-\sum_i p_{t,i}\log p_{t,i}.
$$

主图为 2×2 数据集面板；实线是题目 mean，阴影是题目 bootstrap 95% CI。C0 只描述 token distribution sharpness，不把 entropy 下降解释为推理改善。

### C1. Accuracy--K 与闭环稳定性（P0，新增）

这是同模型 Latent CoT 的主要行为实验。

**主扫描。** 对相同题目和 decoder budget 比较五种方法：

$$
K\in\{0,4,8,16,32,50,100\}.
$$

`K=0` 是无 latent recurrence 的直接回答基线。文本条件必须同时报告“相同序列位置数”和“近似相同 FLOPs”两种口径，避免把额外计算误判为接口优势。

**逐题保存。** dataset、item ID、method、$K$、最终文本、解析结果、correctness、生成 token 数、wall time、OOM/NaN/提前停止原因，以及：

- 每步 hidden norm；
- 相邻 step cosine；
- entropy 与 top-10 mass；
- special-token mass；
- 重复/周期诊断；
- kernel denominator 最小值。

**主指标。** task accuracy/pass rate、parse success、failure rate、相对 `text` 和 `soft` 的 paired difference。主图：accuracy--$K$、failure-rate--$K$；次图：hidden norm 与相邻 cosine 随 step 变化。

**解释。** C1 才回答 latent recurrence 是否产生有用任务行为。即使 kernel 几乎等于 soft，若二者均差于 text，也只能说明近似准确，不能宣称 soft interface 是好的 CoT。

### C2. Kernel--soft 闭环误差累积（P0，新增）

从相同 prompt/KV state 同时启动配对的 soft 与 kernel rollout。由于两条轨迹从第一步后就可能面对不同 hidden，分别保存

$$
\delta_t^e=\frac{\|e_t^{\mathrm{kernel}}-e_t^{\mathrm{soft}}\|_2}
{\|e_t^{\mathrm{soft}}\|_2+10^{-8}},
$$

以及 hidden cosine、readout JS/KL、top-1/top-10 agreement、entropy difference、最终答案 agreement。

**固定扫描。** GSM8K 与 ARC-Challenge 各固定 128 题，$K\in\{16,50,100\}$，$m\in\{512,2048,4096\}$，ORF seed 至少 5 个。每题记录首次超过预注册 divergence threshold 的 step，但 threshold 只用于诊断，不删失后续轨迹。

**主图。** soft--kernel embedding/hidden/readout divergence vs. step；最终答案 agreement vs. $m$。该实验是 S 系列静态保真与 C1 任务结果之间的关键桥梁。

### C3. 文本锚点与动态退化诊断（P1，新增）

在 GSM8K、ARC-Challenge 各固定 20 个案例保存每步：

- readout top-10 token/probability；
- entropy、top-10 mass、special-token mass；
- hidden norm、相邻 cosine；
- 最近历史 state cosine，用于检测周期；
- 对应 text trajectory 的 token 与 input embedding。

定量部分比较高置信 text 位置上的真实下一 token embedding 与 soft/kernel feedback 的 cosine、nearest-token rank；案例图只用于定位重复、special-token collapse、主题突变和错误高置信收敛，不把 top-1 token 当作唯一 latent thought 翻译。

### C4. ORF seed 的闭环敏感性（P1，新增）

固定题目、prompt、$m=2048,\tau=1.0$，仅替换 ORF seed：

$$
\mathrm{seed}\in\{101,202,303,404,505\}.
$$

在 GSM8K、ARC-Challenge 各 128 题上报告最终答案 agreement、accuracy variance、每步 entropy/hidden divergence、异常率和首次分叉 step。S3 的静态 seed variance 不能替代 C4。

### B0. Interface 诊断基线（P2，可选）

只在 C1 的固定小样本上增加，不进入默认全矩阵：

- top-$k$ soft embedding，$k\in\{1,10,100\}$；
- argmax token embedding；
- norm-matched random embedding；
- mean vocabulary embedding。

Top-$k$ 用于判断完整词表 aggregation 是否必要，并作为比 kernel 更简单的速度基线；随机/均值基线用于排除“任意增加一个 embedding 位置都有效”。

## 6. E：端到端效率--质量实验

### E0. Latency、memory 与 break-even（P0，新增）

Kernel 的系统 motivation 必须用端到端测量支持，不能只依赖 S1 microbenchmark。

**方法。** `soft`、`kernel` 的 $m\in\{256,512,1024,2048,4096\}$、top-$k$ soft（若实现）、text。

**测量。** 在相同硬件、dtype、batch、prompt 长度和 $K$ 下记录：

- kernel 离线预计算时间与持久化大小；
- warm-up 后单步及完整 rollout latency；
- peak allocated/reserved GPU memory；
- questions/second 与 tokens-or-steps/second；
- 总能耗（仅在集群可可靠读取时报告）；
- accuracy、失败率及相对 soft 的 paired degradation。

设 soft 每 query 在线成本为 $T_s$，kernel 预计算为 $T_0$、每 query 成本为 $T_k$，则报告经验 break-even：

$$
N_{\mathrm{break-even}}=\frac{T_0}{T_s-T_k},
$$

仅当 $T_s>T_k$ 时定义。主图为 latency--accuracy、memory--accuracy Pareto，并明确是否计入 kernel 预计算。

## 7. M：跨 agent Latent Communication

M 系列区分三件事：接收方能否从 message 恢复私有信息、message 能否改善真实任务、重复跨模型映射是否累积退化。所有条件必须使用同一 B prompt、position ID、attention mask 和生成配置。

### M0. 无题目真实任务压力测试（保留并扩充方法对照）

A 看完整问题并运行固定 $K$ 步；B 不看题目、选项、A 文本或其派生文本，只接收最终 latent message 后回答。保留条件：

- B no-message；
- random-pair message；
- mismatched-pair message；
- matched soft；
- matched kernel；
- matched linear。

报告 accuracy、parse success、相对 no-message 的 paired difference，以及 kernel 相对 soft 的 paired degradation。M0 是高难度压力测试；失败不能单独定位是编码、映射、容量还是读取问题，因此不能替代 M1。

### M1. 受控私有信息恢复（P0，新增）

建立固定合成评测集，每条含 `entity / attribute / value`：A 看到完整事实，B 只看到 entity/attribute query，value 在 16 个平衡类别中取值。模板、entity 和 value 在 train/calibration/test 间分层隔离；数据只用于评测和阈值校准，不训练模型。

**条件。** no message、random pair、mismatched pair、matched soft、matched kernel、matched linear；可选 matched text message 作为带宽更高的参考。

**指标。** top-1/top-3 accuracy、NLL、ECE、混淆矩阵、parse failure、kernel 相对 soft 的 paired difference。若 no-message 明显高于类别先验，先检查模板或数据泄漏，不报告通信结论。

M1 是“message 携带 B 原本不知道的信息”的最干净因果证据。

### M2. 共享问题的真实协作收益（P1，新增）

A、B 都看到同一问题；A 先运行 latent recurrence，B 在自身正常推理上下文中接收 A message 后独立作答。主数据为 GSM8K 与 ARC-Challenge，固定条件：

- B alone；
- A alone；
- B + random/mismatched message；
- B + matched soft；
- B + matched kernel；
- B + matched linear；
- TextMAS communication。

报告 B 相对 B-alone 的 paired accuracy improvement、kernel 相对 soft 的 paired degradation、总 latency、传递 embedding/token 数和失败率。必须提供 compute-matched 对照，例如让 B-alone 使用近似相同额外 forward budget，避免把额外计算当成通信收益。

### M3. 多跳通信退化（P2，新增）

在 M1 test 子集和 GSM8K 固定子集上比较 A→B 一跳与 A→B→A 两跳；每跳保存 receiver hidden/readout、message norm、soft--kernel divergence、top-k agreement 和最终任务结果。

M3 只评价重复重编码的误差累积，不把多跳链称为同模型 Latent CoT。

## 8. G：泛化与最小复现矩阵

### G0. Model direction、edge 与 dataset（P1，新增）

主开发设置仍为 Qwen3-4B；冻结代码后运行最小复现矩阵：

| 维度 | 主设置 | 最小复现 |
| --- | --- | --- |
| 同模型 | 4B→4B | 8B→8B |
| 跨模型方向 | 4B→8B | 8B→4B |
| agent edge | Refiner→Judger | Planner→Critic |
| 数据集 | ARC-Challenge | GSM8K |

仅在 tokenizer token-to-ID vocabulary 完全一致时运行 soft/kernel 跨模型映射；不做未经定义的 tokenizer 概率搬运。G0 复用 S1、C1 的核心指标和 M1 的私有信息 accuracy，不重复所有辅助图。

若算力只允许一个跨模型方向，优先 4B→8B，并将 8B→4B 明确标为未测试，而不是从单向结果推断双向泛化。

## 9. 实验登记表

| ID | 优先级 | 核心问题 | 主输出 |
| --- | --- | --- | --- |
| S0 | 已有 | hidden/embedding 数值尺度如何 | norm metrics 与 histogram |
| S1 | 已有 | kernel 单步是否接近 soft | error ECDF、conditioning、latency |
| S2 | 已有 | distribution error 如何传播 | KL/JS/TV/top-k 与 embedding error |
| S3 | 已有 | 固定 state 的 ORF bias--variance | variance--temperature、MSE |
| S4 | 已有/辅助 | receiver space 几何如何 | PCA 与定量距离 |
| S5 | P1 | 误差是否跨 state/edge 泛化 | 分层 forest plot |
| C0 | 已有 | entropy 如何随 100 step 变化 | 2×2 entropy trajectory |
| C1 | P0 | recurrence 是否完成任务 | accuracy--K、failure--K |
| C2 | P0 | kernel error 是否闭环放大 | divergence--step、answer agreement |
| C3 | P1 | 轨迹如何退化或形成周期 | top-k/special-token/cycle cases |
| C4 | P1 | ORF seed 是否改变闭环行为 | answer agreement、seed bands |
| B0 | P2 | 更简单接口是否足够 | top-k/random/mean baseline |
| E0 | P0 | kernel 是否形成实际效率优势 | latency/memory--accuracy Pareto |
| M0 | 已有/扩充 | 单 message 能否支撑无题目真实答题 | paired accuracy |
| M1 | P0 | message 是否携带私有信息 | accuracy/NLL/ECE/confusion |
| M2 | P1 | message 是否改善真实协作 | paired improvement 与 cost |
| M3 | P2 | 多跳是否累积退化 | hop degradation |
| G0 | P1 | 是否跨模型方向/edge 复现 | 最小复现矩阵 |

## 10. 图表与主报告顺序

主报告按论证链排列，而不是按代码目录排列：

1. S1：kernel vs soft error ECDF 与数值异常率；
2. E0：latency/memory--accuracy Pareto 与 break-even；
3. C1：accuracy--$K$ 和 failure--$K$；
4. C2：soft--kernel divergence vs. step；
5. C0/C3：entropy 与动态退化诊断；
6. M1：私有信息 accuracy/ECE；
7. M2：真实任务 paired improvement；
8. G0：跨模型/edge forest plot；
9. S2--S4、C4、M0/M3 放入机制分析或附录。

每张主图的标题或图注明确编码：线表示何种统计量、阴影/误差条表示何种区间、样本单位是题目还是 state、是否跨 seed 聚合。

## 11. 实施阶段与算力裁剪

### Phase A：P0，形成最小完整论证

必须完成：

```text
S1 + C0 + C1 + C2 + E0 + M1
```

它们分别覆盖单步近似、动态描述、任务结果、误差累积、系统价值和信息传输。若 P0 结果否定核心假设，应先报告失败机制，不继续扩展大规模可视化。

### Phase B：P1，验证稳健性与真实协作

```text
S5 + C3 + C4 + M2 + G0
```

### Phase C：P2，机制与扩展

```text
B0 + M3 + S4 扩展案例
```

资源不足时，优先减少 P1/P2 的样本或复现格点，不删 P0 的关键对照；不得只保留 entropy/PCA 而删除 accuracy、通信和效率结果。

## 12. 停止规则与解释边界

1. soft/kernel 的 temperature、bias、normalization 或 vocabulary 定义不一致时，停止 kernel-vs-soft 结论；
2. tokenizer vocab/ID 不一致时，停止对应跨模型映射；
3. C 系列无法取得真实 pre-unembedding hidden 时，不用生成 token hidden 替代；
4. M 系列 B 看到被禁止的问题/答案派生文本时，该运行无效；
5. M1 no-message 显著高于先验时，先排查泄漏；
6. CUDA timing 未同步或硬件/并发不一致时，不报告效率比较；
7. OOM、NaN、denominator failure、解析失败必须进入分母和失败率；
8. entropy 下降不能单独解释为推理改善；PCA 聚类不能单独解释为语义；单步 mapping error 小不能单独解释为闭环稳定；
9. kernel 与 soft 都差于 text 时，只能得出 kernel 复现了一个较差 soft interface，不能宣称 latent CoT 成功；
10. 正确配对 message 未显著优于 random/mismatched/no-message 时，不宣称 latent communication 成功。

## 13. v4 preview 的验收标准

计划进入正式 v4 前，需要冻结并补齐：

- P0 每个数据集的最终样本数和 seed；
- C1 的 decoder/compute-matching 细节；
- C2 divergence threshold 仅作诊断的具体数值；
- M1 数据模板、split 和泄漏审计；
- E0 硬件、batch、warm-up、同步和预计算计费口径；
- G0 最终可运行的模型方向；
- 所有新增 Parquet schema、summary 字段和预注册图名。

完成上述冻结后，将本文件复制为正式 `plan_v4.md`；正式运行期间仅允许修复实现错误，并在 manifest 中记录变更，不根据中间结果重选主指标或主数据集。
