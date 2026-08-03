# plan v3：Approximator、Latent CoT 与 Latent Communication

## 1. 范围与结论边界

本计划只包含：

- 算子层：S0--S4；
- 同模型 Latent CoT：C0；
- 跨 agent Latent Communication：M0。

不写入其他 C/M 实验。三类结论严格分开：S0--S4 只衡量 `Refiner → Judger` 映射算子的数值性质；C0 只衡量同模型 latent recurrence 中的可读出分布如何随 step 演化；M0 才衡量前一 agent 的 latent 信息能否使未见题目的后一 agent 正确作答。

当前 `exp/approximator` 的具体映射边为四角色顺序 `latent_mas_hybrid` 流程中的：

$$
h_t^R\xrightarrow{W_{\mathrm{out}}^R}\operatorname{softmax}
\xrightarrow{W_{\mathrm{in}}^J}e_t^J,
$$

其中 source 是 Refiner 的 `latent_reply_hidden`，target 是 Judger 的输入 embedding。每次运行首先执行真实的四角色 sequential hybrid 流程并缓存轨迹；S1--S4 从缓存中选择上述 source state。S4 默认另跑一条独立缓存的 TextMAS 轨迹，作为 receiver-side token embedding 几何参考，但不改变 S1--S3 的算子样本。所有写入产物位于 `exp_result/approximator/runs/<run>/`；长期轨迹缓存位于 `exp/cache/`，run-local manifest 与指标位于 `exp_result/approximator/runs/<run>/`。

令 $q=h_t^R$，词表大小为 $V$，$w_i^\top=(W_{\mathrm{out}}^R)_{i,:}$，$c_i=(W_{\mathrm{in}}^J)_{i,:}$，$b_i=b_i^R$。exact reference 为

$$
p_i(q)=\frac{\exp(w_i^\top q/\tau+b_i)}{\sum_{j=1}^{V}\exp(w_j^\top q/\tau+b_j)},
\qquad F(q)=\sum_{i=1}^{V}p_i(q)c_i.
$$

设 block-ORF 矩阵 $\Omega=[\omega_1^\top;\ldots;\omega_m^\top]\in\mathbb R^{m\times d_R}$（当前主设置 $m=2048$、seed 101），随机特征为

$$
\phi_\Omega(x)=\frac{1}{\sqrt m}
\left[\exp\left(\omega_r^\top x-\frac{\|x\|_2^2}{2}\right)\right]_{r=1}^{m}.
$$

令 $\beta=\max_i b_i$。离线预聚合量与 kernel estimate 明确为

$$
S=\sum_{i=1}^{V}c_i\,e^{b_i-\beta}\phi_\Omega(w_i)^\top\in\mathbb R^{d_J\times m},
\qquad z=\sum_{i=1}^{V}e^{b_i-\beta}\phi_\Omega(w_i)\in\mathbb R^m,
$$

$$
\boxed{\displaystyle
\hat F(q)=\frac{S\,\phi_\Omega(q/\tau)}
{z^\top\phi_\Omega(q/\tau)}}.
$$

实现中在线 query feature 会减去其所有 $m$ 个 log-feature 的共同最大值后再取 exp；此公共缩放同时作用于分子和分母，因此不改变 $\hat F(q)$。若分母非有限或不大于 machine epsilon，当前实现返回/抛出数值失败。`exact` 是 kernel 的数值 reference，不是 latent thought 的语义 oracle；`linear` 仅在 S4 中作为几何比较基线。

## 2. 统一运行与统计口径

现有入口为：

```bash
python exp/approximator/run.py \
  --agent_models Qwen/Qwen3-4B \
  --dataset arc_easy --split test --study all
```

一个模型名会复制给 Planner、Critic、Refiner、Judger；也可显式传入四个模型名。当前轨迹生成使用 `prompt=sequential`、`latent_steps=50`、采样解码（默认 `temperature=0.6`、`top_p=0.95`），并以 `probe_seed=42` 抽题、抽 state。每题每类 state 最多保留 20 个；被分析的 Refiner latent state 最多为每题 20 个。

### 2.1 实验登记表（固定主设置）

除非某一条目另有说明，S0--S4 均使用当前 `config.json` 的主设置；这不是建议值，而是本计划对应的可复现实例。C0/M0 尚未在 `exp/` 中实现，以下表格是其首个实现必须遵守的固定规格。

| 实验 | 模型与角色 | 输入数据（固定） | 运行输入/关键参数 | Parquet：逐题/逐 state 原始结果 | JSON：manifest 与汇总统计 | PDF：预注册图 |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | Planner=Critic=Refiner=Judger=`Qwen/Qwen3-4B` | ARC-Easy `test`；seed 42 shuffle 后取 10 题 | `sequential`；50 latent steps；采样 `T=0.6, top_p=0.95`；Judger 最多 512 新 token；每题每 state kind 最多 20 个 state | `s0_hidden_states.parquet`：每个缓存 state 的 item/role/kind/position/hidden norm | `run_manifest.json`；`s0_summary.json`：Refiner $W_{out}$、Judger $W_{in}$ 行范数及按 role/kind 的 hidden-norm 分位数 | `s0_embedding_norm_hist.pdf`；`s0_hidden_norm_hist.pdf` |
| S1 | 同 S0；映射边固定为 Refiner $\to$ Judger | 同 S0 的已缓存 Refiner `latent_reply_hidden` | $m=2048,\tau=1.0$，block-ORF seed 101，chunk 4096；float64 audit 最多 256 state | `s1_mapping.parquet`：每 state 的 $F$ relative-$L_2$/cosine、entropy、confidence、分母/NaN 标记；`s1_single_kernel.parquet`：按 rank band 的单核误差；可选 `s1_performance.parquet`：每次调用 latency | `run_manifest.json`；`s1_summary.json`：题目 cluster 汇总、CI；启用性能时其内包含 latency 汇总 | `s1_f_rel_l2.pdf`；`s1_error_conditioning.pdf` |
| S2 | 同 S1 | 同一份 S1 mapping cache | 主分析不重算映射；可选 calibration：ORF/iid、$m=256/512/1024/2048/4096$、$\tau=0.7/1.0/1.3$、seed 101/202/303/404/505 | `s2_mapping.parquet`：每 state 的 KL/JS/TV/$L_1$/top-k；`s2_single_kernel.parquet`：rank 分层核误差；启用时 `s2_calibration.parquet`：family/$m$/$\tau$/seed/误差 | `run_manifest.json`；`s2_summary.json`；启用时 `s2_calibration_summary.json` | `s2_f_rel_l2.pdf`；`s2_error_propagation.pdf` |
| S3 | 同 S1 | 同一 Refiner state pool；seed 42 shuffle 后至多 50 题、每题至多 16 state | $m=512/1024/2048$；$\tau=0.5,0.6,\ldots,2.0$；ORF seed 1001--1032（32 个） | `s3_variance.parquet`：每 state × $m$ × $\tau$ 的 kernel/$\hat F$ variance、std、bias$^2$、MSE 与 rank band | `run_manifest.json`；`s3_summary.json`（headline）；`s3_grid_summary.json`（$m$ × $\tau$ grid） | `s3_variance_tau.pdf`；`s3_single_kernel_variance_tau.pdf`；`s3_refiner_state_forest.pdf` |
| S4 | 同 S1；另拟合 linear baseline；默认增加真实 TextMAS 对照 | 同题 Refiner latent states 与 TextMAS Refiner 文本在 Judger prompt 中的 token embeddings；按题平衡后最多 2,000 个 | 主 ORF 设置同 S1；linear ridge $10^{-5}$；TextMAS 每角色默认最多 256 token；可选 t-SNE：seed 101 | `s4_embeddings.parquet`；`s4_joint_pca_coordinates.parquet`；启用时 `s4_joint_tsne_coordinates.parquet` | `run_manifest.json`；`s4_summary.json`：四组降维、空间汇总、latent 配对几何及 TextMAS 分布几何 | 四张 `s4_{linear,kernel}_{exact,text}_joint_reduction.pdf` |
| C0 | 单模型 A=`Qwen/Qwen3-4B`，A 同时承担 latent recurrence 与 $W_{out}$ readout | GSM8K `test`；seed 42 shuffle 后取 512 题 | question prompt；greedy；$K=50$；每题保存 50 个 pre-unembedding hidden | `c0_entropy_by_step.parquet`：item ID、step、$H_t$、finite 标记、失败原因；恰为一题一步一行 | `run_manifest.json`；`c0_summary.json`：每个 step 的题数、mean、median、95% CI | `c0_entropy_vs_step.pdf`：横轴 step，纵轴 entropy，含 95% CI |
| M0 | A=`Qwen/Qwen3-4B`，B=`Qwen/Qwen3-4B`；角色为 sender/answerer | ARC-Easy `test`；seed 42 shuffle 后取 512 题 | A 看完整题目并做 $K=50$ latent steps；B 永不接收题目/选项/A 文本；B greedy、最多 32 token；真实 `transfer_via_realignment` 注入 A 最终 hidden | `m0_per_question.parquet`：item ID、condition、配对 source ID、prediction、parsed、correct、message position、生成长度、失败原因；四条件各一行 | `run_manifest.json`；`m0_summary.json`：四条件 accuracy、相对 no-message 的 paired difference 与 95% CI | `m0_accuracy.pdf`：四条件 accuracy 与 paired-difference CI |

所有 JSON、Parquet、PDF 均写入同一 run 目录的 `summaries/`、`metrics/`、`figures/` 子目录；`run_manifest.json` 位于该 run 根目录。Parquet 只存原始可重算记录，JSON 不含 hidden/embedding 大向量而只存参数、计数和聚合统计，PDF 只呈现对应 JSON/Parquet 的可复现可视化。

算子统计以题目为 cluster：先在题内平均 state 指标，再在题间汇总；连续指标输出 mean、median、variance/std、分位数及题目 cluster bootstrap 95% CI。运行 manifest 记录参数、版本、git commit、实现指纹、轨迹/映射缓存命中情况。tokenizer vocabulary 不一致、映射分母非有限或非正、或 float32 对 float64 exact audit 的 p99 相对误差超过 $10^{-4}$ 时停止对应运行。

### 2.2 PDF 图定义

| PDF 文件 | 图的构成与编码 | 它呈现什么 |
| --- | --- | --- |
| `s0_embedding_norm_hist.pdf` | 单面板叠加 histogram；横轴是 embedding row $L_2$ norm，纵轴是 token 数；两条轮廓线分别为 Refiner $W_{out}$ 与 Judger $W_{in}$。 | 两个词表矩阵的行向量数值尺度及其分布是否明显失配。 |
| `s0_hidden_norm_hist.pdf` | 每个 `role × state_kind` 一个面板；横轴为 hidden $L_2$ norm，纵轴为 state 数。 | 真实四角色轨迹中不同来源 state 的尺度分布。 |
| `s1_f_rel_l2.pdf` | 两面板：左为 $\|\hat F-F\|_2/(\|F\|_2+10^{-8})$ 的 density histogram，右为同一量的 ECDF；每个 source 一条线。当前主分析只有 `refiner_latent_reply_hidden` 一条线。 | kernel embedding 映射误差的整体分布与长尾。 |
| `s1_error_conditioning.pdf` | 两个散点面板；纵轴均为 relative-$L_2(F)$ error；左横轴为 exact entropy，右横轴为 exact confidence；每点一个 state。 | 映射误差是否集中在高 entropy 或低置信度的 output distribution。 |
| `s2_f_rel_l2.pdf` | 与 `s1_f_rel_l2.pdf` 相同的 histogram+ECDF，使用 S2 复用的 mapping rows。 | S2 使用的同一主 kernel 误差样本分布，用于与 softmax 指标对照。 |
| `s2_error_propagation.pdf` | 三个散点面板，点均为 state：$(\|p-\hat p\|_1,\mathrm{relative}\ L_2(F))$、$(\mathrm{TV}(p,\hat p),\mathrm{relative}\ L_2(F))$、$(H(p),\mathrm{KL}(p\|\hat p))$。 | 单核/softmax 分布误差是否会传播为 embedding 映射误差。 |
| `s3_variance_tau.pdf` | 单面板折线；横轴 $\tau$，纵轴（log scale）为每个 $m$ 下 state-level $\hat F$ variance 的 median；$m=512/1024/2048$ 各一条线。 | 固定 state、仅换 ORF seed 时，$\hat F$ 方差如何随温度和 feature 数改变。 |
| `s3_single_kernel_variance_tau.pdf` | 单面板折线；横轴 $\tau$，纵轴（log scale）为单核 estimator 方差的 median；每条线一个 exact-softmax rank band，固定 $m=2048$。 | 哪些 token-rank 区域的单核近似对温度最敏感。 |
| `s3_refiner_state_forest.pdf` | 题目级点图；横轴（log scale）为该题所选 state 的平均 $\hat F$ variance，纵轴为 item ID；固定 $m=2048,\tau=1$。 | seed 方差是否被少数题目主导。 |
| `s4_{alignment}_{reference}_joint_reduction.pdf` | alignment 为 linear/kernel，reference 为 exact/text；每张图对 hidden、reference、aligned 三类原始向量独立联合拟合 PCA，并可附 t-SNE。TextMAS 与 latent 按题和数量平衡，但不建立 token-step 语义配对。 | linear/kernel 相对 exact soft embedding 或真实 TextMAS receiver-side token embedding 的几何偏移。 |
| `c0_entropy_vs_step.pdf` | 单面板曲线；横轴 latent step $t=0\ldots49$，纵轴为 $H_t$（nats）；实线为题目 mean，阴影为题目 cluster bootstrap 95% CI；可选虚线为 median。 | 同模型 latent recurrence 的 $W_{out}$-decoded token distribution 尖锐程度如何随 step 演化。 |
| `m0_accuracy.pdf` | 两面板：左为 no-message、random-pair、mismatched-pair、matched-message 四条件的 accuracy 与题目 bootstrap 95% CI；右为后三条件减 no-message 的 paired accuracy difference 与 CI，0 处画参考线。 | B 在不见题目时，正确配对 A latent message 是否比无消息或错误消息更能提高答题准确率。 |

所有图均在同名 `.pdf` 旁写入同名 `.json` artifact context，记录运行参数、模型、数据集、seed 与输入 cache；该 JSON 不是统计 summary，而是图的可追溯元数据。

## 3. 算子层：S0--S4

### S0. 轨迹状态与矩阵范数

**运行。** 对完整缓存轨迹中的所有已采样 state 计算 hidden $L_2$ norm；同时统计 Refiner $W_{\mathrm{out}}$ 行范数和 Judger $W_{\mathrm{in}}$ 行范数。输出按 `role × state_kind` 分面的 hidden-norm histogram，以及两类 embedding row-norm 的叠加 histogram。

**现有产物。** `metrics/s0_hidden_states.parquet`、`summaries/s0_summary.json`、`figures/s0_embedding_norm_hist.pdf`、`figures/s0_hidden_norm_hist.pdf`。

**意义。** 给出当前真实流程中 source/target 的数值尺度背景；不将范数分布解释为推理质量或通信语义。

### S1. exact 映射保真与在线性能

**运行。** 对每一个 Refiner `latent_reply_hidden` 计算 $F(q)$ 和主 ORF kernel $\hat F(q)$，报告

$$
\frac{\|\hat F(q)-F(q)\|_2}{\|F(q)\|_2+10^{-8}},
\qquad
\cos(\hat F(q),F(q)).
$$

同时从 exact softmax 的 rank 1、2--10、11--100、101--1000、$>1000$ 各 band 抽取 key，计算单核绝对误差、相对误差、log 误差与比值。可选性能测试先 warm-up 200 次，再对最多 500 个 state 分别测量 exact 和 kernel 的在线调用 latency。

**现有产物。** `metrics/s1_mapping.parquet`、`metrics/s1_single_kernel.parquet`、可选 `metrics/s1_performance.parquet`，以及 error-conditioning 图和按 source 汇总的 JSON。

**意义。** 判定有限特征 kernel 是否可近似完整 vocab softmax 映射并量化其速度代价；不能证明该映射适合作为 latent CoT 接口。

### S2. softmax 误差传播与 ORF/iid 校准

**运行。** S2 复用 S1 的完整 mapping cache。由 exact softmax $p$ 与随机特征诱导的 $\hat p$ 计算 KL$(p\|\hat p)$、JS、TV、$L_1$、top-1 agreement、top-10/top-100 overlap 与 exact top-10 mass，并绘制这些量与 $\|F-\hat F\|$、entropy 的关系。

仅在显式 `--run_s2_calibration` 时，扫描 ORF 和 iid Gaussian RF：

$$
m\in\{256,512,1024,2048,4096\},\quad
\tau\in\{0.7,1.0,1.3\},\quad
\mathrm{seed}\in\{101,202,303,404,505\}.
$$

**现有产物。** `metrics/s2_mapping.parquet`、`metrics/s2_single_kernel.parquet`、可选 `metrics/s2_calibration.parquet`，以及 `figures/s2_error_propagation.pdf`。

**意义。** 定位单核近似如何传递到 token 分布和 Judger embedding；ORF/iid 扫描是数值校准，不改变主设置，也不构成语义通信证据。

### S3. 固定状态下的 ORF seed variance--temperature

**运行。** 从 Refiner state 中按题目最多选择 16 个 state，最多 50 题；对

$$
m\in\{512,1024,2048\},\quad \tau=0.5,0.6,\ldots,2.0
$$

固定 state，仅替换 32 个 ORF seed（1001--1032）。对每个 state 记录 $\hat F$ 的分量平均 sample variance/std、relative std、bias$^2$、MSE；对 rank 抽样的单核项记录相同的 seed 方差量。

**现有产物。** `metrics/s3_variance.parquet`、`summaries/s3_summary.json`、`summaries/s3_grid_summary.json`，以及 F variance--$\tau$、single-kernel variance--$\tau$、题目级 forest 图。

**意义。** 隔离“同一真实 latent state，仅换 ORF 矩阵”带来的随机不确定性，区分低方差与低偏差；不能外推为闭环 latent CoT 的稳定性。

### S4. Refiner-to-Judger embedding 空间的 Exact/TextMAS 双参考降维

**运行。** 对各 Refiner source state 生成 hidden、exact、linear、kernel 向量；默认另跑同题 sequential TextMAS 至 Refiner，并在真实渲染后的 Judger prompt 中用 offset mapping 定位 Refiner 回复 token，取 Judger input embedding table 的对应行作为 text reference。对每种 alignment 分别独立拟合 `exact-hidden-aligned` 与 `text-hidden-aligned`，形成 linear/exact、linear/text、kernel/exact、kernel/text 四组 PCA/t-SNE。Text token 与 latent state 按题和数量平衡，最多 2,000 个，但不解释为逐 step 语义配对。可用 `--no_s4_text_mas` 关闭默认 TextMAS 对照；可选 t-SNE 参数仍固定为 `init=pca`、`perplexity=50`、`learning_rate=auto`、`max_iter=1500`、seed 101。latent 方法继续逐 state 比较 absolute/relative $L_2$ 与 cosine，TextMAS 使用题目级 centroid 等非配对分布几何。

**现有产物。** `metrics/s4_embeddings.parquet`、`metrics/s4_joint_pca_coordinates.parquet`、可选 t-SNE coordinates、`summaries/s4_summary.json`、四张 alignment × reference reduction PDF，以及独立的 `exp/cache/text_trajectories` 缓存与 run-local manifest。

**意义。** 展示算子输出相对 exact soft embedding 和真实 TextMAS receiver-side token embedding 的几何偏移；不同 reference 的降维独立拟合，坐标不可跨图直接比较。它不是 CoT trajectory 图，也不能表明 hidden state 有可读的语言语义。

## 4. 同模型 Latent CoT：C0

**目标。** 只绘制 latent CoT 中 hidden state 经该同模型 $W_{\mathrm{out}}$ 解码后 entropy 随 latent time step 的变化曲线；C0 的其他分析留空。

**运行定义。** 使用同一个模型完成 question prompt 的 latent rollout。设第 $t$ 个 latent recurrence 进入下一步前的末层 hidden 为 $h_t$，则仅计算

$$
p_t=\operatorname{softmax}(W_{\mathrm{out}}h_t+b),
\qquad
H_t=-\sum_i p_{t,i}\log p_{t,i}.
$$

对每题保存 $t\in\{0,\ldots,K-1\}$ 的 entropy；横轴为 latent time step，纵轴为 $H_t$（nats）。每个 step 先在题目层面保留一条值，再报告题间 mean/median 与 95% cluster bootstrap CI；同时保存原始逐题数据。若某 rollout 提前停止或某 step 非有限，则明确记录有效题目数和失败原因，不能用后续步填补。

**对照与固定项。** 使用 greedy 或固定 seed 的生成配置，并将 model revision、prompt template、数据集 split、$K$、tokenizer、$W_{\mathrm{out}}$ 是否含 bias 写入 manifest。该图不需要 $W_{\mathrm{in}}$、不使用 exact/kernel/linear 映射，也不将 argmax token 当成完整 CoT 文本。

**解释边界。** entropy 下降、上升或震荡只描述 token distribution 的尖锐程度随 recurrence 的变化，不能单独说明推理更正确、更有信息量，或存在唯一可读的 latent thought。

## 5. Latent Communication：M0

**目标。** 前一个 agent A 看到 problem 并完成其 Latent CoT；将 A 的最终 latent message 传给下一个 agent B。B 不接收题目文本，只根据自身固定 role prompt 与该 latent message 直接回答，统计答案准确率。M0 的其他分析留空。

**运行定义。** 对每个题目 $x$：

1. A 接收 $x$ 并运行固定 $K$ 步 Latent CoT，取最终 raw hidden $h_K^A$；
2. 用仓库真实 `transfer_via_realignment` 路径把 $h_K^A$ 注入 B 的 input embedding/cache；
3. B 的 prompt 中不得包含题目、答案、选项、A 的文本回复或任何由题目直接派生的文本；B 直接 greedy 生成答案；
4. 以对应数据集的既有答案解析器评分，并按题目保存 A/B model、message position、$K$、生成长度、解析是否成功、预测与 correctness。

**必要对照。** 至少包含：B 无消息（相同无题目 prompt）、B + 随机配对的 A message、B + 错配 A message、B + 正确配对 A message。所有条件对同一题配对，固定 B 的生成配置；报告 accuracy、相对无消息条件的 paired accuracy difference，以及题目 bootstrap 95% CI。若 B 在无题目条件下因数据/role 泄漏仍能稳定回答，或 prompt 含有题目相关文本，该设置无效，停止并修正。

**解释边界。** 正确配对消息优于无消息、随机配对和错配消息，才可作为“A 的信息经 latent communication 被 B 利用”的证据。它不证明 kernel 近似保真，也不等价于同模型 Latent CoT 成功；本 M0 使用真实仓库传输路径而不是 `exp/approximator` 的 Refiner-to-Judger exact/kernel 消融。

## 6. 交付物与停止条件

每次 S 运行保存 `run_manifest.json`、轨迹 manifest、Parquet 原始指标、JSON 汇总、PDF 图及图的 artifact context。C0/M0 新实现也应采用同样的 run-local 目录、完整参数 manifest 与逐题原始结果，避免只保存聚合曲线或 accuracy。

停止规则：

1. tokenizer vocab/ID 不一致，或 mapping 的 exact audit、分母检查失败时，停止相应 S 运行；
2. C0 中无法取得每步真实 pre-unembedding hidden state 时，不以生成 token hidden 或文本替代；
3. M0 中 B 看到题目或其派生文本、A 的文本回答，或消息配对无法追溯时，不报告 communication accuracy；
4. 所有失败、跳过和无效题目必须在 manifest 与原始 metrics 中保留原因。
