# Sparse OT 加速方法：前后方案与区别

## 1. 结论

当前改动是**更换加速方法**，而不是只把原方法改成增量更新。

- 之前的方法：标准 log-domain Sinkhorn 热启动，然后在当前点的**增量变量**上运行 marginal-scaled L-BFGS-B，最后回到标准 Sinkhorn。
- 当前的方法：标准 log-domain Sinkhorn 热启动，然后运行 residual-driven marginal-scaled Newton-CG，并对 Newton 步执行边际残差回溯，最后仍可回到标准 Sinkhorn。

两者求解的是同一个候选图、同一个 Gibbs kernel 和同一个熵正则 OT 问题；变化的是对偶变量的加速求解器与步长接受机制，不是运输目标本身。

## 2. 共同的 OT 问题

设稀疏候选边集合为 \(E\)，source marginal 为 \(a\)，target marginal 为 \(b\)，边代价为 \(C_{ij}\)，熵正则系数为 \(\varepsilon\)。固定 kernel 为

\[
K_{ij}=\exp(-C_{ij}/\varepsilon),\qquad (i,j)\in E.
\]

用行、列对偶变量 \(\alpha_i,\beta_j\) 表示 coupling：

\[
\Pi_{ij}=K_{ij}\exp(\alpha_i+\beta_j).
\]

需要满足

\[
\Pi\mathbf 1=b,
\qquad
\Pi^{\mathsf T}\mathbf 1=a.
\]

实现以原始两侧 L1 边际残差验收：

\[
r_{\text{row}}=\lVert\Pi\mathbf 1-b\rVert_1,
\qquad
r_{\text{col}}=\lVert\Pi^{\mathsf T}\mathbf 1-a\rVert_1,
\qquad
r=\max(r_{\text{row}},r_{\text{col}}).
\]

只有 \(r\leq\text{tolerance}\) 才算收敛。精度现在按场景分层：toy/dense oracle、有限差分和小图算法回归继续使用 `1e-9` 或原有更严格阈值；真实 2.3M-edge full-vocabulary 构建使用 `2e-3`。这是一项明确的需求调整，不应把 Job 234 描述为在旧 `1e-9` 标准下通过。

两种方案也都先执行标准 log-domain Sinkhorn：交替更新行缩放 \(\alpha\) 和列缩放 \(\beta\)。这样可以快速消除大部分误差，并给后续加速器提供数值较合理的热启动点。

## 3. 之前的方法：增量 scaled L-BFGS-B

### 3.1 为什么叫“增量”

旧方案不会直接让 L-BFGS-B 优化绝对对偶变量。它先冻结当前 Sinkhorn 点 \(x_0=[\alpha,\beta]\)，再优化增量 \(\Delta x\)：

\[
x=x_0+\Delta x.
\]

这样做是为了避免绝对对偶变量很大、正负项严重相消时，对偶目标值在 float64 精度下不可分辨。相对目标使用 `expm1` 计算当前 coupling 相对基点的质量变化：

\[
\phi_{x_0}(\Delta x)
=\sum_{(i,j)\in E}
\Pi^{(0)}_{ij}\left[\exp(\Delta\alpha_i+\Delta\beta_j)-1\right]
-b^{\mathsf T}\Delta\alpha-a^{\mathsf T}\Delta\beta.
\]

其梯度就是 gauge-fixed 的行列边际误差。因此，“增量更新”描述的是旧方案的变量表示和重启方式，并不是一种独立于 L-BFGS-B 的新 OT 方法。

### 3.2 边际缩放

令

\[
D=\operatorname{diag}(\sqrt{b},\sqrt{a})
\]

（固定一个列对偶变量以消除 gauge 后，删去对应维度）。旧方案优化缩放增量

\[
z=D\Delta x,
\qquad
\Delta x=D^{-1}z.
\]

这使稀有 token 与高频 token 的局部尺度更接近，避免极小 marginal 对应的梯度被欧氏尺度淹没。

### 3.3 L-BFGS-B 如何产生更新

L-BFGS-B 用最近少量梯度/变量差分近似逆 Hessian，再由基于目标函数值的 line search 选择步长。旧实现限制 history（默认 `maxcor=3`）和目标函数 evaluation 预算，以控制内存和总计算量。

一次加速结束后，外层重新计算原始两侧 L1 residual。只有候选的 \(r\) 严格小于加速前的 \(r\)，才接受整个 L-BFGS-B 候选；否则丢弃候选，继续标准 Sinkhorn，并在之后进行有界重启。

### 3.4 真实图上暴露的问题

虽然增量目标和边际缩放修复了部分数值问题，但 L-BFGS-B 的内部 line search/termination 仍依赖目标函数值。当目标下降小于 float64 可分辨精度时，它会以 `RELATIVE REDUCTION OF F <= FACTR*EPSMCH` 提前返回，即使严格边际残差仍明显不合格。

已有真实 Slurm 结果包括：

- Job 230：运行约 40 分 50 秒，加速器只做 27 次 evaluation，最终 row residual 为 `4.66e-4`。
- Job 232：21 次加速尝试共耗尽 1,000 次 evaluation，运行约 39 分 06 秒，最终 row residual 为 `1.69e-3`。
- Job 233：即使设置 `ftol=0`，结果仍与 Job 232 相同，继续触发 FACTR 精度平台。

因此，继续增加 L-BFGS-B 的重启次数或关闭 `ftol` 并不能解决根本问题。

## 4. 当前方法：residual-driven scaled Newton-CG

### 4.1 直接使用梯度和 Hessian-vector product

令 gauge-fixed 对偶变量为 \(x\)，边到变量的关联算子为 \(A\)。对偶梯度和 Hessian 可写为

\[
g=A^{\mathsf T}\Pi-m,
\qquad
H=A^{\mathsf T}\operatorname{diag}(\Pi)A,
\]

其中 \(m\) 是拼接后的目标行列 marginal。

当前方法不显式构造稠密 Hessian，而只计算

\[
Hv=A^{\mathsf T}\left(\Pi\odot(Av)\right).
\]

该 Hessian-vector product 只需沿稀疏边扫描，因此内存不会增长为词表维度的平方。

### 4.2 缩放坐标与预条件 CG

仍使用同一个边际尺度矩阵 \(D\)，但当前方法直接在缩放坐标中求 Newton 方向：

\[
\widetilde H=D^{-1}HD^{-1},
\qquad
\widetilde g=D^{-1}g,
\qquad
\widetilde H p=-\widetilde g.
\]

线性系统由预条件共轭梯度（CG）近似求解。预条件器使用当前 row/column mass 形成的 Hessian 对角近似；收敛附近它在缩放坐标中接近 1。默认每次 Newton 方向最多执行 32 次 CG 迭代。

得到缩放方向 \(p\) 后，原变量方向为

\[
d=D^{-1}p.
\]

### 4.3 以残差而不是目标函数值决定步长

当前方法依次尝试

\[
t\in\{1,1/2,1/4,\ldots\},
\]

最多 12 个回溯候选，并通过

\[
\Pi(t)=\Pi\odot\exp(tAd)
\]

直接计算候选行列边际。只有

\[
r(t)<r(0)
\]

才接受该步。

因此，Newton-CG 方向来自同一个对偶目标的二阶信息，但实际步长的接受与停止不依赖微小的目标函数值差，而是直接围绕最终验收指标——原始两侧 L1 residual——进行。

### 4.4 有界预算和失败语义

CG 的每次 Hessian-vector product 与每个回溯候选检查都计入同一个 acceleration evaluation 预算；默认总加速预算仍为 1,000，并且与标准 Sinkhorn 共享 `max_iter=10_000` 总预算。非有限梯度、非有限方向、预算耗尽或没有残差改善时都会显式记录 termination provenance，再由外层决定继续标准 Sinkhorn 或最终严格失败。

## 5. 关键区别

| 维度 | 之前：增量 scaled L-BFGS-B | 当前：residual-driven scaled Newton-CG |
|---|---|---|
| OT 目标与 kernel | 不变 | 不变 |
| 标准 Sinkhorn 热启动 | 有 | 有 |
| 优化变量 | 当前点上的缩放增量 | 当前点上的缩放 Newton 方向 |
| 曲率信息 | 用有限 history 近似逆 Hessian | 用精确 Hessian-vector product，CG 近似解 Newton 方程 |
| 内部步长依据 | 对偶目标函数值 line search | 原始两侧 L1 residual 回溯 |
| 候选接受 | L-BFGS-B 完成后由外层检查一次残差改善 | 每个回溯步直接检查残差并只接受严格改善 |
| 主要停止风险 | 目标值浮点平台触发 FACTR 提前返回 | CG/回溯预算耗尽或找不到残差改善方向 |
| 主要参数 | `acceleration_history_size=3` | `acceleration_cg_iterations=32` |
| 加速预算记账 | 主要记录目标/梯度 evaluation | HVP 和回溯候选共同显式记账 |
| 方法标识 | `sinkhorn-scaled-lbfgs-sinkhorn` | `sinkhorn-scaled-newton-cg-sinkhorn` |

## 6. 计算量和拟合时间

设候选边数为 \(|E|\)，活跃行列变量总数为 \(n\)。

- 一次标准 Sinkhorn 行列 sweep：时间约为 \(O(|E|+n)\)。
- 旧方法一次 L-BFGS-B 目标/梯度 evaluation：约为 \(O(|E|+n)\)。
- 新方法一次 Hessian-vector product：约为 \(O(|E|+n)\)；一次 residual 候选检查也是同阶。

新方法的一次 Newton 更新通常包含多次 CG HVP 和若干回溯检查，因此“单个 Newton 更新”比“一次 L-BFGS evaluation”更贵。但两者都受显式 evaluation 预算限制，而且真正决定总时间的是达到严格 residual 所需的稀疏边扫描总数。

所以不能仅凭代码更复杂就断言总拟合时间一定更长：

- 如果 Newton 方向能快速降低病态稀有边上的残差，总扫描次数可能减少，总时间可能更短。
- 如果 CG 经常达到迭代上限或回溯多次，常数开销会增大，总时间也可能更长。
- 旧方法约 39–41 分钟的真实运行时间不能当作当前方法的测速结果，因为那些 Job 实际运行的是 L-BFGS-B 路径。

## 7. 当前验证状态

当前 Newton-CG 实现已通过本地 Hessian 有限差分、CG 参数/预算、病态稀疏图、facade/artifact/audit 回归以及完整本地测试。真实 Job 234 在 37:11.54 后报告 row/column residual `1.6915304665e-3`/`6.2669379185e-14`、MaxRSS 1,847,260 KiB、0 swap；该数值满足后来确认的 `2e-3` full-vocabulary 要求。

Job 234 的运行配置仍是旧 `1e-9`，因此进程以 Exit 1 结束，checkpoint 保持 `building/fresh`，没有 artifact/audit。当前结论是“数值精度满足新需求，但产物尚未验收”；必须以 `2e-3` 配置重新执行，使构建器正常原子保存并通过独立审计。小图 `1e-9` 数值测试、非有限值检查、可行支撑检查和 provenance 预算要求均不变。
