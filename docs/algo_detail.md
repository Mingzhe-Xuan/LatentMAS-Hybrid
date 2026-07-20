# 核化快速近似：逐步推导

本文展开 \`algo.md\` 中的“核化的快速近似”。目标是近似精确映射

$$
F(h_L^A)=C\,\operatorname{softmax}\!\left(W_{\mathrm{out}}^Aq/\tau+b^A\right),
\qquad
q=\operatorname{Norm}_A(h_L^A),
$$

并将每个 query 的计算从扫描 A 的全词表，改为只依赖随机特征维度 $m$ 的计算。

设 $W_{\mathrm{out}}^A$ 的第 $i$ 行为 $w_i^\top$，且

$$
C=W_{\mathrm{in}}^B T_{A\to B}\in\mathbb R^{d_B\times V_A},
\qquad
c_i=C_{:,i}\in\mathbb R^{d_B}.
$$

本文中的 $c_i$ 是经过词表迁移后对应的 B embedding-space value；核化近似的是 A 的 softmax 加权聚合过程。

## 1. 将精确映射改写为“分子 / 分母”

第 $i$ 个 A token 的 logit 和未归一化权重为

$$
\ell_i(q)=\frac{w_i^\top q}{\tau}+b_i^A,
\qquad
a_i(q)=e^{\ell_i(q)}
=e^{b_i^A}\exp\!\left(w_i^\top\frac q\tau\right).
$$

定义 softmax 的配分函数

$$
Z(q)=\sum_{i=1}^{V_A}a_i(q).
$$

则

$$
p_i^A(q)=\frac{a_i(q)}{Z(q)}.
$$

精确映射可写为

$$
\begin{aligned}
F(q)
&=\sum_{i=1}^{V_A}p_i^A(q)c_i\\
&=\frac{\sum_{i=1}^{V_A}c_i a_i(q)}
{\sum_{i=1}^{V_A}a_i(q)}\\
&=
\frac{
\sum_{i=1}^{V_A}c_i e^{b_i^A}
\exp\!\left(w_i^\top q/\tau\right)}
{
\sum_{i=1}^{V_A}e^{b_i^A}
\exp\!\left(w_i^\top q/\tau\right)}.
\end{aligned}
$$

分子是 $d_B$ 维向量，分母是正标量。精确计算需要对所有 $i=1,\ldots,V_A$ 求 $w_i^\top q$，主成本至少为

$$
O(V_A d_A),
$$

并且分子聚合还要随 $V_A$ 线性增长。

## 2. 将 softmax 中的指数点积看作核

令

$$
x=\frac q\tau,\qquad y_i=w_i.
$$

所有与 query 有关的非线性都来自

$$
k(x,y_i)=\exp(x^\top y_i).
$$

这是 exponential dot-product kernel。若存在正特征映射

$$
\phi:\mathbb R^{d_A}\to\mathbb R_+^m
$$

满足

$$
\exp(x^\top y)\approx\phi(x)^\top\phi(y),
$$

则不必逐个计算 $x^\top y_i$。这里要求特征逐元素非负，是为了让近似的未归一化权重和归一化分母也保持非负。

## 3. 正交随机特征的构造与无偏性

本节以 `orth.md` 为准，直接定义本文使用的随机特征。目标核为

$$
k(x,y)=\exp(x^\top y),
$$

离线预计算和在线推理均采用

$$
\phi_{\mathrm{orth}},
$$

；相对于独立采样方向构成的 $\phi$，它不改变核化映射的代数形式，只改变用于估计核的随机方向的联合采样分布。

### 3.1 单个正交块的方向与长度

令隐状态/输出头维度为

$$
d=d_A.
$$

一个正交块至多含有 $d$ 个特征方向。先采样高斯矩阵

$$
G\in\mathbb R^{d\times d},
\qquad G_{ab}\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
$$

并作 QR 分解

$$
G=QR.
$$

采用带固定符号约定的 QR 分解，使 $Q$ 服从 Haar 均匀的正交矩阵分布。于是

$$
Q\in\mathbb R^{d\times d},
\qquad Q^\top Q=QQ^\top=I_d.
$$

记 $q_r=Q_{:,r}$ 为第 $r$ 列，则

$$
q_r^\top q_s=\delta_{rs}.
$$

因此 $\{q_r\}_{r=1}^{d}$ 是球面上的两两正交单位方向。

只使用单位方向会改变随机向量的边缘分布，因此还必须恢复高斯半径。标准高斯向量具有极坐标分解

$$
g=\rho u,
\qquad
u\sim\operatorname{Unif}(\mathbb S^{d-1}),
\qquad
\rho\sim\chi_d,
$$

且 $u$ 和 $\rho$ 独立。故对每个方向采样独立长度

$$
\rho_r\overset{\mathrm{iid}}{\sim}\chi_d,
$$

并定义正交随机方向

$$
\boxed{
\omega_r^{\mathrm{orth}}=\rho_rq_r,
\qquad r=1,\ldots,d.
}
$$

实践中，也可取 QR 分解前原高斯矩阵对应列的长度：若 $g_r=G_{:,r}$，则

$$
\rho_r=\lVert g_r\rVert_2.
$$

关键性质是每个方向的边缘分布仍保持标准高斯：

$$
\omega_r^{\mathrm{orth}}\overset{d}{=}\mathcal N(0,I_d).
$$

但是不同 $r$ 的向量不独立；它们的方向被强制正交。这种相关性是降低方差而非引入偏差的来源。

### 3.2 多块构造

若特征数 $m>d$，单组方向无法全部正交。写成

$$
m=Bd+r,
\qquad 0\le r<d.
$$

对每个完整块 $b=1,\ldots,B$ 独立生成 $Q^{(b)}$、$\{\rho_s^{(b)}\}_{s=1}^{d}$，并构成行方向矩阵

$$
\Omega^{(b)}=
\operatorname{Diag}(\rho_1^{(b)},\ldots,\rho_d^{(b)})
(Q^{(b)})^\top
\in\mathbb R^{d\times d}.
$$

第 $b$ 块中第 $s$ 行是

$$
(\omega_{b,s}^{\mathrm{orth}})^\top
=\rho_s^{(b)}(q_s^{(b)})^\top.
$$

最后一个部分块仅保留另一个独立块的前 $r$ 行，并沿行拼接：

$$
\Omega=
\begin{bmatrix}
\Omega^{(1)}\\
\vdots\\
\Omega^{(B)}\\
\Omega^{(\mathrm{partial})}
\end{bmatrix}
\in\mathbb R^{m\times d}.
$$

块内方向两两正交，块间相互独立。若 $m\le d$，仅需一块并取其前 $m$ 个方向。

### 3.3 用 ORF 定义正特征

令 $\omega_r^{\mathrm{orth}\top}$ 为 $\Omega$ 的第 $r$ 行，定义

$$
\boxed{
\phi_{\mathrm{orth}}(x)=
\frac1{\sqrt m}
\left[
\exp\!\left(
(\omega_r^{\mathrm{orth}})^\top x
-\frac{\lVert x\rVert_2^2}{2}
\right)
\right]_{r=1}^{m}.
}
$$

由于每个 $\omega_r^{\mathrm{orth}}$ 的边缘仍是 $\mathcal N(0,I_d)$，对任意固定 $x,y$ 有

$$
\mathbb E\left[
\exp\!\left(
(\omega_r^{\mathrm{orth}})^\top(x+y)
-\frac{\lVert x\rVert_2^2+\lVert y\rVert_2^2}{2}
\right)
\right]
=\exp(x^\top y).
$$

即使各项并不独立，线性期望仍给出

$$
\mathbb E\left[
\phi_{\mathrm{orth}}(x)^\top
\phi_{\mathrm{orth}}(y)
\right]
=\exp(x^\top y).
$$

因此 ORF 保留了普通正随机特征对未归一化 softmax kernel 的无偏性。

设单项为

$$
Z_r=
\exp\!\left(
(\omega_r^{\mathrm{orth}})^\top(x+y)
-\frac{\lVert x\rVert_2^2+\lVert y\rVert_2^2}{2}
\right).
$$

则估计量方差为

$$
\operatorname{Var}\left(\frac1m\sum_{r=1}^{m}Z_r\right)
=
\frac1{m^2}
\left(
\sum_r\operatorname{Var}(Z_r)
+\sum_{r\ne s}\operatorname{Cov}(Z_r,Z_s)
\right).
$$

ORF 通过球面方向的均匀覆盖避免相近方向重复抽样，通常降低第二项造成的总体方差。这里应表述为实践上及相关理论设置中的方差降低趋势，而非对所有 $x,y,m$ 无条件声称严格更小。

### 3.4 ORF 下的核化映射

因此，本文所有实际特征计算使用

$$
u=\phi_{\mathrm{orth}}(q/\tau),
$$

并在离线阶段构造

$$
S=
\sum_{i=1}^{V_A}
 c_i e^{b_i^A}
\phi_{\mathrm{orth}}(w_i)^\top,
$$

$$
z=
\sum_{i=1}^{V_A}
 e^{b_i^A}
\phi_{\mathrm{orth}}(w_i).
$$

最终在线映射为

$$
\boxed{
\hat h_0^B=
\frac{S\phi_{\mathrm{orth}}(q/\tau)}
{z^\top\phi_{\mathrm{orth}}(q/\tau)}.
}
$$

除随机方向的离线采样由 iid 高斯改为分块 ORF 外，$S,z$ 的预聚合方式、在线复杂度

$$
O\bigl(m(d_A+d_B)\bigr),
$$

以及温度、bias 和数值稳定性处理均不变。应保存生成的 $\Omega$（或生成它的随机种子及 QR/半径约定），使 query 特征与所有预计算 key 特征严格使用同一组 ORF。

## 4. 将核近似代入每个 softmax 权重

对在线 query 定义

$$
u=\phi_{\mathrm{orth}}(q/\tau)\in\mathbb R_+^m,
$$

并对每个固定词表 key 定义

$$
k_i=\phi_{\mathrm{orth}}(w_i)\in\mathbb R_+^m.
$$

于是

$$
\exp(w_i^\top q/\tau)\approx k_i^\top u,
$$

从而

$$
a_i(q)\approx
\hat a_i(q)=e^{b_i^A}k_i^\top u.
$$

直接代入精确式，得到

$$
\hat F(q)=
\frac{\sum_{i=1}^{V_A}c_i e^{b_i^A}k_i^\top u}
{\sum_{i=1}^{V_A}e^{b_i^A}k_i^\top u}.
$$

这一步只完成了核替换；若仍按 $i$ 求和，在线仍需扫描全词表。关键在于 $w_i,c_i,b_i^A$ 都是固定的，只有 $u$ 随 query 变化。

## 5. 将分子的词表求和移到离线阶段

令近似分子为

$$
N(q)=\sum_{i=1}^{V_A}c_i e^{b_i^A}k_i^\top u.
$$

因为

$$
c_i e^{b_i^A}k_i^\top\in\mathbb R^{d_B\times m},
$$

可利用矩阵乘法的分配律：

$$
\begin{aligned}
N(q)
&=\sum_{i=1}^{V_A}
\left(c_i e^{b_i^A}k_i^\top\right)u\\
&=\left(
\sum_{i=1}^{V_A}c_i e^{b_i^A}k_i^\top
\right)u.
\end{aligned}
$$

预先定义固定矩阵

$$
\boxed{
S=\sum_{i=1}^{V_A}
c_i e^{b_i^A}\phi_{\mathrm{orth}}(w_i)^\top
\in\mathbb R^{d_B\times m}.
}
$$

则在线分子变成一次矩阵向量乘法：

$$
N(q)=Su.
$$

$S$ 汇总了全部 A 词表 key、输出 bias 和映射到 B embedding 空间的 value。

## 6. 将分母的词表求和移到离线阶段

近似配分函数为

$$
\hat Z(q)=
\sum_{i=1}^{V_A}e^{b_i^A}k_i^\top u.
$$

同样地，

$$
\begin{aligned}
\hat Z(q)
&=
\left(
\sum_{i=1}^{V_A}e^{b_i^A}k_i
\right)^\top u.
\end{aligned}
$$

定义固定向量

$$
\boxed{
z=\sum_{i=1}^{V_A}e^{b_i^A}\phi_{\mathrm{orth}}(w_i)
\in\mathbb R^m.
}
$$

于是

$$
\hat Z(q)=z^\top u.
$$

由于 $e^{b_i^A}>0$、$\phi_{\mathrm{orth}}(w_i)\ge0$ 与 $u\ge0$，理论上

$$
z^\top u>0.
$$

这保证近似仍可作为非负权重的归一化平均理解。

## 7. 得到最终快速公式

把分子和分母结果合并：

$$
\boxed{
\hat h_0^B=\hat F(q)
=\frac{Su}{z^\top u},
\qquad
u=\phi_{\mathrm{orth}}(q/\tau).
}
$$

维度核对如下：

$$
\begin{array}{c|c}
\text{量} & \text{维度}\\
\hline
q,w_i & d_A\\
u,\phi_{\mathrm{orth}}(w_i),z & m\\
S & d_B\times m\\
Su & d_B\\
z^\top u & 1\\
\hat h_0^B & d_B
\end{array}
$$

因此输出正好属于 B 的 token embedding 空间。之后若 B 的原始输入流程还包含 position embedding、embedding scale 或 LayerNorm，应继续照 B 的既有流程处理。

## 8. 离线、在线计算步骤

### 8.1 离线预计算

固定模型参数、词表迁移矩阵、随机特征参数和温度后：

1. 对每个 A token 取得 $w_i$、$b_i^A$ 与 $c_i=C_{:,i}$；
2. 计算 $k_i=\phi_{\mathrm{orth}}(w_i)$、$\alpha_i=e^{b_i^A}$；
3. 累加

$$
S\leftarrow S+c_i\alpha_i k_i^\top,
\qquad
z\leftarrow z+\alpha_i k_i.
$$

初始值为

$$
S=0_{d_B\times m},
\qquad z=0_m.
$$

可按词表 block 累加，避免物化完整的 $C$：

$$
S\leftarrow S+\sum_{i\in I_b}c_i\alpha_i k_i^\top,
\qquad
z\leftarrow z+\sum_{i\in I_b}\alpha_i k_i.
$$

若 $T_{A\to B}$ 的每列是稀疏的，也可直接由

$$
c_i=W_{\mathrm{in}}^B T_{:,i}
$$

计算当前列，而无需存储 $C$。

### 8.2 在线推理

给定新的 $h_L^A$：

1. $q\leftarrow\operatorname{Norm}_A(h_L^A)$；
2. $u\leftarrow\phi_{\mathrm{orth}}(q/\tau)$；
3. $n\leftarrow Su$；
4. $d\leftarrow z^\top u$；
5. 返回 $n/d$。

即

$$
h_L^A\longmapsto q\longmapsto u
\longmapsto\frac{Su}{z^\top u}.
$$

在线路径没有长度为 $V_A$ 的 logits、概率向量或词表扫描。

## 9. 复杂度与存储变化

精确式的单 query 主成本为

$$
O(V_A d_A)+O(V_A d_B).
$$

前一项是输出头 logits，后一项是对 value 的加权聚合。核化后，在线成本为

$$
\underbrace{O(md_A)}_{\phi_{\mathrm{orth}}(q/\tau)}
+\underbrace{O(md_B)}_{Su}
+\underbrace{O(m)}_{z^\top u}
=O\!\left(m(d_A+d_B)\right).
$$

需要保存的主要统计量为

$$
S\in\mathbb R^{d_B\times m},
\qquad z\in\mathbb R^m,
$$

即 $O(d_Bm)$ 个标量。加速有效的前提通常是

$$
m\ll V_A.
$$

代价被转移到离线阶段：必须遍历词表构造 $S,z$，其随机特征计算约为 $O(V_Amd_A)$。当同一组统计量服务于大量 online query 时，这一成本可被摊销。

## 10. 温度与 bias 不能遗漏

这里的 $\tau$ 是该映射及其核近似的**超参数**，用于控制从 A 输出 logits 导出的 soft token 分布的尖锐程度；它不必等于模型在文本生成接口中使用的 temperature。后者通常只在需要从分布中采样离散 token 时参与采样，而本文的映射直接传递条件期望 embedding，不进行 sampling。若系统同时存在生成采样，应将 $\tau$ 与 top-$k$、top-$p$、典型采样、重复惩罚等策略作为一个联合设计来调节，并以实际通信目标和下游 B 的表现为准。

温度在核中作用于 query：

$$
u=\phi_{\mathrm{orth}}(q/\tau).
$$

不能把它替换为对最终 $u$ 的简单常数缩放，因为一般地

$$
\exp(w_i^\top q/\tau)
\ne\gamma(\tau)\exp(w_i^\top q).
$$

bias 是每个 token 的独立权重

$$
\alpha_i=e^{b_i^A},
$$

必须同时进入 $S$ 和 $z$。漏掉它会近似另一个目标：

$$
\operatorname{softmax}(W_{\mathrm{out}}^Aq/\tau),
$$

而不是原来的

$$
\operatorname{softmax}(W_{\mathrm{out}}^Aq/\tau+b^A).
$$

若对所有 logits 同时加常数 $\beta$，则 $S,z$ 同时乘以 $e^\beta$，最终输出不变：

$$
\frac{(e^\beta S)u}{(e^\beta z)^\top u}
=\frac{Su}{z^\top u}.
$$

## 11. 数值稳定性

正特征中要计算

$$
\omega_r^\top x-\frac{\lVert x\rVert_2^2}{2}.
$$

这类指数可能上溢或下溢。对某次在线计算，将所有 query 特征同乘 $e^{-s}$ 不会改变输出：

$$
\frac{S(e^{-s}u)}{z^\top(e^{-s}u)}
=\frac{Su}{z^\top u}.
$$

因此可在计算 $u$ 的 log-feature 后减去一个公共最大值再指数化。离线侧也应使用一致的特征定义和高精度累加；若对不同 block 使用不同缩放，必须恢复相应比例后再累加进 $S,z$。

理论上 $z^\top u$ 为正。实现中若

$$
z^\top u\le\varepsilon,
$$

通常应诊断为数值下溢、特征尺度异常或精度不足，而不是正常的 softmax 行为。

## 12. 与 proof.md 对齐的核估计误差与输出误差

令本文需近似的单核为

$$
k_i(q)=\exp(w_i^\top q/\tau),
\qquad
x=w_i,\quad y=q/\tau.
$$

因此 $k_i(q)=e^{x^\top y}$。以下 iid 结论对应 proof.md 的单核证明；随后说明当前 ORF 实现的区别。

### 12.1 单核无偏性与精确 iid 方差

令

$$
\omega_j\overset{\mathrm{iid}}{\sim}\mathcal N(0,I_{d_A}),
$$

$$
Z_j=\exp\!\left(
\omega_j^\top(x+y)
-\frac{\lVert x\rVert_2^2+\lVert y\rVert_2^2}{2}
\right),
\qquad
\widehat k_m=\frac1m\sum_{j=1}^{m}Z_j.
$$

由高斯矩母函数，

$$
\mathbb E[Z_j]
=
e^{-(\lVert x\rVert_2^2+\lVert y\rVert_2^2)/2}
e^{\lVert x+y\rVert_2^2/2}
=e^{x^\top y}=k_i(q).
$$

故 $\mathbb E[\widehat k_m]=k_i(q)$。计算二阶矩：

$$
\mathbb E[Z_j^2]
=
e^{-\lVert x\rVert_2^2-\lVert y\rVert_2^2}
e^{2\lVert x+y\rVert_2^2}
=
k_i(q)^2e^{\lVert x+y\rVert_2^2}.
$$

所以

$$
\operatorname{Var}(Z_j)
=
k_i(q)^2\left[e^{\lVert x+y\rVert_2^2}-1\right].
$$

iid 样本独立，因而

$$
\boxed{
\operatorname{Var}[\widehat k_m(w_i,q/\tau)]
=
\frac{e^{2w_i^\top q/\tau}}{m}
\left[e^{\lVert w_i+q/\tau\rVert_2^2}-1\right].
}
$$

相对均方误差为

$$
\boxed{
\mathbb E\left[
\left(\frac{\widehat k_m}{k_i(q)}-1\right)^2
\right]
=
\frac{e^{\lVert w_i+q/\tau\rVert_2^2}-1}{m}.
}
$$

### 12.2 Chebyshev 界与范数统一界

对任意 $\epsilon>0$，

$$
\Pr\left(
|\widehat k_m-k_i(q)|\ge\epsilon k_i(q)
\right)
\le
\frac{e^{\lVert w_i+q/\tau\rVert_2^2}-1}{m\epsilon^2}.
$$

故以概率至少 $1-\delta$ 获得相对误差不超过 $\epsilon$ 的充分条件是

$$
\boxed{
m\ge
\frac{e^{\lVert w_i+q/\tau\rVert_2^2}-1}
{\delta\epsilon^2}.
}
$$

若 $\lVert w_i\rVert_2\le R_w$、$\lVert q\rVert_2\le R_q$，则

$$
\left\lVert w_i+\frac q\tau\right\rVert_2
\le R_w+\frac{R_q}{\tau},
$$

从而

$$
\boxed{
\Pr\left(
\frac{|\widehat k_m-k_i(q)|}{k_i(q)}
\ge\epsilon
\right)
\le
\frac{\exp((R_w+R_q/\tau)^2)-1}{m\epsilon^2}.
}
$$

对全部 $V_A$ 个 key 采用并集界，将单项失败概率取为 $\delta/V_A$，可得充分条件

$$
m\ge
\frac{V_A[\exp((R_w+R_q/\tau)^2)-1]}
{\delta\epsilon^2}.
$$

这说明高范数、低温度和大词表都会显著增大所需特征数。

### 12.3 当前 ORF 的适用方式

当前算法使用正交随机特征，每一个

$$
\omega_r^{\mathrm{orth}}\overset d=\mathcal N(0,I_{d_A}),
$$

故仍有

$$
\mathbb E[\widehat k_m^{\mathrm{orth}}(x,y)]
=e^{x^\top y}.
$$

但同一正交块内的项不独立。因此第 12.1 节的精确 iid 方差式与第 12.2 节的 Chebyshev 界不能直接当作 ORF 的严格定理。ORF 的方差应写为

$$
\operatorname{Var}(\widehat k_m^{\mathrm{orth}})
=
\frac1{m^2}
\left[
\sum_{r=1}^{m}\operatorname{Var}(Z_r)
+
\sum_{r\ne s}\operatorname{Cov}(Z_r,Z_s)
\right].
$$

ORF 通过更均匀的方向覆盖通常降低总体方差；上述 iid 结论应作为严格基线和保守诊断。

### 12.4 从逐项相对误差到映射误差

若所有 token 满足

$$
(1-\eta)k_i(q)\le\widehat k_i(q)
\le(1+\eta)k_i(q),
\qquad0<\eta<1,
$$

则乘上 bias 后仍有

$$
(1-\eta)a_i(q)\le\widehat a_i(q)
\le(1+\eta)a_i(q).
$$

令 $\widehat a_i=a_ir_i$、$\bar r=\sum_ip_ir_i$，则

$$
\widehat p_i=\frac{p_ir_i}{\bar r},
\qquad
\lVert\widehat p-p\rVert_1
\le\frac{2\eta}{1-\eta}.
$$

令

$$
D_C=\max_{i,r}\lVert c_i-c_r\rVert_2,
$$

即得

$$
\boxed{
\lVert\widehat F(q)-F(q)\rVert_2
\le D_C\frac{\eta}{1-\eta}.
}
$$

这完成从单核估计误差、softmax 分布误差到 B embedding-space 映射误差的证明链。

## 13. 适用边界

这一预聚合之所以成立，是因为

$$
w_i,\ c_i,\ b_i^A
$$

均与 query 无关，query 只通过 $u=\phi_{\mathrm{orth}}(q/\tau)$ 出现。若使用条件化的 $T(q)$、条件化 value，或其他使 $c_i$ 随 query 改变的机制，通常不能复用一组静态 $S,z$。

最终，原本的全词表 softmax 加权平均被替换为

$$
\boxed{
C\,\operatorname{softmax}\!\left(W_{\mathrm{out}}^Aq/\tau+b^A\right)
\approx
\frac{S\phi_{\mathrm{orth}}(q/\tau)}{z^\top\phi_{\mathrm{orth}}(q/\tau)}.
}
$$

词表相关工作被放在离线阶段，在线只保留特征计算、一次矩阵向量乘法和一次标量归一化。

## 14. 词表迁移矩阵 $T_{A\to B}$：详细构造算法

本节展开 `algo.md` 中的词表迁移矩阵。该矩阵处于 soft-token communication 路线：它将 A 的 token 概率分布转换成 B 的 token 概率分布，随后再由 B 的输入 embedding 合成为 B 的输入向量。它不是严格的纯 latent 映射。

### 14.1 输入、输出与概率质量约束

令 A、B 的词表大小分别为 $V_A,V_B$。迁移矩阵的形状为

$$
T=T_{A\to B}\in\mathbb R_+^{V_B\times V_A}.
$$

列索引 $i$ 对应一个 A token，行索引 $j$ 对应一个 B token。$T_{ji}$ 的含义是：在 A token 为 $i$ 的条件下，将其概率质量分配给 B token $j$ 的比例。因此每一列都必须是一个概率分布：

$$
T_{ji}\ge0,
\qquad
\sum_{j=1}^{V_B}T_{ji}=1
\qquad(i=1,\ldots,V_A).
$$

即

$$
\mathbf1_{V_B}^\top T=\mathbf1_{V_A}^\top.
$$

给定任意 A 分布 $p^A\in\Delta^{V_A-1}$，令

$$
p^B=Tp^A.
$$

则 $p^B$ 非负，且

$$
\mathbf1_{V_B}^\top p^B
=\mathbf1_{V_B}^\top Tp^A
=\mathbf1_{V_A}^\top p^A=1.
$$

故 $p^B\in\Delta^{V_B-1}$，概率质量严格守恒。进一步，精确通信映射可以按两步理解：

$$
p^A\xrightarrow{\ T\ }p^B\xrightarrow{\ W_{\mathrm{in}}^B\ } h_0^B,
$$

即

$$
h_0^B=W_{\mathrm{in}}^BTp^A.
$$

对单个 A token $i$，对应的 B embedding-space value 是

$$
c_i=W_{\mathrm{in}}^BT_{:,i}
=\sum_{j=1}^{V_B}T_{ji}(W_{\mathrm{in}}^B)_{:,j}.
$$

所以 $c_i$ 是 B token embeddings 的凸组合。

### 14.2 以语义特征定义运输代价

OT 不能直接从 token ID 得知语义相似性，需要先给 A、B token 建立可比较的特征。令

$$
a_i\in\mathbb R^r,
\qquad
b_j\in\mathbb R^r
$$

分别是 A token $i$ 和 B token $j$ 的特征。它们可由 token 字符串、字节/字符片段、词典释义、经锚点对齐的静态 embedding，或共同语料上的上下文表示构造。必须确保二者确实位于可比较的特征空间；未经对齐的两个模型 embedding 通常不能直接作欧氏距离比较。

常见的代价定义为

$$
D_{ji}=d(b_j,a_i).
$$

例如，对归一化特征，可取 cosine distance：

$$
D_{ji}=1-\frac{b_j^\top a_i}{\lVert b_j\rVert_2\lVert a_i\rVert_2+\eta};
$$

对同一坐标系中的特征，也可取平方欧氏距离：

$$
D_{ji}=\lVert b_j-a_i\rVert_2^2.
$$

代价越小，表示将 A token $i$ 的质量分给 B token $j$ 越合理。对特殊 token、空白前缀、控制符和 tokenizer 的边界标记，宜额外设定规则或较高的错配代价，防止它们被语义近邻错误吸收。

### 14.3 设定 OT 的两侧先验

取严格为正的词表先验

$$
\mu\in\Delta^{V_A-1},
\qquad
\nu\in\Delta^{V_B-1},
\qquad
\mu_i>0,\ \nu_j>0.
$$

$\mu_i$ 是 A token $i$ 的参考质量，$\nu_j$ 是 B token $j$ 的目标质量。它们可以来自同一代表性语料上的 token frequency；没有可靠语料时，可用平滑后的均匀先验。严格正性是后续除以 $\mu_i$ 的必要条件。

先验在此并非 online 分布 $p^A$。它们只用于离线构造一个全局耦合，使迁移矩阵在参考分布下满足

$$
T\mu=\nu.
$$

对于任意实际 query 的 $p^A$，仍使用 $p^B=Tp^A$，并不要求其边缘恰为 $\nu$。

### 14.4 熵正则最优传输问题

求耦合矩阵 $\Pi\in\mathbb R_+^{V_B\times V_A}$：

$$
\begin{aligned}
\Pi^*=arg\min_{\Pi\ge0}\quad
&\langle D,\Pi\rangle
+\varepsilon\sum_{j=1}^{V_B}\sum_{i=1}^{V_A}
\Pi_{ji}(\log\Pi_{ji}-1)\\
\text{s.t.}\quad
&\Pi\mathbf1_{V_A}=\nu,\\
&\Pi^\top\mathbf1_{V_B}=\mu.
\end{aligned}
$$

其中

$$
\langle D,\Pi\rangle=\sum_{j,i}D_{ji}\Pi_{ji}.
$$

约束分别表示 B 侧行边缘和 A 侧列边缘。熵正则系数 $\varepsilon>0$ 控制两种倾向：较小的 $\varepsilon$ 更接近最小运输代价、但耦合更尖锐且数值更困难；较大的 $\varepsilon$ 更平滑、更容易求解、但可能将质量分散到语义较远的 token。

注意 $\Pi$ 不是迁移条件概率。其列和是 $\mu_i$，而不是 $1$：

$$
\sum_j\Pi_{ji}^*=\mu_i.
$$

### 14.5 从 OT 耦合得到列随机迁移矩阵

对每个 A token 条件化，即按列边缘归一化：

$$
\boxed{
T=\Pi^*\operatorname{Diag}(\mu)^{-1},
\qquad
T_{ji}=\frac{\Pi_{ji}^*}{\mu_i}.
}
$$

列随机性可直接验证：

$$
\sum_{j=1}^{V_B}T_{ji}
=\frac{1}{\mu_i}\sum_{j=1}^{V_B}\Pi_{ji}^*
=\frac{\mu_i}{\mu_i}=1.
$$

同时，OT 的 B 边缘可写为

$$
T\mu
=\Pi^*\operatorname{Diag}(\mu)^{-1}\mu
=\Pi^*\mathbf1_{V_A}
=\nu.
$$

因此 $T$ 是以 A token 为条件的软对齐规则，而 $\Pi^*$ 是在参考先验下的联合分布：

$$
\Pi_{ji}^*=P(B=j,A=i),
\qquad
T_{ji}=P(B=j\mid A=i).
$$

### 14.6 Sinkhorn 求解的逐步形式

定义 Gibbs kernel

$$
K_{ji}=\exp\left(-\frac{D_{ji}}{\varepsilon}\right).
$$

熵正则 OT 的解可写为

$$
\Pi^*=\operatorname{Diag}(r)K\operatorname{Diag}(s),
$$

其中 $r\in\mathbb R_+^{V_B}$、$s\in\mathbb R_+^{V_A}$ 是缩放向量。将行、列边缘代入：

$$
r\odot(Ks)=\nu,
qquad
s\odot(K^\top r)=\mu.
$$

由此得到 Sinkhorn 迭代：

$$
r^{(t+1)}=\nu\oslash(Ks^{(t)}),
$$

$$
s^{(t+1)}=\mu\oslash(K^\top r^{(t+1)}),
$$

其中 $\odot$、$\oslash$ 分别表示逐元素乘法与除法。初始化可取

$$
s^{(0)}=\mathbf1_{V_A}.
$$

迭代至边缘残差足够小：

$$
\lVert\Pi^{(t)}\mathbf1-\nu\rVert_1\le\mathrm{tol},
\qquad
\lVert(\Pi^{(t)})^\top\mathbf1-\mu\rVert_1\le\mathrm{tol},
$$

其中

$$
\Pi^{(t)}=\operatorname{Diag}(r^{(t)})K\operatorname{Diag}(s^{(t)}).
$$

当 $D/\varepsilon$ 数值较大时，直接形成 $K$ 可能下溢；应采用 log-domain Sinkhorn，使用 log-sum-exp 完成缩放更新。对于超大词表，完整 $V_B\times V_A$ 代价矩阵本身不可承受，通常要先为每个 A token 检索少量 B 候选，或使用块计算、低秩/核化 OT 等实现。

### 14.7 稀疏化：每列 top-$k$ 后重新归一化

稠密 $T$ 的存储为 $O(V_AV_B)$，且构造

$$
C=W_{\mathrm{in}}^BT
$$

成本很高。对每一列 $i$，令 $J_i$ 是 $T_{:,i}$ 中质量最大的 $k$ 个行索引。保留并重新归一化：

$$
\tilde T_{ji}=
\begin{cases}
\displaystyle\frac{T_{ji}}{\sum_{r\in J_i}T_{ri}}, & j\in J_i,\\
0, & j\notin J_i.
\end{cases}
$$

于是仍有

$$
\tilde T_{ji}\ge0,
\qquad
\sum_j\tilde T_{ji}=1.
$$

令被截去的列质量为

$$
\rho_i=1-\sum_{j\in J_i}T_{ji}.
$$

则该列的总变差距离为

$$
\operatorname{TV}(T_{:,i},\tilde T_{:,i})=\rho_i.
$$

因此 $k$ 不应只按内存选定，也应通过 $\rho_i$ 的分位数、最大值及下游误差决定。稀疏后，单列 value 可高效计算：

$$
\tilde c_i
=\sum_{j\in J_i}\tilde T_{ji}(W_{\mathrm{in}}^B)_{:,j},
$$

成本从访问 $V_B$ 个 embedding 降为 $O(kd_B)$。

### 14.8 完整离线算法

给定 A 输出词表、B 输入词表、特征构造函数、先验 $\mu,\nu$、正则系数 $\varepsilon$ 和可选稀疏度 $k$：

1. 构造 token 特征 $\{a_i\}_{i=1}^{V_A}$、$\{b_j\}_{j=1}^{V_B}$；
2. 构造代价 $D_{ji}=d(b_j,a_i)$，并对特殊 token 施加必要约束；
3. 使 $\mu,\nu$ 非负、归一化且（为条件化安全起见）每项为正；
4. 由 $K_{ji}=e^{-D_{ji}/\varepsilon}$ 和 Sinkhorn 迭代求 $\Pi^*$；
5. 计算 $T=\Pi^*\operatorname{Diag}(\mu)^{-1}$；
6. 可选：逐列 top-$k$ 稀疏化并重新归一化，得到 $\tilde T$；
7. 用最终矩阵构造 $C=W_{\mathrm{in}}^BT$，或按列即时计算 $c_i$；
8. 再以这些 $c_i$ 构造核化近似所需的 $S,z$。

第 8 步说明了两类离线工作之间的依赖：先确定 $T$，才可确定 $C$ 的 value 列，最后才能做第 5 节的词表预聚合。

### 14.9 必要验证与边界条件

至少检查：

$$
\min_{j,i}T_{ji}\ge-\delta_{\mathrm{num}},
$$

$$
\max_i\left|\sum_jT_{ji}-1\right|\le\delta_{\mathrm{col}},
$$

以及对随机测试分布 $p^A$：

$$
\left|\mathbf1^\top Tp^A-1\right|\le\delta_{\mathrm{mass}},
\qquad
\min_j(Tp^A)_j\ge-\delta_{\mathrm{num}}.
$$

还应检查 OT 边缘：

$$
\lVert T\mu-\nu\rVert_1,
$$

并人工抽样检查每列 top 候选是否具有合理语义和 tokenizer 边界。若 A、B 使用完全相同的 tokenizer 和 token ID，则无需 OT：

$$
T=I_{V_A}.
$$

若 token 集合相同但 ID 排列不同，则令 $P$ 为相应置换矩阵并取

$$
T=P.
$$

最后，$T$ 的质量与核近似误差是两个不同问题：前者决定 soft-token 通信本身是否语义合理；后者只衡量在给定 $T$ 时，$\hat F$ 对精确 $F$ 的计算近似程度。
