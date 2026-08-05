# 实验命令

本文档分别说明主任务 `run.sh` 系列和解析实验 `exp.sh` 系列。除特别说明外，命令均在仓库根目录执行。

> PBS 环境变量使用 `qsub -v "A=x,B=y" script.sh`，变量列表前不要加 `--`。

## 1. `run.sh` 系列：主任务与端到端评估

`run.sh` 的完整 suite 包含：

- `baseline`：sequential、hierarchical；
- `text_mas`：sequential、hierarchical；
- `latent_mas`：sequential/hierarchical × identical/linear/kernel/soft。

### 1.1 全量实验

`run_all.sh` 是完整矩阵的推荐入口。Qwen3-8B 和 Qwen3-14B 各运行全部 9 个 dataset；Qwen3-4B 只运行 6 个 dataset，跳过 `aime2024`、`aime2025`、`gpqa`。因此共有 `(9 + 9 + 6) × 12 = 288` 个数组子任务。每个子任务只运行一个配置，并由 `#PBS -J 1-288%3` 将全局并发上限设为 3。

数组按模型分段，顺序为：

```text
1–108:   Qwen3-8B，全部 9 个 dataset
109–216: Qwen3-14B，全部 9 个 dataset
217–288: Qwen3-4B，仅 arc_challenge、arc_easy、gsm8k、humanevalplus、mbppplus、medqa
```

提交方式：

```bash
bash run_all.sh
# 或直接提交
qsub run_all.sh
```

若希望完全排除耗时较长的 `aime2024`、`aime2025`、`gpqa`，使用快速数组入口：

```bash
bash run_all_fast.sh
# 或
qsub run_all_fast.sh
```

`run_all_fast.sh` 对 8B、14B、4B 都只运行 `arc_challenge`、`arc_easy`、`gsm8k`、`humanevalplus`、`mbppplus`、`medqa`，因此数组为 `6 × 3 × 12 = 216` 个子任务，PBS 指令是 `#PBS -J 1-216%3`。它复用同一套 `state/` 配置日志和根目录 `state.txt` 进度账本，也支持：

```bash
bash run_all_fast.sh --force_all
qsub -v "FORCE_ALL=true" run_all_fast.sh
```

12 个配置固定为：

```text
baseline   × sequential/hierarchical × identical
text_mas   × sequential/hierarchical × identical
latent_mas × sequential/hierarchical × identical/linear/kernel/soft
```

不要额外传 `-J`，否则会覆盖脚本内的数组范围或并发限制。也可以让单个 `run.sh` 作业串行执行相同的过滤后矩阵，但它不是 288-task 独立排队方式：

```bash
qsub -v "FULL_EXP=true" run.sh
```

### 1.2 状态文件、跳过与总进度

每个配置的完整 stdout/stderr 平铺写入根目录 `state/`，不创建 dataset、model 或 method 子目录：

```text
state/<dataset>_<method>_<prompt>_<model>_state.txt
```

模型名中的 `/` 会转换为 `_`。LatentMAS 把 alignment 合并到 method，例如：

```text
state/arc_easy_latent_mas_kernel_sequential_Qwen_Qwen3-4B_state.txt
```

默认只检查当前配置的精确状态文件；存在时只跳过该配置。旧文件 `state_<dataset>.txt` 不参与判断。强制忽略已有配置状态文件：


```bash
bash run_all.sh --force_all
qsub -v "FORCE_ALL=true" run_all.sh
```

根目录 `state.txt` 是并发安全的追加式 TSV 账本，通过 `flock` 防止三个子任务互相覆盖。字段依次为 `timestamp`、`job_id`、`array_index`、`dataset`、`method`、`prompt`、`alignment`、`model`、`status`、`detail`；状态为 `STARTED`、`SKIPPED`、`COMPLETED`、`FAILED`。常用查看方式：

```bash
column -t -s $'\t' state.txt | less -S
tail -n 30 state.txt
awk -F '\t' 'NR > 1 { count[$9]++ } END { for (s in count) print s, count[s] }' state.txt
```

method、prompt 或 alignment 非法时会在创建配置状态文件前退出；`baseline` 和 `text_mas` 只允许 `identical`，`latent_mas` 允许 `identical`、`linear`、`kernel`、`soft`。

### 1.3 单任务与单配置入口

当前数据集脚本为：

```text
run_aime2024.sh       run_aime2025.sh
run_arc_challenge.sh  run_arc_easy.sh
run_gpqa.sh           run_gsm8k.sh
run_humanevalplus.sh  run_mbppplus.sh
run_medqa.sh
```

例如，只在 ARC-Easy 上运行全部模型和完整 method suite：

```bash
qsub run_arc_easy.sh
```

只运行 `run.sh` 当前默认配置：

```bash
qsub run.sh
```

该入口默认写入 `state/run_state.txt`，不会覆盖数组任务使用的根进度账本 `state.txt`；可用 `STATE_FILE` 覆盖其日志路径。

### 1.4 method 消融

一个标准 `run.sh` suite 已经自动包含 identical、linear、kernel、soft，并同时包含 baseline 与 TextMAS，因此完整 method 消融无需分别提交：

```bash
qsub -v "TASK=arc_easy,MODEL_NAME=Qwen/Qwen3-8B" run.sh
```

若需要把每个 alignment 作为独立命令运行，可直接调用 `run.py`：

```bash
for method in identical linear kernel soft; do
  python3 run.py --method latent_mas --align_method "${method}" --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --latent_steps 10 --kernel_features 1024 --kernel_temperature 1.0 --kernel_seed 42 --kernel_chunk_size 4096 --soft_temperature 1.0 --soft_chunk_size 32 --seed 42 --trust_remote_code
done
```

baseline、TextMAS 和 LatentMAS 的总体方法消融：

```bash
python3 run.py --method baseline --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --seed 42 --trust_remote_code
python3 run.py --method text_mas --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --seed 42 --trust_remote_code
python3 run.py --method latent_mas --align_method kernel --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --latent_steps 10 --kernel_features 1024 --seed 42 --trust_remote_code
```

### 1.5 dataset 消融

使用已有 PBS 包装脚本逐数据集提交：

```bash
for script in run_aime2024.sh run_aime2025.sh run_arc_challenge.sh run_arc_easy.sh run_gpqa.sh run_gsm8k.sh run_humanevalplus.sh run_mbppplus.sh run_medqa.sh; do
  qsub "${script}"
done
```

也可以固定单一模型，通过 `run.sh` 逐任务提交：

```bash
for dataset in aime2024 aime2025 arc_challenge arc_easy gpqa gsm8k humanevalplus mbppplus medqa; do
  qsub -v "TASK=${dataset},MODEL_NAME=Qwen/Qwen3-8B,FULL_EXP=false" run.sh
done
```

### 1.6 max_samples 消融

`run.sh` 和 `run_all.sh` 都允许通过 PBS 变量覆盖 `MAX_SAMPLES`。例如分别提交完整矩阵：

```bash
for max_samples in 10 30 50 100; do
  qsub -v "MAX_SAMPLES=${max_samples},FORCE_ALL=true" run_all.sh
done
```

`--max_samples -1` 表示使用该 split 的全部样本。

### 1.7 kernel_features 消融

`KERNEL_FEATURES` 同样可传入 `run.sh` 或 `run_all.sh`：

```bash
for kernel_features in 256 512 1024 2048 4096; do
  qsub -v "KERNEL_FEATURES=${kernel_features},FORCE_ALL=true" run_all.sh
done
```

## 2. `exp.sh` 系列：解析实验

`exp.sh` 当前可运行 `approximator` 和 `latent_cot`。`latent_comm` 分支仍在脚本中，但当前仓库没有 `exp/latent_comm/run.py`，不要提交该 target。

### 2.1 全量实验

Approximator：在 ARC-Easy 上运行 S0–S4：

```bash
qsub -v "EXP_TARGET=approximator,STUDY=all,DATASET=arc_easy,SPLIT=test,AGENT_MODELS=Qwen/Qwen3-4B,MAX_QUESTIONS=50,MAX_STATES_PER_QUESTION=50,M=2048,TAU=1.0,ORF_SEED=101,LATENT_STEPS=50,PROBE_SEED=42" exp.sh
```

默认配置可简写为：

```bash
qsub -v "EXP_TARGET=approximator" exp.sh
```

C0：自动运行 GSM8K、MBPP+、ARC-Challenge 和 AIME 2025，每个数据集均比较 identical、linear、soft、kernel、text。前三个数据集使用 `test` split；AIME 2025 自动使用其 `train` split：

```bash
qsub -v "EXP_TARGET=latent_cot,DATASET=all,SPLIT=test,MODEL_NAME=Qwen/Qwen3-4B,MAX_QUESTIONS=50,LATENT_STEPS=100,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
```

默认配置可简写为：

```bash
qsub -v "EXP_TARGET=latent_cot" exp.sh
```

### 2.2 method 消融

C0 的一个作业会自动运行 identical、linear、soft、kernel、text 五条独立 recurrence；当前没有单 method 参数：

```bash
qsub -v "EXP_TARGET=latent_cot,DATASET=all,MAX_QUESTIONS=50,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
```

结果表中的 `alignment` 列区分五种方法，四幅数据集子图也会分别绘制五条曲线。其中 `soft` 是完整 softmax 期望 embedding，不进行 token sampling 或 argmax；`kernel` 近似的正是该映射。

Approximator 没有 `--method` 参数；method 相关分析由 study 决定。S4 对比 linear 与 kernel，并使用 hidden/exact 作为参照：

```bash
qsub -v "EXP_TARGET=approximator,STUDY=s4,DATASET=arc_easy,MAX_QUESTIONS=50,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
```

S1、S2、S3 的核近似分析可分别提交：

```bash
for study in s1 s2 s3; do
  qsub -v "EXP_TARGET=approximator,STUDY=${study},DATASET=arc_easy,MAX_QUESTIONS=50,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
done
```

不要给 approximator 或 latent_cot 设置 `METHOD=...`：当前两个 Python 入口都不读取该变量。

### 2.3 dataset 消融

Approximator 支持 `arc_easy`、`arc_challenge`、`gsm8k`、`medqa`、`mbppplus`、`gpqa`：

```bash
for dataset in arc_easy arc_challenge gsm8k medqa mbppplus gpqa; do
  qsub -v "EXP_TARGET=approximator,STUDY=all,DATASET=${dataset},SPLIT=test,MAX_QUESTIONS=50,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
done
```

C0 支持 `gsm8k`、`mbppplus`、`arc_challenge`、`aime2025`，也支持 `all` 联合 2×2 子图：

```bash
for dataset in gsm8k mbppplus arc_challenge aime2025; do
  qsub -v "EXP_TARGET=latent_cot,DATASET=${dataset},SPLIT=test,MAX_QUESTIONS=50,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
done
```

### 2.4 max_samples 消融

`exp.sh` 中对应变量名是 `MAX_QUESTIONS`，并且表示每个数据集的最大样本数：

```bash
for max_samples in 10 25 50 100; do
  qsub -v "EXP_TARGET=approximator,STUDY=all,DATASET=arc_easy,MAX_QUESTIONS=${max_samples},M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
done
```

```bash
for max_samples in 10 25 50 100; do
  qsub -v "EXP_TARGET=latent_cot,DATASET=all,MAX_QUESTIONS=${max_samples},M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
done
```

例如 `DATASET=all,MAX_QUESTIONS=50` 表示每个数据集最多抽取 50 个样本；AIME 2025 只有 30 题，因此会使用全部可用题目。

### 2.5 kernel_features 消融

PBS 变量 `M` 对应 Python 参数 `--kernel_features`：

```bash
for kernel_features in 256 512 1024 2048 4096; do
  qsub -v "EXP_TARGET=approximator,STUDY=all,DATASET=arc_easy,MAX_QUESTIONS=50,M=${kernel_features},TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
done
```

```bash
for kernel_features in 256 512 1024 2048 4096; do
  qsub -v "EXP_TARGET=latent_cot,DATASET=all,MAX_QUESTIONS=50,M=${kernel_features},TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
done
```

增大 `M` 会增加 kernel alignment 的构建时间和显存占用。不同 `M`、`TAU`、`ORF_SEED` 和 chunk size 会进入缓存身份，不会错误复用同一轨迹。

## 3. 快速冒烟测试

主任务：

```bash
python3 run.py --method latent_mas --align_method kernel --model_name Qwen/Qwen3-4B --task arc_easy --prompt sequential --max_samples 2 --split test --latent_steps 2 --kernel_features 256 --seed 42 --trust_remote_code
```

解析实验：

```bash
qsub -v "EXP_TARGET=latent_cot,DATASET=gsm8k,MAX_QUESTIONS=2,LATENT_STEPS=3,M=256" exp.sh
```

实验输出分别写入 `result/`、`logging/` 和 `exp_result/`；主任务的配置日志位于 `state/`、总进度位于根目录 `state.txt`，解析实验日志位于 `exp_state.txt`。
