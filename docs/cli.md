# 实验命令

本文档分别说明主任务 `run.sh` 系列和解析实验 `exp.sh` 系列。除特别说明外，命令均在仓库根目录执行。

> PBS 环境变量使用 `qsub -v "A=x,B=y" script.sh`，变量列表前不要加 `--`。

## 1. `run.sh` 系列：主任务与端到端评估

`run.sh` 的完整 suite 包含：

- `baseline`：sequential、hierarchical；
- `text_mas`：sequential、hierarchical；
- `latent_mas`：sequential/hierarchical × identical/linear/kernel。

### 1.1 全量实验

在 `params_dict.json` 的全部 9 个任务和 Qwen3-4B/8B/14B 上提交全矩阵：

```bash
qsub -v "FULL_EXP=true" run.sh
```

更推荐按数据集拆成独立 PBS 作业。下面的提交脚本会为全部数据集分别调用 `qsub`：

```bash
bash run_all.sh
```

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

### 1.2 method 消融

一个标准 `run.sh` suite 已经自动包含 identical、linear、kernel，并同时包含 baseline 与 TextMAS，因此完整 method 消融无需分别提交：

```bash
qsub -v "TASK=arc_easy,MODEL_NAME=Qwen/Qwen3-8B" run.sh
```

若需要把每个 alignment 作为独立命令运行，可直接调用 `run.py`：

```bash
for method in identical linear kernel; do
  python3 run.py --method latent_mas --align_method "${method}" --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --latent_steps 10 --kernel_features 1024 --kernel_temperature 1.0 --kernel_seed 42 --kernel_chunk_size 4096 --seed 42 --trust_remote_code
done
```

baseline、TextMAS 和 LatentMAS 的总体方法消融：

```bash
python3 run.py --method baseline --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --seed 42 --trust_remote_code
python3 run.py --method text_mas --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --seed 42 --trust_remote_code
python3 run.py --method latent_mas --align_method kernel --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --latent_steps 10 --kernel_features 1024 --seed 42 --trust_remote_code
```

### 1.3 dataset 消融

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

### 1.4 max_samples 消融

注意：当前 `run.sh` 写死了 `MAX_SAMPLES=30`，所以 `qsub -v MAX_SAMPLES=... run.sh` 不会生效。请直接调用 `run.py`：

```bash
for max_samples in 10 30 50 100; do
  python3 run.py --method latent_mas --align_method kernel --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples "${max_samples}" --split test --latent_steps 10 --kernel_features 1024 --seed 42 --trust_remote_code
done
```

`--max_samples -1` 表示使用该 split 的全部样本。

### 1.5 kernel_features 消融

注意：当前 `run.sh` 写死了 `KERNEL_FEATURES=1024`，所以 `qsub -v KERNEL_FEATURES=... run.sh` 不会生效。请直接调用 `run.py`：

```bash
for kernel_features in 256 512 1024 2048 4096; do
  python3 run.py --method latent_mas --align_method kernel --model_name Qwen/Qwen3-8B --task arc_easy --prompt sequential --max_samples 30 --split test --latent_steps 10 --kernel_features "${kernel_features}" --kernel_temperature 1.0 --kernel_seed 42 --kernel_chunk_size 4096 --seed 42 --trust_remote_code
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

C0：自动运行 GSM8K 和 MBPP+，每个数据集均比较 identical、linear、kernel：

```bash
qsub -v "EXP_TARGET=latent_cot,DATASET=all,SPLIT=test,MODEL_NAME=Qwen/Qwen3-4B,MAX_QUESTIONS=50,LATENT_STEPS=50,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
```

默认配置可简写为：

```bash
qsub -v "EXP_TARGET=latent_cot" exp.sh
```

### 2.2 method 消融

C0 的一个作业会自动运行 identical、linear、kernel 三条独立 recurrence；当前没有单 method 参数：

```bash
qsub -v "EXP_TARGET=latent_cot,DATASET=all,MAX_QUESTIONS=50,M=2048,TAU=1.0,ORF_SEED=101,PROBE_SEED=42" exp.sh
```

结果表中的 `alignment` 列区分三种方法，两幅数据集子图也会分别绘制三条曲线。

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

C0 支持 `gsm8k`、`mbppplus`，也支持 `all` 联合双子图：

```bash
for dataset in gsm8k mbppplus; do
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

例如 `DATASET=all,MAX_QUESTIONS=50` 表示最多抽取 50 个 GSM8K 样本和 50 个 MBPP+ 样本。

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

实验输出分别写入 `result/`、`logging/` 和 `exp_result/`；PBS 运行日志分别记录到 `state*.txt` 与 `exp_state.txt`。
