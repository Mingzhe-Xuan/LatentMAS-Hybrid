# Qwen3-8B 实验结果对比

本文档汇总指定实验目录中 `summary.json` 的平均结果。所有实验模型均为 `Qwen/Qwen3-8B`，split 均为 `test`。ARC 与 GSM8K 结果来自 1 次运行（seed 42）；HumanEval+ 结果为 4 次运行（seeds 42、43、44、45）的平均值。

指标定义：

- `Timing (total)`：`average.timing.total_seconds`，单位为秒，越低越好。
- `Accuracy`：`average.results.accuracy`，越高越好；括号内保留原始小数值。
- `Text output tokens`：`average.results.tokens.text_output.total`，越低表示文本输出开销越少。
- `Latent MAS` 的变体由 align method 区分；Baseline 和 Text MAS 的 align method 均为 `identical`。

## ARC Challenge

样本数：1,172。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 5064.7749 | 91.8089% (0.918089) | 906,338 |
| Baseline | identical | sequential | 5212.3415 | 91.4676% (0.914676) | 917,627 |
| Latent MAS | identical | hierarchical | 4579.7108 | 94.1980% (0.941980) | 709,781 |
| Latent MAS | identical | sequential | 4286.1966 | 94.5392% (0.945392) | 655,187 |
| Latent MAS | kernel | hierarchical | 4295.5560 | 94.9659% (0.949659) | 706,149 |
| Latent MAS | kernel | sequential | 4519.4786 | 94.7099% (0.947099) | 668,007 |
| Latent MAS | linear | hierarchical | 4187.1180 | 94.1980% (0.941980) | 633,886 |
| Latent MAS | linear | sequential | **4170.0498** | 92.9181% (0.929181) | **600,878** |
| Text MAS | identical | hierarchical | 21667.3036 | **95.6485% (0.956485)** | 3,209,203 |
| Text MAS | identical | sequential | 14028.8080 | 94.9659% (0.949659) | 2,713,264 |

## ARC Easy

样本数：2,376。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 7530.3345 | 97.5168% (0.975168) | 1,447,708 |
| Baseline | identical | sequential | 7483.7147 | 97.6852% (0.976852) | 1,453,956 |
| Latent MAS | identical | hierarchical | 6083.6464 | 98.4848% (0.984848) | 1,173,125 |
| Latent MAS | identical | sequential | 6031.3467 | 98.2744% (0.982744) | 1,112,327 |
| Latent MAS | kernel | hierarchical | 6086.8404 | **98.6953% (0.986953)** | 1,174,263 |
| Latent MAS | kernel | sequential | **5784.9140** | 98.4428% (0.984428) | 1,114,326 |
| Latent MAS | linear | hierarchical | 6349.6668 | 97.5589% (0.975589) | 1,069,841 |
| Latent MAS | linear | sequential | 5808.8872 | 97.6431% (0.976431) | **1,038,856** |
| Text MAS | identical | hierarchical | 33573.9832 | 98.4428% (0.984428) | 5,164,912 |
| Text MAS | identical | sequential | 23360.1524 | 98.5269% (0.985269) | 5,192,883 |

## GSM8K

样本数：1,319。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 6075.5844 | 88.6277% (0.886277) | 1,215,567 |
| Baseline | identical | sequential | 6179.5661 | 89.1585% (0.891585) | 1,202,688 |
| Latent MAS | identical | hierarchical | 5394.1932 | 91.5845% (0.915845) | 907,048 |
| Latent MAS | identical | sequential | 5473.4753 | 91.5845% (0.915845) | 828,472 |
| Latent MAS | kernel | hierarchical | 5562.1265 | 91.4329% (0.914329) | 914,685 |
| Latent MAS | kernel | sequential | 5565.9670 | 91.4329% (0.914329) | 843,169 |
| Latent MAS | linear | hierarchical | 5249.0986 | 91.2055% (0.912055) | 862,365 |
| Latent MAS | linear | sequential | **5179.9020** | 91.1296% (0.911296) | **794,843** |
| Text MAS | identical | hierarchical | 29302.2938 | 93.4799% (0.934799) | 4,164,489 |
| Text MAS | identical | sequential | 20874.6561 | **93.5557% (0.935557)** | 3,117,053 |

## HumanEval+

样本数：164；表中数据为 4 次运行的平均值。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 3148.1668 | 77.2866% (0.772866) | 361,720.75 |
| Baseline | identical | sequential | 3200.1302 | 76.9817% (0.769817) | 366,481.50 |
| Latent MAS | identical | hierarchical | 3314.5298 | 82.6219% (0.826219) | 276,880.75 |
| Latent MAS | identical | sequential | **3104.6432** | 84.1463% (0.841463) | 273,600.75 |
| Latent MAS | kernel | hierarchical | 3345.2881 | 84.2988% (0.842988) | 277,720.75 |
| Latent MAS | kernel | sequential | 3109.7878 | 85.2134% (0.852134) | 270,003.00 |
| Latent MAS | linear | hierarchical | 3698.6554 | 22.8659% (0.228659) | 281,938.75 |
| Latent MAS | linear | sequential | 3492.6929 | 18.2927% (0.182927) | **261,758.00** |
| Text MAS | identical | hierarchical | 13921.3377 | 85.5183% (0.855183) | 1,259,946.75 |
| Text MAS | identical | sequential | 8666.9324 | **89.6342% (0.896342)** | 669,085.25 |

表格中的粗体表示各数据集的单项最优值。
