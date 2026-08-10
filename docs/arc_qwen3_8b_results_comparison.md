# 实验结果汇总对比

本文档汇总当前 `result` 目录所有子文件夹中的 `summary.json`。共收录 **115** 组结果；所有 split 均为 `test`。

指标定义：

- `Timing (total)`：`average.timing.total_seconds`，单位为秒，越低越好。
- `Accuracy`：`average.results.accuracy`，越高越好；括号内为原始小数值。
- `Text output tokens`：`average.results.tokens.text_output.total`，越低表示文本输出开销越少。
- 表格中的数值均来自 `average`；多次运行的结果可能带小数。
- 粗体表示同一任务、同一模型下的单项最优值。

## ARC Challenge

### Qwen/Qwen3-8B

样本数：1172；汇总组数：10；重复次数：1；seeds：42。

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

### Qwen/Qwen3-14B

样本数：1172；汇总组数：12；重复次数：1；seeds：42。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 5750.6135 | 93.8567% (0.938567) | 733,063 |
| Baseline | identical | sequential | 5714.9885 | 93.6007% (0.936007) | 722,640 |
| Latent MAS | identical | hierarchical | 4640.2323 | 95.6485% (0.956485) | 508,940 |
| Latent MAS | identical | sequential | 5019.7200 | 95.9898% (0.959898) | 543,042 |
| Latent MAS | kernel | hierarchical | 4435.0784 | 95.9044% (0.959044) | 505,172 |
| Latent MAS | kernel | sequential | 5292.5312 | 95.9044% (0.959044) | 557,068 |
| Latent MAS | linear | hierarchical | 4023.3889 | 94.1980% (0.941980) | 452,426 |
| Latent MAS | linear | sequential | 5867.9655 | 91.8089% (0.918089) | 545,409 |
| Latent MAS | soft | hierarchical | **3663.3481** | **96.1604% (0.961604)** | **401,696** |
| Latent MAS | soft | sequential | 4295.6182 | 95.7338% (0.957338) | 456,794 |
| Text MAS | identical | hierarchical | 26155.1683 | 95.6485% (0.956485) | 2,864,029 |
| Text MAS | identical | sequential | 15605.3501 | 95.7338% (0.957338) | 2,349,056 |

## ARC Easy

### Qwen/Qwen3-8B

样本数：2376；汇总组数：10；重复次数：1；seeds：42。

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

### Qwen/Qwen3-14B

样本数：2376；汇总组数：12；重复次数：1；seeds：42。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 8178.1512 | 97.9798% (0.979798) | 1,130,679 |
| Baseline | identical | sequential | 7955.2348 | 97.8535% (0.978535) | 1,153,916 |
| Latent MAS | identical | hierarchical | 6009.8087 | 98.6953% (0.986953) | 816,437 |
| Latent MAS | identical | sequential | 6568.5204 | 98.4848% (0.984848) | 900,323 |
| Latent MAS | kernel | hierarchical | 5901.7041 | 98.6953% (0.986953) | 808,146 |
| Latent MAS | kernel | sequential | 6394.6719 | 98.6953% (0.986953) | 889,990 |
| Latent MAS | linear | hierarchical | 5637.8405 | 98.0640% (0.980640) | 741,440 |
| Latent MAS | linear | sequential | 6693.9885 | 97.5589% (0.975589) | 895,331 |
| Latent MAS | soft | hierarchical | **4937.7954** | **98.7374% (0.987374)** | **657,581** |
| Latent MAS | soft | sequential | 5710.1207 | 98.4007% (0.984007) | 793,873 |
| Text MAS | identical | hierarchical | 36300.5634 | 98.3586% (0.983586) | 4,593,589 |
| Text MAS | identical | sequential | 25696.2812 | 98.5690% (0.985690) | 4,383,505 |

## GSM8K

### Qwen/Qwen3-8B

样本数：1319；汇总组数：10；重复次数：1；seeds：42。

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

### Qwen/Qwen3-14B

样本数：1319；汇总组数：12；重复次数：1；seeds：42。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 6823.2254 | 89.9924% (0.899924) | 1,030,779 |
| Baseline | identical | sequential | 6691.9661 | 90.3715% (0.903715) | 1,030,176 |
| Latent MAS | identical | hierarchical | 6098.9767 | 91.9636% (0.919636) | 756,606 |
| Latent MAS | identical | sequential | 6649.8624 | 92.1152% (0.921152) | 761,722 |
| Latent MAS | kernel | hierarchical | 6054.5927 | 92.1152% (0.921152) | 753,662 |
| Latent MAS | kernel | sequential | 6772.4794 | **93.1008% (0.931008)** | 763,101 |
| Latent MAS | linear | hierarchical | 8955.9452 | 86.7324% (0.867324) | 887,491 |
| Latent MAS | linear | sequential | 8181.6414 | 88.8552% (0.888552) | 789,031 |
| Latent MAS | soft | hierarchical | **6033.5254** | 91.5087% (0.915087) | 682,389 |
| Latent MAS | soft | sequential | 6124.0597 | 92.1911% (0.921911) | **679,853** |
| Text MAS | identical | hierarchical | 35410.3608 | 92.4943% (0.924943) | 3,792,489 |
| Text MAS | identical | sequential | 20254.0311 | 92.2669% (0.922669) | 2,552,889 |

## HumanEval+

### Qwen/Qwen3-8B

样本数：164；汇总组数：12；重复次数：4；seeds：42, 43, 44, 45。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 3148.1668 | 77.2866% (0.772866) | 361,720.75 |
| Baseline | identical | sequential | 3200.1302 | 76.9817% (0.769817) | 366,481.50 |
| Latent MAS | identical | hierarchical | 3314.5298 | 82.6219% (0.826219) | 276,880.75 |
| Latent MAS | identical | sequential | 3104.6432 | 84.1463% (0.841463) | 273,600.75 |
| Latent MAS | kernel | hierarchical | 3345.2881 | 84.2988% (0.842988) | 277,720.75 |
| Latent MAS | kernel | sequential | 3109.7878 | 85.2134% (0.852134) | 270,003 |
| Latent MAS | linear | hierarchical | 3698.6554 | 22.8659% (0.228659) | 281,938.75 |
| Latent MAS | linear | sequential | 3492.6929 | 18.2927% (0.182927) | 261,758 |
| Latent MAS | soft | hierarchical | 3010.4093 | 57.6219% (0.576219) | 192,776.50 |
| Latent MAS | soft | sequential | **2741.8782** | 81.5549% (0.815549) | **152,343.75** |
| Text MAS | identical | hierarchical | 13921.3377 | 85.5183% (0.855183) | 1,259,946.75 |
| Text MAS | identical | sequential | 8666.9324 | **89.6342% (0.896342)** | 669,085.25 |

### Qwen/Qwen3-14B

样本数：164；汇总组数：12；重复次数：4；seeds：42, 43, 44, 45。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | **3626.1177** | 81.5549% (0.815549) | 343,898 |
| Baseline | identical | sequential | 3683.1824 | 84.1463% (0.841463) | 339,744 |
| Latent MAS | identical | hierarchical | 3903.8169 | 88.4146% (0.884146) | 241,649.50 |
| Latent MAS | identical | sequential | 3956.5086 | 86.8902% (0.868902) | 260,819 |
| Latent MAS | kernel | hierarchical | 3866.0392 | 87.9573% (0.879573) | 233,565.25 |
| Latent MAS | kernel | sequential | 3895.0828 | 88.2622% (0.882622) | 255,827 |
| Latent MAS | linear | hierarchical | 5239.5834 | 19.5122% (0.195122) | 445,547.50 |
| Latent MAS | linear | sequential | 4804.6432 | 39.0244% (0.390244) | 341,322.50 |
| Latent MAS | soft | hierarchical | 3856.5704 | 88.8720% (0.888720) | **221,114** |
| Latent MAS | soft | sequential | 3692.7442 | 88.1097% (0.881097) | 240,093.50 |
| Text MAS | identical | hierarchical | 18272.6283 | 89.9390% (0.899390) | 1,188,630 |
| Text MAS | identical | sequential | 9396.9305 | **90.7012% (0.907012)** | 613,612.50 |

## MBPP+

### Qwen/Qwen3-8B

样本数：378；汇总组数：11；重复次数：4；seeds：42, 43, 44, 45。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 6550.2281 | 73.7434% (0.737434) | 722,646.75 |
| Baseline | identical | sequential | 6570.0886 | 74.2063% (0.742063) | 716,459.25 |
| Latent MAS | identical | hierarchical | 7676.6528 | 70.9656% (0.709656) | 542,623.75 |
| Latent MAS | kernel | hierarchical | 7424.3879 | 72.4868% (0.724868) | 564,454.25 |
| Latent MAS | kernel | sequential | 6998.8064 | 75.2645% (0.752645) | 553,177.50 |
| Latent MAS | linear | hierarchical | 7407.9916 | 17.5265% (0.175265) | 414,118.25 |
| Latent MAS | linear | sequential | 7276.2998 | 29.3651% (0.293651) | 491,330.25 |
| Latent MAS | soft | hierarchical | 7259.9011 | 67.1958% (0.671958) | 413,224.75 |
| Latent MAS | soft | sequential | **5116.7007** | 71.9577% (0.719577) | **262,830.25** |
| Text MAS | identical | hierarchical | 29595.6652 | 79.4312% (0.794312) | 2,342,089.50 |
| Text MAS | identical | sequential | 19895.1947 | **79.8941% (0.798941)** | 1,399,429 |

### Qwen/Qwen3-14B

样本数：378；汇总组数：2；重复次数：4；seeds：42, 43, 44, 45。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | **7681.7218** | **76.5873% (0.765873)** | 643,769.75 |
| Baseline | identical | sequential | 7846.1192 | 75.4630% (0.754630) | **643,176** |

## MedQA

### Qwen/Qwen3-8B

样本数：300；汇总组数：12；重复次数：1；seeds：42。

| Method | Align method | Topology | Timing (total, s) ↓ | Accuracy ↑ | Text output tokens ↓ |
|---|---|---|---:|---:|---:|
| Baseline | identical | hierarchical | 5161.2272 | 76.6667% (0.766667) | 559,782 |
| Baseline | identical | sequential | 5009.3528 | 75.3333% (0.753333) | 551,843 |
| Latent MAS | identical | hierarchical | 4660.1764 | 75.3333% (0.753333) | 417,209 |
| Latent MAS | identical | sequential | 5107.5381 | 76.3333% (0.763333) | 561,562 |
| Latent MAS | kernel | hierarchical | **4316.2145** | 71.0000% (0.710000) | 402,727 |
| Latent MAS | kernel | sequential | 5237.6816 | 76.3333% (0.763333) | 554,541 |
| Latent MAS | linear | hierarchical | 4725.4246 | 65.3333% (0.653333) | 387,280 |
| Latent MAS | linear | sequential | 5094.4982 | 75.0000% (0.750000) | 580,086 |
| Latent MAS | soft | hierarchical | 5245.4710 | 67.6667% (0.676667) | **374,115** |
| Latent MAS | soft | sequential | 5257.7842 | 76.6667% (0.766667) | 561,827 |
| Text MAS | identical | hierarchical | 19572.9969 | **78.6667% (0.786667)** | 1,922,939 |
| Text MAS | identical | sequential | 9187.7308 | 75.6667% (0.756667) | 1,020,685 |

数据快照日期：2026-08-10。仅收录实际存在且可解析的 `summary.json`。
