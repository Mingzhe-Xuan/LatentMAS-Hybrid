# ⚡️ LatentMAS-Hybrid (Fork)

> **Note:** This is a fork of the original [LatentMAS](https://github.com/Gen-Verse/LatentMAS) repository.

## 🌟 Contribution: Heterogeneous Latent Communication

This fork extends the framework to support **Hybrid Multi-Agent Systems**, where agents can use **different model checkpoints** (from the same family) while still communicating via latent representations.

### Motivation
The original LatentMAS uses a single model backbone for all agents. While this demonstrates the efficiency of latent communication, it functionally resembles recursive Chain-of-Thought reasoning rather than true multi-agent collaboration. All agents share the same capabilities and limitations, which means the system cannot leverage the core advantage of multi-agent systems: **specialization**.

In a truly heterogeneous MAS, different agents can have different strengths. For example, a large, capable model (e.g., 7B parameters) might excel at high-level planning and reasoning, while a smaller, faster model (e.g., 1.5B parameters) could efficiently handle code generation or execution tasks. This division of labor could allow the system to be both performant and efficient, without forcing a single model size that either wastes compute on simple tasks or underperforms on complex ones.

This fork enables such heterogeneous collaboration by allowing agents to use different model (caveat: using the same tokenizer) while still communicating through latent representations.

### Limitations
**Performance vs. Efficiency Tradeoff:** The current implementation introduces overhead through cross-model alignment and re-encoding of context at each model switch. While the goal is to leverage specialized models for better task performance, the computational cost of model switching may offset gains from using smaller models. 

Key questions remain:
- Does the performance improvement from specialization justify the alignment overhead?
- Can the alignment process be optimized (e.g., cached alignments, learned lightweight adapters)?
- What is the optimal agent-to-model assignment strategy for different task types?

These questions require extensive benchmarking across diverse tasks and model combinations, which is compute-intensive and remains future work.

### Method: Cross-Model Alignment
To enable latent communication between different models without training adapters, we extend the linear alignment mechanism. We align Model A's output to Model B's input space via their shared vocabulary.

**The Math:**
For cross-model transfer, we solve for a transformation matrix that maps Model A's hidden states to Model B's input embeddings:

$$W_{cross} = (W_{out,A}^T W_{out,A} + \lambda I)^{-1} W_{out,A}^T W_{in,B}$$

This effectively maps the latent state $h_A$ to the embedding in Model B that corresponds to the same semantic concept.

### Usage
Use the `latent_mas_hybrid` method and specify `--agent_models`:

```bash
python run.py \
  --method latent_mas_hybrid \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --agent_models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --task gsm8k \
  --prompt sequential
```

---


## 🛠️ Getting Started

This repository provides all code for reproducing LatentMAS, TextMAS, and baseline single-agent experiments across GSM8K, AIME24/25, GPQA, ARC-Easy/Challenge, MBPP+, HumanEval+, and MedQA.

### ⚙️ Setup Environment Variables

We recommend setting your HF cache directory to avoid repeated downloads:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
````

Models and datasets will automatically be downloaded into `$HF_HOME`.


### 📦 Install Packages

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas

pip install -r requirements.txt
```

If you want **vLLM support**, also install:

```bash
pip install vllm
```

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Gen-Verse/LatentMAS.git
cd LatentMAS
```

### 2. Repository Structure

```
LatentMAS/
│── run.py                 # Main entry for experiments
│── models.py              # Wrapper for HF + vLLM + latent realignment
│── methods/
│   ├── baseline.py        # Single-agent baseline
│   ├── text_mas.py        # Token-space multi-agent method
│   └── latent_mas.py      # Latent-space multi-agent 
│   └── latent_mas_hybrid.py # Latent-space multi-heterogeneous-agent 
│── prompts.py             # Prompt constructors
│── data.py                # Dataset loaders
│── data/                  # Provided data + figures (We give medqa.json as an example here)
│── utils.py               # Answer parsing / timeout / helpers
│── example_logs/          # Example logs from LatentMAS
│── requirements.txt
```


## 🧪 Running Experiments (standard HF backend)

### 🔹 **Baseline (single model)**

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --max_new_tokens 2048
```


### 🔹 **TextMAS (text based multi-agent system)**

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```


### 🔹 **LatentMAS (our latent mas method)**

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```

#### Notes:

* **`--latent_steps`** ∈ [0, 80]
  Tune for best performance.
* **`--latent_space_realign`**
  Enables latent→embedding alignment
  We treat this as a **hyperparameter** — enable/disable depending on task/model:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --latent_space_realign --max_new_tokens 2048
```


## 📘 Example Logs

Two example LatentMAS logs are provided for reference purposes:

* `example_logs/qwen3_14b_mbppplus_sequential.txt`
* `example_logs/qwen3_14b_humanevalplus_hierarchical.txt`


Please refer to additional experiment logs [here](https://drive.google.com/drive/folders/1evGv5YAmLb4YM_D9Yu0ABa1nfqHC5N-l?usp=drive_link).
You can open them to view the full agent interaction traces and outputs.


## ⚡ vLLM Integration

LatentMAS supports vLLM for faster inference.

### 🔹 Baseline with vLLM

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 TextMAS with vLLM

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 LatentMAS with vLLM

LatentMAS supports a **hybrid HF + vLLM pipeline** for fast inference:
- vLLM handles **final text generation** (with prefix caching, tensor parallelism, etc.)
- A HuggingFace model handles **latent-space rollout** and hidden-state alignment

For this setup, we recommend using two GPUs:
- One GPU for vLLM (`--device`, e.g., `cuda:0`)
- One GPU for the auxiliary HF model (`--device2`, e.g., `cuda:1`)

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048 \
  --use_vllm \
  --use_second_HF_model \
  --enable_prefix_caching \
  --device2 cuda:1
```

**📍Important Note:**

> vLLM does **not** officially support modifying KV-cache or prompting via latent embeddings.
> We modify the partial inner package inside vLLM backend for our method implementation.
> Note minor numeric differences may arise compared to offical HF backend due to different decoding (generation) strategies. Please Use the HF backend to reproduce the official published results.

## 🌐 Awesome Works based on LatentMAS

1. KNN-LatentMAS: [Blog](https://bookmaster9.github.io/kNN-latentMAS/) and [Code](https://github.com/Bookmaster9/kNN-latentMAS).

## 📚 Citation

💫 If you find **LatentMAS** helpful, please kindly give us a star ⭐️ and cite below. Thanks!

```
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## 🤝 Ackowledgement 

This code is partially based on the amazing work of [vLLM](https://github.com/vllm-project/vllm).
