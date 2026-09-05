# Kernel Analysis Implementation Plan

## 1. Goal and scope

Implement a standalone analysis framework under `analysis/` for five claims
about Kernel alignment:

1. entropy of probed logits;
2. Kernel scalability with the number of latent steps;
3. variance of Kernel-aligned states;
4. stability of latent communication under perturbations;
5. Sender/Receiver model-size performance for Qwen3-8B and Qwen3-14B across
   all nine repository datasets.

The main experiment uses one latent **Sender** and one text-generating
**Receiver**. A four-agent Planner/Critic/Refiner/Judger rollout is deliberately
excluded from the primary analysis because role transitions and accumulated
prompts would confound the properties of the alignment operator and prevent
valid prefix reuse.

Code under `analysis/` must not import modules under `exp/`. Production modules
such as `models.py`, `alignment.py`, `data.py`, `prompts.py`, and `utils.py` may
be reused. Existing `exp/` code listed in this document is a reference for
implementation patterns only.

Implementation must follow the workflow and development rules in
`analysis/AGENTS.md`.

## 2. Experimental protocol

### 2.1 Datasets

| Dataset | Split | Questions | Receiver generation repeats |
| --- | --- | ---: | ---: |
| AIME 2024 | `train` | all 30 | 3, seeds 42--44 |
| HumanEval+ | `test` | all 164 | 3, seeds 42--44 |
| ARC-Challenge | `test` | all test questions | 3, seeds 42--44 |

Dataset order and contents are fingerprinted before trajectory collection. The
same ordered questions are used by every downstream condition.

The repetition policy is global: every dataset uses exactly three Receiver
generation seeds, 42--44. Sender hidden trajectories remain deterministic and
are not recollected across these repeats.

### 2.2 Model and Kernel configuration

- Sender and Receiver model: `Qwen/Qwen3-8B`;
- Sender recurrence: Kernel alignment;
- maximum Sender latent steps: `Kmax=160`;
- Kernel features: `m=2048`;
- Kernel temperature: `tau=0.6`;
- Kernel seed used to generate the canonical Sender trajectory: `101`;
- Kernel chunk size: `4096`;
- generation temperature/top-p: task configuration, default `0.6/0.95`;
- Receiver always sees the original question;
- Sender trajectory is deterministic and does not depend on Receiver generation
  seed.

The role prompts are not reimplemented in `analysis/`. The Sender calls
`build_agent_message_sequential_latent_mas` with `role="planner"`; the Receiver
calls the same function with `role="judger"`. Both calls use
`method="latent_mas"` and an args/config object containing the effective task
and model name. The rendered messages and their SHA256 hashes are stored in the
corresponding cache manifests. Any change in prompt contents invalidates the
cache identity.

The Sender runs once to `Kmax` for each question and stores:

```text
h_1, h_2, ..., h_160
```

Throughout this document, a **hidden state** has one exact meaning: it is the
final-layer hidden vector at the current sequence position immediately before
the model readout/output head (`lm_head`). In the Hugging Face execution path
used by `run.sh`, this is exactly:

```python
outputs.hidden_states[-1][:, -1, :]
```

Equivalently, if `R` is the model readout layer, the probed logits are
`R(h_t) = W_out h_t + b`. A cached hidden state is never an input-embedding
vector, an intermediate-layer activation, a KV-cache tensor, logits, or an
already aligned vector. The indexing follows the recurrence used by
`run.sh -> run.py -> LatentMAS -> ModelWrapper.generate_latent_batch`: `h_0` is
the final prompt-position hidden state before the first latent feedback, and
stored `h_t` for `t >= 1` is the corresponding immediately-before-readout
hidden state after latent feedback step `t`. The canonical trajectory stores
`h_1, ..., h_Kmax`; `h_0` may be retained only as explicitly labelled
diagnostic metadata and is not part of a transmitted prefix.

This definition applies without exception to Sender trajectory caches,
Receiver transfer inputs before alignment, entropy probes, offline variance
analysis, perturbation inputs, and fields named `hidden` or `hidden_state` in
schemas and task interfaces. Implementations must capture it from the final
hidden-state output before applying the readout or any alignment operation.

Because the Receiver never changes the Sender state, every prefix is a valid
shorter Kernel trajectory:

```text
K = 10, 20, 40, 80, 160
```

This prefix property is the reason the two-agent design can reuse trajectories.

### 2.3 Receiver protocol

For a given question, the Receiver:

1. renders the task-specific `judger` prompt from `prompts.py`, containing the
   original question;
2. prefills that textual prompt;
3. appends the aligned Sender-state prefix in chronological order;
4. generates the final text answer;
5. evaluates the answer using the task-specific evaluator.

`K=0` skips step 3 and is the Receiver-only baseline. Different `K`, alignment,
or perturbation conditions require separate Receiver generation, but never a
new Sender trajectory.

The Sender prompt is always the task-specific `planner` prompt from
`prompts.py`. Custom analysis-only Planner or Judger wording is prohibited, so
the two-agent study remains comparable to the repository's main LatentMAS runs.

### 2.4 Qwen3-8B/Qwen3-14B Sender--Receiver matrix

A separate model-size experiment fixes Kernel communication at `K=40` and
evaluates all four ordered pairs:

```text
Qwen3-8B  -> Qwen3-8B
Qwen3-8B  -> Qwen3-14B
Qwen3-14B -> Qwen3-8B
Qwen3-14B -> Qwen3-14B
```

The Receiver always sees the original question. For every Receiver model, a
`K=0` Receiver-only baseline is collected once per dataset and generation seed;
it is shared by both possible Sender models. No text-transfer condition is
needed.

The experiment covers all nine datasets exposed by the main runner:

| Dataset key | Split | Repeats | Generation seeds |
| --- | --- | ---: | --- |
| `aime2024` | `train` | 3 | 42--44 |
| `aime2025` | `train` | 3 | 42--44 |
| `arc_challenge` | `test` | 3 | 42--44 |
| `arc_easy` | `test` | 3 | 42--44 |
| `gpqa` (GPQA-Diamond) | `test` | 3 | 42--44 |
| `gsm8k` | `test` | 3 | 42--44 |
| `humanevalplus` | `test` | 3 | 42--44 |
| `mbppplus` | `test` | 3 | 42--44 |
| `medqa` | repository-local training data | 3 | 42--44 |

This gives `9 * 3 = 27` dataset-seed combinations. Task-specific maximum answer tokens
and generation batch sizes follow `params_dict.json`, but the transmitted
Kernel prefix is fixed at `K=40` for comparability.

Cross-model Kernel alignment maps the Sender output space into the Receiver
input-embedding space. Before running a model pair, the implementation must
verify that both Qwen3 tokenizers have identical token-to-row semantics. A
vocabulary size match alone is insufficient; a stable vocabulary mapping hash
must match, otherwise the pair fails explicitly.

### 2.5 Metrics required for every experiment

Every condition that produces Receiver answers records the same three headline
metrics at question level and in its summary:

1. **Accuracy**
   - question row: boolean `correct` plus prediction and gold answer;
   - summary: `processed`, `correct`, and `accuracy=correct/processed`;
   - HumanEval+/MBPP+ use executed tests; other tasks use their normalized
     answer evaluator.
2. **Time**
   - `total_seconds` for the complete logical inference condition;
   - `seconds_per_problem=total_seconds/processed`;
   - component fields for Sender prefix computation, alignment, Receiver
     prefill, Receiver text decoding, and correctness evaluation;
   - model-loading time is excluded, matching the timer placement in `run.py`.
3. **Output tokens / unembedding evaluations**
   - follow the metric semantics of the execution path launched by `run.sh`:
     one output token is one per-sample hidden vector evaluated by the full
     vocabulary readout/unembedding multiplication `h @ W_out.T + b` for
     decoding or alignment;
   - Receiver autoregressive generation therefore contributes one count per
     generated completion ID, including the first generated EOS when present;
   - a latent step contributes one count only when it evaluates the full
     vocabulary output head. Exact Soft contributes one per aligned hidden
     state, ordinary Kernel and Linear contribute zero, and a sampled Kernel
     entropy/readout check contributes one when it is actually executed. This
     must use the same conventions as `models.latent_vocab_decode_steps`;
   - count per evaluated sample, not per batched CUDA kernel launch: multiplying
     a batch of `B` hidden vectors by the unembedding matrix contributes `B`;
   - exclude prompt-prefill positions, input embeddings, KV-cache entries,
     Kernel random-feature operations that avoid `W_out`, and latent steps that
     never perform a full-vocabulary readout;
   - report `total`, `average_per_problem`, and components for Sender
     recurrence, Sender-to-Receiver transfer/alignment, and Receiver decoding.

The canonical summary schema is:

```json
{
  "results": {
    "processed": 0,
    "correct": 0,
    "accuracy": 0.0,
    "output_tokens": {
      "total": 0,
      "average_per_problem": 0.0,
      "components": {
        "sender_recurrence": 0,
        "transfer_alignment": 0,
        "receiver_decode": 0
      }
    }
  },
  "timing": {
    "total_seconds": 0.0,
    "seconds_per_problem": 0.0,
    "components": {
      "sender_prefix_seconds": 0.0,
      "alignment_seconds": 0.0,
      "receiver_prefill_seconds": 0.0,
      "receiver_text_decode_seconds": 0.0,
      "evaluation_seconds": 0.0
    }
  }
}
```

Receiver decode counts come from
`ModelWrapper.last_generation_metrics["output_token_counts"]`, which counts
the completion IDs returned by generation. Latent/alignment counts follow
`models.latent_vocab_decode_steps` and explicit instrumentation at every other
full-vocabulary `W_out` evaluation. Retokenizing decoded text is only a
validation fallback for the Receiver decode component when generation metrics
are unavailable; it can never reconstruct latent/alignment unembedding
evaluations. Every question row stores the three components so that the summary
is their exact sum.

Offline cache-only diagnostics, such as entropy and variance probes, record
their own `probe_unembedding_evaluations` for computational provenance. They do
not alter the linked answer-producing condition's `output_tokens` block.

Caching must not make reported inference time artificially small. Every Sender
question shard stores cumulative measured time at each latent step, allowing a
Receiver condition at prefix `K` to reconstruct its logical Sender time. The
primary `total_seconds` is the sum of logical Sender-prefix time and Receiver
condition time. The actual cache-assisted PBS wall time is stored separately as
`execution_wall_seconds` for operational accounting and is never plotted as
model inference performance.

Pure cache-only analyses such as entropy and alignment variance do not generate
new answers. Their reports must link and reproduce the accuracy, logical time,
and output-token block from the canonical Receiver cache used by that
analysis, with `metric_source="linked_receiver_cache"`; they must not rerun the
Receiver or fabricate condition-specific performance for offline feature/seed
cells.

## 3. Exact trajectory and evaluation counts

### 3.1 Sender hidden trajectories

Sender recurrence is deterministic, so only three dataset-level caches are
collected:

| Cache | Question-level Sender trajectories |
| --- | ---: |
| AIME 2024 | 30 |
| HumanEval+ | 164 |
| ARC-Challenge | all test questions |
| **Dataset-level cache count** | **3** |

Each question is an independently validated cache shard. A failed job resumes
missing question shards rather than recollecting completed trajectories.
Sender shards include cumulative `sender_prefix_seconds` at every stored step,
so timing for any reused prefix can be reconstructed without rerunning Sender.

### 3.2 Kernel-only scaling

Scaling evaluates only Kernel prefixes:

```text
K = 0, 10, 20, 40, 80, 160
```

There are six Receiver conditions for every dataset-generation-seed pair. With
`3 + 3 + 3 = 9` dataset-seed pairs, scaling requires:

```text
9 * 6 = 54 Receiver evaluation cells
```

No linear, soft, identical, or text scaling curves are collected.

### 3.3 Perturbation stability

Stability is evaluated at `K=40`. Source hidden states are perturbed before
alignment:

```text
h_tilde = h + alpha * ||h||_2 / sqrt(d) * epsilon
epsilon ~ Normal(0, I)
alpha in {0, 0.01, 0.05, 0.10}
```

Kernel, soft, and linear receive the same standard-Gaussian direction for a
given dataset, item, role, step, and repeat. Kernel `alpha=0, K=40` is already a
scaling condition. Soft and linear each need one clean `K=40` condition, and all
three methods need three nonzero perturbation conditions:

`text` is not part of perturbation stability: it does not consume a continuous
aligned hidden state at the intervention site, so the same Gaussian perturbation
is not well-defined or directly comparable. `identical` is also excluded from
the formal stability matrix.

```text
2 clean controls + 3 methods * 3 nonzero alpha = 11 cells per dataset-seed
9 * 11 = 99 Receiver evaluation cells
```

The formal total is therefore:

```text
3 Sender trajectory caches
54 Kernel scaling Receiver cells
99 stability Receiver cells
= 153 unique Receiver evaluation cells
```

Entropy and alignment-variance analysis perform no new rollout or generation.

### 3.4 Additional model-size performance matrix

The model-size experiment needs one Sender trajectory per Sender model and
dataset at a usable `K>=40`:

```text
2 Sender models * 9 datasets = 18 logical Sender sources
```

The existing Qwen3-8B `Kmax=160` caches for AIME 2024, HumanEval+, and
ARC-Challenge already provide their `K=40` prefix. Therefore only 15 new Sender
caches are collected:

```text
9 Qwen3-14B dataset caches
+ 6 Qwen3-8B caches for the remaining datasets
= 15 new Sender caches
```

For each of the 27 dataset-seed combinations, Receiver evaluation contains four
ordered Sender/Receiver pairs and two source-independent `K=0` baselines:

```text
27 * (4 model pairs + 2 Receiver-only baselines)
= 162 logical Receiver cells
```

The three primary datasets already contain 9 dataset-seed combinations. Their
Qwen3-8B-to-Qwen3-8B `K=40` results and Qwen3-8B `K=0` baselines are shared with
the primary scaling matrix:

```text
162 logical cells - 9 reused 8B->8B cells - 9 reused 8B baselines
= 144 new Receiver cells
```

Across the primary analyses and the nine-dataset model-size experiment, the
deduplicated formal workload is:

```text
18 total Sender caches (3 existing primary + 15 additional)
297 total Receiver cells (153 primary + 144 additional)
```

## 4. Cache design

### 4.1 Directory layout

```text
analysis_cache/
|-- datasets/
|-- sender_trajectories/
|   `-- <sender_cache_id>/
|       |-- manifest.json
|       |-- questions.parquet
|       `-- states/
|           |-- item_0000.safetensors
|           `-- ...
|-- receiver_evaluations/
|   `-- <receiver_cache_id>/
|       |-- manifest.json
|       `-- answers.parquet
`-- alignment_statistics/
```

Results derived from these caches are written separately:

```text
analysis_result/<task>/<run_id>/
|-- run_manifest.json
|-- metrics/
|-- summaries/
|-- figures/
`-- provenance/
```

### 4.2 Sender cache identity

The Sender cache ID includes:

- dataset name, split, ordered-content fingerprint, and selection policy;
- model ID, resolved revision, and tokenizer fingerprint;
- Sender prompt version;
- recurrence alignment and all Kernel parameters;
- `Kmax`;
- dtype and trajectory schema version.

It does not include Receiver generation seed, Receiver maximum tokens, `K`
prefix, or perturbation strength.

### 4.3 Receiver cache identity

The Receiver cache ID includes:

- referenced Sender manifest hash;
- model and Receiver prompt fingerprint;
- prefix length `K`;
- transfer alignment and its parameters;
- perturbation site, `alpha`, and deterministic noise-key scheme;
- Receiver generation parameters and seed;
- evaluator version.

`alpha=0` is canonicalized to `clean`, so the Kernel stability baseline resolves
to exactly the same cache ID as Kernel scaling at `K=40`.

### 4.4 Integrity and resume rules

- Write question shards and manifests through a temporary path followed by an
  atomic rename.
- Store SHA256, tensor schema, question count, state count, and completion state
  in every manifest.
- Reject incompatible complete caches instead of overwriting them.
- Resume incomplete caches at question granularity.
- Use a lock per cache ID and reject duplicate cache IDs while building the PBS
  matrix.
- A cache hit must result in zero model forwards.

## 5. Package and task interfaces

### 5.1 Proposed layout

```text
analysis/
|-- README.md
|-- plan.md
|-- configs/
|   `-- kernel_analysis.yaml
|-- core/
|   |-- config.py
|   |-- schemas.py
|   |-- datasets.py
|   |-- cache.py
|   |-- sender.py
|   |-- receiver.py
|   |-- interventions.py
|   |-- evaluation.py
|   |-- statistics.py
|   `-- artifacts.py
|-- tasks/
|   |-- collect_sender_trajectories.py
|   |-- evaluate_kernel_scaling.py
|   |-- evaluate_perturbation_stability.py
|   |-- evaluate_sender_receiver_performance.py
|   |-- analyze_logit_entropy.py
|   |-- analyze_kernel_scaling.py
|   |-- analyze_aligned_state_variance.py
|   |-- analyze_perturbation_stability.py
|   |-- analyze_sender_receiver_performance.py
|   `-- build_kernel_analysis_report.py
|-- pbs/
|   |-- analysis_job.pbs
|   |-- build_job_matrix.py
|   `-- submit_analysis.sh
`-- tests/
```

### 5.2 Core Python interfaces

The exact dataclass fields may evolve, but the module boundaries should remain
stable.

```python
# analysis/core/datasets.py
def load_analysis_items(dataset: str, split: str) -> DatasetSnapshot: ...

# analysis/core/schemas.py
def build_role_messages(
    *, role: Literal["planner", "judger"], question: str, task: str, model_name: str
) -> list[dict[str, str]]: ...

# analysis/core/sender.py
def collect_sender_item(
    item: AnalysisItem,
    model: ModelWrapper,
    config: SenderConfig,
) -> SenderItemTrajectory: ...

# analysis/core/cache.py
class SenderTrajectoryStore:
    def resolve(self, identity: SenderCacheIdentity) -> CacheHandle: ...
    def missing_item_ids(self, handle: CacheHandle) -> list[int]: ...
    def write_item(self, handle: CacheHandle, item: SenderItemTrajectory) -> None: ...
    def finalize(self, handle: CacheHandle) -> SenderManifest: ...
    def validate(self, handle: CacheHandle) -> SenderManifest: ...

# analysis/core/receiver.py
def evaluate_receiver_item(
    item: AnalysisItem,
    sender: SenderItemTrajectory,
    config: ReceiverCondition,
    model: ModelWrapper,
) -> ReceiverItemResult: ...

# analysis/core/receiver.py
def validate_cross_model_alignment(
    sender: ModelWrapper,
    receiver: ModelWrapper,
) -> TokenizerCompatibilityReport: ...

# analysis/core/interventions.py
def perturb_source_states(
    hidden: torch.Tensor,
    *,
    alpha: float,
    noise_key: NoiseKey,
) -> tuple[torch.Tensor, PerturbationDiagnostics]: ...

# analysis/core/statistics.py
def paired_question_bootstrap(...): ...
def nested_question_seed_bootstrap(...): ...

# analysis/core/artifacts.py
def summarize_condition_metrics(
    rows: Sequence[ReceiverItemResult],
    *,
    execution_wall_seconds: float,
) -> ConditionMetrics: ...
```

Task entry points accept `--config`, `--job-spec`, `--cache-only`, and
`--force` consistently. Analysis tasks default to strict cache-only mode and
must never fall back to model rollout.

`build_role_messages` is a narrow adapter around
`build_agent_message_sequential_latent_mas`; it must pass exactly
`role="planner"` for Sender and `role="judger"` for Receiver. It may normalize
the args object but may not modify the returned system or user message text.

## 6. Analysis definitions

### 6.1 Probed-logit entropy

Read cached Kernel Sender states and compute:

```text
p_t = softmax(W_out h_t + b)
H_t = -sum_v p_t(v) log p_t(v)
```

Report per dataset:

- entropy curve with question-cluster bootstrap intervals;
- entropy AUC;
- early-to-late entropy change;
- linear and robust slope estimates;
- non-finite and missing-state counts.

This measures whether the Kernel recurrence remains non-degenerate over a long
latent horizon. It does not by itself claim reasoning correctness.

### 6.2 Kernel scalability

Use the cached prefixes at `K=0,10,20,40,80,160` and report:

- accuracy/pass@1 with paired confidence intervals;
- paired changes between adjacent `K` values;
- logical inference time, output-token/unembedding evaluations, and
  latent-prefix length;
- accuracy per second and per latent state;
- the smallest `K` inside the best observed confidence region.

Only Kernel is evaluated across this grid. The analysis establishes whether
additional Kernel latent computation remains useful, saturates, or degrades.

### 6.3 Kernel aligned-state variance

Use the same cached Sender states and evaluate offline:

```text
m in {256, 512, 1024, 2048, 4096}
32 independent Kernel seeds
tau = 0.6
```

For every state, compare `A_kernel(h)` with exact `A_soft(h)` and report:

- mean coordinate variance over Kernel seeds;
- mean squared and relative L2 error;
- cosine similarity;
- error tail probabilities;
- log-log slope of variance against `m`.

PCA may be included as visualization but is not accepted as the primary
variance evidence.

### 6.4 Perturbation stability

At `K=40`, apply paired noise to the cached source hidden states before Kernel,
soft, or linear alignment. Report:

- achieved relative perturbation norm;
- source-to-aligned-state amplification ratio;
- aligned-state cosine and relative L2 change;
- answer flip rate and accuracy degradation;
- entropy change where applicable;
- Kernel-minus-soft and Kernel-minus-linear difference-in-differences;
- paired or nested question/seed bootstrap intervals.

This is a stability analysis of the Sender-to-Receiver communication channel.
It does not test how a perturbation changes future Sender recurrence states.

### 6.5 Sender/Receiver model-size performance

At fixed Kernel `K=40`, report every ordered Qwen3-8B/Qwen3-14B pair separately
for all nine datasets. Primary metrics are:

- accuracy or pass@1 with question-cluster or nested question/seed intervals;
- paired gain over the corresponding Receiver-only `K=0` baseline;
- Sender-size effect at a fixed Receiver:
  `14B->R - 8B->R`;
- Receiver-size effect at a fixed Sender:
  `S->14B - S->8B`;
- heterogeneous-transfer difference relative to the same-size pair with the
  same Receiver;
- Sender-by-Receiver interaction (difference-in-differences);
- Sender time, Receiver time, total output-token/unembedding evaluations with
  their Sender/transfer/Receiver components, and transmitted latent state
  count.

Results are reported per dataset, as an equal-weight nine-dataset macro-average,
and by task family (math, code, multiple choice, and general QA). The analysis
must not pool raw questions across datasets because their sizes differ greatly.

The experiment supports claims about which model size is more useful as the
thinking Sender and which is more useful as the answering Receiver. It does not
attribute a cross-model difference solely to model capacity unless the paired
Receiver baseline and interaction estimates agree.

## 7. PBS interface

### 7.1 Worker template

`analysis/pbs/analysis_job.pbs` is the only PBS worker template. Its PBS
directives, environment setup, working-directory handling, Hugging Face cache
variables, GPU diagnostics, CUDA UUID normalization, and Python launch
conventions must follow the repository-root `run.sh`. It replaces only
`run.sh`'s experiment-suite argument construction with a validated analysis
task name and one JSONL matrix row. When this document and `run.sh` differ on
cluster launch mechanics, `run.sh` is authoritative; analysis-specific array,
cache, validation, logging, and dependency behavior in this section remains
additional and mandatory.

The implementation must use the following template:

```bash
#!/usr/bin/env bash
###############################################################################
# analysis_job.pbs - array worker for standalone Kernel analyses
#
# Example:
#   qsub -J 1-18%3 \
#     -v TASK_NAME=collect_sender_trajectories,JOB_MATRIX=analysis/jobs/sender.jsonl \
#     analysis/pbs/analysis_job.pbs
###############################################################################

#PBS -N kernel_analysis
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

set -euo pipefail

TASK_NAME="${TASK_NAME:-}"
JOB_MATRIX="${JOB_MATRIX:-}"
CONFIG_PATH="${CONFIG_PATH:-analysis/configs/kernel_analysis.yaml}"
DEVICE="${DEVICE:-cuda}"
CACHE_MODE="${CACHE_MODE:-reuse}"
ANALYSIS_CACHE_ROOT="${ANALYSIS_CACHE_ROOT:-analysis_cache}"
ANALYSIS_RESULT_ROOT="${ANALYSIS_RESULT_ROOT:-analysis_result}"
STATE_ROOT="${STATE_ROOT:-state/analysis}"
LEDGER_PATH="${LEDGER_PATH:-${STATE_ROOT}/analysis_jobs.tsv}"

if [[ -z "${PBS_ARRAY_INDEX:-}" ]] ||
   ! [[ "${PBS_ARRAY_INDEX}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PBS_ARRAY_INDEX must be a positive one-based integer." >&2
    exit 2
fi
if [[ -z "${TASK_NAME}" || -z "${JOB_MATRIX}" ]]; then
    echo "ERROR: TASK_NAME and JOB_MATRIX are required." >&2
    exit 2
fi

# A fixed map is safer and clearer than accepting an arbitrary TASK_FILE.
case "${TASK_NAME}" in
    collect_sender_trajectories)
        ENTRY="analysis/tasks/collect_sender_trajectories.py" ;;
    evaluate_kernel_scaling)
        ENTRY="analysis/tasks/evaluate_kernel_scaling.py" ;;
    evaluate_perturbation_stability)
        ENTRY="analysis/tasks/evaluate_perturbation_stability.py" ;;
    evaluate_sender_receiver_performance)
        ENTRY="analysis/tasks/evaluate_sender_receiver_performance.py" ;;
    analyze_logit_entropy)
        ENTRY="analysis/tasks/analyze_logit_entropy.py" ;;
    analyze_kernel_scaling)
        ENTRY="analysis/tasks/analyze_kernel_scaling.py" ;;
    analyze_aligned_state_variance)
        ENTRY="analysis/tasks/analyze_aligned_state_variance.py" ;;
    analyze_perturbation_stability)
        ENTRY="analysis/tasks/analyze_perturbation_stability.py" ;;
    analyze_sender_receiver_performance)
        ENTRY="analysis/tasks/analyze_sender_receiver_performance.py" ;;
    build_kernel_analysis_report)
        ENTRY="analysis/tasks/build_kernel_analysis_report.py" ;;
    *)
        echo "ERROR: unsupported TASK_NAME=${TASK_NAME}" >&2
        exit 2 ;;
esac

case "${CACHE_MODE}" in
    reuse|cache-only|force) ;;
    *)
        echo "ERROR: CACHE_MODE must be reuse, cache-only, or force." >&2
        exit 2 ;;
esac

# Environment setup intentionally mirrors the repository-root run.sh.
module purge
module load python/3.12.13
source /home/n2501945g/LatentMAS-Hybrid/.venv/bin/activate

cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}" || exit 1
REPO_ROOT="$(pwd -P)"
if [[ ! -d analysis || ! -f "${ENTRY}" ]]; then
    echo "ERROR: submit from the repository root; missing ${ENTRY}." >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/home/n2501945g/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# PBS may expose UUIDs while PyTorch expects local integer device indices.
if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "GPU-"; then
    GPU_COUNT="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)"
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
    export CUDA_VISIBLE_DEVICES
fi

if [[ "${DEVICE}" == cuda* ]]; then
    nvidia-smi -L || { echo "ERROR: nvidia-smi failed." >&2; exit 1; }
fi

if [[ ! -f "${JOB_MATRIX}" || ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: missing JOB_MATRIX or CONFIG_PATH." >&2
    exit 2
fi

MATRIX_ABS="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${JOB_MATRIX}")"
CONFIG_ABS="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${CONFIG_PATH}")"
case "${MATRIX_ABS}" in
    "${REPO_ROOT}"/analysis/jobs/*) ;;
    *) echo "ERROR: JOB_MATRIX must be under analysis/jobs/." >&2; exit 2 ;;
esac
case "${CONFIG_ABS}" in
    "${REPO_ROOT}"/analysis/configs/*) ;;
    *) echo "ERROR: CONFIG_PATH must be under analysis/configs/." >&2; exit 2 ;;
esac

ROW_COUNT="$(python3 -c 'from pathlib import Path; import sys; print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))' "${MATRIX_ABS}")"
if (( PBS_ARRAY_INDEX > ROW_COUNT )); then
    echo "ERROR: PBS_ARRAY_INDEX=${PBS_ARRAY_INDEX} exceeds ${ROW_COUNT} rows." >&2
    exit 2
fi

JOB_SLUG="$(printf '%s' "${PBS_JOBID:-local}" | tr -c 'A-Za-z0-9._-' '_')"
STATE_DIR="${STATE_ROOT}/${TASK_NAME}"
STATE_PATH="${STATE_DIR}/${JOB_SLUG}_${PBS_ARRAY_INDEX}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LEDGER_PATH}")"

append_progress() {
    local status="$1"
    local detail="${2//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    (
        flock -x 9
        if [[ ! -s "${LEDGER_PATH}" ]]; then
            printf 'timestamp\tjob_id\tarray_index\ttask\tstatus\tdetail\n' >&9
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date --iso-8601=seconds)" "${PBS_JOBID:-local}" \
            "${PBS_ARRAY_INDEX}" "${TASK_NAME}" "${status}" "${detail}" >&9
    ) 9>> "${LEDGER_PATH}"
}

ARGS=(
    --config "${CONFIG_ABS}"
    --job-matrix "${MATRIX_ABS}"
    --job-index "${PBS_ARRAY_INDEX}"
    --cache-root "${ANALYSIS_CACHE_ROOT}"
    --result-root "${ANALYSIS_RESULT_ROOT}"
    --device "${DEVICE}"
)
case "${CACHE_MODE}" in
    cache-only) ARGS+=(--cache-only) ;;
    force) ARGS+=(--force) ;;
esac

{
    echo "========================================================================"
    echo "PBS job/index : ${PBS_JOBID:-local}/${PBS_ARRAY_INDEX}"
    echo "Task          : ${TASK_NAME}"
    echo "Entry         : ${ENTRY}"
    echo "Matrix        : ${MATRIX_ABS} (${ROW_COUNT} rows)"
    echo "Config        : ${CONFIG_ABS}"
    echo "Cache mode    : ${CACHE_MODE}"
    echo "Host          : $(hostname)"
    echo "Started       : $(date --iso-8601=seconds)"
    echo "========================================================================"
} > "${STATE_PATH}"

append_progress STARTED "state=${STATE_PATH}"
set +e
python3 "${ENTRY}" "${ARGS[@]}" >> "${STATE_PATH}" 2>&1
STATUS=$?
set -e

# Python tasks use 10 for a validated cache hit. PBS still sees success so
# afterok dependencies can proceed.
if (( STATUS == 0 )); then
    append_progress COMPLETED "state=${STATE_PATH}"
elif (( STATUS == 10 )); then
    append_progress SKIPPED "validated cache hit; state=${STATE_PATH}"
    STATUS=0
else
    append_progress FAILED "exit=${STATUS}; state=${STATE_PATH}"
fi

echo "Finished: $(date --iso-8601=seconds); exit=${STATUS}" >> "${STATE_PATH}"
exit "${STATUS}"
```

All task programs must implement the arguments used above. A matrix row is read
by its one-based `--job-index`; blank lines are forbidden by the matrix builder.
Exit code `0` means work completed, `10` means a fully validated cache hit, and
any other code means failure. The worker records `STARTED`, `SKIPPED`,
`COMPLETED`, or `FAILED` in a `flock`-protected TSV ledger.

### 7.2 Submission dependency chain

`analysis/pbs/submit_analysis.sh` submits:

```text
18 deduplicated Sender collection jobs
(3 primary Kmax=160, 15 additional model/dataset K=40)
        |
        | afterok
        v
297 deduplicated Receiver evaluation jobs
(180 primary, 144 additional model-size cells)
        |
        | afterok
        v
cache-only entropy / variance / scaling / stability / model-pair analysis jobs
        |
        | afterok
        v
combined report
```

Expected commands:

```bash
bash analysis/pbs/submit_analysis.sh --stage all
bash analysis/pbs/submit_analysis.sh --stage collect
bash analysis/pbs/submit_analysis.sh --stage evaluate
bash analysis/pbs/submit_analysis.sh --stage analyze
bash analysis/pbs/submit_analysis.sh --stage model-pairs
bash analysis/pbs/submit_analysis.sh --dataset aime2024 --smoke
```

The matrix is generated as JSONL, not as parallel Bash arrays. Matrix generation
must print the number of cells by task and dataset, detect duplicate effective
cache IDs, and support a dry run before calling `qsub`.

The formal job matrices are:

| Matrix | Task name | Rows | Mode |
| --- | --- | ---: | --- |
| `sender.jsonl` | `collect_sender_trajectories` | 18 | reuse |
| `kernel_scaling.jsonl` | `evaluate_kernel_scaling` | 54 | reuse |
| `perturbation.jsonl` | `evaluate_perturbation_stability` | 99 | reuse |
| `model_pairs.jsonl` | `evaluate_sender_receiver_performance` | 144 | reuse |
| `entropy_analysis.jsonl` | `analyze_logit_entropy` | 3 | cache-only |
| `scaling_analysis.jsonl` | `analyze_kernel_scaling` | 3 | cache-only |
| `variance_analysis.jsonl` | `analyze_aligned_state_variance` | 3 | cache-only |
| `stability_analysis.jsonl` | `analyze_perturbation_stability` | 3 | cache-only |
| `model_pair_analysis.jsonl` | `analyze_sender_receiver_performance` | 9 | cache-only |
| `report.jsonl` | `build_kernel_analysis_report` | 1 | cache-only |

The three Receiver evaluation arrays contain `54 + 99 + 144 = 297` unique
cells. Reused model-pair cells are deliberately absent from `model_pairs.jsonl`;
the analyzer resolves them through their canonical scaling cache identities.

`submit_analysis.sh` should use the following submission structure after
`build_job_matrix.py` has generated and validated all matrices:

```bash
#!/usr/bin/env bash
set -euo pipefail

PBS_TEMPLATE="analysis/pbs/analysis_job.pbs"
MAX_RUNNING="${MAX_RUNNING:-3}"
COMMON_VARS="CONFIG_PATH=analysis/configs/kernel_analysis.yaml"

submit_array() {
    local task_name="$1"
    local matrix="$2"
    local rows="$3"
    local cache_mode="$4"
    local dependency="${5:-}"
    local -a command=(
        qsub -J "1-${rows}%${MAX_RUNNING}"
        -v "TASK_NAME=${task_name},JOB_MATRIX=${matrix},CACHE_MODE=${cache_mode},${COMMON_VARS}"
    )
    if [[ -n "${dependency}" ]]; then
        command+=(-W "depend=afterok:${dependency}")
    fi
    command+=("${PBS_TEMPLATE}")
    "${command[@]}"
}

SENDER_JOB="$(submit_array collect_sender_trajectories \
    analysis/jobs/sender.jsonl 18 reuse)"

SCALING_JOB="$(submit_array evaluate_kernel_scaling \
    analysis/jobs/kernel_scaling.jsonl 54 reuse "${SENDER_JOB}")"
PERTURB_JOB="$(submit_array evaluate_perturbation_stability \
    analysis/jobs/perturbation.jsonl 99 reuse "${SENDER_JOB}")"
PAIR_JOB="$(submit_array evaluate_sender_receiver_performance \
    analysis/jobs/model_pairs.jsonl 144 reuse "${SENDER_JOB}")"

ENTROPY_JOB="$(submit_array analyze_logit_entropy \
    analysis/jobs/entropy_analysis.jsonl 3 cache-only "${SCALING_JOB}")"
SCALING_ANALYSIS_JOB="$(submit_array analyze_kernel_scaling \
    analysis/jobs/scaling_analysis.jsonl 3 cache-only "${SCALING_JOB}")"
VARIANCE_JOB="$(submit_array analyze_aligned_state_variance \
    analysis/jobs/variance_analysis.jsonl 3 cache-only "${SCALING_JOB}")"
STABILITY_JOB="$(submit_array analyze_perturbation_stability \
    analysis/jobs/stability_analysis.jsonl 3 cache-only "${PERTURB_JOB}")"
PAIR_ANALYSIS_JOB="$(submit_array analyze_sender_receiver_performance \
    analysis/jobs/model_pair_analysis.jsonl 9 cache-only "${PAIR_JOB}")"

REPORT_DEPENDENCY="${ENTROPY_JOB}:${SCALING_ANALYSIS_JOB}:${VARIANCE_JOB}:${STABILITY_JOB}:${PAIR_ANALYSIS_JOB}"
REPORT_JOB="$(submit_array build_kernel_analysis_report \
    analysis/jobs/report.jsonl 1 cache-only "${REPORT_DEPENDENCY}")"

printf 'sender=%s scaling=%s perturbation=%s model_pairs=%s report=%s\n' \
    "${SENDER_JOB}" "${SCALING_JOB}" "${PERTURB_JOB}" "${PAIR_JOB}" "${REPORT_JOB}"
```

The implementation may add `--stage`, `--dataset`, `--smoke`, and `--dry-run`
filtering around this core, but must not change the task names, matrix row
counts, cache modes, or dependency ordering for the formal `--stage all` run.
If the target PBS installation requires `afterokarray` rather than `afterok`
for array parents, `submit_analysis.sh` may expose that operator as a single
cluster-specific configuration value; it must never remove the dependency.

## 8. Existing code references

These locations may guide implementation. References under `exp/` describe
patterns only and must not become runtime imports from `analysis/`.

### 8.1 Production interfaces that may be reused

| Need | Reference | Intended use |
| --- | --- | --- |
| Model/tokenizer wrapper | `models.py:163` (`ModelWrapper`) | Load the shared Qwen model and tokenizer |
| KV-cache length handling | `models.py:62` (`_past_length`) | Construct attention masks when appending latent embeddings |
| Alignment state lifecycle | `models.py:321`, `models.py:347` | Follow model-pair alignment-state construction and reuse |
| Apply model realignment | `models.py:355` | Reference the production hidden-to-input mapping behavior |
| Kernel state construction | `alignment.py:71` | Build ORF Kernel alignment for the canonical Sender and offline variance grid |
| Exact soft state | `alignment.py:115` | Construct the exact oracle used by the variance analysis |
| Linear state | `alignment.py:150` | Perturbation-control mapping |
| Apply alignment | `alignment.py:268` | Map cached source hidden states into Receiver input space |
| Soft entropy helper | `alignment.py:205` | Check numerical conventions for probability/entropy calculations |
| Planner/Judger prompts | `prompts.py:9` | Call the sequential builder with `role="planner"` for Sender and `role="judger"` for Receiver |
| Single-agent prompt | `prompts.py:697` | Alternative minimal Receiver prompt reference |
| AIME 2024 loader | `data.py:34` | Dataset registry implementation |
| AIME 2025 loader | `data.py:21` | Dataset registry implementation |
| ARC-Easy loader | `data.py:60` | Dataset registry implementation |
| ARC-Challenge loader | `data.py:97` | Dataset registry implementation |
| GPQA-Diamond loader | `data.py:47` | Dataset registry implementation |
| GSM8K loader | `data.py:8` | Dataset registry implementation |
| MBPP+ loader | `data.py:153` | Dataset registry implementation |
| HumanEval+ loader | `data.py:177` | Dataset registry implementation |
| MedQA loader | `data.py:202` | Dataset registry implementation |
| Answer normalization | `utils.py:44` | AIME and ARC normalization |
| Markdown Python extraction | `utils.py:50` | HumanEval+ answer extraction |
| Sandboxed timeout execution | `utils.py:61` | HumanEval+ correctness evaluation |
| Per-role metric schema | `utils.py:82` (`build_agent_metrics`) | Match output-token/unembedding-evaluation and phase-timing field semantics |
| Accuracy aggregation | `run.py:53` | Use `correct/processed` with explicit counts |
| Receiver output fallback count | `run.py:60` | Fallback only for the Receiver decode component when generation metrics are absent |
| Token/timing aggregation | `run.py:69` (`summarize_role_metrics`) | Emit totals and averages per problem |
| Run-level time summary | `run.py:524`--`561` | Match `total_seconds` and `seconds_per_sample` conventions |
| Generation completion counts | `models.py:460`--`464` | Prefer generated completion token IDs via `last_generation_metrics` |
| Latent full-vocabulary readout counts | `models.py:37`--`48`, `models.py:708`--`715` | Count only latent steps that actually multiply by the full unembedding matrix |
| Existing task evaluation branches | `methods/latent_mas.py:299`, `methods/latent_mas.py:316`, `methods/latent_mas.py:332` | Match current task semantics while centralizing them in `analysis/core/evaluation.py` |

### 8.2 Behavior references that must not be imported

| Need | Reference | Pattern to reproduce independently |
| --- | --- | --- |
| Latent recurrence loop | `methods/latent_mas.py:102` and `methods/latent_mas.py:165` | Observe the placement of hidden-state capture relative to feedback |
| Immediately-before-readout hidden state | `models.py:582`, `models.py:674` | Use exactly `outputs.hidden_states[-1][:, -1, :]`, matching the path launched by `run.sh` |
| Hybrid hidden-state collection | `exp/approximator/trajectory.py:253` | Audit which hidden state is causally used for alignment |
| Cache identity construction | `exp/approximator/run.py:396`, `exp/approximator/run.py:483` | Stable readable identity plus manifest payload |
| Cache validation and reuse | `exp/approximator/run.py:565`, `exp/approximator/run.py:615` | Strict validation and cache-hit behavior |
| Atomic cache write | `exp/approximator/run.py:653`--`665` | Temporary file followed by atomic replacement |
| Entropy observer | `exp/latent_cot/mas_analysis.py:285` | Output-head entropy computation and failure recording |
| Paired perturbation CI | `exp/latent_cot/c5_gaussian_robustness.py:341` | Paired question bootstrap shape |
| Two-level repeated-seed CI | `exp/latent_cot/c4_noise_ablation.py:493` | Nested seed/question resampling pattern |
| Perturbation diagnostics | `exp/latent_cot/c5_gaussian_robustness.py:242`--`243` | Relative noise norm and original/perturbed cosine |
| Four ordered 8B/14B pairs | `exp/latent_comm/README.md:3`--`17` | Cross-model matrix and visible-question Receiver semantics |
| Cross-model aligned-message construction | `exp/latent_comm/run.py:533`--`538` | Sender-output to Receiver-input mapping behavior |
| Visible Receiver execution | `exp/latent_comm/run.py:554`--`685` | Prompt visibility audit, embedding injection, and result fields |
| Sender/answer cache separation | `exp/latent_comm/run.py:277`--`324`, `exp/latent_comm/run.py:423`--`465` | Separate reusable Sender and Receiver identities |
| PBS array indexing | `run_all.sh:97`--`148` | Submission/worker split and array-index validation |
| Locked progress ledger | `run_all.sh:208`--`218` | `flock`-protected status records |
| PBS environment and launch | `run.sh` | Authoritative PBS directives, module/venv setup, working directory, Hugging Face caches, diagnostics, CUDA UUID normalization, and Python launch conventions |

When behavior is copied or adapted, it must be moved into a small, tested
interface under `analysis/core/`; tasks should not duplicate it.

## 9. Implementation phases

### Phase 1: contracts and cache

1. Add YAML configuration and strict dataclasses.
2. Add the dataset snapshot/fingerprint layer.
3. Implement Sender and Receiver cache identities.
4. Implement atomic sharded stores, validation, locking, and resume.

### Phase 2: two-agent execution

1. Implement deterministic Kernel Sender recurrence to `Kmax=160`.
2. Store exactly the immediately-before-readout final-layer hidden state
   `outputs.hidden_states[-1][:, -1, :]`, with `h_1` denoting the state after
   the first latent feedback step; never substitute aligned embeddings, logits,
   KV tensors, or intermediate-layer activations.
3. Implement Receiver prompt prefill, aligned-prefix injection, generation, and
   task evaluation.
4. Verify `K=20` output uses exactly the prefix of the cached `K=160` Sender
   trajectory.

### Phase 3: analyses

1. Implement cache-only entropy analysis.
2. Implement Kernel-only scaling aggregation.
3. Implement offline Kernel seed/feature variance analysis with soft oracle.
4. Implement paired perturbation evaluation and comparative stability summary.
5. Implement the four-pair Qwen3-8B/Qwen3-14B performance matrix and paired
   Sender/Receiver capacity contrasts.
6. Implement dataset-specific and macro-average plots and tables.

### Phase 4: PBS and verification

1. Generate an 18-row deduplicated Sender matrix and a 297-row deduplicated
   Receiver matrix, while retaining task-level counts for the primary and
   model-size experiments.
2. Add the PBS template, dependencies, dry run, and progress ledger.
3. Run a smoke matrix with one question per dataset and `Kmax=4`.
4. Run formal jobs only after all smoke tests and cache-reuse checks pass.

## 10. Tests and acceptance criteria

The implementation is accepted only if:

- an AST/import test proves no module under `analysis/` imports `exp`;
- every cached and analyzed hidden state is the final-layer, last-position
  `outputs.hidden_states[-1][:, -1, :]` immediately before `lm_head`, and a
  fake-model sentinel test fails if an embedding, logits, KV tensor, or a
  different layer is captured instead;
- a complete cache hit performs zero model forwards;
- partial collection resumes only missing question shards;
- every Sender manifest proves that its prompt was built with the `planner`
  branch of `prompts.py`, and every Receiver manifest proves use of the
  task-specific `judger` branch;
- the primary three-dataset analysis requires exactly three Qwen3-8B Sender
  cache identities and 153 Receiver identities;
- the combined nine-dataset plan contains exactly 18 Sender cache identities
  and 297 deduplicated Receiver identities;
- the PBS matrices have exact row counts `18/54/99/144/3/3/3/3/9/1`, contain
  no duplicate effective cache IDs, and pass a dry-run/static shell test;
- the PBS worker rejects unknown task names, out-of-range indices, matrices
  outside `analysis/jobs/`, and configs outside `analysis/configs/`;
- the PBS worker preserves the repository-root `run.sh` PBS resource
  directives and environment/launch behavior, with only the documented
  analysis array, cache, validation, logging, and dependency extensions;
- task exit code `10` is recorded as `SKIPPED` while remaining successful for
  PBS dependencies; all other nonzero codes are recorded as `FAILED`;
- Kernel scaling contains no non-Kernel positive-`K` conditions;
- perturbation stability contains exactly `kernel`, `soft`, and `linear`, with
  no `text` or `identical` conditions;
- `alpha=0/kernel/K=40` is shared by scaling and stability;
- the model-size matrix contains all four ordered 8B/14B pairs on all nine
  datasets and exactly two source-independent Receiver baselines per
  dataset-seed;
- the existing three-dataset `8B->8B/K=40` and `8B/K=0` cells are reused rather
  than duplicated;
- cross-model alignment aborts unless the Sender and Receiver tokenizer row
  mappings are semantically identical;
- perturbations are identical across Kernel, soft, and linear for the same noise
  key;
- entropy and variance tasks run in strict cache-only mode;
- corrupt hashes, incompatible model revisions, or dataset changes fail loudly;
- HumanEval+ execution failures are recorded rather than discarded;
- every answer-producing condition contains accuracy, logical
  `total_seconds`, `seconds_per_problem`, output-token total,
  output-token average per problem, and Sender-recurrence/transfer-alignment/
  Receiver-decode components;
- output tokens equal the per-sample number of full-vocabulary
  `h @ W_out.T + b` evaluations used for decoding or alignment: Receiver
  completion IDs count one each (including the first EOS), exact Soft latent
  mappings count one per state, ordinary Kernel/Linear latent mappings count
  zero, and sampled entropy/readout checks count when executed;
- output-token counts exclude prompt-prefill positions, input embeddings, KV
  tensors, and Kernel random-feature operations that do not evaluate the full
  vocabulary; batch size changes do not change per-question counts;
- fake-model metric tests instrument the output head and prove that every
  counted unembedding evaluation is included exactly once and that no
  non-readout latent step is counted;
- cached-prefix timing reconstructs logical Sender cost, while actual PBS wall
  time is separately labeled `execution_wall_seconds`;
- cache-only entropy and variance reports link the canonical Receiver
  performance block and never trigger generation;
- every metric, summary, and figure links to the relevant cache manifest hash;
- unit tests, a fake-model end-to-end test, and the three-dataset PBS smoke test
  pass before formal submission.
