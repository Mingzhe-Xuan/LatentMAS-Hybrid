# Planner → Judger STT 伪代码

## 1. 目标与符号

给定同一道题 `problem`：

- sender 模型 `A`（Qwen3-8B） 充当 `planner`；
- receiver 模型 `B`（Mistral-Nemo-Instruct-2407）充当 `judger`；
- 两个模型都显式接收题目，并在各自原生 chat template 中开启 CoT；
- planner 先生成 `sender_plan`；
- `sender_prompt + sender_plan` 的全部有效位置经过 exact soft-token
  transport（STT）；
- aligned sender context 拼在 receiver 原生提示词之前，最后由 judger 自己
  思考并回答。

记：

- `V_A`、`V_B`：sender/receiver tokenizer 词表大小；
- `S_A`、`S_B`：artifact 保存的 source/target active token ID 列表；
- `n_A = |S_A|`、`n_B = |S_B|`：T 两侧 active support 大小；当前正式
  artifact 的 source support 覆盖完整 sender tokenizer，而 target support
  只包含允许参与 transport 的 receiver token；
- `d_B`：receiver hidden size；
- `T ∈ R^(n_B × n_A)`：有向稀疏 transport，布局固定为
  `[target, source]`；
- `E_B ∈ R^(V_B × d_B)`：receiver input embedding table；
- `tau > 0`：通信 softmax 温度；
- `T[j, i]`：source token `i` 向 target token `j` 的 transport 权重。

下面采用 row-vector 记法。对 sender hidden state `h_t`：

```text
p_A_full[t] = softmax(LMHead_A(h_t) / tau)  # [V_A]
p_A[t] = GATHER(p_A_full[t], S_A)            # [n_A]
p_B[t] = p_A[t] @ T^T                       # [n_B]
z_B[t] = p_B[t] @ E_B[S_B]                  # [d_B]
```

`z_B[t]` 是传给 receiver 的连续 embedding，不在模型之间采样或传递离散
target token。

## 2. 顶层算法

```text
ALGORITHM PlannerJudgerSTT(problem, model_A, tokenizer_A,
                            model_B, tokenizer_B,
                            artifact_path, tau,
                            sender_budget, receiver_budget):

    # ---------- A. 加载并验证有向 T ----------
    artifact ← LOAD_NPZ(artifact_path, allow_pickle = false)
    REQUIRE artifact.schema is complete
    REQUIRE artifact.shape == [len(artifact.target_token_ids),
                               len(artifact.source_token_ids)]
    REQUIRE artifact.source_token_ids covers 0 .. len(tokenizer_A)-1
    REQUIRE every artifact.target_token_id is within tokenizer_B vocabulary
    REQUIRE artifact.source_fingerprint == FINGERPRINT(tokenizer_A)
    REQUIRE artifact.target_fingerprint == FINGERPRINT(tokenizer_B)
    REQUIRE artifact direction == A → B
    T ← artifact.CSC_matrix                      # 保持稀疏，禁止 dense 化

    # ---------- B. 同一道题分别构造双角色提示词 ----------
    sender_messages ← analysis.core.schemas.BUILD_ROLE_MESSAGES(
        role = "planner", question = problem, task = dataset,
        model_name = sender_model_id,
    )
    receiver_messages ← analysis.core.schemas.BUILD_ROLE_MESSAGES(
        role = "judger", question = problem, task = dataset,
        model_name = receiver_model_id,
    )

    sender_prompt_text ← analysis.core.schemas.RENDER_ROLE_PROMPT(
        model_A, sender_messages, sender_model_id,
    )
    receiver_prompt_text ← analysis.core.schemas.RENDER_ROLE_PROMPT(
        model_B, receiver_messages, receiver_model_id,
    )

    sender_prompt_ids, sender_prompt_mask ← tokenizer_A.ENCODE(
        sender_prompt_text,
        add_special_tokens = false,
    )
    receiver_prompt_ids, receiver_prompt_mask ← tokenizer_B.ENCODE(
        receiver_prompt_text,
        add_special_tokens = false,
    )

    # ---------- C. planner 显式生成自己的 CoT ----------
    sender_full_ids ← model_A.GENERATE(
        input_ids = sender_prompt_ids,
        attention_mask = sender_prompt_mask,
        do_sample = false,
        max_new_tokens = sender_budget,
        use_cache = true,
    )

    REQUIRE sender_full_ids starts_with sender_prompt_ids
    sender_plan_ids ← sender_full_ids AFTER sender_prompt_ids
    REQUIRE LENGTH(sender_plan_ids) > 0

    sender_full_mask ← CONCAT(
        sender_prompt_mask,
        ONES_LIKE(sender_plan_ids),
    )

    # 这里的 sender_full_ids 正好表示：sender prompt + sender think。
    # 重新 forward 是必要的，因为需要完整 context 每个位置的状态。
    sender_hidden ← model_A.FORWARD(
        input_ids = sender_full_ids,
        attention_mask = sender_full_mask,
        output_hidden_states = true,
        use_cache = false,
        no_grad = true,
    ).last_hidden_state

    # ---------- D. 对完整 sender context 执行 exact STT ----------
    aligned_sender ← EMPTY([batch, sender_full_length, model_B.hidden_size])

    FOR each batch item b:
        FOR each position t WHERE sender_full_mask[b, t] == 1:
            h ← sender_hidden[b, t]                         # [d_A]
            logits_A ← model_A.LM_HEAD(h)                  # [V_A]
            logits_A ← CROP_TRAILING_HARDWARE_PADDING(logits_A,
                                                       len(tokenizer_A))
            p_A_full ← SOFTMAX(logits_A / tau)             # [V_A]
            p_A ← GATHER(p_A_full, artifact.source_token_ids) # [n_A]

            # T 的布局是 [target, source]。
            # row-vector 等价写法：p_B = p_A @ T^T。
            p_B ← SPARSE_RIGHT_MULTIPLY(p_A, TRANSPOSE(T)) # [n_B]
            E_B_active ← GATHER_ROWS(model_B.INPUT_EMBEDDINGS,
                                     artifact.target_token_ids)
            z_B ← p_B @ E_B_active                         # [d_B]

            aligned_sender[b, t] ← z_B

    # 当前 exact 主协议不做 causal shift：每个 sender 有效位置直接对齐。
    REQUIRE causal_shift == false
    aligned_sender ← PACK_VALID(aligned_sender, sender_full_mask)

    # ---------- E. 拼接 receiver 自己的原生提示词 ----------
    receiver_native_embeddings ← model_B.EMBED(receiver_prompt_ids)
    receiver_native_embeddings ← PACK_VALID(receiver_native_embeddings,
                                              receiver_prompt_mask)

    receiver_inputs_embeds ← CONCAT_ALONG_SEQUENCE(
        aligned_sender,                  # aligned sender prompt
                                         # + aligned sender think
        receiver_native_embeddings,      # receiver native prompt，含同一道题
    )

    receiver_attention_mask ← ONES_FOR_EACH_VALID_EMBEDDING(
        receiver_inputs_embeds
    )
    receiver_position_ids ← CUMSUM(receiver_attention_mask) - 1
    receiver_position_ids ← MASK_PADDING_AS_ZERO(receiver_position_ids,
                                                  receiver_attention_mask)

    REQUIRE prefix order == [
        "aligned_sender_prompt",
        "aligned_sender_plan",
        "receiver_native_prompt",
    ]
    REQUIRE receiver never receives sender token IDs

    # ---------- F. judger prefill，并显式执行自己的推理/回答 ----------
    state ← model_B.FORWARD(
        inputs_embeds = receiver_inputs_embeds,
        attention_mask = receiver_attention_mask,
        position_ids = receiver_position_ids,
        use_cache = true,
        no_grad = true,
    )

    next_logits ← state.last_token_logits
    kv_cache ← state.kv_cache
    answer_ids ← EMPTY_SEQUENCE()

    FOR step IN 1 .. receiver_budget:
        next_id ← ARGMAX(next_logits)               # 当前 benchmark 为 greedy
        APPEND(answer_ids, next_id)

        IF next_id IN tokenizer_B.eos_token_ids:
            BREAK

        decode_state ← model_B.FORWARD(
            input_ids = [next_id],
            past_key_values = kv_cache,
            attention_mask = APPEND_ONE(receiver_attention_mask),
            position_ids = [NUMBER_OF_PREVIOUS_VALID_POSITIONS],
            use_cache = true,
            no_grad = true,
        )
        next_logits ← decode_state.last_token_logits
        kv_cache ← decode_state.kv_cache
        receiver_attention_mask ← APPEND_ONE(receiver_attention_mask)

    sender_plan_text ← tokenizer_A.DECODE(sender_plan_ids)
    receiver_answer_text ← tokenizer_B.DECODE(answer_ids)

    RETURN {
        sender_plan: sender_plan_text,
        answer: receiver_answer_text,
        sender_prompt_tokens: COUNT_ONES(sender_prompt_mask),
        sender_plan_tokens: LENGTH(sender_plan_ids),
        aligned_sender_tokens: COUNT_ONES(sender_full_mask),
        receiver_prompt_tokens: COUNT_ONES(receiver_prompt_mask),
        receiver_output_tokens: LENGTH(answer_ids),
    }
```

## 3. 向量化的 exact STT 核心

真实实现不会对每个 token 使用 Python 循环。设某个 query chunk 的 sender
hidden states 为 `H ∈ R^(B × L × d_A)`：

```text
FUNCTION ExactSTTChunk(H, LMHead_A, sparse_T, E_B, tau):
    logits_A ← LMHead_A(H)                    # [B, L, V_A]
    p_A_full ← SOFTMAX(logits_A / tau, axis = -1)
    p_A ← GATHER(p_A_full, S_A)               # [B, L, n_A]

    # T: [n_B, n_A]；row-vector 运算使用 T^T。
    p_B ← SPARSE_MATMUL(p_A, T^T)            # [B, L, n_B]
    Z_B ← p_B @ E_B[S_B]                     # [B, L, d_B]
    RETURN Z_B
```

为控制峰值内存，可以沿 `L` 或 target vocabulary 分块，但每个分块仍必须：

1. 使用完整 source vocabulary softmax；
2. 使用 T 中全部有效稀疏边；
3. 按原位置顺序拼回；
4. 与未分块公式给出相同的 exact 结果。

因此分块是等价内存优化，不是 top-m、hard mapping、ORF 或其他近似。

## 4. Batch 与 padding

```text
FUNCTION PackPlannerAndJudger(aligned_sender, sender_mask,
                               receiver_embeddings, receiver_mask):
    FOR each batch item b:
        sender_valid ← aligned_sender[b][sender_mask[b] == 1]
        receiver_valid ← receiver_embeddings[b][receiver_mask[b] == 1]
        packed[b] ← CONCAT(sender_valid, receiver_valid)

    padded, mask ← RIGHT_PAD_TO_BATCH_MAX(packed, value = 0)
    position_ids ← CUMSUM(mask, axis = sequence) - 1
    position_ids[mask == 0] ← 0
    RETURN padded, mask, position_ids
```

这一步不能直接拼接两侧的物理 padded tensor，否则中间 padding 会进入
receiver context，破坏 prefix 顺序与 KV-cache position。

## 5. 必须保持的不变量

```text
T.shape == [len(target_token_ids), len(source_token_ids)]
source_token_ids covers the complete sender tokenizer vocabulary
all target_token_ids are valid receiver tokenizer IDs
artifact.source_fingerprint == fingerprint(tokenizer_A)
artifact.target_fingerprint == fingerprint(tokenizer_B)
sender_role == "planner"
receiver_role == "judger"
sender_full_context == sender_prompt + sender_plan
causal_shift == false
receiver_prefix == aligned(sender_prompt + sender_plan) + receiver_native_prompt
receiver_native_prompt explicitly contains problem
do_sample == false                       # 当前正式 benchmark
no discrete source IDs enter receiver
no dense materialization of full T
```

任何 fingerprint、方向、shape、prompt prefix 或生成前缀检查失败都应立即停止，
不能通过转置 T、删除校验、丢弃 sender prompt、跳过 receiver 原生题目或降低为
hard token 映射来继续运行。
