# Lessons

- 分析缓存必须将确定性 Sender 轨迹与随机 Receiver 生成分开寻址；否则 generation seed 会不必要地重复昂贵的 Sender forward。
- 缓存隐藏态的语义应由捕获点和 tensor schema 同时约束：只允许 latent feedback 后的 `hidden_states[-1][:, -1, :]`，不能从已对齐 embedding 反推。
