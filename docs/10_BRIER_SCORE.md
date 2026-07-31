# Brier Score

## Definition

多分类 Brier Score 衡量整个预测概率向量与真实 one-hot 向量之间的平方误差：

```text
Brier = mean(sum((probability_c - one_hot_c)^2))
```

分数越低越好。与 ECE 不同，Brier Score 不依赖置信度分桶，并且同时考虑所有类别的概率分配。

## 当前练习

完成：

```text
src/evaluation/brier.py::multiclass_brier_score
```

要求：

1. 检查 `probabilities` 为 `[N, C]`，`targets` 为 `[N]`；
2. 检查样本数一致、输入非空；
3. 检查概率在 `[0, 1]`，标签索引合法；
4. 用 `scatter_` 构造 one-hot 标签；
5. 对类别维度求平方误差，再对样本求平均。

测试命令：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m unittest tests.unit.test_brier -v
```

完成后，校准评测报告会同时包含：

- `ece`：分桶校准误差，越低越好；
- `brier_score`：完整概率向量平方误差，越低越好；
- `accuracy`：分类准确率。
