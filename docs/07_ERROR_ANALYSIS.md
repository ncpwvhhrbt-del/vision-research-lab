# Error Analysis

本阶段的目标是把“模型准确率是多少”推进到“模型具体错在哪里”。

## 第一层：类别级错误

已有的 `test_report.json` 包含混淆矩阵。矩阵约定为：

- 行：真实类别
- 列：预测类别
- 对角线：预测正确
- 非对角线：分类错误

因此，`matrix[bird, dog]` 表示真实类别是 bird、但被预测为 dog 的样本数量。

## 当前练习

请完成 `src/evaluation/error_analysis.py` 中的 `rank_confusions`：

1. 检查矩阵是否为方阵；
2. 检查类别名称数量是否匹配；
3. 检查 `top_k` 是否为正数；
4. 排除对角线；
5. 按错误数量从大到小排序；
6. 用类别名称替换整数索引；
7. 对相同数量使用真实类别索引、预测类别索引作为稳定的次级排序。

完成后运行：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m unittest tests.unit.test_error_analysis -v
```

这是一个纯 CPU 的分析步骤，不需要重新训练模型。

## 命令行分析

测试通过后，可以直接分析已有的最佳模型报告：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.analyze_errors `
  --report outputs\augmentation_sweeps_formal\crop_flip_seed_2024\test_report.json `
  --top-k 10
```

脚本会在同一目录生成 `error_summary.json`，其中每一项表示：

```text
真实类别 -> 预测类别: 错误数量
```
