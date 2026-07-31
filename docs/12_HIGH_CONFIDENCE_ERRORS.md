# High-Confidence Error Analysis

## Why it matters

一个低置信度错误可能只是模型犹豫；一个 99% 置信度但预测错误的样本，则说明模型的决策边界或概率表达存在问题。这类样本适合用于：

- 发现类别相似性；
- 检查标签或数据质量；
- 分析分布外样本；
- 研究模型校准失败案例。

## 当前练习

完成 `src/evaluation/high_confidence.py` 中的 `rank_high_confidence_errors`：

1. 对 logits 做 softmax；
2. 找到最大概率及其预测类别；
3. 只保留预测错误的样本；
4. 按置信度从高到低排序；
5. 输出样本索引、真实类别、预测类别和置信度。

测试命令：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m unittest tests.unit.test_high_confidence -v
```

## 完整测试集分析

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.analyze_high_confidence `
  --config configs\experiment\cosine_crop_flip.yaml `
  --checkpoint outputs\augmentation_sweeps_formal\crop_flip_seed_2024\best.pt `
  --top-k 50 `
  --output outputs\augmentation_sweeps_formal\crop_flip_seed_2024\high_confidence_errors.json
```

这里的 `index` 是测试 DataLoader 拼接后的行号。若要展示原始图像，需要进一步把该行号映射回 CIFAR-10 测试集样本。
