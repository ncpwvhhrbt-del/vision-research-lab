# Confidence Calibration

## Research question

当模型说“我有 90% 把握”时，它是否真的大约有 90% 的正确率？

准确率只衡量预测对错，不能衡量概率是否可信。校准评测对自动驾驶、医学辅助和工程决策等场景尤其重要。

## ECE

Expected Calibration Error 将置信度划分为若干区间。对每个非空区间，比较：

- `bin accuracy`：该区间内实际预测正确的比例；
- `bin confidence`：该区间内平均预测置信度。

最后按每个区间的样本比例加权：

```text
ECE = sum(number_in_bin / total * |accuracy - confidence|)
```

ECE 越接近 0，表示置信度越可靠。

## 当前练习

完成 `src/evaluation/calibration.py` 中的 `expected_calibration_error`，然后运行：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m unittest tests.unit.test_calibration -v
```

暂时不需要重新训练模型。下一步会把测试集 logits、最大 softmax 概率和预测正确性接入这个指标。

## 下一步练习：可靠性图数据

ECE 只有一个数，无法告诉我们具体哪个置信度区间出现了偏差。请完成
`src/evaluation/calibration.py` 中的 `calibration_bin_stats`，返回每个非空区间的：

- `bin`：区间编号；
- `count`：样本数量；
- `confidence`：平均置信度；
- `accuracy`：实际准确率。

完成后运行：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m unittest tests.unit.test_calibration_bins -v
```

## 命令行入口

先运行 smoke test：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate_calibration `
  --config configs\experiment\cosine_crop_flip.yaml `
  --checkpoint outputs\augmentation_sweeps_formal\crop_flip_seed_2024\best.pt `
  --max-test-batches 2
```

正式评测：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate_calibration `
  --config configs\experiment\cosine_crop_flip.yaml `
  --checkpoint outputs\augmentation_sweeps_formal\crop_flip_seed_2024\best.pt
```

生成可靠性图：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.plot_calibration `
  --report outputs\augmentation_sweeps_formal\crop_flip_seed_2024\calibration_report.json `
  --output reports\crop_flip_reliability.png
```
