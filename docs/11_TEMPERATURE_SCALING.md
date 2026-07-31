# Temperature Scaling

## Purpose

神经网络的 softmax 概率经常过度自信。温度缩放只学习一个标量 `T`：

```text
calibrated_logits = logits / T
probabilities = softmax(calibrated_logits)
```

- `T > 1`：概率分布更平滑，通常降低过度自信；
- `T < 1`：概率分布更尖锐；
- 分类最大值通常不变，因此预测类别和 Accuracy 理论上不变。

## 数据隔离

温度参数必须只在验证集上学习。测试集不能参与温度拟合，否则 ECE 和 Brier Score 会产生数据泄漏。

## 当前练习

完成 `src/evaluation/temperature_scaling.py` 中的 `fit_temperature`：

1. 检查 logits 和 targets 的形状；
2. 用 `log_temperature` 参数保证 `temperature=exp(log_temperature)>0`；
3. 在验证集 logits 上最小化交叉熵；
4. 返回正的 Python `float` 温度值。

完成基础函数后运行：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m unittest tests.unit.test_temperature_scaling -v
```

## 完整评测

使用验证集拟合温度、测试集比较校准前后结果：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate_temperature_scaling `
  --config configs\experiment\cosine_crop_flip.yaml `
  --checkpoint outputs\augmentation_sweeps_formal\crop_flip_seed_2024\best.pt `
  --output outputs\augmentation_sweeps_formal\crop_flip_seed_2024\temperature_scaling_report.json
```

注意：脚本中的 `fit_temperature` 只接收验证集 logits；测试集只用于最终比较。
