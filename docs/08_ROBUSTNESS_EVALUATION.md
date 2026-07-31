# Robustness Evaluation

## Research question

模型在输入受到轻微随机扰动时，准确率下降多少？数据增强是否让模型对这种扰动更稳定？

## Perturbation protocol

本阶段使用高斯噪声：

```text
perturbed_image = image + std * Normal(0, 1)
```

CIFAR-10 输入已经经过标准化，因此 `std` 的单位是标准化后的张量单位，不是 0 到 255 的原始像素单位。

建议比较：

- `std = 0.00`：干净测试集基线；
- `std = 0.05`：轻微扰动；
- `std = 0.10`：中等扰动；
- `std = 0.20`：较强扰动。

## 当前练习

完成 `src/evaluation/robustness.py` 中的 `add_gaussian_noise`：

1. 拒绝负的 `std`；
2. 使用 `torch.randn_like(images)` 创建同形状、同设备、同数据类型的噪声；
3. 返回 `images + std * noise`。

完成后运行：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m unittest tests.unit.test_robustness -v
```

这一步仍然是 CPU/GPU 推理评测，不需要重新训练模型。

## 命令行入口

完成测试后，用最佳增强模型运行一次 smoke test：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate_robustness `
  --config configs\experiment\cosine_crop_flip.yaml `
  --max-test-batches 2
```

正式评测命令：

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate_robustness `
  --config configs\experiment\cosine_crop_flip.yaml
```
