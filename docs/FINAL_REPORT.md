# Vision Research Lab — Final Phase 1–3 Report

## Research question

在固定数据、训练预算和评测协议下，训练策略与数据增强能否同时改善 CIFAR-10 分类准确率、鲁棒性和置信度可靠性？

## Main results

### Architecture baseline

| Model | Parameters | Test accuracy |
|---|---:|---:|
| SimpleCNN | 288,746 | 79.42% |
| ResNet-18 | 11,173,962 | 82.98% |

### Training strategy

| Scheduler | Mean accuracy across 3 seeds |
|---|---:|
| Fixed learning rate | 79.75% ± 0.29% |
| StepLR | 87.63% ± 0.01% |
| CosineAnnealingLR | 88.98% ± 0.37% |

### Data augmentation

| Augmentation | Mean accuracy across 3 seeds |
|---|---:|
| None | 86.46% ± 0.22% |
| Random crop | 88.00% ± 0.08% |
| Random crop + horizontal flip | 88.98% ± 0.37% |

The strongest recorded configuration is adapted ResNet-18 with cosine annealing,
random crop and horizontal flip: **89.35%** on the test set for seed 2024.

## Reliability results

For that strongest checkpoint:

- Clean accuracy: 89.35%;
- Gaussian noise `std=0.05`: 88.50%;
- Gaussian noise `std=0.10`: 83.81%;
- Gaussian noise `std=0.20`: 61.87%;
- ECE before temperature scaling: 0.0255;
- ECE after validation-set temperature scaling: 0.0045;
- Brier Score before scaling: 0.1547;
- Brier Score after scaling: 0.1530;
- Learned temperature: 1.2579.

## Conclusions

1. Learning-rate scheduling and basic augmentation substantially improve generalization under this protocol.
2. The model is reasonably calibrated but mildly overconfident; temperature scaling improves calibration without changing the predicted class.
3. Strong input noise causes a large accuracy drop, so clean-test accuracy alone is not an adequate reliability claim.
4. Results are limited to CIFAR-10, the adapted ResNet-18, this training budget and this perturbation protocol. They are not universal claims about all vision tasks.

## Reproduction

The formal experiments are launched by:

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.sweep_schedulers
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.sweep_augmentations --seeds 42,123,2024
```

Reliability evaluation uses the saved best checkpoint and does not retrain:

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate_robustness `
  --config configs\experiment\cosine_crop_flip.yaml `
  --checkpoint outputs\augmentation_sweeps_formal\crop_flip_seed_2024\best.pt
```

Generated checkpoints, logs and local reports remain under `outputs/` and are
excluded from Git. Public summary tables and selected figures belong under
`benchmarks/` and `reports/`.
