# Data Augmentation Ablation

## Research question

在固定模型、优化器、学习率策略和训练轮数的前提下，训练数据增强是否能改善 CIFAR-10 的泛化能力？

## Protocol

- Model: adapted ResNet-18
- Dataset: CIFAR-10
- Scheduler: cosine annealing
- Epochs: 30
- Seeds: 42, 123, 2024
- Validation split: fixed split from the project data configuration
- Test set: unchanged CIFAR-10 test set
- Reported metric: test accuracy

## Results

| Training augmentation | Mean test accuracy | Std. deviation | Best seed | Best accuracy |
|---|---:|---:|---:|---:|
| None | 86.46% | 0.22% | 123 | 86.73% |
| Random crop | 88.00% | 0.08% | 2024 | 88.10% |
| Random crop + horizontal flip | 88.98% | 0.37% | 2024 | 89.35% |

All nine runs evaluated 10,000 test samples. The raw per-seed records are stored in `augmentation_seed_records.json` and the aggregated records in `augmentation_seed_summary.csv`.

## Interpretation

Relative to no augmentation:

- Random crop improves mean accuracy by **1.54 percentage points**.
- Random crop plus horizontal flip improves mean accuracy by **2.52 percentage points**.
- The combined policy is the strongest configuration in this experiment.

This supports the hypothesis that simple, label-preserving image transformations improve generalization on CIFAR-10. The conclusion is limited to this model, training protocol, and dataset; it is not evidence that every augmentation policy will help every vision task.

The standard deviation is still important: the best combined-policy seed reaches 89.35%, while the worst reaches 88.47%. Therefore, future model comparisons should keep the seed set fixed and report mean and standard deviation rather than comparing only one lucky run.

## Reproduction

From the repository root:

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.sweep_augmentations --seeds 42,123,2024
```

Generated checkpoints and reports belong under `outputs/` and are intentionally excluded from Git commits.
