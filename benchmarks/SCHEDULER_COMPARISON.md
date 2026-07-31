# Learning-Rate Scheduler Comparison

All runs use the same SimpleCNN, CIFAR-10 protocol, SGD optimizer, 30-epoch
budget, and three seeds (`42`, `123`, `2024`). Only the epoch-level learning
rate schedule changes.

| Scheduler | Mean test accuracy | Standard deviation |
|---|---:|---:|
| Fixed learning rate | 79.75% | 0.29% |
| StepLR | 87.63% | 0.01% |
| CosineAnnealingLR | 88.98% | 0.37% |

CosineAnnealingLR is best across all three seeds. Its mean test improvement
over fixed learning rate is 9.22 percentage points; StepLR improves by 7.88
points. The next experiment changes data augmentation while holding the
CosineAnnealingLR training protocol fixed.
