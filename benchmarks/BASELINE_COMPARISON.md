# Baseline Comparison

Both models used the same CIFAR-10 data protocol, 30 epochs, SGD with
learning rate `0.1`, weight decay `0.0005`, seed `42`, and best-validation
checkpoint selection.

| Model | Parameters | Best validation | Test accuracy |
|---|---:|---:|---:|
| SimpleCNN | 288,746 | 80.50% | 79.42% |
| ResNet-18 | 11,173,962 | 84.58% | 82.98% |

ResNet-18 improves test accuracy by 3.56 percentage points, at roughly 38.7x
the parameter count. This is a baseline observation, not yet a causal claim:
the next experiments will control the learning-rate schedule and compare the
accuracy-efficiency trade-off more systematically.
