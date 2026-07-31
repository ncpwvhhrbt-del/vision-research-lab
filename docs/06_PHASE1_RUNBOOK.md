# Phase 1 Runbook

This document defines the reproducible baseline workflow. The CIFAR-10 files
remain outside Git; the data config points to the local dataset root.

## Smoke run

Use a few batches to validate the complete pipeline:

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.train `
  --config configs\experiment\baseline_simple_cnn.yaml `
  --epochs 1 `
  --max-train-batches 2 `
  --max-val-batches 1
```

Then evaluate one test batch:

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate `
  --config configs\experiment\baseline_simple_cnn.yaml `
  --max-test-batches 1
```

Expected artifacts under `outputs/baseline_simple_cnn/`:

- `best.pt`: model and optimizer checkpoint
- `history.jsonl`: one record per completed epoch
- `test_report.json`: overall accuracy, per-class accuracy, and confusion matrix

## Full SimpleCNN baseline

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.train `
  --config configs\experiment\baseline_simple_cnn.yaml
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate `
  --config configs\experiment\baseline_simple_cnn.yaml
```

## Full ResNet-18 baseline

```powershell
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.train `
  --config configs\experiment\baseline_resnet18.yaml
D:\Anaconda_envs\envs\torch_clean\python.exe -m scripts.evaluate `
  --config configs\experiment\baseline_resnet18.yaml
```

## Verification contract

Before reporting a baseline, verify:

1. all unit, integration, and smoke tests pass;
2. the experiment configuration is preserved with the result;
3. the checkpoint epoch and metric history are present;
4. the test report is produced only after model selection is complete;
5. training, validation, and test data remain separated.

The smoke result is a pipeline check, not a model-quality claim. Formal
comparisons require fixed epochs, seeds, hardware notes, and complete test-set
evaluation.
